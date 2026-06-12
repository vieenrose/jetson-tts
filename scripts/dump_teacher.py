#!/usr/bin/env python3
"""Dump (z, g, 8kHz-wav) distillation pairs from the MeloTTS zh_en teacher.

For each corpus utterance we run the SAME model.infer() forward and capture:
  - z_dec = (z * y_mask)   [192, T']     (exact decoder input)
  - g     = emb_g(sid)     [256, 1]      (constant: single speaker -> saved once)
  - o     = dec(z_dec, g)  44.1 kHz waveform, then soxr-resampled to 8 kHz mono
z and o come from one forward pass so they stay paired through z_p's noise sampling.

Sharding: --shard k --num-shards N processes utterances with index%N==k, so you can run
one process per GPU. Resumable: skips ids whose .z.npy + .wav already exist.
"""
import argparse, os, sys, json, numpy as np, torch, soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from student.audio_config import TEACHER_SR, TARGET_SR, Z_CHANNELS, G_CHANNELS

import librosa  # soxr resampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/text/corpus.tsv")
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--noise-scale", type=float, default=0.6)
    ap.add_argument("--noise-scale-w", type=float, default=0.8)
    ap.add_argument("--sdp-ratio", type=float, default=0.2)
    ap.add_argument("--z-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--zero-bert", action="store_true",
                    help="zero bert/ja_bert to match the sherpa-onnx deployed model (REQUIRED "
                         "for a faithful drop-in: the stock binary never computes BERT)")
    ap.add_argument("--keep-wav44k", action="store_true", help="also save teacher 44.1k wav")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap utterances")
    args = ap.parse_args()

    from melo.api import TTS
    import re
    model = TTS(language="ZH", device=args.device)
    hps = model.hps
    assert hps.data.sampling_rate == TEACHER_SR
    spk = list(hps.data.spk2id.values())[0]
    sub = f"shard{args.shard:02d}"
    outdir = os.path.join(args.out, sub)
    os.makedirs(outdir, exist_ok=True)

    rows = [l.rstrip("\n").split("\t", 1) for l in open(args.corpus, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if len(r) == 2]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]

    manifest_path = os.path.join(outdir, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest_path):
        for l in open(manifest_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass

    g_saved = os.path.exists(os.path.join(args.out, "g.npy"))
    mf = open(manifest_path, "a", encoding="utf-8")
    np_dtype = np.float16 if args.z_dtype == "float16" else np.float32
    n_ok, n_frames_tot = 0, 0

    for idx, (uid, text) in enumerate(rows):
        pieces = model.split_sentences_into_pieces(text, model.language, quiet=True)
        for pj, t in enumerate(pieces):
            sid_uid = uid if len(pieces) == 1 else f"{uid}_p{pj}"
            if sid_uid in done:
                continue
            if model.language in ["EN", "ZH_MIX_EN"]:
                t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            try:
                bert, ja_bert, phones, tones, lang_ids = \
                    __import__("melo.utils", fromlist=["x"]).get_text_for_tts_infer(
                        t, model.language, hps, args.device, model.symbol_to_id)
                with torch.no_grad():
                    x = phones.to(args.device).unsqueeze(0)
                    tn = tones.to(args.device).unsqueeze(0)
                    lid = lang_ids.to(args.device).unsqueeze(0)
                    be = bert.to(args.device).unsqueeze(0)
                    jb = ja_bert.to(args.device).unsqueeze(0)
                    if args.zero_bert:        # match deployed sherpa-onnx model (bert=0)
                        be = torch.zeros_like(be)
                        jb = torch.zeros_like(jb)
                    xl = torch.LongTensor([phones.size(0)]).to(args.device)
                    sp = torch.LongTensor([spk]).to(args.device)
                    o, attn, y_mask, (z, z_p, m_p, logs_p) = model.model.infer(
                        x, xl, sp, tn, lid, be, jb,
                        sdp_ratio=args.sdp_ratio, noise_scale=args.noise_scale,
                        noise_scale_w=args.noise_scale_w, length_scale=1.0)
                    z_dec = (z * y_mask)                       # [1,192,T']
                    g = model.model.emb_g(sp).unsqueeze(-1)    # [1,256,1]
                    wav44 = o[0, 0].detach().cpu().float().numpy()
                    z_np = z_dec[0].detach().cpu().float().numpy()  # [192,T']
            except Exception as e:
                print(f"[skip {sid_uid}] {type(e).__name__}: {e}", file=sys.stderr)
                continue

            if not g_saved:
                np.save(os.path.join(args.out, "g.npy"),
                        g[0, :, 0].detach().cpu().float().numpy())  # [256]
                g_saved = True

            wav8 = librosa.resample(wav44, orig_sr=TEACHER_SR, target_sr=TARGET_SR,
                                    res_type="soxr_hq").astype(np.float32)
            wav8 = np.clip(wav8, -1.0, 1.0)
            Tp = z_np.shape[1]
            np.save(os.path.join(outdir, f"{sid_uid}.z.npy"), z_np.astype(np_dtype))
            sf.write(os.path.join(outdir, f"{sid_uid}.wav"), wav8, TARGET_SR, subtype="PCM_16")
            if args.keep_wav44k:
                sf.write(os.path.join(outdir, f"{sid_uid}.src44k.wav"), wav44, TEACHER_SR, subtype="PCM_16")
            mf.write(json.dumps({"id": sid_uid, "text": t, "z_frames": Tp,
                                 "n8k": int(len(wav8)), "dur": len(wav8) / TARGET_SR},
                                ensure_ascii=False) + "\n")
            n_ok += 1
            n_frames_tot += Tp
            if n_ok % 200 == 0:
                mf.flush()
                print(f"[{sub}] {n_ok} done, ~{n_frames_tot/86.1328/3600:.2f} h", flush=True)

    mf.close()
    print(f"[{sub}] DONE {n_ok} pairs, ~{n_frames_tot/86.1328/3600:.3f} h audio")


if __name__ == "__main__":
    main()
