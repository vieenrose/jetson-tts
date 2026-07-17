"""Phase-2 go/no-go: how much does causalizing the MB-iSTFT decoder's convs cost,
with NO finetune (warm-start weights unchanged)?

Full-utterance inference (no chunking yet) so the ONLY variable is symmetric->causal
padding. iSTFT/PQMF are frame-local overlap-add and identical here; chunk-seam state
is a separate later test. Measures X-ASR CER: baseline vs causalized decoder.

A small gap => warm-start finetune is cheap (green-light Phase 2). A blow-up => the
causal receptive-field shift is severe and the finetune must travel far (amber).

Run: CUDA_VISIBLE_DEVICES=1 python causal_gonogo.py --n 40
"""
import os, sys, json, argparse, re
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
sys.path.insert(0, "/home/luigi/MB-iSTFT-VITS")
import utils, commons
from models import SynthesizerTrn
from causal_patch import causalize_generator


def synth(net_g, rows, add_blank, dev, out_dir, noise_scale=0.667):
    os.makedirs(out_dir, exist_ok=True)
    man = []
    with torch.no_grad():
        for r in rows:
            p, t, l = list(r["phone_ids"]), list(r["tone_ids"]), list(r["lang_ids"])
            if add_blank:
                p = commons.intersperse(p, 0); t = commons.intersperse(t, 0); l = commons.intersperse(l, 0)
            phone = torch.LongTensor(p).unsqueeze(0).to(dev)
            tone = torch.LongTensor(t).unsqueeze(0).to(dev)
            lang = torch.LongTensor(l).unsqueeze(0).to(dev)
            xlen = torch.LongTensor([phone.size(1)]).to(dev)
            o, *_ = net_g.infer(phone, tone, lang, xlen, noise_scale=noise_scale, length_scale=1.0)
            wav = o.squeeze().cpu().numpy().astype(np.float32)
            wp = os.path.join(out_dir, f"{r['id']}.wav")
            sf.write(wp, wav, 16000)
            man.append({"id": r["id"], "wav": wp, "text": r["text"]})
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", default="/home/luigi/MB-iSTFT-VITS/configs/zhtw_mb_istft_16k_xinran.json")
    ap.add_argument("--ckpt", default="/home/luigi/mbvits_run/xinran_final_ckpt/G_400000.pth")
    ap.add_argument("--jsonl", default="/home/luigi/mbvits_run/val_xinran.jsonl")
    ap.add_argument("--out", default="/home/luigi/mbvits_run/causal_gonogo")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--cpu", action="store_true", help="synth on CPU (uncontended while GPU trains)")
    a = ap.parse_args()

    hps = utils.get_hparams_from_file(a.c); d = hps.data
    add_blank = getattr(d, "add_blank", False); dev = "cpu" if a.cpu else "cuda:0"
    rows = [json.loads(x) for x in open(a.jsonl) if x.strip()][: a.n]
    print(f"[gonogo] {len(rows)} utts, add_blank={add_blank}", flush=True)

    def build():
        g = SynthesizerTrn(88, d.filter_length // 2 + 1,
                           hps.train.segment_size // d.hop_length, **hps.model.__dict__).to(dev)
        sd = torch.load(a.ckpt, map_location=dev); g.load_state_dict(sd["model"]); g.eval()
        return g

    base_g = build()
    base_man = synth(base_g, rows, add_blank, dev, os.path.join(a.out, "baseline"))

    caus_g = build()
    causalize_generator(caus_g.dec)
    caus_man = synth(caus_g, rows, add_blank, dev, os.path.join(a.out, "causal"))

    for name, man in [("baseline", base_man), ("causal", caus_man)]:
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for m in man: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[gonogo] wrote wavs+manifests under {a.out}; score with score_gonogo.py", flush=True)


if __name__ == "__main__":
    main()
