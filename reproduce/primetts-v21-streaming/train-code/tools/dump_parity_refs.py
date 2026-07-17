#!/usr/bin/env python
"""PyTorch parity reference dump for the ggml-CUDA port of MB-iSTFT-VITS.

Runs SynthesizerTrn.infer on the FIXED (phone,tone,lang) inputs from
parity_inputs.json and saves per-MODULE intermediate tensors as .npy so the
ggml port can check cosine-sim > 0.99 module-by-module, starting with the
riskiest kernel (relative-position attention).

Determinism: model.eval() + noise_scale=0 (z_p == expanded m_p, no RNG), so the
refs are exactly reproducible and the ggml port needs no RNG kernel.

Per utterance NN we dump:
  emb            [1,h,t]     3-embedding sum * sqrt(h), transposed  (encoder input)
  attn{0..5}     [1,h,t]     output of each rel-pos MHA sublayer     <-- riskiest
  enc            [1,h,t]     TextEncoder output
  m_p,logs_p     [1,d,t]     text-side prior stats
  logw,w_ceil    [1,1,t]     deterministic DurationPredictor / length regulator
  m_p_exp,logs_p_exp [1,d,T'] length-regulated prior
  z_p            [1,d,T']    flow input (== m_p_exp at noise_scale=0)
  z_flow         [1,d,T']    flow output (reverse)
  o_mb           [1,sub,Ls]  per-subband iSTFT waveforms (pre-PQMF)
  wav            [1,1,L]     final waveform (post-PQMF synthesis)

Run in .venv (torch 2.10). Decoder PQMF hardcodes .cuda, so a GPU is required;
use GPU1 lightly (does not disturb training):
  CUDA_VISIBLE_DEVICES=1 /home/luigi/jetson-tts/.venv/bin/python tools/dump_parity_refs.py
"""
import os, sys, json, glob
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import commons
from models import SynthesizerTrn

INPUTS = os.environ.get("PARITY_INPUTS", "/home/luigi/mbvits_run/parity_inputs.json")
OUTDIR = os.environ.get("PARITY_OUTDIR", "/home/luigi/mbvits_run/parity_refs")
CONFIG = os.environ.get("PARITY_CONFIG", os.path.join(_ROOT, "configs", "zhtw_mb_istft_16k.json"))
MODEL_DIR = os.path.join(_ROOT, "logs", "zhtw_mbistft_16k")
CKPT_OVERRIDE = os.environ.get("PARITY_CKPT")  # explicit G_*.pth, else latest in MODEL_DIR


def latest_ckpt(model_dir):
    fl = glob.glob(os.path.join(model_dir, "G_*.pth"))
    fl.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    return fl[-1]


def norm(t):
    return float(t.detach().float().norm().cpu())


def main():
    ckpt = CKPT_OVERRIDE or latest_ckpt(MODEL_DIR)
    cfg = json.load(open(CONFIG)); m = cfg["model"]; d = cfg["data"]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert dev.startswith("cuda"), "decoder PQMF requires CUDA; run with CUDA_VISIBLE_DEVICES=1"

    net = SynthesizerTrn(88, d["filter_length"] // 2 + 1, cfg["train"]["segment_size"] // d["hop_length"],
                         **m).to(dev)
    sd = torch.load(ckpt, map_location=dev, weights_only=False)["model"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    net.eval()
    # Streaming refs: the band encoder is baked in at construction via
    # attentions._STREAM_ENC (env MBVITS_STREAM_ENC); the causal decoder is a
    # runtime patch that must be applied here to match training/inference.
    if os.environ.get("MBVITS_STREAM_ENC") == "1":
        from causal_patch import causalize_generator
        causalize_generator(net.dec, verbose=True)
        print("[stream] band encoder (env) + causal decoder applied")
    print(f"[model] loaded {ckpt} on {dev}")

    # hooks: capture each rel-pos MHA sublayer output (conv_o output, [1,h,t])
    attn_out = {}
    def mk(i):
        def hook(mod, inp, out):
            attn_out[i] = out.detach().float().cpu().numpy()
        return hook
    for i, layer in enumerate(net.enc_p.encoder.attn_layers):
        layer.register_forward_hook(mk(i))

    data = json.load(open(INPUTS))["rows"]
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = {"checkpoint": os.path.abspath(ckpt), "noise_scale": 0.0, "length_scale": 1.0,
                "device": dev, "utterances": []}
    hidden = m["hidden_channels"]

    torch.manual_seed(1234)
    for row in data:
        idx = row["idx"]
        x = torch.LongTensor([row["phone_ids"]]).to(dev)
        tone = torch.LongTensor([row["tone_ids"]]).to(dev)
        lang = torch.LongTensor([row["lang_ids"]]).to(dev)
        xl = torch.LongTensor([x.size(1)]).to(dev)
        dumps = {}
        with torch.no_grad():
            # ---- embeddings (encoder input) ----
            import math
            emb = (net.enc_p.emb_phone(x) + net.enc_p.emb_tone(tone) + net.enc_p.emb_lang(lang)) \
                  * math.sqrt(hidden)                      # [1,t,h]
            emb_t = emb.transpose(1, -1)                   # [1,h,t]
            dumps["emb"] = emb_t

            # ---- TextEncoder (hooks capture per-layer attn outputs) ----
            enc, m_p, logs_p, x_mask = net.enc_p(x, tone, lang, xl)
            dumps["enc"] = enc; dumps["m_p"] = m_p; dumps["logs_p"] = logs_p

            # ---- deterministic DurationPredictor + length regulator ----
            logw = net.dp(enc, x_mask, g=None)             # [1,1,t]
            w = torch.exp(logw) * x_mask * 1.0
            w_ceil = torch.ceil(w)
            dumps["logw"] = logw; dumps["w_ceil"] = w_ceil
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)
            m_p_e = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p_e = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
            dumps["m_p_exp"] = m_p_e; dumps["logs_p_exp"] = logs_p_e

            # ---- flow (reverse), noise_scale=0 -> z_p == m_p_e ----
            z_p = m_p_e + torch.randn_like(m_p_e) * torch.exp(logs_p_e) * 0.0
            dumps["z_p"] = z_p
            z = net.flow(z_p, y_mask, g=None, reverse=True)
            dumps["z_flow"] = z

            # ---- decoder (multiband iSTFT + PQMF synthesis) ----
            o, o_mb = net.dec((z * y_mask)[:, :, :None])
            dumps["o_mb"] = o_mb; dumps["wav"] = o

        # attach captured attention outputs
        for i in sorted(attn_out):
            dumps[f"attn{i}"] = torch.from_numpy(attn_out[i])
        attn_out.clear()

        # save + report
        rec = {"idx": idx, "text": row["text"], "T_text": int(x.size(1)),
               "T_frames": int(y_lengths.item()), "modules": {}}
        order = ["emb"] + [f"attn{i}" for i in range(len(net.enc_p.encoder.attn_layers))] + \
                ["enc", "m_p", "logs_p", "logw", "w_ceil", "m_p_exp", "logs_p_exp",
                 "z_p", "z_flow", "o_mb", "wav"]
        for name in order:
            t = dumps[name]
            arr = t.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(t) else t
            np.save(os.path.join(OUTDIR, f"utt{idx:02d}_{name}.npy"), arr)
            rec["modules"][name] = {"shape": list(arr.shape), "norm": round(norm(torch.from_numpy(arr)), 4)}
        manifest["utterances"].append(rec)
        print(f"[utt{idx:02d}] T_text={rec['T_text']:3d} T_frames={rec['T_frames']:4d}  "
              f"enc={rec['modules']['enc']['shape']} wav={rec['modules']['wav']['shape']} "
              f"| attn0_norm={rec['modules']['attn0']['norm']:.3f} wav_norm={rec['modules']['wav']['norm']:.3f}")

    with open(os.path.join(OUTDIR, "parity_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # print module shape/norm table for utt00 (the reference layout)
    print("\n==== utt00 per-module shapes + norms (ggml port checks cosine>0.99 in this order) ====")
    for name, info in manifest["utterances"][0]["modules"].items():
        print(f"  {name:12s} shape={str(info['shape']):22s} L2norm={info['norm']}")
    print(f"\n[write] {OUTDIR}/  ({len(manifest['utterances'])} utts x {len(order)} modules "
          f"= {len(manifest['utterances'])*len(order)} .npy files)")


if __name__ == "__main__":
    main()
