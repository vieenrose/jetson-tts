#!/usr/bin/env python
"""Export a MB-iSTFT-VITS (3-embedding zh-TW/en) G_*.pth checkpoint to GGUF,
keeping ONLY the tensors that run at inference (SynthesizerTrn.infer) and
FOLDING weight_norm (g*v/||v|| -> plain .weight) so the ggml-CUDA kernels see
plain conv/linear weights.

INFERENCE submodules kept:  enc_p (TextEncoder), dp (deterministic
DurationPredictor), flow (ResidualCouplingBlocks, run in reverse), dec
(Multiband_iSTFT_Generator).
TRAINING-only submodules DROPPED:  enc_q (PosteriorEncoder / 16-layer WN),
discriminators (not in G_*.pth anyway), and any StochasticDurationPredictor.

Also bakes the FIXED (non-learned) DSP tensors the decoder needs so the C++
port does not have to re-derive them:
  * pqmf.synthesis_filter  (subbands,1,taps+1)  -> conv1d
  * pqmf.updown_filter     (subbands,subbands,subbands) -> conv_transpose1d
  * istft.window           (hann, win_length=gen_istft_n_fft) -> iSTFT

Usage:
  python tools/convert_mbistft_to_gguf.py \
      --ckpt logs/zhtw_mbistft_16k/G_105000.pth \
      --config configs/zhtw_mb_istft_16k.json \
      --out /home/luigi/mbvits_run/mbistft_zhtw_16k.gguf

Runs on CPU only (no CUDA needed).
"""
import argparse, json, os, sys, glob
import numpy as np
import torch

# repo root on path (for pqmf design fn)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

INCLUDE_PREFIXES = ("enc_p.", "dp.", "flow.", "dec.")
EXCLUDE_PREFIXES = ("enc_q.", "disc", "sdp")
ARCH = "mbistft-vits"


def is_fp16_weight(name, tensor):
    """A tensor eligible for F16 storage: a conv/linear weight matrix.
    Keeps embeddings (.emb_*), norms (.gamma/.beta), biases (.bias), the
    rel-pos tables (emb_rel_*) and the baked pqmf/istft DSP buffers in F32
    (small and/or precision-sensitive)."""
    return (name.endswith(".weight") and tensor.dim() >= 2
            and ".emb_" not in name)


def latest_ckpt(model_dir):
    fl = glob.glob(os.path.join(model_dir, "G_*.pth"))
    if not fl:
        raise FileNotFoundError(f"no G_*.pth under {model_dir}")
    fl.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    return fl[-1]


def fold_weight_norm(sd):
    """Return a new dict where every (X.weight_g, X.weight_v) pair is folded to
    plain X.weight = _weight_norm(v, g, dim=0). All other tensors copied as-is.
    Returns (folded_dict, n_folded)."""
    out = {}
    n_folded = 0
    consumed = set()
    for k in sd:
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            vk = base + ".weight_v"
            assert vk in sd, f"weight_g without weight_v: {k}"
            g = sd[k].float()
            v = sd[vk].float()
            try:
                w = torch._weight_norm(v, g, 0)
            except Exception:
                # manual fallback: norm over all dims except dim 0
                dims = list(range(1, v.dim()))
                norm = v.pow(2).sum(dim=dims, keepdim=True).sqrt()
                w = g * v / norm
            out[base + ".weight"] = w.contiguous()
            consumed.add(k); consumed.add(vk)
            n_folded += 1
    for k in sd:
        if k in consumed:
            continue
        if k.endswith(".weight_v"):  # its _g was handled (or asserted)
            continue
        out[k] = sd[k].float().contiguous()
    return out, n_folded


def design_pqmf_tensors(subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
    """Recompute PQMF synthesis + updown filters (numpy, no CUDA) exactly as
    pqmf.PQMF does, so they can be baked into the GGUF."""
    from pqmf import design_prototype_filter
    h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
    h_synthesis = np.zeros((subbands, len(h_proto)))
    for k in range(subbands):
        h_synthesis[k] = 2 * h_proto * np.cos(
            (2 * k + 1) * (np.pi / (2 * subbands)) *
            (np.arange(taps + 1) - ((taps - 1) / 2)) -
            (-1) ** k * np.pi / 4)
    synthesis_filter = h_synthesis.astype(np.float32)[:, None, :]  # (sub,1,taps+1)
    updown = np.zeros((subbands, subbands, subbands), dtype=np.float32)
    for k in range(subbands):
        updown[k, k, 0] = 1.0
    return synthesis_filter, updown


def istft_hann_window(win_length):
    from scipy.signal import get_window
    return get_window("hann", win_length, fftbins=True).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to G_*.pth (default: latest under --model_dir)")
    ap.add_argument("--model_dir", default=os.path.join(_ROOT, "logs", "zhtw_mbistft_16k"))
    ap.add_argument("--config", default=os.path.join(_ROOT, "configs", "zhtw_mb_istft_16k.json"))
    ap.add_argument("--out", default="/home/luigi/mbvits_run/mbistft_zhtw_16k.gguf")
    ap.add_argument("--vocab", type=int, default=88)
    ap.add_argument("--fp16", action="store_true",
                    help="store the big conv/linear .weight tensors as GGML F16 "
                         "(~half the size); embeddings, norms, biases and the "
                         "baked pqmf/istft DSP buffers stay F32.")
    args = ap.parse_args()

    ckpt = args.ckpt or latest_ckpt(args.model_dir)
    with open(args.config) as f:
        cfg = json.load(f)
    m = cfg["model"]; d = cfg["data"]

    print(f"[load] {ckpt}")
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    iteration = raw.get("iteration") if isinstance(raw, dict) else None
    # strip DDP 'module.' prefix if present
    sd = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() }

    # ---- partition: inference-only tensors ----
    inf = {k: v for k, v in sd.items() if k.startswith(INCLUDE_PREFIXES)}
    dropped = {k for k in sd if not k.startswith(INCLUDE_PREFIXES)}
    n_encq = sum(1 for k in dropped if k.startswith("enc_q."))
    print(f"[split] total ckpt tensors={len(sd)}  kept(pre-fold)={len(inf)}  dropped={len(dropped)} (enc_q={n_encq})")

    # hard guards -- the two things that silently break VITS ports
    for k in inf:
        assert not k.startswith(EXCLUDE_PREFIXES), f"training tensor leaked into inference set: {k}"
    assert not any(".flows." in k for k in inf if k.startswith("dp.")), \
        "dp.*.flows.* present -> this is a StochasticDurationPredictor (use_sdp). Expected deterministic DP."
    assert any(k == "dp.conv_1.weight" for k in inf), "deterministic DurationPredictor tensors missing"

    folded, n_folded = fold_weight_norm(inf)
    # after folding there must be NO weight_g/weight_v left
    assert not any(k.endswith(("weight_g", "weight_v")) for k in folded), "weight_norm not fully folded"
    print(f"[fold] weight_norm pairs folded = {n_folded}  tensors after fold = {len(folded)}")

    # ---- baked fixed DSP tensors ----
    subbands = int(m["subbands"]); istft_nfft = int(m["gen_istft_n_fft"])
    synth, updown = design_pqmf_tensors(subbands=subbands)
    hann = istft_hann_window(istft_nfft)
    derived = {
        "pqmf.synthesis_filter": torch.from_numpy(synth),   # (sub,1,63)
        "pqmf.updown_filter":    torch.from_numpy(updown),  # (sub,sub,sub)
        "istft.window":          torch.from_numpy(hann),    # (n_fft,)
    }
    for k, v in derived.items():
        folded[k] = v.float().contiguous()
    print(f"[derived] baked fixed DSP tensors: {list(derived.keys())}")

    # ---- write GGUF ----
    try:
        from gguf import GGUFWriter
        have_gguf = True
    except Exception as e:
        have_gguf = False
        print(f"[warn] gguf package unavailable ({e}); writing flat binary + manifest instead")

    total_params = sum(v.numel() for v in folded.values())
    total_bytes = sum(v.numel() * (2 if (args.fp16 and is_fp16_weight(k, v)) else 4)
                      for k, v in folded.items())

    manifest = {
        "architecture": ARCH,
        "source_checkpoint": os.path.abspath(ckpt),
        "iteration": iteration,
        "dtype": "F16-weights" if args.fp16 else "F32",
        "n_tensors": len(folded),
        "n_params": int(total_params),
        "bytes_f32": int(total_bytes),
        "weight_norm_pairs_folded": n_folded,
        "excluded_training_modules": {
            "enc_q_tensors": n_encq, "sdp": 0, "discriminators": 0,
        },
        "hparams": {
            "n_vocab": args.vocab, "num_tones": m.get("num_tones", 6), "num_langs": m.get("num_langs", 2),
            "hidden_channels": m["hidden_channels"], "filter_channels": m["filter_channels"],
            "inter_channels": m["inter_channels"], "n_heads": m["n_heads"], "n_layers": m["n_layers"],
            "kernel_size": m["kernel_size"], "window_size": 4,
            "k_channels": m["hidden_channels"] // m["n_heads"],
            "resblock": m["resblock"], "resblock_kernel_sizes": m["resblock_kernel_sizes"],
            "resblock_dilation_sizes": m["resblock_dilation_sizes"],
            "upsample_rates": m["upsample_rates"], "upsample_initial_channel": m["upsample_initial_channel"],
            "upsample_kernel_sizes": m["upsample_kernel_sizes"],
            "gen_istft_n_fft": istft_nfft, "gen_istft_hop_size": m["gen_istft_hop_size"], "subbands": subbands,
            "sampling_rate": d["sampling_rate"], "hop_length": d["hop_length"], "filter_length": d["filter_length"],
            "add_blank": d.get("add_blank", True), "blank_id": 0,
            "flow_n_flows": 4, "flow_wn_layers": 4, "flow_wn_kernel": 5, "flow_wn_dilation_rate": 1,
            "pqmf_taps": 62, "pqmf_cutoff_ratio": 0.15, "pqmf_beta": 9.0,
        },
        "tensors": {k: {"shape": list(v.shape),
                        "dtype": ("F16" if (args.fp16 and is_fp16_weight(k, v)) else "F32")}
                    for k, v in folded.items()},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    man_path = os.path.splitext(args.out)[0] + ".manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[manifest] {man_path}")

    if have_gguf:
        w = GGUFWriter(args.out, ARCH)
        hp = manifest["hparams"]
        w.add_uint32("mbistft.n_vocab", hp["n_vocab"])
        w.add_uint32("mbistft.num_tones", hp["num_tones"])
        w.add_uint32("mbistft.num_langs", hp["num_langs"])
        w.add_uint32("mbistft.hidden_channels", hp["hidden_channels"])
        w.add_uint32("mbistft.filter_channels", hp["filter_channels"])
        w.add_uint32("mbistft.inter_channels", hp["inter_channels"])
        w.add_uint32("mbistft.n_heads", hp["n_heads"])
        w.add_uint32("mbistft.n_layers", hp["n_layers"])
        w.add_uint32("mbistft.kernel_size", hp["kernel_size"])
        w.add_uint32("mbistft.window_size", hp["window_size"])
        w.add_uint32("mbistft.k_channels", hp["k_channels"])
        w.add_uint32("mbistft.subbands", hp["subbands"])
        w.add_uint32("mbistft.gen_istft_n_fft", hp["gen_istft_n_fft"])
        w.add_uint32("mbistft.gen_istft_hop_size", hp["gen_istft_hop_size"])
        w.add_uint32("mbistft.upsample_initial_channel", hp["upsample_initial_channel"])
        w.add_uint32("mbistft.sampling_rate", hp["sampling_rate"])
        w.add_uint32("mbistft.hop_length", hp["hop_length"])
        w.add_array("mbistft.upsample_rates", hp["upsample_rates"])
        w.add_array("mbistft.upsample_kernel_sizes", hp["upsample_kernel_sizes"])
        w.add_array("mbistft.resblock_kernel_sizes", hp["resblock_kernel_sizes"])
        w.add_bool("mbistft.add_blank", bool(hp["add_blank"]))
        w.add_string("mbistft.resblock", str(hp["resblock"]))
        if iteration is not None:
            w.add_uint32("mbistft.iteration", int(iteration))
        n_f16 = 0
        for name, t in folded.items():
            if args.fp16 and is_fp16_weight(name, t):
                arr = np.ascontiguousarray(t.numpy().astype(np.float16))
                n_f16 += 1
            else:
                arr = np.ascontiguousarray(t.numpy().astype(np.float32))
            w.add_tensor(name, arr)  # GGUFWriter infers F16/F32 from arr.dtype
        if args.fp16:
            print(f"[fp16] stored {n_f16}/{len(folded)} conv/linear weights as F16")
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        sz = os.path.getsize(args.out)
        print(f"[gguf] wrote {args.out}  ({sz/1e6:.2f} MB on disk)")
    else:
        # flat binary fallback: little-endian f32 blobs concatenated in manifest order
        bin_path = os.path.splitext(args.out)[0] + ".f32.bin"
        offset = 0
        with open(bin_path, "wb") as f:
            for k, v in folded.items():
                arr = np.ascontiguousarray(v.numpy().astype(np.float32))
                manifest["tensors"][k]["offset"] = offset
                manifest["tensors"][k]["nbytes"] = int(arr.nbytes)
                f.write(arr.tobytes()); offset += arr.nbytes
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[flatbin] wrote {bin_path} ({offset/1e6:.2f} MB) + manifest with offsets")

    # ---- report ----
    print("\n==== EXPORT SUMMARY ====")
    print(f"  source ckpt      : {ckpt} (iteration={iteration})")
    print(f"  exported tensors : {len(folded)}  ({total_params/1e6:.2f}M params, {total_bytes/1e6:.2f} MB f32)")
    print(f"  weight_norm fold : {n_folded} conv/linear layers folded to plain .weight")
    print(f"  EXCLUDED         : enc_q={n_encq} tensors, discriminators=0 (not in G), sdp=0")
    by_mod = {}
    for k in folded:
        top = k.split(".")[0]
        by_mod[top] = by_mod.get(top, 0) + 1
    print(f"  by module        : {by_mod}")
    assert "enc_q" not in by_mod, "enc_q leaked!"
    print("  guards           : no enc_q / no weight_g|weight_v / deterministic DP  OK")


if __name__ == "__main__":
    main()
