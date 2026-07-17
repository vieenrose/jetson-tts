# Reproduce PrimeTTS v2.1 — Streaming (causal vocoder + streaming encoder)

This bundle is the **self-contained recipe** for the streaming variant of PrimeTTS v2.1:
an MB-iSTFT-VITS (zh-TW + English, 16 kHz, ~34.7M train / 27.45M infer) whose decoder is
converted to a **causal cached-conv vocoder** (Phase 2) and whose encoder is optionally made
**token-streaming** (Phase 3), for true intra-phrase incremental output on Jetson Nano gen-1.

Everything you need to go from source to a streaming ONNX/GGUF model is here or on the Hub.
The design rationale lives in [`../../docs/streaming-arch-design.md`](../../docs/streaming-arch-design.md);
the ggml/Nano port spec is [`PORT_NOTES.md`](PORT_NOTES.md).

---

## What is already published (don't retrain unless you want to)

Trained weights + exports are on the Hub — **[huggingface.co/Luigi/PrimeTTS](https://huggingface.co/Luigi/PrimeTTS)**:

| Path on HF | What it is |
|---|---|
| `v2_mbistft_16k/primetts_v2_xinran.{onnx,gguf}` | base single-speaker v2 (xinran teacher, CER 0.027) — the warm-start source |
| `v21_mbistft_16k/primetts_v21_3voice.onnx`      | base v2.1 (3-voice multi-speaker) |
| `v21_streaming/{v21_enc,v21_dec}.onnx`          | **v2.1 streaming split** (run enc once, decode per chunk) |
| `v2stream_streaming/{v2stream_enc,v2stream_dec}.onnx` + `.gguf` + `onnx_stream.py` | streaming causal-vocoder model + reference runner |
| `v2streamclean_streaming/…`                     | non-causal "clean" streaming split (16-frame lookahead) |
| `scripts/frontend_bopomofo.py`, `symbol_table.json`, `text_norm.py` | text frontend |
| `docs/streaming-arch-design.md`                 | architecture design |

The one thing **not** on the Hub is the modified **training code** (the causalization /
streaming-encoder logic patched into MB-iSTFT-VITS). That gap is what this bundle fills:
[`mbistft-vits-streaming.patch`](mbistft-vits-streaming.patch) + [`train-code/`](train-code) +
[`launch/`](launch) + [`configs/`](configs).

---

## 0. Prerequisites

- **Env:** `jetson-tts/.venv` — build via [`../../docs/ENV_SETUP.md`](../../docs/ENV_SETUP.md)
  (torch + the MB-iSTFT-VITS deps). GPU **1 only** on this box (GPU 0 is the ASR project → all
  scripts set `CUDA_VISIBLE_DEVICES=1`).
- **Base model repo:** clone upstream and apply the patch:
  ```bash
  git clone https://github.com/MasayaKawamura/MB-iSTFT-VITS.git
  cd MB-iSTFT-VITS && git checkout df2f8d3063f83c22e04d2c0066fa2129d26da9a1
  git apply /path/to/reproduce/primetts-v21-streaming/mbistft-vits-streaming.patch
  cp /path/to/reproduce/primetts-v21-streaming/train-code/*.py .
  cp /path/to/reproduce/primetts-v21-streaming/train-code/tools/*.py tools/
  cp /path/to/reproduce/primetts-v21-streaming/configs/*.json configs/
  # build monotonic_align: cd monotonic_align && python setup.py build_ext --inplace
  ```
- **Frontend:** [`frontend/frontend_bopomofo.py`](frontend/frontend_bopomofo.py) (88 symbols / 6 tones
  / 2 langs) + `symbol_table.json`. Needs **G2PWModel** (download from the G2PW release) for zh
  polyphone disambiguation, plus the zh-TW lexicon overrides in
  [`../../data/tw_readings/`](../../data/tw_readings/) (tracked in this repo).

## 1. Corpus (skip if warm-starting from HF weights)

The base model is trained on the **xinran** single-voice corpus (VoxCPM2 teacher; see the voice
lineage + legality in [`../../docs/voice-provenance/`](../../docs/voice-provenance/) and the
`corpus-multivoice-trap` memory — train on ONE consistent timbre). Corpus is regenerable via the
teacher-dump scripts (`scripts/dump_teacher.py` here + `build_cc0_corpus.py` on the training box).
The streaming finetunes **reuse the same corpus** — no new audio, only phrase-boundary metadata.

## 2. Base v2 / v2.1 (the warm-start source)

```bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 \
  train_latest.py -c configs/zhtw_mb_istft_16k_xinran.json -m zhtw_mbistft_16k_xinran
```
Gate on **resynth CER** (not MCD/spectra — see the `aligner-subsyllable-split-lesson` memory).
Final = `G_400000`, all-set CER **0.027**. (This is `v2_mbistft_16k` / `v21_mbistft_16k` on HF.)

## 3. Phase 2 — streaming causal vocoder (warm-start finetune)

`launch/launch_causal.sh`: `MBVITS_CAUSAL=1` warm-starts `net_g` from the v2 `G/D_400000`,
converts the decoder Conv1d's to **left-causal**, and freezes enc/flow/dp
(`MBVITS_CAUSAL_FREEZE=1`, default). Losses unchanged (mel + adversarial + feature-matching) plus a
chunk-boundary consistency term. Config: `configs/zhtw_mbistft_16k_xinran_causal.json`.
→ checkpoint `keep_causal_36k_{G,D}.pth`.

## 4. Phase 3 — token-streaming encoder (optional)

`launch/launch_streamenc.sh`: `MBVITS_STREAM_ENC=1 MBVITS_ENC_LOOKAHEAD=5` warm-starts from the
causal-vocoder checkpoint; decoder stays frozen+causal, enc/flow/dp train with band-attention +
causal FFN. Config: `configs/zhtw_mbistft_16k_xinran_streamenc.json`.
→ checkpoint `keep_streamenc_10k_{G,D}.pth`. Gate with `launch/gate_streamenc.sh` +
`train-code/causal_gonogo.py` (resynth CER at each step; cold ~1.0 → ~0.075 ≈ baseline at 10k).

## 5. Export + verify

```bash
# split into streaming enc/dec (enc once, dec per chunk)
python tools/export_onnx_stream_split.py --ckpt keep_streamenc_10k_G.pth --out v2stream_split/
# ggml/GGUF for the RapidSpeech.cpp Nano runtime
python tools/convert_mbistft_to_gguf.py keep_streamenc_10k_G.pth primetts_v2stream_xinran_f32.gguf
# parity fixtures
python tools/gen_parity_inputs.py   # -> parity_inputs.json   (CPU, .venv-breezy)
python tools/dump_parity_refs.py    # -> parity_refs/*.npy    (GPU1, .venv)
```

## 6. Streaming inference (reference runner)

[`../../streaming/onnx_stream.py`](../../streaming/onnx_stream.py) is the exact orchestration a
sherpa-onnx / RapidSpeech.cpp C++ runner mirrors: run `enc` once, then decode `z` in **24-frame
chunks via overlap-save** (`LEFT=64, RIGHT=16, HOP=256`), keeping the middle `(b-a)*256` samples.
Validated **bit-exact** vs the monolithic model (cos 1.000000, maxerr ~1e-6). First audio after
enc + one chunk instead of the whole utterance. `streaming/chunker.py` does the upstream
phrase-chunking; `streaming/stream_decoder.py` wraps the decode loop.

```bash
python -m streaming.onnx_stream --enc v2stream_split/v2stream_enc.onnx \
                                --dec v2stream_split/v2stream_dec.onnx --i 0
```

## 7. Nano / ggml deployment

`PORT_NOTES.md` is the full tensor-level spec for the RapidSpeech.cpp `mbistft-vits` graph
(inference subset = 27.45M params / 283 learned + 3 baked DSP tensors, fp32, no int8/fp16 —
matches the A57/Maxwell reality; conv_transpose→zero-stuff+conv1d and iSTFT already ported).
