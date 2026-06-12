# third_party

- `sherpa_melo_export.py` — reference MeloTTS→ONNX export script from
  [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (`scripts/melo-tts/export-onnx.py`),
  Apache-2.0. Used to match the exact drop-in I/O contract. Reused verbatim by
  `scripts/export_full_onnx.py`.
- `MeloTTS/` (gitignored) — cloned from https://github.com/myshell-ai/MeloTTS (MIT) and patched
  for zh/en-only lazy imports via `scripts/melo_zh_en_lazyimport.patch`.
