# Environment setup (training host, reproducible)

Host: dual RTX 5090 (sm_120), Ryzen 9 9950X3D (32 threads), 249 GB RAM, Ubuntu, py3.12.
Target (NOT this host): Jetson Nano gen1, 4× Cortex-A57, ONNX Runtime CPU via sherpa-onnx.

## Why an isolated venv
System python has torch 2.10+cu128 (sm_120 OK) but numpy 2.4.3, which conflicts with
MeloTTS's older audio stack. We use a dedicated venv with its own cu128 torch + numpy<2.

## Steps
```bash
cd /home/luigi/jetson-tts
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install "numpy<2"
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# MeloTTS deps, zh/en only (no JP/KR/FR/ES g2p backends), transformers bumped for py3.12 wheels:
pip install -r requirements-melo.txt          # transformers==4.44.2, librosa 0.10.x, ...
git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git third_party/MeloTTS
git -C third_party/MeloTTS apply ../../scripts/melo_zh_en_lazyimport.patch   # see below
pip install -e third_party/MeloTTS --no-deps
```
Exact pinned set: `requirements-freeze.txt`.

## MeloTTS patch (`scripts/melo_zh_en_lazyimport.patch`)
Stock MeloTTS imports ALL language g2p backends at module load (JP MeCab/pykakasi/fugashi,
KR jamo, FR/ES gruut). zh_en needs none of them. The patch makes those imports lazy WITHOUT
changing zh/en behaviour:
- `text/cleaner.py`: language module map → lazy per-language import.
- `text/__init__.py:get_bert`: import only the requested language's BERT backend.
- `text/english.py`: inline the tiny `distribute_phone` helper instead of importing japanese.py.
- `text/japanese.py`: defer MeCab/pykakasi/JP-BERT-tokenizer to first actual JP use.

## Teacher facts confirmed at runtime
- `TTS(language='ZH')` → internal `ZH_MIX_EN`, single speaker (spk2id `{'ZH':1}`), sr 44100.
- BERT for ZH_MIX_EN feature = `bert-base-multilingual-uncased` (auto-downloaded to HF cache).
- Checkpoint `checkpoint.pth` + `config.json` auto-downloaded from `myshell-ai/MeloTTS-Chinese`.
- Smoke test: 「幫您轉接給 Kevin 陳經理,他的分機是 533。」 → 4.41 s @ 44.1 kHz, clean. ✓

## ONNX Runtime (export/bench target)
Device sherpa-onnx statically links a pinned ORT (~1.17). Install ORT **1.17.x** here for
host benchmarking/compat checks; export at **opset 17**. (Installed separately in the eval step.)
