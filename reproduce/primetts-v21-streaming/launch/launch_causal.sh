#!/bin/bash
# Streaming Phase-2: causal vocoder finetune, warm-started from v2 G/D_400000.
# GPU 1 ONLY (GPU 0 belongs to the ASR project). Single-GPU DDP.
# warmstart_{G,D}.pth already staged in logs/$NAME by setup; MBVITS_CAUSAL=1 loads
# them, causalizes dec, and freezes enc/flow/dp (MBVITS_CAUSAL_FREEZE=1 default).
set -euo pipefail
NAME="zhtw_mbistft_16k_xinran_causal"
REPO=/home/luigi/MB-iSTFT-VITS
PY=/home/luigi/jetson-tts/.venv/bin/python
LOG=/home/luigi/mbvits_run/train_${NAME}.log
cd "$REPO"

setsid env CUDA_VISIBLE_DEVICES=1 MBVITS_CAUSAL=1 "$PY" -m torch.distributed.run --nproc_per_node=1 \
  train_latest.py -c "configs/${NAME}.json" -m "$NAME" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched $NAME (pid $!)  ->  tail -f $LOG"
echo "look for: '[causal] warm-started net_g', '[causal] N Conv1d ... left-causal', '[causal] froze'"
