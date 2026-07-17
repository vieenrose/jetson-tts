#!/bin/bash
# Phase-3: token-level streaming ENCODER finetune (band attn + causal FFN), warm-started
# from the causal-vocoder ckpt. GPU 1 ONLY. dec frozen+causal; enc/flow/dp train.
set -euo pipefail
NAME=zhtw_mbistft_16k_xinran_streamenc
REPO=/home/luigi/MB-iSTFT-VITS
PY=/home/luigi/jetson-tts/.venv/bin/python
LOG=/home/luigi/mbvits_run/train_${NAME}.log
cd "$REPO"
setsid env CUDA_VISIBLE_DEVICES=1 MBVITS_STREAM_ENC=1 MBVITS_ENC_LOOKAHEAD=5 \
  "$PY" -m torch.distributed.run --nproc_per_node=1 \
  train_latest.py -c "configs/${NAME}.json" -m "$NAME" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched $NAME (pid $!) -> tail -f $LOG"
