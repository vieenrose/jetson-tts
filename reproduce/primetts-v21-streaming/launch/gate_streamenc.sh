#!/bin/bash
set -uo pipefail
REPO=/home/luigi/MB-iSTFT-VITS
CK=$REPO/logs/zhtw_mbistft_16k_xinran_streamenc
OUT=/home/luigi/mbvits_run/se_trace; RES=/home/luigi/mbvits_run/se_trace.tsv
LOG=/home/luigi/mbvits_run/train_zhtw_mbistft_16k_xinran_streamenc.log
mkdir -p "$OUT"; : > "$RES"; cd /home/luigi/jetson-tts
for STEP in 2000 5000 10000 20000 35000 50000 70000 90000; do
  G=$CK/G_${STEP}.pth
  until [ -f "$G" ] || grep -qE "Traceback|Killed" "$LOG"; do sleep 45; done
  [ -f "$G" ] || { echo "died before $STEP">>"$RES"; break; }
  sleep 5
  ( cd "$REPO" && MBVITS_STREAM_ENC=1 MBVITS_ENC_LOOKAHEAD=5 /home/luigi/moss-nano-venv/bin/python \
      causal_gonogo.py --cpu --n 40 --ckpt "$G" --out "$OUT/s$STEP" ) >/dev/null 2>&1
  L=$(/home/luigi/moss-nano-venv/bin/python mossnano/zhtw8k/score_gonogo.py "$OUT/s$STEP" 2>/dev/null | grep "^causal")
  echo -e "${STEP}\t${L}" | tee -a "$RES"
done
echo "SE-TRACE-DONE" | tee -a "$RES"
