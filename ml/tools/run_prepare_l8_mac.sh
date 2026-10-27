#!/bin/zsh
# Brush GPT L8's training sequences on the Mac GPU (tools/prepare_l8.py; owner, 2026-10-20): a fresh python process
# every 50 cats (one process slows from 2 to 5 s a cat by its 200th) (MPS memory, as tools/run_prepare_l_mac.sh), until every L5 sequence has its L8 sequence.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools
caffeinate -ims -w $$ &
OUT=data/processed/brush_l/cats8
echo "prepare L8 started $(date)"
while true; do
  out=$(python3 tools/prepare_l8.py --repaint data/processed/repaint/kaggle_t4_run1/out data/processed/repaint/kaggle_t4_run2/out \
        data/processed/repaint/mac --plans data/processed/sketch_plan/afhq --l5 data/processed/brush_l/cats --out $OUT \
        --n 50 --device mps 2>&1 | grep -c '^{')
  n=$(ls $OUT | grep '\.npz$' | grep -vc tmp)
  echo "$(date '+%H:%M') +$out (have $n)"
  [[ $out -eq 0 ]] && break                           # nothing left (or only failures)
done
echo "prepare L8 finished $(date)"
