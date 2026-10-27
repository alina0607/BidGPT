#!/bin/zsh
# Brush GPT L's training sequences on the Mac GPU (tools/prepare_l.py; owner, 2026-10-19). A fresh python process every
# 25 cats (MPS memory, as tools/run_repaint_mac.sh); cats whose sketch plan is not ready yet are picked up on a later
# pass, so this keeps going until every repainted AFHQ cat has a sequence.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools
caffeinate -ims -w $$ &
OUT=data/processed/brush_l/cats
echo "prepare L started $(date)"
while true; do
  n=$(ls $OUT 2>/dev/null | grep -c '\.npz$' | grep -v tmp)
  [[ ${n:-0} -ge 5547 ]] && break
  out=$(python3 tools/prepare_l.py --repaint data/processed/repaint/kaggle_t4_run1/out data/processed/repaint/kaggle_t4_run2/out \
        data/processed/repaint/mac --plans data/processed/sketch_plan/afhq --out $OUT --n 25 --device mps 2>&1 | grep '^{' | wc -l)
  echo "$(date '+%H:%M') +$out (have ${n:-0})"
  [[ $out -eq 0 ]] && sleep 300                       # waiting for sketch plans
done
echo "prepare L finished $(date)"
