#!/bin/zsh
# The digital eye material on the Mac GPU (owner, 2026-10-24: every cat's eyes repainted in the style of
# runs/eye_layers/digital6/steps_000031.png overnight, Kaggle and the Mac together): tools/eye_data_l8.py --style digital
# --order inside_out --device mps from the last cat backwards, while Kaggle's two T4 go from the first forwards
# (data/kaggle/eye_digital/kernel); a cat painted by both is kept once when merged. The Mac reads the photos directly,
# Kaggle the uint8 pack of tools/eye_pack.py: the same inputs up to 8-bit rounding (at most 1/510 per pixel).
# A fresh python process every 100 cats (the fit's memory grows). Paintings in data/processed/brush_l/eye_l8_digital/.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools
caffeinate -ims -w $$ &
OUT=data/processed/brush_l/eye_l8_digital
echo "mac digital eyes started $(date)"
while true; do
  out=$(python3 tools/eye_data_l8.py --style digital --order inside_out --device mps --reverse --threads 2 \
        --max-cats 100 --out $OUT 2>&1 | grep '^{' | tail -1)
  echo "$(date '+%H:%M') $out (have $(ls $OUT | grep -c '^afhq.*[0-9]\.npz$'))"
  [[ "$out" == *'"done"'* ]] && break
  [[ "$out" != *'max_cats'* ]] && { echo "unexpected exit, stopping"; exit 1; }
done
echo "mac digital eyes finished $(date)"
