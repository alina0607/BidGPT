#!/bin/zsh
# The eye model's bases on the Mac GPU (tools/gen_l8_bases.py; owner, 2026-10-21): a fresh python process every 4
# chunks of 64 cats (MPS memory), until every cat of data_L8.npz has its L8 painting. The CAT repaint is paused meanwhile
# (one GPU job at a time); restart it afterwards with tools/run_repaint_cat_mac.sh.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools
caffeinate -ims -w $$ &
echo "L8 bases started $(date)"
while true; do
  out=$(python3 tools/gen_l8_bases.py --device mps --max-chunks 4 2>&1 | grep '^{' | tail -1)
  echo "$(date '+%H:%M') $out (have $(ls data/processed/brush_l/l8_gen | grep -c '^afhq.*[0-9]\.npz$'))"
  [[ "$out" == *'"done"'* ]] && break
  [[ "$out" != *'max_chunks'* ]] && { echo "unexpected exit, stopping"; exit 1; }
done
echo "L8 bases finished $(date)"
