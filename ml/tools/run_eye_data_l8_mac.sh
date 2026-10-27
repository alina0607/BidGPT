#!/bin/zsh
# The eye model's training data on the Mac CPU (tools/eye_data_l8.py; owner, 2026-10-21): fits the four eye layers on
# every L8 base as the bases appear (tools/run_gen_l8_bases_mac.sh writes them on the GPU), two CPU threads, no GPU.
# Every 10 minutes it picks up the new bases; it ends once all 5,547 bases exist and every cat whose eyes the review
# keeps has its file in data/processed/brush_l/eye_l8/. A new run skips the cats already done.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTHONPATH=.:tools
echo "eye data started $(date)"
while true; do
  bases=$(ls data/processed/brush_l/l8_gen | grep -c '^afhq.*[0-9]\.npz$')
  out=$(python3 tools/eye_data_l8.py --threads 2 2>&1 | grep -v -i warn | tail -1)
  have=$(ls data/processed/brush_l/eye_l8 | grep -c '^afhq.*[0-9]\.npz$')
  echo "$(date '+%H:%M') bases $bases, eye files $have: $out"
  [[ "$out" != *'"done"'* ]] && { echo "unexpected exit, stopping"; exit 1; }
  [[ $bases -ge 5547 ]] && break
  sleep 600
done
echo "eye data finished $(date)"
