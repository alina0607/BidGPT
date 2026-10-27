#!/bin/zsh
# The eye material's base with L8 kept 3 px inside the plan's outlines (owner, 2026-10-26; tools/eye_rebase.py): every
# file of eye_l8_rim_hybrid still on the old hybrid base gets the new base and is drawn again (strokes unchanged).
# One process on the CPU (2 threads) next to the Mac repaint (--reverse as an argument: a second one from the other end); a fresh python process every 200 cats. Run again after
# the repaint has finished (tools/run_e8_chain.sh does) to do the cats painted since. Log: brush_l/eye_rebase.log
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTHONPATH=.:tools
caffeinate -ims -w $$ &
OUT=data/processed/brush_l/eye_l8_rim_hybrid
echo "eye rebase started $(date)"
while true; do
  out=$(nice python3 tools/eye_rebase.py --dir $OUT --max-cats 200 --threads 2 $@ 2>&1 | grep '^{' | tail -1)
  echo "$(date '+%m-%d %H:%M') $out"
  [[ "$out" == *'"done"'* ]] && break
  [[ "$out" != *'max_cats'* ]] && { echo "unexpected exit, stopping"; exit 1; }
done
echo "eye rebase finished $(date)"
