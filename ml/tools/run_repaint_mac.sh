#!/bin/zsh
# Mac GPU share of the AFHQ repaint (owner, 2026-10-18): shards 10-11 (AFHQ part), then shards 9 and 4 from their
# last cat backwards, meeting the two Kaggle T4s (which go forwards through 0-4 and 5-9) in the middle.
# (Shards 10-11 done 2026-10-19.)
# Each python process paints at most 25 cats and exits, and this loop starts the next: a single long process on MPS
# kept memory it did not give back (7 GB and 28 GB of swap after about 100 cats; 1,500 s a cat instead of 35).
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools
caffeinate -ims -w $$ &
echo "mac repaint started $(date)"
run_until_done() {
  while true; do
    out=$(python3 tools/repaint.py "$@" --device mps --max-cats 25 --out data/processed/repaint/mac 2>&1 | grep -v -i warn | tail -1)
    echo "$(date '+%H:%M') $out"
    [[ "$out" == *'"done"'* ]] && break
    [[ "$out" != *'max_cats'* ]] && { echo "unexpected exit, stopping"; exit 1; }
  done
}
# 2026-10-19: after Kaggle's first session, the Mac paints only its share of the cats still missing
# (data/processed/repaint/todo_mac.txt, tools/repaint_merge.py); Kaggle's second session paints the rest.
# 13:00: Kaggle's second session waited over an hour for two T4s (nothing else running, quota 13 of 30 h used), so
# the Mac also takes Kaggle's 627 cats, from the far end: Kaggle goes forwards (3, 4, 8), the Mac backwards (9, 8, 4, 3).
run_until_done --shards 3,4,8,9 --reverse --source afhq --reverse-cats --only data/processed/repaint/todo_mac_all.txt
echo "mac repaint finished $(date)"
