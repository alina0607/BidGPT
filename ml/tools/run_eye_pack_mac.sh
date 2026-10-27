#!/bin/zsh
# Packs every cat for the digital eye material (tools/eye_pack.py), four parts in parallel, a fresh python process per
# file of 350 cats (the memory of one process grows by about 10 MB a cat). Ends when every part says done.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTHONPATH=.:tools
OUT=data/kaggle/eye_digital/pack
echo "eye pack started $(date)"
for i in 0 1 2 3; do
  (
    while true; do
      out=$(python3 tools/eye_pack.py --part $i/4 --out $OUT --max-chunks 1 2>&1 | grep '^{' | tail -1)
      [[ "$out" == *'"done"'* ]] && break
      [[ "$out" != *'max_chunks'* ]] && { echo "part $i: unexpected exit: $out"; break; }
    done
  ) &
done
wait
echo "eye pack: $(ls $OUT | grep -c '^pack_.*[0-9]\.npz$') files $(date)"
echo "eye pack finished $(date)"
