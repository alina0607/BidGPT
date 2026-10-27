#!/bin/zsh
# Eye material for E3 on the Mac CPU (owner, 2026-10-22: the eye model's next change is the stroke order): every cat of
# data/processed/brush_l/eye_l8/ fitted again with tools/eye_data_l8.py --order inside_out (iris, pupil and highlight
# from each eye's centre outwards; the outline still around), into data/processed/brush_l/eye_l8_inside/. Three
# interleaved parts of two threads each, no GPU (E2 trains on it meanwhile). Each part starts a fresh python process
# every 100 cats: one process grew to 5 GB after about 650 cats (attempt 1), next to E2's training on a 34 GB Mac.
# A new run skips the cats already done.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTHONPATH=.:tools
OUT=data/processed/brush_l/eye_l8_inside
mkdir -p $OUT
echo "E3 eye material started $(date)"
for i in 0 1 2; do
  (
    while true; do
      out=$(python3 tools/eye_data_l8.py --order inside_out --threads 2 --part $i/3 --max-cats 100 --out $OUT 2>&1 \
            | grep -v -i warn | tail -1)
      [[ "$out" == *'"done"'* ]] && break
      [[ "$out" != *'max_cats'* ]] && { echo "part $i: $out"; break; }
    done
  ) &
done
wait
have=$(ls $OUT | grep -c '^afhq.*[0-9]\.npz$')
want=$(ls data/processed/brush_l/eye_l8 | grep -c '^afhq.*[0-9]\.npz$')
echo "E3 eye material: $have of $want cats $(date)"
[[ $have -ge $want ]] && echo "E3 eye material finished $(date)" || echo "E3 eye material incomplete, unexpected exit $(date)"
