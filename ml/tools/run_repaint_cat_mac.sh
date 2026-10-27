#!/bin/zsh
# Mac GPU share of the CAT-dataset repaint (owner, 2026-10-20): shards 11-30 from the last cat backwards, while Colab
# (tools/prepare_colab_repaint.py) goes forwards from shard 11; the two meet in the middle (stop this when they do;
# tools/repaint_merge.py keeps one copy of a cat painted twice). Same loop as tools/run_repaint_mac.sh: at most 25 cats
# per python process (MPS memory). Paintings in data/processed/repaint/mac_cat/.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools
caffeinate -ims -w $$ &
echo "mac CAT repaint started $(date)"
while true; do
  out=$(python3 tools/repaint.py --shards 11-30 --reverse --reverse-cats --source cat --device mps --max-cats 25 \
        --out data/processed/repaint/mac_cat 2>&1 | grep -v -i warn | tail -1)
  echo "$(date '+%H:%M') $out"
  [[ "$out" == *'"done"'* ]] && break
  [[ "$out" != *'max_cats'* ]] && { echo "unexpected exit, stopping"; exit 1; }
done
echo "mac CAT repaint finished $(date)"
