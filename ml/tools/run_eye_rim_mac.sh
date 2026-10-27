#!/bin/zsh
# The eye material for training (owner, 2026-10-25): every cat repainted on the Mac GPU with the hybrid base (the
# whole-face repaint outside the plan's eye outlines, L8 inside: eye_data_l8.hybrid_base) and the digital painter with
# the new rim (eye_layers "digital_rim_long": a dark, smooth lid line along the photo's rim, fibres kept off it). Kaggle's
# quota is kept for training. One GPU job at a time: the CAT repaint's loop is stopped and its batch allowed to finish
# first. A fresh python process every 25 cats (the fit's memory grows: at 100 cats two processes reached 35 GB each and swapped, 2026-10-26). Out: data/processed/brush_l/eye_l8_rim_hybrid/.
# Do not edit tools/eye_data_l8.py, eye_layers.py, eye_layers_l.py, eye_crop_styles.py or repaint.py while this runs:
# each new process reads them again.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools
caffeinate -ims -w $$ &
OUT=data/processed/brush_l/eye_l8_rim_hybrid
echo "eye rim hybrid started $(date)"
pkill -f "tools/run_repaint_cat_mac.sh" && echo "CAT repaint loop stopped; waiting for its batch $(date)"
while pgrep -f "tools/repaint.py --shards 11-30" >/dev/null; do sleep 30; done
echo "CAT repaint paused $(date)"
while true; do
  out=$(python3 tools/eye_data_l8.py --style digital_rim_long --order inside_out --base hybrid --device mps --threads 2 \
        --max-cats 25 --out $OUT 2>&1 | grep '^{' | tail -1)
  echo "$(date '+%m-%d %H:%M') $out (have $(ls $OUT | grep -c '^afhq.*[0-9]\.npz$'))"
  [[ "$out" == *'"done"'* ]] && break
  [[ "$out" != *'max_cats'* ]] && { echo "unexpected exit, stopping"; exit 1; }
done
echo "eye rim hybrid finished $(date)"
