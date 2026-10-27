#!/bin/zsh
# Eye model E5 on the Mac GPU (owner, 2026-10-23): E3 with its training cats doubled by left-right mirrors
# (tools/eye_data_mirror.py: data_E3.npz + 4,875 mirrored training cats = data_E5.npz; validation and test unchanged),
# everything else as E3 (split paint, inside-out order, network, batch 8 x 2 and the other training settings).
# One GPU job at a time: it stops the CAT repaint's loop, waits for its batch, trains E5, evaluates it as E3 (E3's
# material as the upper bound; free L8 samples from E1's saved run), measures it, and restarts the CAT repaint.
# Any step that fails stops the chain (and the CAT repaint stays paused). Log: data/processed/brush_l/e5_chain.log
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
setopt pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools BRUSHGPT_DEVICE=mps
caffeinate -ims -w $$ &
B=data/processed/brush_l
echo "E5 chain started $(date)"
[[ -f $B/data_E5.npz ]] || { echo "no $B/data_E5.npz, stopping $(date)"; exit 1; }
pkill -f "tools/run_repaint_cat_mac.sh" && echo "CAT repaint loop stopped; waiting for its batch to finish $(date)"
while pgrep -f "tools/repaint.py --shards 11-30" >/dev/null; do sleep 60; done
echo "CAT repaint paused $(date)"
echo "training E5 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 train --name E5 --paint split --batch 8 --accum 2 --data $B/data_E5.npz 2>&1 \
  | grep -v -i warn
[[ -f runs/brushgpt/E5/best.pt ]] || { echo "no E5 checkpoint, stopping $(date)"; exit 1; }
echo "evaluating E5 (real plans) $(date)"
python3 tools/eye_eval_e1.py --run E5 --stage real --eyes $B/eye_l8_inside 2>&1 | grep -v -i warn | tail -3 || exit 1
echo "evaluating E5 (free L8 samples, E1's faces) $(date)"
python3 tools/eye_eval_e1.py --run E5 --stage free --n 128 --l8-from runs/brushgpt/E1/eval/free_images.npz 2>&1 \
  | grep -v -i warn | tail -4 || exit 1
echo "measuring E5 $(date)"
python3 tools/eye_measure_e1.py --run E5 2>&1 | grep -v -i warn | tail -1 || exit 1
echo "E5 done; restarting the CAT repaint $(date)"
nohup tools/run_repaint_cat_mac.sh >> data/processed/repaint/mac_cat.log 2>&1 &
echo "E5 chain finished $(date)"
