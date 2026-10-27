#!/bin/zsh
# Eye model E4 on the Mac GPU (owner, 2026-10-22): the better of E2 / E3 with one change, each stroke's position written
# as distance and direction from its eye's centre (eyegpt_l8 --coords polar). Which material it builds on is the one
# argument (data/processed/brush_l/eye_l8 for E2's, eye_l8_inside for E3's), chosen by the rule fixed before E3's results
# (runs/brushgpt/eye_night_2026-10-22.md). Same cats, split, network and training settings (batch 8 x 2) as E2/E3.
# One GPU job at a time: it stops the CAT repaint's loop, waits for its batch, trains E4, evaluates it as E2/E3 (its
# own material through its own tokens as the upper bound; free L8 samples from E1's saved run), measures it, and
# restarts the CAT repaint. Any step that fails stops the chain. Log: data/processed/brush_l/e4_chain.log
#
#     nohup tools/run_e4_mac.sh data/processed/brush_l/eye_l8_inside >> data/processed/brush_l/e4_chain.log 2>&1 &
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
setopt pipefail
EYES=${1:?usage: run_e4_mac.sh <eye material folder>}
[[ -d $EYES ]] || { echo "no folder $EYES"; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools BRUSHGPT_DEVICE=mps
caffeinate -ims -w $$ &
B=data/processed/brush_l
echo "E4 chain started on $EYES $(date)"
pkill -f "tools/run_repaint_cat_mac.sh" && echo "CAT repaint loop stopped; waiting for its batch to finish $(date)"
while pgrep -f "tools/repaint.py --shards 11-30" >/dev/null; do sleep 60; done
echo "CAT repaint paused $(date)"
echo "building data_E4 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 data --paint split --coords polar --eyes $EYES --out $B/data_E4.npz 2>&1 \
  | grep -v -i warn || exit 1
echo "training E4 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 train --name E4 --paint split --coords polar --batch 8 --accum 2 \
  --data $B/data_E4.npz 2>&1 | grep -v -i warn
[[ -f runs/brushgpt/E4/best.pt ]] || { echo "no E4 checkpoint, stopping $(date)"; exit 1; }
echo "evaluating E4 (real plans) $(date)"
python3 tools/eye_eval_e1.py --run E4 --stage real --eyes $EYES 2>&1 | grep -v -i warn | tail -3 || exit 1
echo "evaluating E4 (free L8 samples, E1's faces) $(date)"
python3 tools/eye_eval_e1.py --run E4 --stage free --n 128 --l8-from runs/brushgpt/E1/eval/free_images.npz 2>&1 \
  | grep -v -i warn | tail -4 || exit 1
echo "measuring E4 $(date)"
python3 tools/eye_measure_e1.py --run E4 2>&1 | grep -v -i warn | tail -1 || exit 1
echo "E4 done; restarting the CAT repaint $(date)"
nohup tools/run_repaint_cat_mac.sh >> data/processed/repaint/mac_cat.log 2>&1 &
echo "E4 chain finished $(date)"
