#!/bin/zsh
# Eye model E2 on the Mac GPU (owner, 2026-10-22): E1 with the colour written as four tokens (r, g, b, opacity:
# eyegpt_l8 --paint split), everything else as E1 (same cats, split, network body and training settings). One GPU job at
# a time: it stops the CAT repaint's loop, waits for the repaint batch in progress to finish its cats (nothing painted
# is lost), then trains E2, evaluates it as E1 (real plans; free L8 samples taken from E1's saved run, so both eye
# models get the very same faces and L8 is not sampled again), measures speckle / left-right / dark eyes
# (tools/eye_measure_e1.py), and only then restarts the CAT repaint. Any step that fails stops the chain (and the CAT
# repaint stays paused). Log: data/processed/brush_l/e2_chain.log
# Batch 8 x 2 accumulation steps (16 cats per update, as E1): the split sequences are 1,888 positions instead of
# 1,312, and batch 16 needed 33 GB on the 34 GB Mac and swapped (attempt 1, 2026-10-22: no step 100 in 100 minutes).
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
setopt pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools BRUSHGPT_DEVICE=mps
caffeinate -ims -w $$ &
B=data/processed/brush_l
echo "E2 chain started $(date)"
pkill -f "tools/run_repaint_cat_mac.sh" && echo "CAT repaint loop stopped; waiting for its batch to finish $(date)"
while pgrep -f "tools/repaint.py --shards 11-30" >/dev/null; do sleep 60; done
echo "CAT repaint paused $(date)"
if [[ ! -f $B/data_E2.npz ]]; then
  echo "building data_E2 $(date)"
  python3 -m sellanything_ml.sketch.eyegpt_l8 data --paint split --out $B/data_E2.npz 2>&1 | grep -v -i warn || exit 1
fi
echo "training E2 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 train --name E2 --paint split --batch 8 --accum 2 --data $B/data_E2.npz 2>&1 | grep -v -i warn
[[ -f runs/brushgpt/E2/best.pt ]] || { echo "no E2 checkpoint, stopping $(date)"; exit 1; }
echo "evaluating E2 (real plans) $(date)"
python3 tools/eye_eval_e1.py --run E2 --stage real 2>&1 | grep -v -i warn | tail -3 || exit 1
echo "evaluating E2 (free L8 samples, E1's faces) $(date)"
python3 tools/eye_eval_e1.py --run E2 --stage free --n 128 --l8-from runs/brushgpt/E1/eval/free_images.npz 2>&1 \
  | grep -v -i warn | tail -4 || exit 1
echo "measuring E2 $(date)"
python3 tools/eye_measure_e1.py --run E2 2>&1 | grep -v -i warn | tail -1 || exit 1
echo "E2 done; restarting the CAT repaint $(date)"
nohup tools/run_repaint_cat_mac.sh >> data/processed/repaint/mac_cat.log 2>&1 &
echo "E2 chain finished $(date)"
