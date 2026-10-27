#!/bin/zsh
# Eye model E3 on the Mac GPU (owner, 2026-10-22): E2 (split paint) with one change, the eye material's stroke order
# (tools/eye_data_l8.py --order inside_out: iris, pupil and highlight from each eye's centre outwards). Same cats,
# split, network and training settings as E1/E2. Waits for the E2 chain (tools/run_e2_mac.sh) and the E3 eye material
# (tools/run_eye_data_e3_mac.sh, CPU) to finish; one GPU job at a time, so it then stops the CAT repaint's loop
# (which the E2 chain restarted), waits for the batch in progress to finish, trains E3, evaluates it as E2 (its own
# material as the upper bound; free L8 samples from E1's saved run), measures it, and restarts the CAT repaint.
# Any step that fails stops the chain (and the CAT repaint stays paused). Log: data/processed/brush_l/e3_chain.log
# Batch 8 x 2 accumulation steps (16 cats per update, as E1): the split sequences are 1,888 positions instead of
# 1,312, and batch 16 needed 33 GB on the 34 GB Mac and swapped (attempt 1, 2026-10-22: no step 100 in 100 minutes).
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
setopt pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools BRUSHGPT_DEVICE=mps
caffeinate -ims -w $$ &
B=data/processed/brush_l
echo "E3 chain waiting $(date)"
until grep -q "E2 chain finished\|stopping" $B/e2_chain.log && grep -q "E3 eye material finished\|unexpected exit" $B/eye_e3.log; do
  sleep 120
done
grep -q "stopping" $B/e2_chain.log && { echo "the E2 chain stopped with an error, E3 not started $(date)"; exit 1; }
grep -q "unexpected exit" $B/eye_e3.log && { echo "the E3 eye material is incomplete, E3 not started $(date)"; exit 1; }
sleep 30
pkill -f "tools/run_repaint_cat_mac.sh" && echo "CAT repaint loop stopped; waiting for its batch to finish $(date)"
while pgrep -f "tools/repaint.py --shards 11-30" >/dev/null; do sleep 60; done
echo "CAT repaint paused $(date)"
echo "building data_E3 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 data --paint split --eyes $B/eye_l8_inside --out $B/data_E3.npz 2>&1 \
  | grep -v -i warn || exit 1
echo "training E3 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 train --name E3 --paint split --batch 8 --accum 2 --data $B/data_E3.npz 2>&1 | grep -v -i warn
[[ -f runs/brushgpt/E3/best.pt ]] || { echo "no E3 checkpoint, stopping $(date)"; exit 1; }
echo "evaluating E3 (real plans) $(date)"
python3 tools/eye_eval_e1.py --run E3 --stage real --eyes $B/eye_l8_inside 2>&1 | grep -v -i warn | tail -3 || exit 1
echo "evaluating E3 (free L8 samples, E1's faces) $(date)"
python3 tools/eye_eval_e1.py --run E3 --stage free --n 128 --l8-from runs/brushgpt/E1/eval/free_images.npz 2>&1 \
  | grep -v -i warn | tail -4 || exit 1
echo "measuring E3 $(date)"
python3 tools/eye_measure_e1.py --run E3 2>&1 | grep -v -i warn | tail -1 || exit 1
echo "E3 done; restarting the CAT repaint $(date)"
nohup tools/run_repaint_cat_mac.sh >> data/processed/repaint/mac_cat.log 2>&1 &
echo "E3 chain finished $(date)"
