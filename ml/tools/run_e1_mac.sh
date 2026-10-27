#!/bin/zsh
# Eye model E1 on the Mac GPU (owner, 2026-10-21: condition A, no mirror, train on the Mac). One GPU job at a time, so
# it waits for the L8 bases (tools/run_gen_l8_bases_mac.sh) and the eye material (tools/run_eye_data_l8_mac.sh) to
# finish, then: build data_E1.npz, train E1, evaluate (real plans, then free L8 samples), and only then restart the CAT
# repaint (tools/run_repaint_cat_mac.sh). Any step that fails stops the chain (and the CAT repaint stays paused).
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
setopt pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools BRUSHGPT_DEVICE=mps
caffeinate -ims -w $$ &
B=data/processed/brush_l
echo "E1 chain waiting $(date)"
until grep -q "L8 bases finished\|unexpected exit" $B/l8_gen.log && grep -q "eye data finished\|unexpected exit" $B/eye_l8.log; do sleep 120; done
grep -q "unexpected exit" $B/l8_gen.log $B/eye_l8.log && { echo "a step before E1 stopped with an error $(date)"; exit 1; }
echo "building data_E1 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 data --out $B/data_E1.npz 2>&1 | grep -v -i warn || exit 1
echo "training E1 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 train --name E1 --data $B/data_E1.npz 2>&1 | grep -v -i warn
[[ -f runs/brushgpt/E1/best.pt ]] || { echo "no E1 checkpoint, stopping $(date)"; exit 1; }
echo "evaluating E1 (real plans) $(date)"
python3 tools/eye_eval_e1.py --run E1 --stage real 2>&1 | grep -v -i warn | tail -3 || exit 1
echo "evaluating E1 (free L8 samples) $(date)"
python3 tools/eye_eval_e1.py --run E1 --stage free --n 128 2>&1 | grep -v -i warn | tail -4 || exit 1
echo "E1 done; restarting the CAT repaint $(date)"
nohup tools/run_repaint_cat_mac.sh >> data/processed/repaint/mac_cat.log 2>&1 &
echo "E1 chain finished $(date)"
