#!/bin/zsh
# Paper-ideas P6 / direction B for run J: where in the coarse-to-fine painting the errors come from
# (tools/stage_continuation.py). Waits for the J driver (tools/run_brushgpt_J_mac.sh) to write "J finished", so the
# GPU is free (one GPU job at a time), then scores J on the same 64 AFHQ validation cats run I was scored on (I on the
# CPU, 2026-10-16), each model in its own stroke order, and J on 64 CAT-dataset validation cats.
#
#     cd ml && (nohup tools/run_stage_continuation_J.sh >> runs/brushgpt/J_stage_continuation_driver.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
R=runs/brushgpt
J=J_cat_aligned_points_eyes
DATA=data/processed/brushgpt/tokens_J_aligned_reordered.npz
I_NPZ=$R/I_aligned_points_eyes/stage_continuation_afhq_t1.0.npz
until grep -q "J finished" $R/J_driver.log 2>/dev/null && ! pgrep -f run_brushgpt_J_mac.sh > /dev/null; do
  sleep 300
done
[ -f $I_NPZ ] || { echo "no $I_NPZ: run I first $(date)"; exit 1; }
echo "J stage continuation start $(date)"
caffeinate -i python3 tools/stage_continuation.py --run $J --data $DATA --ids-from $I_NPZ --n 64 --batch 64 \
  > $R/$J/stage_continuation_afhq_t1.0.log 2>&1
echo "J afhq exit $? $(date)"
caffeinate -i python3 tools/stage_continuation.py --run $J --data $DATA --source cat --n 64 --batch 64 \
  > $R/$J/stage_continuation_cat_t1.0.log 2>&1
echo "J cat exit $? $(date)"
