#!/bin/zsh
# FID / KID to the AFHQ test paintings and IconShop's Uniqueness / Novelty for run J (tools/brushgpt_fid_diversity.py),
# next to run I's (measured on the CPU, 2026-10-16). One GPU job at a time: waits for the stage-continuation script
# (tools/run_stage_continuation_J.sh), which itself waits for the J driver, to write "J cat exit". J is scored on its
# own token file; its AFHQ validation and test paintings are the same pictures as I's.
#
#     cd ml && (nohup tools/run_fid_diversity_J.sh >> runs/brushgpt/J_fid_diversity_driver.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
R=runs/brushgpt
J=J_cat_aligned_points_eyes
until grep -q "J cat exit" $R/J_stage_continuation_driver.log 2>/dev/null && ! pgrep -f run_stage_continuation_J.sh > /dev/null; do
  sleep 300
done
echo "J fid start $(date)"
caffeinate -i python3 tools/brushgpt_fid_diversity.py --run $J --data data/processed/brushgpt/tokens_J_aligned_reordered.npz \
  > $R/$J/fid_diversity_t1.0.log 2>&1
echo "J fid exit $? $(date)"
