#!/bin/zsh
# B4f9 on the Mac (owner, 2026-10-16; docs/b4f9-line-first-plan.md): B4's data with every scene written line first
# (prompt format 9), trained with B4's recipe from the same pre-trained model, so the only difference from B4 is the
# scene's order. Then B4 and B4f9 evaluated on the same held-out prompts with three sampling seeds each, and the verdict
# the plan fixed before the run (tools/compare_b4f9.py).
#
# Started by hand when the owner says so. One GPU job at a time: it refuses to start while brush GPT J's driver or
# any other training is running.
#
#     cd ml && (nohup tools/run_b4f9_mac.sh > runs/sft_b4f9/run.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=.
J_PID=runs/brushgpt/J_driver.pid
if [ -f $J_PID ] && kill -0 "$(cat $J_PID)" 2>/dev/null; then
  echo "brush GPT J's driver (pid $(cat $J_PID)) is still running: not starting $(date)"; exit 1
fi
if pgrep -f "sellanything_ml.llm.sft|brushgpt train|paint_cat_dataset" > /dev/null; then
  echo "another GPU job is running: not starting $(date)"; pgrep -fl "sellanything_ml.llm.sft|brushgpt train|paint_cat_dataset"
  exit 1
fi
mkdir -p runs/sft_b4f9/eval
echo "B4f9 started $(date)"
caffeinate -ims -w $$ &

if [ ! -f data/processed/buyers_b4f9/meta.json ]; then
  python3 tools/make_line_first.py > runs/sft_b4f9/data_report.json || { echo "data failed $(date)"; exit 1; }
fi
echo "data ready $(date)"

python3 -m sellanything_ml.llm.sft --init runs/pretrain_large/final.pt --data data/processed/buyers_b4f9 \
  --run-dir runs/sft_b4f9 --epochs 3 --batch 16 --lr 3e-4 --eval-every 500 > runs/sft_b4f9/train.log 2>&1
echo "b4f9 sft exit $? $(date)"

for name in b4:runs/sft_b4/best.pt b4f9:runs/sft_b4f9/best.pt; do
  for set in old new; do
    for k in 0 1 2; do
      out=runs/sft_b4f9/eval/${name%%:*}_val_${set}_s$k
      python3 tools/eval_buyer_b1.py ${name#*:} --data runs/eval_sets_b2/val_${set}_format5.jsonl --seed-offset $k \
        --out $out.json --outputs ${out}_outputs.jsonl > $out.log 2>&1
      echo "${name%%:*} eval $set seed $k exit $? $(date)"
    done
  done
done
python3 tools/compare_b4f9.py runs/sft_b4f9/eval --out runs/sft_b4f9/eval/compare.json > runs/sft_b4f9/eval/compare.log 2>&1
echo "compare exit $? $(date)"
tail -1 runs/sft_b4f9/eval/compare.log
echo "B4f9 finished $(date)"
