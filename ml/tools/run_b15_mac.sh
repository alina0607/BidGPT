#!/bin/zsh
# B15 (paper-ideas, owner 2026-10-14): the data-size curve. B4's recipe on half of B4's training sales
# (tools/make_half_set.py, data/processed/buyers_b4_half), same pre-trained start, same hyperparameters and epochs,
# so the only difference from B4 is the number of sales. Stops the cat painting, trains (one GPU job at a time),
# restarts the painting, then evaluates on the CPU with the same seeds as the B3 / B4 measurements in runs/measure_b1
# (CPU and MPS sample differently, so all models compared there are run on the CPU).
#
#     cd ml && (nohup tools/run_b15_mac.sh > runs/sft_b4_half/run.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=.
R=runs/sft_b4_half; E=runs/eval_sets_b2; M=runs/measure_b1
mkdir -p $R/eval
kill $(cat data/processed/cat_brush_face/paint.pid) 2>/dev/null
pkill -f "paint_cat_dataset.py data/processed/cat_brush_face"
sleep 10
echo "painting stopped $(date)"
python3 -m sellanything_ml.llm.sft --init runs/pretrain_large/final.pt --data data/processed/buyers_b4_half \
  --run-dir $R --epochs 3 --batch 16 --lr 3e-4 --eval-every 500 > $R/train.log 2>&1
echo "b4_half sft exit $? $(date)"
(nohup tools/run_mac_paint.sh >> data/processed/cat_brush_face/paint_run.log 2>&1 &)
echo "painting restarted $(date)"
for s in old new b3; do
  python3 tools/eval_buyer_b1.py $R/best.pt --data $E/val_${s}_format5.jsonl --device cpu \
    --out $M/b4half_val_${s}_cpu.json --outputs $M/b4half_val_${s}_outputs.jsonl > /dev/null 2>&1
  echo "eval $s exit $? $(date)"
done
for v in original player; do
  python3 tools/eval_buyer_b1.py $R/best.pt --data $E/player/$v.jsonl --device cpu \
    --out $E/player/b4half_$v.json --outputs $E/player/b4half_${v}_outputs.jsonl > /dev/null 2>&1
done
python3 tools/player_style_test.py compare $E/player/b4half_original_outputs.jsonl $E/player/b4half_player_outputs.jsonl > /dev/null
echo "b15 finished $(date)"
