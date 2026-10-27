#!/bin/zsh
# The scene-quota experiment on the Mac (owner, 2026-10-13): B4 = B3's sales plus sales_b3.jsonl (the sales written
# to the scene quotas, 2500-2599, and the human-seller trial 2650-2659), prompt format 5, trained with B3's recipe
# from the same pre-trained model, so the only difference from B3 is the added sales. Then the held-out evaluation
# on the same prompts as B1-B3 (runs/eval_sets_b2) and the scene probes for B3 and B4.
#
# Stops the cat painting, trains (one GPU job at a time), restarts the painting, then runs the CPU probes.
#
#     cd ml && (nohup tools/run_b4_mac.sh > runs/sft_b4/run.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=.
mkdir -p runs/sft_b4/eval
python3 -m sellanything_ml.buyers.sales_sft --sources b4 --out data/processed/buyers_b4 > runs/sft_b4/data_report.json
echo "data built $(date)"
kill $(cat data/processed/cat_brush_face/paint.pid) 2>/dev/null
pkill -f "paint_cat_dataset.py data/processed/cat_brush_face"
sleep 10
echo "painting stopped $(date)"
python3 -m sellanything_ml.llm.sft --init runs/pretrain_large/final.pt --data data/processed/buyers_b4 \
  --run-dir runs/sft_b4 --epochs 3 --batch 16 --lr 3e-4 --eval-every 500 > runs/sft_b4/train.log 2>&1
echo "b4 sft exit $? $(date)"
for set in old new; do
  python3 tools/eval_buyer_b1.py runs/sft_b4/best.pt --data runs/eval_sets_b2/val_${set}_format5.jsonl \
    --out runs/sft_b4/eval/b4_val_${set}.json > runs/sft_b4/eval/b4_val_${set}.log 2>&1
  echo "b4 eval $set exit $? $(date)"
done
(nohup tools/run_mac_paint.sh >> data/processed/cat_brush_face/paint_run.log 2>&1 &)
echo "painting restarted $(date)"
for name in b3:runs/sft_b3_resume/best.pt b4:runs/sft_b4/best.pt; do
  python3 tools/probe_scene_list.py ${name#*:} --out runs/sft_b4/eval/scene_list_${name%%:*}.jsonl \
    > runs/sft_b4/eval/scene_list_${name%%:*}.log 2>&1
  echo "scene probe ${name%%:*} exit $? $(date)"
done
echo "b4 experiment finished $(date)"
