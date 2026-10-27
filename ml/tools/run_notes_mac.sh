#!/bin/zsh
# The notes experiment on the Mac (owner, 2026-10-13): B3n = B3's 360 sales in prompt format 7 (older answers kept as
# notes, and a note task), trained with B3's recipe, so the only difference from B3 is the notes. Then the held-out
# evaluation on the same prompts (runs/eval_sets_b3n = runs/eval_sets_b2 with notes) and the misread probe for both.
#
# Waits for the knowledge experiment (tools/run_knowledge_mac.sh) to finish, stops the cat painting, trains, and
# restarts the painting at the end. One GPU job at a time.
#
#     cd ml && (nohup tools/run_notes_mac.sh > runs/sft_b3n/run.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=.
until grep -q "knowledge experiment finished" runs/knowledge/run.log 2>/dev/null; do sleep 60; done
echo "knowledge done, stopping the painting $(date)"
kill $(cat data/processed/cat_brush_face/paint.pid) 2>/dev/null
pkill -f "paint_cat_dataset.py data/processed/cat_brush_face"
sleep 10
# B3 stopped at step ~5300 of 7089 (MPS out of memory, before the masked-loss fix of 2026-10-13): finish it from its
# step-5000 checkpoint into its own folder, so the B3 that B3n is compared with also ran the whole schedule
mkdir -p runs/sft_b3_resume/eval runs/sft_b3n/eval
python3 -m sellanything_ml.llm.sft --init runs/sft_b3/last.pt --start-step 5000 --data data/processed/buyers_b3 \
  --run-dir runs/sft_b3_resume --epochs 3 --batch 16 --lr 3e-4 --eval-every 500 > runs/sft_b3_resume/train.log 2>&1
echo "b3 resume exit $? $(date)"
for set in old new; do
  python3 tools/eval_buyer_b1.py runs/sft_b3_resume/best.pt --data runs/eval_sets_b2/val_${set}_format5.jsonl \
    --out runs/sft_b3_resume/eval/b3_val_${set}.json > runs/sft_b3_resume/eval/b3_val_${set}.log 2>&1
done
python3 -m sellanything_ml.llm.sft --init runs/pretrain_large/final.pt --data data/processed/buyers_b3_notes \
  --run-dir runs/sft_b3n --epochs 3 --batch 16 --lr 3e-4 --eval-every 500 > runs/sft_b3n/train.log 2>&1
echo "b3n sft exit $? $(date)"
for set in old new; do
  python3 tools/eval_buyer_b1.py runs/sft_b3n/best.pt --data runs/eval_sets_b3n/val_${set}_format7.jsonl \
    --out runs/sft_b3n/eval/b3n_val_${set}.json > runs/sft_b3n/eval/b3n_val_${set}.log 2>&1
  echo "b3n eval $set exit $? $(date)"
done
python3 tools/probe_misread.py runs/sft_b3/best.pt --out runs/sft_b3n/eval/misread_b3.jsonl
python3 tools/probe_misread.py runs/sft_b3n/best.pt --sets runs/eval_sets_b3n --out runs/sft_b3n/eval/misread_b3n.jsonl
(nohup tools/run_mac_paint.sh >> data/processed/cat_brush_face/paint_run.log 2>&1 &)
echo "painting restarted, notes experiment finished $(date)"
