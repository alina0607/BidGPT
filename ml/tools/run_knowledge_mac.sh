#!/bin/zsh
# The knowledge experiment on the Mac (owner, 2026-10-13), the same runs tools/modal_knowledge.py would make on Modal:
# B2k  = pretrain_large → one pass of extra pre-training on FineWeb-Edu + the cat knowledge texts → SFT as B2;
# B2k0 = the same with FineWeb-Edu only (the control). Then tools/eval_knowledge.sh.
#
# Waits for B3 to finish (its after_training.sh evaluates it and restarts the cat painting), stops the painting
# again, runs everything one GPU job at a time, and restarts the painting at the end.
#
#     cd ml && (nohup tools/run_knowledge_mac.sh > runs/knowledge/run.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=.
until grep -q "painting restarted" runs/sft_b3/eval/after_training.log 2>/dev/null; do sleep 60; done
echo "B3 done, stopping the painting $(date)"
kill $(cat data/processed/cat_brush_face/paint.pid) 2>/dev/null
pkill -f "paint_cat_dataset.py data/processed/cat_brush_face"
sleep 10
for name in b2k b2k0; do
  mix=data/processed/pretrain_knowledge; [ $name = b2k0 ] && mix=data/processed/pretrain_knowledge_control
  mkdir -p runs/knowledge/$name/pretrain runs/knowledge/$name/sft
  python3 -m sellanything_ml.llm.pretrain --init runs/pretrain_large/final.pt --data-dir $mix \
    --run-dir runs/knowledge/$name/pretrain --tokens 5551500 --batch 16 --accum 2 --block 512 --lr 3e-4 \
    --warmup 20 --eval-every 100 > runs/knowledge/$name/pretrain/train.log 2>&1
  echo "$name pretrain exit $? $(date)"
  python3 -m sellanything_ml.llm.sft --init runs/knowledge/$name/pretrain/final.pt --data data/processed/buyers_b2 \
    --run-dir runs/knowledge/$name/sft --epochs 3 --batch 16 --lr 3e-4 --eval-every 500 \
    > runs/knowledge/$name/sft/train.log 2>&1
  echo "$name sft exit $? $(date)"
done
tools/eval_knowledge.sh
(nohup tools/run_mac_paint.sh >> data/processed/cat_brush_face/paint_run.log 2>&1 &)
echo "painting restarted, knowledge experiment finished $(date)"
