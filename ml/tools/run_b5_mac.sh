#!/bin/zsh
# Buyer GPT B5 on the Mac (owner, 2026-10-28): prompt format 8 (the buyer tasks plus the seller's answer, decide and
# reply) on the B5 sources, which add the sales written since B4 (sales_b3_decide.jsonl: 443 sales, 2690-2699 held
# out). Same pre-trained model and learning rate as B4. Two epochs, not three: B4's validation loss was lowest at
# step 6,000 of 9,156 (about two epochs) and flat after. --accum 2 halves the memory per step (format 8 prompts reach
# 768 tokens; B3n ran out of MPS memory at that length) and keeps the batch at 16. SFT mixing (B16) is left for its own
# run so its effect can be measured apart from the new sales.
#
# Format 8's buyer prompts are identical to format 5's, so B5 is scored on the same held-out prompts as B1 to B4
# (runs/eval_sets_b2), on the CPU like B4's numbers in runs/measure_b1. The Mac is not painting cats (owner); this script does not start or stop the painting.
#
#     cd ml && (nohup tools/run_b5_mac.sh > runs/sft_b5/run.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=.
mkdir -p runs/sft_b5/eval
[ -f data/processed/buyers_b5/train.jsonl ] || \
  python3 -m sellanything_ml.buyers.sales_sft --sources b5 --prompt-format 8 --out data/processed/buyers_b5 \
    > runs/sft_b5/data_report.json
echo "training starts $(date)"
python3 -m sellanything_ml.llm.sft --init runs/pretrain_large/final.pt --data data/processed/buyers_b5 \
  --run-dir runs/sft_b5 --epochs 2 --batch 16 --accum 2 --lr 3e-4 --eval-every 500 > runs/sft_b5/train.log 2>&1
echo "b5 sft exit $? $(date)"
M=runs/measure_b1 E=runs/eval_sets_b2
for s in old new b3; do       # on the CPU like B4's numbers in runs/measure_b1 (CPU and MPS sample differently)
  python3 tools/eval_buyer_b1.py runs/sft_b5/best.pt --data $E/val_${s}_format5.jsonl --device cpu \
    --out $M/b5_val_${s}_cpu.json --outputs $M/b5_val_${s}_outputs.jsonl > runs/sft_b5/eval/b5_val_${s}.log 2>&1
  echo "b5 eval $s exit $? $(date)"
done
for v in original player; do
  python3 tools/eval_buyer_b1.py runs/sft_b5/best.pt --data $E/player/$v.jsonl --device cpu \
    --out $E/player/b5_$v.json --outputs $E/player/b5_${v}_outputs.jsonl > /dev/null 2>&1
  echo "b5 player $v exit $? $(date)"
done
python3 tools/player_style_test.py compare $E/player/b5_original_outputs.jsonl $E/player/b5_player_outputs.jsonl > /dev/null
echo "b5 finished $(date)"
