#!/bin/zsh
# After B5 (tools/run_b5_mac.sh): its seller side. The seller's answers and decisions and the buyers' replies on the
# same held-out sales as the buyer sets (runs/eval_sets_b5, tools/make_eval_sets_seller.py), then 20 sales with the
# model on both sides on cats no written sale has (seeds 5000-5019, tools/self_play.py). CPU only.
#
#     cd ml && (nohup tools/run_b5_seller_mac.sh > runs/sft_b5/seller.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
M=runs/measure_b1
[ -f runs/eval_sets_b5/val_b3_seller.jsonl ] || python3 tools/make_eval_sets_seller.py
for s in old new b3; do
  python3 tools/eval_buyer_b1.py runs/sft_b5/best.pt --data runs/eval_sets_b5/val_${s}_seller.jsonl --device cpu \
    --out $M/b5_seller_${s}_cpu.json --outputs $M/b5_seller_${s}_outputs.jsonl > runs/sft_b5/eval/b5_seller_${s}.log 2>&1
  echo "b5 seller eval $s exit $? $(date)"
done
python3 tools/self_play.py runs/sft_b5/best.pt --sales 20 --out runs/sft_b5/self_play > runs/sft_b5/eval/self_play.log 2>&1
echo "b5 self-play exit $? $(date)"
