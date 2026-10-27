#!/bin/zsh
# The knowledge experiment's evaluation on the Mac, after tools/modal_knowledge.py brought back runs/knowledge/{b2k,b2k0}.
# Same checks as B2: the held-out evaluation on the Mac's GPU (B2's numbers came from the same device and seeds), then
# on the CPU the knowledge probe (also on the two pre-training stages) and the misread probe. Waits for any other GPU
# job first (one GPU job at a time).
#
#     cd ml && (nohup tools/eval_knowledge.sh > runs/knowledge/eval.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "sellanything_ml.llm.sft|eval_buyer_b1.py" >/dev/null; do sleep 60; done
for name in b2k b2k0; do
  for set in old new; do
    PYTHONPATH=. python3 tools/eval_buyer_b1.py runs/knowledge/$name/sft/best.pt \
      --data runs/eval_sets_b2/val_${set}_format5.jsonl --out runs/knowledge/$name/eval_val_${set}.json \
      > runs/knowledge/$name/eval_val_${set}.log 2>&1
    echo "$name eval $set exit $? $(date)"
  done
done
PYTHONPATH=. python3 tools/probe_knowledge.py runs/pretrain_large/final.pt runs/knowledge/b2k/pretrain/final.pt \
  runs/knowledge/b2k0/pretrain/final.pt runs/sft_b2/best.pt runs/knowledge/b2k/sft/best.pt \
  runs/knowledge/b2k0/sft/best.pt --out runs/knowledge/probe_knowledge.json
for name in b2k b2k0; do
  PYTHONPATH=. python3 tools/probe_misread.py runs/knowledge/$name/sft/best.pt --out runs/knowledge/$name/misread.jsonl
done
echo "all done $(date)"
