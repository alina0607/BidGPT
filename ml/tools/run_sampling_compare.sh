#!/bin/zsh
# Three ways to pick each word, on the same held-out prompts (owner, 2026-10-28; notebook 10): greedy (always the
# likeliest), top-p 0.9 and the game's top-k 50, both at temperature 0.7; plus top-k 50 on other seeds, to show how much
# sampling alone moves a number. B4f9 and B5, the old, new and new-style sets, CPU only. The four settings run side by
# side, two threads each.
#
#     cd ml && (nohup tools/run_sampling_compare.sh > runs/measure_b1/sampling/run.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=2
OUT=runs/measure_b1/sampling
mkdir -p $OUT
run() {   # setting name, then the sampling flags
  local name=$1; shift
  for m in b4f9 b5; do
    for s in old new b3; do
      python3 tools/eval_buyer_b1.py runs/sft_$m/best.pt --data runs/eval_sets_b2/val_${s}_format5.jsonl --device cpu \
        "$@" --out $OUT/${m}_${s}_${name}.json --outputs $OUT/${m}_${s}_${name}_outputs.jsonl > /dev/null 2>&1
      echo "$name $m $s exit $? $(date)"
    done
  done
}
run greedy --temperature 0 &
run topp90 --top-p 0.9 --top-k 0 &
run topk50 &
run topk50_seed1 --seed-offset 1 &
wait
echo "sampling compare finished $(date)"
