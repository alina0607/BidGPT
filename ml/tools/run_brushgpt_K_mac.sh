#!/bin/zsh
# Brush GPT run K on the Mac (owner, 2026-10-18: make the eyes clearer before the fur-flow painter). K = run I's recipe
# and data (tokens_runs_HI.npz: AFHQ, aligned faces, face points first, eye-region loss x3, d512 x 7 layers) with one
# change: the eye weight is left off the x and y tokens of eye strokes (--eye-weight-position 0), so only their angle,
# size and colour count three times.
#
# Why: weighting an eye stroke's position also teaches the model to put strokes in the eye region more often. Run F
# (D + eye weight) paints 134 strokes per painting there against D's 74 and the real paintings' 86 (64 samples each,
# eyes found by the eye judge), and I and J paint 171 and 166 against the real 85 (tools/eye_zone_strokes.py), while
# their eye scores barely move (D 0.416, F 0.421; I 0.495, J 0.501 at temperature 1). K asks whether learning how eye
# strokes look, without being pushed to paint more of them, makes the eyes clearer.
#
# Differences from I besides the option: batch 1 x accumulation 16 on the Mac (I ran 8 x 2 on an A100; the 16 rows per
# step are the same, but the weighted loss is normalised per painting rather than per 8 paintings), and fp32 on MPS
# instead of fp16 on CUDA. Everything else as I: seed 0, at most 8,000 steps, evaluation every 100, patience 8.
#
# Verdict, fixed before the run (compared with I, on the same validation data and the same judge):
#   mechanism: K's 292 generated paintings put fewer than 128 strokes per painting in the eye region (128 is halfway
#              between the real paintings' 85.1 and I's 170.6);
#   eyes:      at temperature 1, K's eye score minus I's is above 2 standard errors of the difference ("helped"),
#              below -2 ("hurt"), otherwise "no clear change". Temperatures 0.8 and 0.6 and the FID are reported.
#
# One GPU job at a time: refuses to start while another training runs. The FID runs on the CPU afterwards (on MPS
# its rendering trips the slow-down watchdog, as it did for I and J).
#
#     cd ml && (nohup tools/run_brushgpt_K_mac.sh >> runs/brushgpt/K_driver.log 2>&1 &)
#     cd ml && (EVAL_ONLY=1 nohup tools/run_brushgpt_K_mac.sh >> runs/brushgpt/K_eval_driver.log 2>&1 &)
#
# 2026-10-18: K was trained on a Kaggle T4 with I's batch (8 x 2, fp16; ml/data/kaggle/brushgpt_k/) and evaluated
# here with EVAL_ONLY=1; the Mac copy above (1 x 16) was stopped at step 700 and kept as K_eye_weight_no_position_mac.
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=.
NAME=K_eye_weight_no_position
DATA=data/processed/brushgpt/tokens_runs_HI.npz
R=runs/brushgpt/$NAME
if pgrep -f "sellanything_ml.llm.sft|brushgpt train|paint_cat_dataset" > /dev/null; then
  echo "another GPU job is running: not starting $(date)"; pgrep -fl "sellanything_ml.llm.sft|brushgpt train|paint_cat_dataset"
  exit 1
fi
echo $$ > runs/brushgpt/K_driver.pid
echo "K driver started $(date)"
caffeinate -ims -w $$ &
mkdir -p $R

# EVAL_ONLY=1: skip training and evaluate the run already in $R (the Kaggle T4 run, copied in from its output)
for try in $( [ -n "$EVAL_ONLY" ] || seq 1 20 ); do
  python3 -m sellanything_ml.sketch.brushgpt train --name $NAME --data $DATA \
    --d-model 512 --n-layers 7 --n-heads 8 --n-kv-heads 8 --dropout 0.1 --lr 3e-4 --batch 1 --accum 16 \
    --max-steps 8000 --warmup 100 --eval-every 100 --patience 8 --stop-after-min 2400 --keep-every 1000 \
    --order overlap --heads 1 --points 1 --eye-weight 3 --eye-weight-position 0 --resume 1 >> $R/train.log 2>&1
  code=$?
  echo "K train try $try exit $code $(date)"
  [ $code -eq 0 ] && break
  sleep 120
done
[ -f $R/best.pt ] || { echo "no best.pt, stopping $(date)"; exit 1; }
shasum -a 256 $R/best.pt

for run in $NAME I_aligned_points_eyes; do
  for t in 1.0 0.8 0.6; do
    python3 tools/eye_judge.py --run $run --n 128 --temperature $t --data $DATA \
      2>/dev/null | tail -1 >> $R/eye_judge.jsonl
    echo "eye judge $run t$t exit $? $(date)"
  done
done
nice -n 19 env BRUSHGPT_DEVICE=cpu python3 tools/brushgpt_fid_diversity.py --run $NAME --data $DATA \
  > $R/fid_diversity_t1.0.log 2>&1
echo "K fid exit $? $(date)"
python3 tools/eye_zone_strokes.py --data $DATA --runs $NAME I_aligned_points_eyes J_cat_aligned_points_eyes \
  --out $R/eye_zone_strokes.json > /dev/null
echo "eye zone strokes exit $? $(date)"

python3 - $R <<'PY'
import json, math, sys
r = sys.argv[1]
judge = {}
for line in open(f"{r}/eye_judge.jsonl"):
    d = json.loads(line)
    judge.update({k: v for k, v in d.items() if k != "real_paintings"})
k, i = judge["K_eye_weight_no_position_t1.0"], judge["I_aligned_points_eyes_t1.0"]
diff = k["eye_conf_mean"] - i["eye_conf_mean"]
se = math.hypot(k["eye_conf_se"], i["eye_conf_se"])
eyes = "helped" if diff > 2 * se else "hurt" if diff < -2 * se else "no clear change"
zone = json.load(open(f"{r}/eye_zone_strokes.json"))
kz = zone["K_eye_weight_no_position"]["total_mean"]
fid = json.load(open(f"{r}/fid_diversity_t1.0.json"))["fid_to_test"]["generated"]
verdict = {"eye_zone_strokes": {"K": kz, "I": zone["I_aligned_points_eyes"]["total_mean"], "real": zone["real_val"]["total_mean"],
                                "mechanism": "fewer than 128" if kz < 128 else "not fewer than 128"},
           "eye_score_t1.0": {"K": k["eye_conf_mean"], "I": i["eye_conf_mean"], "diff": round(diff, 4), "se": round(se, 4), "eyes": eyes},
           "fid_to_test": {"K": fid}}
json.dump(verdict, open(f"{r}/verdict.json", "w"), indent=1)
print("verdict:", json.dumps(verdict))
PY
echo "K finished $(date)"
