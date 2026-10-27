#!/bin/zsh
# E8 / E8L end to end (owner, 2026-10-25: once the Mac repaint is done, check every cat, build data_E8 (the 79 unclear
# cats left out of training, training cats mirrored), upload to Kaggle and train E8 / E8L automatically). Then, while
# Kaggle trains, the Mac goes back to the CAT repaint; when Kaggle is done the runs are downloaded, evaluated on the Mac
# as E1-E5 were (real plans with the run's own material as the upper bound; free L8 samples from E1's saved run;
# tools/eye_measure_e1.py) and compared with E5, and the CAT repaint is restarted. Any failed step stops the chain.
# 2026-10-26 (owner): before the check, every cat's base is redone with L8 kept 3 px inside the plan's outlines
# (tools/run_eye_rebase_mac.sh; base_kind hybrid_in3), and 90 more cats (rim_off) are left out of training
# (eye_exclude_2026-10-26.tsv: the 79 of 10-24 and these).
# Log: data/processed/brush_l/e8_chain.log
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
setopt pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.:tools BRUSHGPT_DEVICE=mps
K=/opt/anaconda3/bin/kaggle
B=data/processed/brush_l
M=$B/eye_l8_rim_hybrid
KD=data/kaggle/eye_train
echo "E8 chain waiting for the repaint $(date)"
until grep -q "eye rim hybrid finished\|unexpected exit" $B/eye_rim_hybrid.log; do sleep 300; done
grep -q "unexpected exit" $B/eye_rim_hybrid.log && { echo "the repaint stopped with an error, stopping $(date)"; exit 1; }
caffeinate -ims -w $$ &
echo "waiting for the last painting and the first rebase pass $(date)"
while pgrep -f "tools/eye_data_l8.py" >/dev/null || pgrep -f "tools/run_eye_rebase_mac.sh" >/dev/null; do sleep 60; done
echo "rebasing the cats painted since $(date)"
tools/run_eye_rebase_mac.sh >> $B/eye_rebase.log 2>&1
tail -1 $B/eye_rebase.log | grep -q "eye rebase finished" || { echo "the rebase stopped with an error, stopping $(date)"; exit 1; }
echo "checking every cat $(date)"
python3 tools/eye_material_check.py --dir $M --style digital_rim_long --base hybrid_in3 2>&1 | grep -v -i warn || { echo "material check failed, stopping $(date)"; exit 1; }
echo "building data_E8 $(date)"
python3 -m sellanything_ml.sketch.eyegpt_l8 data --paint split --style digital --eyes $M \
  --exclude $B/eye_exclude_2026-10-26.tsv --out $B/data_E8_nomirror.npz 2>&1 | grep -v -i warn || exit 1
python3 tools/eye_data_mirror.py --data $B/data_E8_nomirror.npz --eyes $M --check 6 2>&1 | grep -v -i warn | tee $B/e8_mirror_check.json
grep -q '"grammar_ok": 6' $B/e8_mirror_check.json || { echo "mirror check failed, stopping $(date)"; exit 1; }
python3 tools/eye_data_mirror.py --data $B/data_E8_nomirror.npz --eyes $M --out $B/data_E8.npz 2>&1 | grep -v -i warn || exit 1
echo "uploading data_E8 $(date)"
mkdir -p $KD/data && ln -f $B/data_E8.npz $KD/data/data_E8.npz
echo '{"title": "eye train data", "id": "alinachuang/eye-train-data", "licenses": [{"name": "other"}]}' > $KD/data/dataset-metadata.json
$K datasets create -p $KD/data 2>&1 | grep -v outdated | grep -v "%|" | tail -2
for i in $(seq 1 60); do $K datasets files alinachuang/eye-train-data 2>/dev/null | grep -q "data_E8.npz" && break; sleep 60; done
$K datasets files alinachuang/eye-train-data 2>/dev/null | grep -q "data_E8.npz" || { echo "dataset not ready, stopping $(date)"; exit 1; }
sleep 60
$K kernels push -p $KD/kernel 2>&1 | grep -v outdated
sleep 180
s=$($K kernels status alinachuang/eye-gpt-e8-train 2>&1 | grep -v outdated); echo "$s $(date)"
nohup tools/run_repaint_cat_mac.sh >> data/processed/repaint/mac_cat.log 2>&1 &
echo "CAT repaint restarted while Kaggle trains $(date)"
while true; do
  s=$($K kernels status alinachuang/eye-gpt-e8-train 2>&1 | grep -v outdated)
  [[ "$s" == *COMPLETE* || "$s" == *ERROR* || "$s" == *CANCEL* ]] && break
  sleep 900
done
echo "$s $(date)"
mkdir -p runs/kaggle_e8
$K kernels output alinachuang/eye-gpt-e8-train -p runs/kaggle_e8 2>&1 | grep -v outdated | tail -2
for r in E8 E8L; do [[ -f runs/kaggle_e8/out/$r/best.pt ]] && cp -R runs/kaggle_e8/out/$r runs/brushgpt/; done
ls runs/brushgpt/E8/best.pt runs/brushgpt/E8L/best.pt || { echo "no checkpoints, stopping $(date)"; exit 1; }
pkill -f "tools/run_repaint_cat_mac.sh" && echo "CAT repaint loop stopped for the evaluation $(date)"
while pgrep -f "tools/repaint.py --shards 11-30" >/dev/null; do sleep 30; done
for r in E8 E8L; do
  echo "evaluating $r $(date)"
  python3 tools/eye_eval_e1.py --run $r --stage real --eyes $M 2>&1 | grep -v -i warn | tail -2 || exit 1
  python3 tools/eye_eval_e1.py --run $r --stage free --n 128 --eyes $M --l8-from runs/brushgpt/E1/eval/free_images.npz 2>&1 \
    | grep -v -i warn | tail -2 || exit 1
  python3 tools/eye_measure_e1.py --run $r 2>&1 | grep -v -i warn | tail -1 || exit 1
done
python3 tools/eye_compare_runs.py --runs E5 E8 E8L --out runs/brushgpt/compare_E5_E8_E8L --n 6 2>&1 | grep -v -i warn
nohup tools/run_repaint_cat_mac.sh >> data/processed/repaint/mac_cat.log 2>&1 &
echo "E8 chain finished $(date)"
