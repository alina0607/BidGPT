#!/bin/zsh
# When the sketch plans and the L5 sequences are both done, build data_L5.npz, upload it with the code as the Kaggle
# dataset brush-l5-data, and push the training kernel brush-gpt-l5-train (owner's go-ahead, 2026-10-19 20:30).
# Every 5 minutes: restarts the sketch-plan batches if they stopped before all cats were planned; "done" means no
# sketch-plan process is left, every cat has a plan or a recorded failure, and every planned cat has a sequence or a
# recorded failure. Log: data/processed/brush_l/autolaunch.log.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
export PYTHONPATH=.:tools
caffeinate -ims -w $$ &
LOG=data/processed/brush_l/autolaunch.log
PL=data/processed/sketch_plan/afhq
CATS=data/processed/brush_l/cats
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
count() { find "$1" -name '*.npz' ! -name '*.tmp.npz' | wc -l | tr -d ' '; }
lines() { [[ -f $1 ]] && wc -l < $1 | tr -d ' ' || echo 0; }
log "watching"
while true; do
  plans=$(count $PL); pfail=$(lines $PL/failed.jsonl)
  running=$(pgrep -f "sketch_plan.py --batch" | wc -l | tr -d ' ')
  if (( running == 0 && plans + pfail < 5547 )); then
    for g in 0-3 4-7 8-11; do
      nohup python tools/sketch_plan.py --batch $g --source afhq --threads 2 --out $PL >> data/processed/sketch_plan/afhq_$g.log 2>&1 &
    done
    log "sketch plans restarted at $plans planned, $pfail failed"
    sleep 300; continue
  fi
  seqs=$(count $CATS); sfail=$(lines $CATS/failed.jsonl)
  log "plans $plans (+$pfail failed, $running running), sequences $seqs (+$sfail failed)"
  if (( running == 0 && plans + pfail >= 5547 && seqs + sfail >= plans )); then break; fi
  sleep 300
done
log "both done: building the dataset"
pkill -f run_prepare_l_mac.sh
while pgrep -f "tools/prepare_l.py" > /dev/null; do sleep 10; done
python -m sellanything_ml.sketch.brushgpt_l data --cats $CATS --layers 5 --out data/processed/brush_l/data_L5.npz >> $LOG 2>&1 || { log "BUILD FAILED"; exit 1; }
K=data/kaggle/brush_l5
rm -rf $K/data/ml $K/data/data_L5.npz
mkdir -p $K/data/ml
rsync -a --exclude __pycache__ --exclude '*.pyc' sellanything_ml $K/data/ml/
cp data/processed/brush_l/data_L5.npz $K/data/
log "uploading $(du -sh $K/data | cut -f1)"
kaggle datasets create -p $K/data -r zip >> $LOG 2>&1 || { log "UPLOAD FAILED"; exit 1; }
for i in $(seq 1 60); do
  kaggle datasets status alinachuang/brush-l5-data 2>&1 | grep -q ready && break
  sleep 60
done
log "dataset: $(kaggle datasets status alinachuang/brush-l5-data 2>&1 | tail -1)"
kaggle kernels push -p $K/kernel >> $LOG 2>&1 || { log "PUSH FAILED"; exit 1; }
sleep 120
log "kernel: $(kaggle kernels status alinachuang/brush-gpt-l5-train 2>&1 | tail -1)"
log "autolaunch done"
