#!/bin/zsh
# Waits for brush GPT J's driver to finish (its training and evaluation), then starts B4f9 (owner, 2026-10-16:
# "J 的 driver 一結束就自動開始 B4f9"). One GPU job at a time: it starts only after two checks five minutes apart
# both find no J driver and no other GPU job, so a J that is restarted in between is waited for too.
#
#     cd ml && (nohup tools/wait_then_b4f9.sh > runs/sft_b4f9/wait.log 2>&1 &)
cd "$(dirname "$0")/.." || exit 1
echo $$ > runs/sft_b4f9/wait.pid
echo "waiting for brush GPT J's driver $(date)"
caffeinate -ims -w $$ &
busy() {
  { [ -f runs/brushgpt/J_driver.pid ] && kill -0 "$(cat runs/brushgpt/J_driver.pid)" 2>/dev/null; } ||
    pgrep -f "sellanything_ml.llm.sft|brushgpt train|paint_cat_dataset|eye_judge" > /dev/null
}
while true; do
  while busy; do sleep 60; done
  echo "nothing running $(date); checking again in 5 minutes"
  sleep 300
  busy || break
  echo "a GPU job started again $(date); waiting"
done
echo "starting B4f9 $(date)"
tools/run_b4f9_mac.sh > runs/sft_b4f9/run.log 2>&1
echo "run_b4f9_mac.sh exit $? $(date)"
