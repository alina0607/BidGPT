#!/bin/zsh
# Restarts the Mac CAT repaint (tools/run_repaint_cat_mac.sh) once the L8 bases are done (owner, 2026-10-21: one GPU job
# at a time; the CAT repaint was paused for the bases). If the bases stop with an error instead, nothing is started.
cd "/Users/alinachuang/Digipen fall 2026/sell-anything/ml" || exit 1
LOG=data/processed/brush_l/l8_gen.log
echo "waiting for the L8 bases $(date)"
until grep -q "L8 bases finished\|unexpected exit" $LOG; do sleep 120; done
if grep -q "unexpected exit" $LOG; then echo "L8 bases stopped with an error, CAT repaint not restarted $(date)"; exit 1; fi
sleep 60
echo "L8 bases finished; restarting the CAT repaint $(date)"
nohup tools/run_repaint_cat_mac.sh >> data/processed/repaint/mac_cat.log 2>&1 &
