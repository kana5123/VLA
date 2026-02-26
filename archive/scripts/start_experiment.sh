#!/bin/bash
# One-shot experiment launcher — run this manually:
#   bash start_experiment.sh
cd /home/kana5123/ATLASVLA
nohup bash run_experiment.sh --gpus 1,2,3,4 --num_eval_episodes 200 > outputs/experiment_run.log 2>&1 &
echo "Experiment started! PID: $!"
echo "Monitor with: tail -f outputs/experiment_run.log"
