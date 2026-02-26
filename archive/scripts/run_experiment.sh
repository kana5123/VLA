#!/bin/bash
# Wrapper to run adapter experiment within conda environment
set -e

# Use conda env Python directly (conda run has issues with --include-user-site)
export PATH="/home/kana5123/miniconda3/envs/interp/bin:$PATH"
export PYTHONPATH="/home/kana5123/ATLASVLA:$PYTHONPATH"

cd /home/kana5123/ATLASVLA

echo "Python: $(which python)"
python -c "import torch; import accelerate; import absl; print(f'torch {torch.__version__}, accelerate {accelerate.__version__}, absl OK')"

# Run experiment with provided arguments
python run_adapter_experiment.py "$@"
