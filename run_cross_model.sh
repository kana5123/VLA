#!/bin/bash
# Cross-model attention sink extraction — GPU parallel
# Uses Bridge V2 dataset (cached at /ceph_data/kana5123/bridge_data_cache/)
set -e
export HF_HOME=/ceph_data/kana5123/hf_cache
export PATH="/home/kana5123/miniconda3/envs/interp/bin:$PATH"
export PYTHONPATH="/home/kana5123/ATLASVLA:$PYTHONPATH"
cd /home/kana5123/ATLASVLA

# Also search home cache for already-downloaded models
export TRANSFORMERS_CACHE=/ceph_data/kana5123/hf_cache/hub
export HF_HUB_CACHE=/ceph_data/kana5123/hf_cache/hub

DATASET="bridge_v2"

run_model() {
    local gpu=$1
    local model=$2
    echo "[GPU $gpu] Starting $model on $DATASET at $(date)"
    CUDA_VISIBLE_DEVICES=$gpu python cross_model_extract.py \
        --model "$model" --dataset "$DATASET" --device cuda:0 \
        --episode 0 --step 0 2>&1 | tee "outputs/cross_model_analysis/${model}_${DATASET}.log"
    echo "[GPU $gpu] Done: $model at $(date)"
}

echo "============================================"
echo "Cross-Model Sink Extraction — Bridge V2"
echo "Start: $(date)"
echo "============================================"

# GPU 5: OpenVLA-7B (~15GB) — already extracted, skip
echo "[GPU 5] OpenVLA already extracted, skipping"

# GPU 6: ECoT-7B (~15GB)
run_model 6 ecot-7b &
PID_ECOT=$!

# GPU 7: TraceVLA (9GB) → SpatialVLA (9GB) → SmolVLA (2GB) sequential
(
    run_model 7 tracevla-phi3v
    run_model 7 spatialvla-4b
    # SmolVLA needs special loading — skip for now if fails
    run_model 7 smolvla-base || echo "[GPU 7] SmolVLA failed (may need custom loader)"
) &
PID_SMALL=$!

echo "Waiting for extractions..."
echo "  GPU 6 (ECoT): PID $PID_ECOT"
echo "  GPU 7 (Small): PID $PID_SMALL"

wait $PID_ECOT
echo "ECoT extraction complete!"

wait $PID_SMALL
echo "Small model extractions complete!"

echo "============================================"
echo "All extractions complete at $(date)!"
echo "============================================"

# List results
echo ""
echo "Results:"
find outputs/cross_model_analysis/ -name "*_perhead.json" -exec ls -la {} \;
