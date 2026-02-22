#!/bin/bash
# Setup LIBERO benchmark dependencies.
#
# Run from the ATLASVLA directory:
#   bash setup_libero.sh
#
# Prerequisites: conda env 'interp' must be activated.

set -e

CONDA_ENV="interp"
PYTHON="/home/kana5123/miniconda3/envs/${CONDA_ENV}/bin/python"
PIP="/home/kana5123/miniconda3/envs/${CONDA_ENV}/bin/pip"

echo "=== Installing LIBERO dependencies into conda env: ${CONDA_ENV} ==="

# 1. MuJoCo (Apache 2.0 since v2.1.2)
echo "[1/4] Installing MuJoCo..."
${PIP} install mujoco>=3.0.0

# 2. robosuite (MuJoCo-based robot simulation)
echo "[2/4] Installing robosuite..."
${PIP} install robosuite>=1.4.0

# 3. LIBERO benchmark
echo "[3/4] Installing LIBERO..."
${PIP} install libero

# 4. EGL rendering setup (for headless GPU rendering)
echo "[4/4] Setting up EGL rendering..."
export MUJOCO_GL=egl
echo 'export MUJOCO_GL=egl' >> ~/.bashrc

# Verify installation
echo ""
echo "=== Verification ==="
${PYTHON} -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
${PYTHON} -c "import robosuite; print(f'robosuite: {robosuite.__version__}')"
${PYTHON} -c "import libero; print('LIBERO: OK')"

echo ""
echo "=== LIBERO setup complete ==="
echo "Run: MUJOCO_GL=egl python libero_eval.py --model openvla-7b --suite libero_spatial"
