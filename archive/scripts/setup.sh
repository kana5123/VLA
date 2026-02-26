#!/bin/bash
set -e

echo "========================================="
echo " ATLASVLA Environment Setup"
echo "========================================="

# ── CUDA 12.1 ──────────────────────────────────────────────
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

# ── 가상환경 생성 ───────────────────────────────────────────
echo ""
echo "[1/3] 가상환경 생성 중..."
python3 -m venv venv
source venv/bin/activate

# ── pip 업그레이드 ──────────────────────────────────────────
echo "[2/3] pip 업그레이드 중..."
pip install --upgrade pip

# ── 패키지 설치 ─────────────────────────────────────────────
echo "[3/3] 패키지 설치 중..."
pip install torch torchvision --index-url "$TORCH_INDEX"
pip install -r requirements.txt

# ── 완료 ────────────────────────────────────────────────────
echo ""
echo "========================================="
echo " 설치 완료!"
echo "========================================="
echo ""
echo "사용법:"
echo "  source venv/bin/activate"
echo "  python run_pipeline.py --all"
echo ""
