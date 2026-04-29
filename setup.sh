#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh  —  One-time environment setup for the QC Evaluation Pipeline
#
# Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "════════════════════════════════════════════════════════"
echo "  QC Pipeline — Environment Setup"
echo "════════════════════════════════════════════════════════"

# ── 1. Ensure pip is available ────────────────────────────────────────────────
echo "[1/4] Checking pip …"
if ! command -v pip3 &>/dev/null && ! python3 -m pip --version &>/dev/null 2>&1; then
    echo "pip not found. Installing …"
    sudo apt-get update -qq && sudo apt-get install -y python3-pip
fi
PIP="python3 -m pip"

# ── 2. (Optional) Create a virtual environment ────────────────────────────────
echo "[2/4] Setting up virtual environment …"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created at ./venv"
else
    echo "  Virtual environment already exists."
fi
source venv/bin/activate

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "[3/4] Installing Python dependencies …"
pip install --upgrade pip --quiet

# Install PyTorch (CPU build by default; comment out and use next line for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

# Install CLIP directly from GitHub
pip install git+https://github.com/openai/CLIP.git --quiet

# Install remaining deps from requirements.txt (excluding torch lines)
pip install \
    pymongo \
    python-dotenv \
    opencv-python \
    Pillow \
    transformers \
    accelerate \
    sentencepiece \
    mediapipe \
    numpy \
    tqdm \
    loguru \
    --quiet

echo "[3/4] Dependencies installed ✓"

# ── 4. Create logs dir ────────────────────────────────────────────────────────
echo "[4/4] Creating logs directory …"
mkdir -p logs

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Activate the environment:  source venv/bin/activate"
echo "  Run single pass:           python main.py"
echo "  Run as daemon:             python main.py --daemon --interval 120"
echo "════════════════════════════════════════════════════════"
