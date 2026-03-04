#!/usr/bin/env bash
# deploy_to_server.sh
# Deploy test_slm_optimizer.ipynb and dependencies to Ubuntu server.
# Usage: bash deploy_to_server.sh [user@host] [remote_dir]
#
# Requirements (local): ssh, rsync
# The server only needs: Ubuntu, nvidia driver, internet access

set -euo pipefail

REMOTE="${1:-ubuntu@192.222.59.8}"
REMOTE_DIR="${2:-~/slm_project}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== SLM Project Server Deployment ==="
echo "Target: $REMOTE:$REMOTE_DIR"
echo ""

# ─────────────────────────────────────────────────────────────
# Step 1: Detect CUDA version on server to pick correct PyTorch wheel
# ─────────────────────────────────────────────────────────────
echo "[1/5] Detecting CUDA version on server..."
CUDA_VERSION=$(ssh "$REMOTE" "nvidia-smi 2>/dev/null | grep 'CUDA Version' | awk '{print \$9}'" || echo "")

if [[ -z "$CUDA_VERSION" ]]; then
    echo "WARNING: nvidia-smi not found or no GPU detected. Falling back to CPU PyTorch."
    CUDA_TAG="cpu"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
    MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    echo "  Detected CUDA $CUDA_VERSION"

    if   [[ "$MAJOR" -ge 12 && "$MINOR" -ge 8 ]]; then CUDA_TAG="cu128"; TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    elif [[ "$MAJOR" -ge 12 && "$MINOR" -ge 6 ]]; then CUDA_TAG="cu126"; TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
    elif [[ "$MAJOR" -ge 12 && "$MINOR" -ge 4 ]]; then CUDA_TAG="cu124"; TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    elif [[ "$MAJOR" -ge 12 && "$MINOR" -ge 1 ]]; then CUDA_TAG="cu121"; TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    elif [[ "$MAJOR" -ge 11 && "$MINOR" -ge 8 ]]; then CUDA_TAG="cu118"; TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    else
        echo "WARNING: CUDA $CUDA_VERSION is older than 11.8 — using CPU PyTorch."
        CUDA_TAG="cpu"; TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    fi
    echo "  Using PyTorch wheel index: $TORCH_INDEX_URL"
fi

# ─────────────────────────────────────────────────────────────
# Step 2: Generate server pyproject.toml with correct CUDA index
# ─────────────────────────────────────────────────────────────
echo "[2/5] Generating server pyproject.toml (CUDA tag: $CUDA_TAG)..."
TEMP_PYPROJECT=$(mktemp /tmp/pyproject_XXXXXX.toml)
sed "s|https://download.pytorch.org/whl/cu128|$TORCH_INDEX_URL|g" \
    "$SCRIPT_DIR/pyproject.server.toml" > "$TEMP_PYPROJECT"

# ─────────────────────────────────────────────────────────────
# Step 3: Sync project files to server (exclude .venv, __pycache__, etc.)
# ─────────────────────────────────────────────────────────────
echo "[3/5] Syncing project files..."
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/config $REMOTE_DIR/output"

# Bundle project files into a tar archive and stream to server
tar -czf - \
    -C "$SCRIPT_DIR" \
    test_slm_optimizer.ipynb \
    test_utils.py \
    phase_generators.py \
    optics_utils.py \
    visualization.py \
    wave_propagation.py \
    config.py \
    .python-version \
    $(for f in atf.mat atf_fresnel.mat pupil.mat; do [[ -f "$SCRIPT_DIR/$f" ]] && echo "$f"; done) \
  | ssh "$REMOTE" "tar -xzf - -C $REMOTE_DIR"

# Copy config file
scp "$SCRIPT_DIR/config/base.json" "$REMOTE:$REMOTE_DIR/config/base.json"

# Upload the generated pyproject.toml
scp "$TEMP_PYPROJECT" "$REMOTE:$REMOTE_DIR/pyproject.toml"
rm -f "$TEMP_PYPROJECT"

# ─────────────────────────────────────────────────────────────
# Step 4: Set up uv + Python environment on server
# ─────────────────────────────────────────────────────────────
echo "[4/5] Setting up Python environment on server..."
ssh "$REMOTE" bash -s << 'REMOTE_SETUP'
set -euo pipefail
PROJECT_DIR="$HOME/slm_project"
cd "$PROJECT_DIR"

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "uv version: $(uv --version)"

# Install Python 3.11 and sync dependencies
uv python install 3.11
uv sync --no-dev

echo ""
echo "Environment ready. Packages installed:"
uv run python -c "import torch; print(f'  torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
uv run python -c "import scipy; print(f'  scipy: {scipy.__version__}')"
REMOTE_SETUP

# ─────────────────────────────────────────────────────────────
# Step 5: Print how to start JupyterLab
# ─────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "To start JupyterLab on the server, run:"
echo ""
echo "  ssh -L 8888:localhost:8888 $REMOTE"
echo "  cd ~/slm_project"
echo "  ~/.local/bin/uv run jupyter lab --no-browser --port=8888"
echo ""
echo "Then open http://localhost:8888 in your local browser."
echo "─────────────────────────────────────────────────────────"
