#!/usr/bin/env bash
# deploy_to_server.sh
# Deploy the phase_optimizer package and notebooks to an Ubuntu GPU server.
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
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
    MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    echo "  Detected CUDA $CUDA_VERSION"

    if   [[ "$MAJOR" -ge 12 && "$MINOR" -ge 8 ]]; then TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    elif [[ "$MAJOR" -ge 12 && "$MINOR" -ge 6 ]]; then TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
    elif [[ "$MAJOR" -ge 12 && "$MINOR" -ge 4 ]]; then TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    elif [[ "$MAJOR" -ge 12 && "$MINOR" -ge 1 ]]; then TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    elif [[ "$MAJOR" -ge 11 && "$MINOR" -ge 8 ]]; then TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    else
        echo "WARNING: CUDA $CUDA_VERSION is older than 11.8 — using CPU PyTorch."
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    fi
    echo "  Using PyTorch wheel index: $TORCH_INDEX_URL"
fi

# ─────────────────────────────────────────────────────────────
# Step 2: Generate server pyproject.toml with correct CUDA index
# ─────────────────────────────────────────────────────────────
echo "[2/5] Generating server pyproject.toml..."
TEMP_PYPROJECT=$(mktemp /tmp/pyproject_XXXXXX.toml)
sed "s|https://download.pytorch.org/whl/cu128|$TORCH_INDEX_URL|g" \
    "$SCRIPT_DIR/pyproject.server.toml" > "$TEMP_PYPROJECT"

# ─────────────────────────────────────────────────────────────
# Step 3: Sync project files to server
# ─────────────────────────────────────────────────────────────
echo "[3/5] Syncing project files..."
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/output"

rsync -avz --delete \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.py[cod]' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='output/' \
    --exclude='*.npy' \
    --exclude='*.mat' \
    --exclude='*.pkl' \
    --exclude='*.zip' \
    --exclude='.claude/' \
    --exclude='pyproject.toml' \
    --exclude='uv.lock' \
    "$SCRIPT_DIR/" "$REMOTE:$REMOTE_DIR/"

# Upload the generated pyproject.toml (overrides the excluded one)
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
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
echo "uv version: $(uv --version)"

# Install Python 3.11 and sync dependencies (installs phase_optimizer package)
uv python install 3.11
uv sync --no-dev

echo ""
echo "Environment ready:"
uv run python -c "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import phase_optimizer; print(f'  phase_optimizer: OK')"
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
echo "  ~/.local/bin/uv run jupyter lab --no-browser --port=8888 --notebook-dir=notebooks"
echo ""
echo "Then open http://localhost:8888 in your local browser."
echo "─────────────────────────────────────────────────────────"
