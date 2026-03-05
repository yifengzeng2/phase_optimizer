# Deployment Guide

Step-by-step instructions for deploying the `phase_optimizer` repo on a fresh Linux cloud computing server and running the Jupyter notebooks interactively from a local machine.

---

## Requirements

- Linux server with an NVIDIA GPU (tested on NVIDIA B200, CUDA 13.0)
- SSH access to the server
- Python 3.11+ on the server (`uv` will manage the venv)

---

## 1. Clone the Repository

```bash
git clone https://github.com/yifengzeng2/phase_optimizer.git
cd phase_optimizer
```

---

## 2. Install `uv` (if not already installed)

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

---

## 3. Set Up the Environment

`uv` reads `pyproject.toml` and `uv.lock` to install all dependencies (including PyTorch with CUDA 12.8 support, which is compatible with CUDA 13.x drivers).

```bash
uv sync
```

This creates a `.venv` virtual environment in the project directory and installs ~133 packages including `torch`, `numpy`, `scipy`, `matplotlib`, and `jupyter`.

---

## 4. Register the Jupyter Kernel

```bash
.venv/bin/python -m ipykernel install --user --name phase_optimizer --display-name "Phase Optimizer (Python 3.11)"
```

---

## 5. Start JupyterLab

```bash
.venv/bin/jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --IdentityProvider.token=phaseopt \
  --notebook-dir=/path/to/phase_optimizer \
  --allow-root > /tmp/jupyter.log 2>&1 &
```

Replace `/path/to/phase_optimizer` with the actual path (e.g. `/workspace/phase_optimizer`).

Check it started:

```bash
cat /tmp/jupyter.log | grep "running at"
```

---

## 6. Connect from Your Local Machine

Since the server does not expose port 8888 publicly, use SSH port forwarding.

Run this on your **local machine**:

```bash
ssh -p <SSH_PORT> root@<SERVER_IP> -L 8888:localhost:8888
```

Replace `<SSH_PORT>` and `<SERVER_IP>` with your server's SSH port and public IP.

> Example from this deployment: `ssh -p 41280 root@52.32.147.192 -L 8888:localhost:8888`

Then open in your local browser:

```
http://localhost:8888/lab?token=phaseopt
```

---

## 7. Open the Notebook

In the JupyterLab file browser, navigate to `notebooks/test_slm_optimizer.ipynb`.

When prompted for a kernel, select **"Phase Optimizer (Python 3.11)"**.

---

## 8. Downloading Output Files

To copy output files from the server to your local machine, run on your **local machine**:

```bash
scp -P <SSH_PORT> -r root@<SERVER_IP>:/path/to/phase_optimizer/notebooks/output ./output
```

Or with `rsync` (resumable, preferred for large files):

```bash
rsync -avz -e "ssh -p <SSH_PORT>" root@<SERVER_IP>:/path/to/phase_optimizer/notebooks/output ./output
```

---

## Default Optimization Parameters

The following defaults are set in `phase_optimizer/config.py` and `phase_optimizer/gui/test_gui.py`:

| Parameter     | Value  |
|---------------|--------|
| Learning Rate | 0.316  |
| Iterations    | 120    |

---

## Useful Commands

| Task | Command (on server) |
|------|---------------------|
| Check GPU | `nvidia-smi` |
| Find server IP | `curl http://checkip.amazonaws.com` |
| Find server user | `whoami` |
| Stop JupyterLab | `kill $(pgrep -f "jupyter lab")` |
| View Jupyter logs | `cat /tmp/jupyter.log` |
