# Phase Optimizer

GPU-accelerated SLM phase optimization for super-resolution and extended-DOF multi-aperture imaging.

Generates phase patterns equivalent to an M×M micro-lens array (MLA) on a Spatial Light Modulator (SLM), optimized for uniform PSF arrays, high focusing efficiency, and configurable depth of field.

---

## Project Structure

```
phase_optimizer/
├── phase_optimizer/           # Installable Python package
│   ├── core/                  # Pure optical computation (no GUI, no I/O)
│   │   ├── phase_generator.py # PhaseGenerator class — Fresnel & optimized modes
│   │   ├── wave_propagation.py# ASM / RSC wave propagation algorithms
│   │   └── optics_utils.py    # PSF centers, Gaussian templates, masks, I/O helpers
│   ├── gui/                   # Jupyter ipywidget interfaces
│   │   ├── optimizer_gui.py   # PhaseOptimizerGUI — job configuration & queue
│   │   ├── test_gui.py        # QuickTestGUI — rapid single-run testing
│   │   └── file_selector.py   # NPYFileSelector — browse & select output files
│   ├── batch_processor.py     # process_jobs() + JobBrowserGUI
│   ├── visualization.py       # Plotting utilities (live, 2D, cross-section, efficiency)
│   └── config.py              # Physical constants & default parameters
│
├── config/                    # JSON configuration files
│   ├── base.json              # Default M9 high-res config
│   ├── base_M5.json           # M5 medium-res config
│   ├── 0226config/            # Experimental parameter sweeps
│   └── crop_configs/          # Mask / ROI configurations
│
├── notebooks/                 # Jupyter workflow entry points
│   ├── optimize_multiple.ipynb# Primary: GUI → batch optimize → browse results
│   ├── capture_PSF.ipynb      # Hardware: SLM upload + Z-scan PSF acquisition
│   └── test_slm_optimizer.ipynb# Quick single-run testing
│
├── tests/
│   └── test_checkpoint_gradient.py  # PyTorch gradient checkpoint verification
│
├── output/                    # Generated phase patterns (git-ignored)
├── pyproject.toml
└── deploy_to_server.sh
```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
# Clone the repository
git clone https://github.com/yifengzeng2/phase_optimizer.git
cd phase_optimizer

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

> **GPU note:** PyTorch is configured to use CUDA 12.8. Adjust `pyproject.toml` if your CUDA version differs.

---

## Quick Start

### 1. Interactive Test (single run)

Open `notebooks/test_slm_optimizer.ipynb` in Jupyter and run:

```python
from phase_optimizer.gui.test_gui import QuickTestGUI
gui = QuickTestGUI()
gui.display()
```

Select a mode (Fresnel or Optimized), adjust parameters, and click **Run**.

### 2. Batch Optimization

Open `notebooks/optimize_multiple.ipynb`:

```python
from phase_optimizer.gui.optimizer_gui import PhaseOptimizerGUI, create_optimizer_gui
from phase_optimizer.batch_processor import process_jobs

# Step 1: Configure jobs interactively
gui = create_optimizer_gui(output_dir='./output')

# Step 2: Run all jobs
results = process_jobs(gui, output_dir='./output', upsampling=2.0)
```

Each job saves:
- `output/<job_title>/<job_title>.npy` — 8-bit phase pattern
- `output/<job_title>/<job_title>.json` — parameters
- `output/<job_title>/<job_title>_optimizer.pkl` — optimizer state

### 3. Browse & Visualize Results

```python
from phase_optimizer.batch_processor import browse_jobs
browser = browse_jobs('./output')
optimizer = browser.get_current_optimizer()
```

### 4. Use in Code

```python
import torch
from phase_optimizer.core import PhaseGenerator
from phase_optimizer import config

params = {
    **config.COMMON_DEFAULTS,
    **config.OPTIMIZED_DEFAULTS,
    'shape': config.SLM_SHAPE,
    'M': 5,
    'focal_length': 73.9e-3,
    'two_pi_value': 210,
    # ... other params
}

gen = PhaseGenerator(params, device=torch.device('cuda'), mode='fresnel')
gen.generate()
phase_8bit = gen.update_phase_8bit()
```

---

## Key Parameters

| Parameter | Description |
|---|---|
| `M` | MLA size: M×M sub-apertures |
| `focal_length` | Focal length in meters |
| `overlap_ratio` | Sub-aperture overlap [0, 1). Increases effective aperture, reduces DOF |
| `airy_correction` | Scale factor for diffraction-limited spot target |
| `depth_in_focus` | List of z-depths (in DOF units) to maintain focus |
| `weights` | Dict of loss term weights: `mse`, `eff_mean`, `eff_std`, `depth_in_focus` |

---

## Hardware (not in repo)

The following files interface with lab hardware and are excluded from version control:

- `hardware.py` — SLM, Z-stage, camera control via RPyC
- `meadowlark.py` — Meadowlark SLM driver (requires vendor SDK)
- `nas_mapper.py` — NAS network drive mapping

---

## Tech Stack

- **PyTorch** ≥ 2.8 — GPU-accelerated optimization & wave propagation
- **NumPy / SciPy** — numerical ops
- **Matplotlib + ipywidgets** — interactive Jupyter GUIs
- **uv** — fast Python package management
