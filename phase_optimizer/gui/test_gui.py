# test_utils.py
"""
Utility functions and GUI for phase optimizer testing.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display, clear_output
import ipywidgets as widgets
from typing import Optional, Dict, Any

from ..core.optics_utils import compute_psf_centers, save_dict_as_json, load_dict_from_json, get_best_gpu
from ..core.phase_generator import PhaseGenerator
from ..visualization import plot_phase, plot_2d_comparisons, plot_2d_comparisons_interactive, plot_energy_distribution


# =============================================================================
# QuickTestGUI - Main GUI for testing
# =============================================================================

class QuickTestGUI:
    """Compact tab-based GUI for quick phase optimization testing."""

    BASE_CONFIG_PATH = r"./config/base.json"

    def __init__(self, config_path: Optional[str] = None):
        self.device = get_best_gpu()
        self.optimizer = None
        self.phase_8bit = None
        self.hologram_image = None
        self.config_path = config_path or self.BASE_CONFIG_PATH

        self._build()
        self._load_config(self.config_path)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build(self):
        S = {'description_width': '120px'}
        W = widgets.Layout(width='300px')
        WN = widgets.Layout(width='220px')

        # --- Tab 0: Setup & Run ---
        self.w_mode = widgets.ToggleButtons(
            options=['fresnel', 'optimized', 'hologram'], value='fresnel',
            description='Mode:', style={'description_width': '50px', 'button_width': '90px'}
        )
        self.w_M = widgets.IntSlider(value=5, min=1, max=15, description='M (array):', style=S, layout=W)
        self.w_N = widgets.IntText(value=4096, description='N (pixels):', style=S, layout=WN)
        self.w_focal_length = widgets.FloatText(value=73.9, description='Focal (mm):', style=S, layout=WN)
        self.w_pixel_size  = widgets.FloatText(value=4.4,  description='Pixel (µm):',  style=S, layout=WN)
        self.w_wavelength  = widgets.FloatText(value=520.0, description='λ (nm):',      style=S, layout=WN)
        self.w_upsampling  = widgets.FloatSlider(value=2.0, min=1.0, max=4.0, step=0.5,
                                                  description='Upsampling:', style=S, layout=W, readout_format='.1f')
        self.w_info = widgets.HTML()

        # hologram sub-section
        self.w_hologram_path = widgets.Text(value='', description='Image:', placeholder='path to image',
                                             style=S, layout=widgets.Layout(width='400px'))
        self.w_hologram_load_btn = widgets.Button(description='Load & Preview', button_style='info',
                                                   layout=widgets.Layout(width='130px'))
        self.w_hologram_preview = widgets.Output(layout=widgets.Layout(height='200px'))
        self.hologram_section = widgets.VBox([
            self.w_hologram_path,
            self.w_hologram_load_btn,
            self.w_hologram_preview,
        ])
        self.hologram_section.layout.display = 'none'

        self.w_run_btn = widgets.Button(description='▶  Run', button_style='success',
                                         layout=widgets.Layout(width='140px', height='36px'))
        tab0 = widgets.VBox([
            self.w_mode,
            widgets.HBox([self.w_M, self.w_N]),
            widgets.HBox([self.w_focal_length, self.w_pixel_size, self.w_wavelength]),
            self.w_upsampling,
            self.hologram_section,
            self.w_info,
            self.w_run_btn,
        ], layout=widgets.Layout(padding='10px'))

        # --- Tab 1: Optimizer ---
        self.w_overlap_ratio  = widgets.FloatSlider(value=0.9, min=0.0, max=1.0, step=0.05,
                                                     description='Overlap:', style=S, layout=W, readout_format='.2f')
        self.w_airy_correction = widgets.FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1,
                                                      description='Airy corr:', style=S, layout=W, readout_format='.1f')
        self.w_center_blend   = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1,
                                                     description='Center blend:', style=S, layout=W, readout_format='.1f')
        self.w_randomness     = widgets.FloatSlider(value=0.0, min=0.0, max=0.5, step=0.05,
                                                     description='Randomness:', style=S, layout=W, readout_format='.2f')
        self.w_random_seed    = widgets.IntText(value=42, description='Seed:', style=S, layout=WN)
        self.w_use_seed       = widgets.Checkbox(value=True, description='Use seed', indent=False)
        self.w_depth_in_focus = widgets.Text(value='-0.5, 0.5', description='Depth (DOF):', style=S,
                                              placeholder='-0.5, 0.5', layout=W)
        self.w_dof_correction = widgets.FloatSlider(value=1.0, min=0.1, max=3.0, step=0.1,
                                                     description='DOF corr:', style=S, layout=W, readout_format='.1f')
        self.w_lr  = widgets.FloatLogSlider(value=0.05, base=10, min=-2, max=0, step=0.1,
                                             description='Learn rate:', style=S, layout=W, readout_format='.3f')
        self.w_ni  = widgets.IntSlider(value=500, min=10, max=2000, step=10,
                                        description='Iterations:', style=S, layout=W)
        self.w_mask_count   = widgets.IntSlider(value=0, min=0, max=9, description='Mask count:', style=S, layout=W)
        self.w_interleaving = widgets.Dropdown(options=['checkerboard', 'coarse1', 'coarse2', 'coarse3'],
                                                value='coarse1', description='Interleaving:', style=S, layout=WN)

        ws = {'description_width': '80px'}
        wl = widgets.Layout(width='145px')
        self.w_weight_mse      = widgets.FloatText(value=1.0,  description='MSE:',     style=ws, layout=wl)
        self.w_weight_depth    = widgets.FloatText(value=1.0,  description='Depth:',   style=ws, layout=wl)
        self.w_weight_eff_mean = widgets.FloatText(value=20.0, description='Eff mean:', style=ws, layout=wl)
        self.w_weight_eff_std  = widgets.FloatText(value=50.0, description='Eff std:',  style=ws, layout=wl)

        tab1 = widgets.VBox([
            widgets.HTML('<b>MLA</b>'),
            self.w_overlap_ratio, self.w_airy_correction, self.w_center_blend,
            self.w_randomness, widgets.HBox([self.w_random_seed, self.w_use_seed]),
            widgets.HTML('<b>Depth</b>'),
            self.w_depth_in_focus, self.w_dof_correction,
            widgets.HTML('<b>Training</b>'),
            self.w_lr, self.w_ni,
            widgets.HTML('<b>Mask</b>'),
            self.w_mask_count, self.w_interleaving,
            widgets.HTML('<b>Loss weights</b>'),
            widgets.HBox([self.w_weight_mse, self.w_weight_depth,
                          self.w_weight_eff_mean, self.w_weight_eff_std]),
        ], layout=widgets.Layout(padding='10px'))

        # --- Tab 2: Save / Config ---
        ts = {'description_width': '90px'}
        self.w_phase_save_path = widgets.Text(value='./output', description='Save dir:',
                                               style=ts, layout=widgets.Layout(width='360px'))
        self.w_phase_filename  = widgets.Text(value='', description='Filename:',
                                               placeholder='auto: M5_optimized_phase.npy',
                                               style=ts, layout=widgets.Layout(width='360px'))
        self.w_save_phase_btn  = widgets.Button(description='Save Phase (.npy)', button_style='warning',
                                                 layout=widgets.Layout(width='160px'))
        self.w_config_path     = widgets.Text(value=self.BASE_CONFIG_PATH, description='Config path:',
                                               style=ts, layout=widgets.Layout(width='360px'))
        self.w_load_config_btn = widgets.Button(description='Load Config', layout=widgets.Layout(width='120px'))
        self.w_save_config_btn = widgets.Button(description='Save Config', layout=widgets.Layout(width='120px'))

        tab2 = widgets.VBox([
            widgets.HTML('<b>Phase</b>'),
            self.w_phase_save_path, self.w_phase_filename, self.w_save_phase_btn,
            widgets.HTML('<b>Config</b>'),
            self.w_config_path,
            widgets.HBox([self.w_load_config_btn, self.w_save_config_btn]),
        ], layout=widgets.Layout(padding='10px'))

        # --- Tabs ---
        tabs = widgets.Tab(children=[tab0, tab1, tab2])
        tabs.set_title(0, '⚙ Setup & Run')
        tabs.set_title(1, '🎛 Optimizer')
        tabs.set_title(2, '💾 Save / Config')

        # --- Action buttons (always visible) ---
        self.w_visualize_btn = widgets.Button(description='Visualize', button_style='info',
                                               layout=widgets.Layout(width='110px'))
        self.w_analyze_btn   = widgets.Button(description='Efficiency', button_style='primary',
                                               layout=widgets.Layout(width='110px'))

        # --- Outputs ---
        self.w_log_output  = widgets.Output(
            layout=widgets.Layout(width='100%', max_height='200px', overflow_y='auto',
                                  border='1px solid #ddd', padding='4px'))
        self.w_plot_output = widgets.Output(layout=widgets.Layout(width='100%'))

        self.main_layout = widgets.VBox([
            widgets.HTML('<h2 style="margin:4px 0">DOE Phase Generator</h2>'),
            tabs,
            widgets.HBox([self.w_visualize_btn, self.w_analyze_btn],
                         layout=widgets.Layout(margin='6px 0')),
            self.w_log_output,
            self.w_plot_output,
        ])

        # --- Callbacks ---
        self.w_run_btn.on_click(self._on_run)
        self.w_visualize_btn.on_click(self._on_visualize)
        self.w_analyze_btn.on_click(self._on_analyze)
        self.w_save_phase_btn.on_click(self._on_save_phase)
        self.w_save_config_btn.on_click(self._on_save_config)
        self.w_load_config_btn.on_click(self._on_load_config)
        self.w_mode.observe(self._on_mode_change, names='value')
        self.w_hologram_load_btn.on_click(self._on_load_hologram)
        for w in [self.w_M, self.w_N, self.w_overlap_ratio, self.w_randomness, self.w_pixel_size]:
            w.observe(self._update_info, names='value')

    # ------------------------------------------------------------------
    # Info banner
    # ------------------------------------------------------------------
    def _update_info(self, _=None):
        M, N = self.w_M.value, self.w_N.value
        overlap = self.w_overlap_ratio.value
        px_um = self.w_pixel_size.value
        phys_mm = px_um * N / 1000
        if overlap > 0 and M > 1:
            alpha = (overlap * (N - N / M) + N / M) / (N / M)
            dof_tag = f'α={alpha:.2f}, DOF×{1/alpha**2:.2f}'
        else:
            dof_tag = 'standard'
        rand = self.w_randomness.value
        rand_tag = f'σ={rand * N / M:.1f}px' if rand > 0 else 'off'
        self.w_info.value = (
            f'<div style="background:#f0f4f8;padding:6px 10px;border-radius:4px;'
            f'font-size:0.88em;color:#333;margin:4px 0">'
            f'<b>{M}×{M}</b> · {N}px · {phys_mm:.2f}mm aperture &nbsp;|&nbsp; '
            f'overlap: {dof_tag} &nbsp;|&nbsp; rand: {rand_tag} &nbsp;|&nbsp; {self.device}'
            f'</div>')

    # ------------------------------------------------------------------
    # Params get / set
    # ------------------------------------------------------------------
    def _get_params(self):
        try:
            depth_list = [float(x.strip()) for x in self.w_depth_in_focus.value.split(',') if x.strip()]
        except Exception:
            depth_list = [-0.5, 0.5]
        N = self.w_N.value
        return {
            'M': self.w_M.value, 'N': N, 'output_size': N,
            'focal_length': self.w_focal_length.value * 1e-3,
            'pixel_size': self.w_pixel_size.value * 1e-6,
            'wavelength': self.w_wavelength.value * 1e-9,
            'overlap_ratio': self.w_overlap_ratio.value,
            'airy_correction': self.w_airy_correction.value,
            'center_blend': self.w_center_blend.value,
            'randomness': self.w_randomness.value,
            'random_seed': self.w_random_seed.value if self.w_use_seed.value else None,
            'depth_in_focus': depth_list, 'depth_out_focus': None,
            'dof_correction': self.w_dof_correction.value,
            'lr': self.w_lr.value, 'ni': self.w_ni.value,
            'show_iters': max(10, self.w_ni.value // 10),
            'mask_count': self.w_mask_count.value,
            'interleaving': self.w_interleaving.value,
            'masked_airy_correction': 1.0, 'focusing_eff_correction': 1.0, 'psf_energy_level': 1.0,
            'upsampling': self.w_upsampling.value,
            'weights': {
                'mse': self.w_weight_mse.value,
                'depth_in_focus': self.w_weight_depth.value,
                'depth_out_focus': 0.0,
                'masked': 1.0 if self.w_mask_count.value > 0 else 0.0,
                'eff_mean': self.w_weight_eff_mean.value,
                'eff_std': self.w_weight_eff_std.value,
            },
            'shape': [N, N], 'roi_center_x': N // 2, 'roi_center_y': N // 2, 'two_pi_value': 255,
        }

    def _set_params(self, p):
        self.w_M.value             = p.get('M', 5)
        self.w_N.value             = p.get('N', 4096)
        self.w_focal_length.value  = p.get('focal_length', 0.0739) * 1e3
        self.w_pixel_size.value    = p.get('pixel_size', 4.4e-6) * 1e6
        self.w_wavelength.value    = p.get('wavelength', 520e-9) * 1e9
        self.w_overlap_ratio.value = p.get('overlap_ratio', 0.9)
        self.w_airy_correction.value = p.get('airy_correction', 1.0)
        self.w_center_blend.value  = p.get('center_blend', 0.0)
        self.w_randomness.value    = p.get('randomness', 0.0)
        self.w_upsampling.value    = p.get('upsampling', 2.0)
        seed = p.get('random_seed')
        self.w_use_seed.value = seed is not None
        if seed is not None:
            self.w_random_seed.value = seed
        self.w_depth_in_focus.value = ', '.join(str(d) for d in p.get('depth_in_focus', [-0.5, 0.5]))
        self.w_dof_correction.value = p.get('dof_correction', 1.0)
        self.w_lr.value  = p.get('lr', 0.05)
        self.w_ni.value  = p.get('ni', 500)
        self.w_mask_count.value   = p.get('mask_count', 0)
        self.w_interleaving.value = p.get('interleaving', 'coarse1')
        w = p.get('weights', {})
        self.w_weight_mse.value      = w.get('mse', 1.0)
        self.w_weight_depth.value    = w.get('depth_in_focus', 1.0)
        self.w_weight_eff_mean.value = w.get('eff_mean', 20.0)
        self.w_weight_eff_std.value  = w.get('eff_std', 50.0)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_mode_change(self, change):
        self.hologram_section.layout.display = 'flex' if change['new'] == 'hologram' else 'none'
        self._update_info()

    def _on_load_hologram(self, _):
        path = self.w_hologram_path.value.strip()
        with self.w_log_output:
            if not path or not os.path.exists(path):
                print(f"File not found: {path}"); return
            try:
                from PIL import Image
                img = np.array(Image.open(path))
                self.hologram_image = img
                print(f"Loaded: {path}  shape={img.shape}")
            except Exception as e:
                print(f"Load error: {e}"); return
        with self.w_hologram_preview:
            clear_output(wait=True); plt.close('all')
            gray = img[..., :3] @ [0.2989, 0.5870, 0.1140] if img.ndim == 3 else img.astype(np.float32)
            H, W = gray.shape
            if H != W:
                s = max(H, W); pad = np.zeros((s, s), np.float32)
                pad[(s-H)//2:(s-H)//2+H, (s-W)//2:(s-W)//2+W] = gray; gray = pad
            fig, ax = plt.subplots(1, 2, figsize=(7, 3))
            ax[0].imshow(img); ax[0].set_title('Original'); ax[0].axis('off')
            ax[1].imshow(gray, cmap='gray'); ax[1].set_title(f'Greyscale ({gray.shape[0]}×{gray.shape[0]})'); ax[1].axis('off')
            plt.tight_layout(); plt.show()

    def _on_run(self, _):
        with self.w_log_output:
            clear_output()
            params = self._get_params()
            mode = self.w_mode.value
            print(f"Running {mode}: M={params['M']}, N={params['N']}, device={self.device}")
            try:
                if mode == 'hologram':
                    if self.hologram_image is None:
                        print("Load a hologram image first."); return
                    params['hologram_image'] = self.hologram_image
                self.optimizer = PhaseGenerator(params, device=self.device, mode=mode)
                self.optimizer.generate(mode=mode,
                                        upsampling=self.w_upsampling.value,
                                        visualize=False)
                self.phase_8bit = self.optimizer.update_phase_8bit()
                print(f"Done. phase shape={self.phase_8bit.shape}, dtype={self.optimizer.phase.dtype}")
            except Exception as e:
                import traceback; traceback.print_exc()
        self._update_info()

    def _on_visualize(self, _):
        if self.optimizer is None:
            with self.w_log_output: print("Run first!"); return
        with self.w_plot_output:
            clear_output(wait=True); plt.close('all')
            plot_phase(self.optimizer); plt.show()
            plot_2d_comparisons_interactive(self.optimizer, upsampling=1.0)

    def _on_analyze(self, _):
        if self.optimizer is None:
            with self.w_log_output: print("Run first!"); return
        with self.w_plot_output:
            clear_output(wait=True); plt.close('all')
            plot_energy_distribution(self.optimizer, upsampling=3); plt.show()
            if self.optimizer.history:
                for k, v in self.optimizer.history.items():
                    if v: print(f"  {k}: {v[0]:.4f} → {v[-1]:.4f}")

    def _on_save_phase(self, _):
        with self.w_log_output:
            if self.optimizer is None or self.optimizer.phase is None:
                print("No phase — run first."); return
            try:
                save_dir = self.w_phase_save_path.value.strip() or './output'
                os.makedirs(save_dir, exist_ok=True)
                name = self.w_phase_filename.value.strip()
                if not name:
                    name = f"M{self.w_M.value}_{self.w_mode.value}_phase"
                fname = name if name.endswith('.npy') else name + '.npy'
                fpath = os.path.join(save_dir, fname)
                np.save(fpath, self.optimizer.phase)
                p = self.optimizer.phase
                print(f"Saved: {fpath}  {p.shape} {p.dtype}  [{p.min():.3f}, {p.max():.3f}] rad")
            except Exception as e:
                print(f"Save error: {e}")

    def _on_save_config(self, _):
        with self.w_log_output:
            try:
                path = self.w_config_path.value.strip() or self.BASE_CONFIG_PATH
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                save_dict_as_json(self._get_params(), path)
                print(f"Config saved: {path}")
            except Exception as e:
                print(f"Save error: {e}")

    def _on_load_config(self, _):
        with self.w_log_output:
            try:
                path = self.w_config_path.value.strip() or self.BASE_CONFIG_PATH
                self._set_params(load_dict_from_json(path))
                self._update_info()
                print(f"Config loaded: {path}")
            except Exception as e:
                print(f"Load error: {e}")

    def _load_config(self, path: str):
        """Load config on startup (silent if missing)."""
        if os.path.exists(path):
            try:
                params = load_dict_from_json(path)
                self._set_params(params)
                print(f"Loaded config from: {path}")
                print(f"Applied {len(params)} config parameters")
            except Exception as e:
                print(f"Warning: could not load {path}: {e}")

    def display(self):
        self._update_info()
        display(self.main_layout)


# =============================================================================
# Visualization helper functions
# =============================================================================

def show_psf_centers_plot(
    M: int, N: int, overlap_ratio: float, center_blend: float,
    randomness: float, random_seed: Optional[int], device: torch.device
):
    """Show PSF centers visualization comparing reference vs randomized positions."""
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    result_ref = compute_psf_centers(
        M=M, overlap_ratio=overlap_ratio, center_blend=center_blend,
        z_ratio=1.0, N=N, output_size=N, device=device, randomness=0.0
    )
    centers_ref = result_ref['centers_pixel'].cpu().numpy()

    ax = axes[0]
    ax.scatter(centers_ref[:, 1], centers_ref[:, 0], c='blue', s=100, label='PSF Centers')
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Reference (No Randomness) - {N}x{N}')
    ax.grid(True, alpha=0.3); ax.legend()

    result_rand = compute_psf_centers(
        M=M, overlap_ratio=overlap_ratio, center_blend=center_blend,
        z_ratio=1.0, N=N, output_size=N, device=device,
        randomness=randomness, random_seed=random_seed
    )
    centers_rand = result_rand['centers_pixel'].cpu().numpy()

    ax = axes[1]
    ax.scatter(centers_ref[:, 1], centers_ref[:, 0], c='lightgray', s=100, marker='x', label='Reference')
    ax.scatter(centers_rand[:, 1], centers_rand[:, 0], c='red', s=100, label='Randomized')
    if randomness > 0:
        for i in range(len(centers_ref)):
            ax.annotate('', xy=(centers_rand[i, 1], centers_rand[i, 0]),
                       xytext=(centers_ref[i, 1], centers_ref[i, 0]),
                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Randomness = {randomness}, Seed = {random_seed}')
    ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout(); plt.show()

    if randomness > 0 and 'random_offsets' in result_rand:
        offsets = result_rand['random_offsets'].cpu().numpy()
        print(f"\nOffset Stats: mean=({offsets[:, 0].mean():.2f}, {offsets[:, 1].mean():.2f}), "
              f"std=({offsets[:, 0].std():.2f}, {offsets[:, 1].std():.2f}), max={np.abs(offsets).max():.2f} px")


def show_psf_slices_interactive(optimizer, upsampling: float = 2.0):
    """Show individual PSF slices with template overlay and adjustable slice position.

    Features:
    - Global intensity map with highlighted PSF regions
    - Fine control: Adjust slice position ±pixels around PSF center
    - Template overlay with different colors (orange=template, blue=actual)
    - Group navigation for viewing multiple PSFs
    """
    import matplotlib.patches as patches

    M, N = optimizer.M, optimizer.N
    N_up = int(N * upsampling)
    randomness = getattr(optimizer, 'randomness', 0.0)
    random_seed = getattr(optimizer, 'random_seed', None)

    center_info = compute_psf_centers(
        M=M, overlap_ratio=optimizer.overlap_ratio, center_blend=optimizer.center_blend,
        z_ratio=1.0, N=N_up, output_size=N_up, device=optimizer.device,
        randomness=randomness, random_seed=random_seed
    )
    centers = center_info['centers_pixel'].cpu().numpy()  # [M*M, 2] format: [x, y]

    with torch.no_grad():
        I_opt = optimizer.forward(upsampling=upsampling).cpu().numpy()

    # Get template if available
    template = None
    if hasattr(optimizer, 'total_psfs') and optimizer.total_psfs is not None:
        from scipy.ndimage import zoom as scipy_zoom
        t = optimizer.total_psfs.cpu().numpy()
        if t.shape[0] != N_up:
            template = scipy_zoom(t, upsampling, order=1)
        else:
            template = t

    pitch = N_up / M
    slice_half_width = int(pitch * 0.5)
    num_psfs = M * M
    psfs_per_group = 5
    num_groups = (num_psfs + psfs_per_group - 1) // psfs_per_group

    # Create widgets for slice position control
    style = {'description_width': '100px'}
    layout = widgets.Layout(width='350px')

    w_fine_offset = widgets.IntSlider(
        value=0, min=-int(pitch * 0.3), max=int(pitch * 0.3), step=1,
        description='Fine Adj (px):', style=style, layout=layout,
        continuous_update=False
    )
    w_log_range = widgets.FloatSlider(
        value=4, min=1, max=6, step=0.5,
        description='Log Range:', style=style, layout=widgets.Layout(width='280px'),
        continuous_update=False, readout_format='.1f'
    )
    w_position_label = widgets.HTML(value='<i>Slice at PSF center (adjust with Fine slider)</i>')

    group_selector = widgets.IntSlider(
        value=0, min=0, max=num_groups - 1, description='PSF Group:',
        layout=widgets.Layout(width='400px'), continuous_update=False
    )
    prev_btn = widgets.Button(description='◀ Prev', layout=widgets.Layout(width='80px'))
    next_btn = widgets.Button(description='Next ▶', layout=widgets.Layout(width='80px'))
    reset_btn = widgets.Button(description='Reset', layout=widgets.Layout(width='80px'))

    plot_out = widgets.Output()

    def compute_slices(fine_offset):
        """Compute PSF slices with given offset from center."""
        psf_slices = []
        for i in range(num_psfs):
            # centers format is [x, y], so centers[i, 0]=x, centers[i, 1]=y
            cx, cy = int(centers[i, 0]), int(centers[i, 1])
            # Apply fine offset to y (row) direction for horizontal slice
            cy_actual = cy + fine_offset
            x_start, x_end = max(0, cx - slice_half_width), min(N_up, cx + slice_half_width)

            if 0 <= cy_actual < N_up:
                slice_data = I_opt[cy_actual, x_start:x_end]
                if slice_data.max() > 0:
                    slice_data = slice_data / slice_data.max()
                template_slice = None
                if template is not None and 0 <= cy_actual < template.shape[0]:
                    template_slice = template[cy_actual, x_start:x_end]
                    if template_slice.max() > 0:
                        template_slice = template_slice / template_slice.max()
            else:
                slice_data = np.zeros(max(1, x_end - x_start))
                template_slice = None

            psf_slices.append({
                'data': slice_data, 'template': template_slice,
                'index': i, 'row': i // M, 'col': i % M,
                'cx': cx, 'cy': cy, 'cy_actual': cy_actual,
                'x_start': x_start, 'x_end': x_end
            })
        return psf_slices

    def plot_group(group_idx, fine_offset, log_range):
        plt.close('all')
        psf_slices = compute_slices(fine_offset)

        start_idx = group_idx * psfs_per_group
        end_idx = min((group_idx + 1) * psfs_per_group, num_psfs)
        group_psfs = psf_slices[start_idx:end_idx]
        n_cols = len(group_psfs)

        # Create figure with global view on left, slices on right
        fig = plt.figure(figsize=(5 + 3.5 * n_cols, 6))
        gs = fig.add_gridspec(2, n_cols + 1, width_ratios=[1.5] + [1] * n_cols)

        # Left: Global intensity map (spans both rows)
        ax_global = fig.add_subplot(gs[:, 0])
        vmin = I_opt.max() * (10 ** (-log_range))
        im = ax_global.imshow(I_opt, cmap='hot', norm=mcolors.LogNorm(vmin=vmin, vmax=I_opt.max()))
        ax_global.set_title(f'Global Intensity (log, {log_range:.0f} dec)', fontsize=10)
        ax_global.axis('off')

        # Draw thin dashed rectangles for current group PSFs
        colors = plt.cm.tab10(np.linspace(0, 1, n_cols))
        for j, psf in enumerate(group_psfs):
            # Rectangle around PSF region - thin dashed line
            rect = patches.Rectangle(
                (psf['x_start'], psf['cy_actual'] - 2),
                psf['x_end'] - psf['x_start'], 5,
                linewidth=1, linestyle='--', edgecolor=colors[j], facecolor='none'
            )
            ax_global.add_patch(rect)
            # Label
            ax_global.text(psf['x_start'], psf['cy_actual'] - 5, f"#{psf['index']}",
                          color=colors[j], fontsize=8, fontweight='bold')

        # Right: PSF slices
        for j, psf in enumerate(group_psfs):
            for row, (log_scale, ylabel) in enumerate([(False, 'Intensity'), (True, 'Log Intensity')]):
                ax = fig.add_subplot(gs[row, j + 1])
                x = np.arange(len(psf['data']))

                # Plot template first (background)
                if psf['template'] is not None:
                    if log_scale:
                        ax.semilogy(x, psf['template'] + 1e-6, 'orange', lw=2, alpha=0.8, label='Template')
                    else:
                        ax.fill_between(x, psf['template'], alpha=0.4, color='orange', label='Template')
                        ax.plot(x, psf['template'], 'orange', lw=1.5, alpha=0.8)

                # Plot actual PSF
                if log_scale:
                    ax.semilogy(x, psf['data'] + 1e-6, color=colors[j], lw=1.5, label='Actual')
                else:
                    ax.plot(x, psf['data'], color=colors[j], lw=1.5, label='Actual')
                    ax.fill_between(x, psf['data'], alpha=0.3, color=colors[j])

                if row == 0:
                    title = f"#{psf['index']} [{psf['row']},{psf['col']}]"
                    if fine_offset != 0:
                        title += f"\ny={psf['cy_actual']} (off={fine_offset:+d})"
                    ax.set_title(title, fontsize=9)
                    ax.legend(loc='upper right', fontsize=7)

                ax.set_xlabel('Pixels', fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if not log_scale:
                    ax.set_ylim(0, 1.15)
                else:
                    ax.set_ylim(1e-4, 2)

        offset_text = f" (offset={fine_offset:+d}px)" if fine_offset != 0 else ""
        fig.suptitle(f'PSF Slices - Group {group_idx + 1}/{num_groups}{offset_text}', fontweight='bold')
        plt.tight_layout()
        plt.show()

    def update_all(change=None):
        fine_offset = w_fine_offset.value
        log_range = w_log_range.value
        w_position_label.value = (
            f'<b>Slice offset:</b> {fine_offset:+d} px from PSF center'
            if fine_offset != 0 else '<i>Slice at PSF center</i>'
        )
        with plot_out:
            clear_output(wait=True)
            plot_group(group_selector.value, fine_offset, log_range)

    def on_prev(b):
        if group_selector.value > 0:
            group_selector.value -= 1

    def on_next(b):
        if group_selector.value < num_groups - 1:
            group_selector.value += 1

    def on_reset(b):
        w_fine_offset.value = 0
        group_selector.value = 0

    # Setup callbacks
    group_selector.observe(update_all, names='value')
    w_fine_offset.observe(update_all, names='value')
    w_log_range.observe(update_all, names='value')
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    reset_btn.on_click(on_reset)

    # Display controls
    controls = widgets.VBox([
        widgets.HTML('<h4>PSF Slice Viewer</h4>'),
        widgets.HBox([group_selector, prev_btn, next_btn]),
        widgets.HBox([w_fine_offset, reset_btn, w_log_range]),
        w_position_label
    ])
    display(controls)
    display(plot_out)

    # Initial plot
    update_all()


def show_psf_patches_interactive(optimizer, upsampling: float = 2.0):
    """Show PSF 2D patches with propagation analysis.

    Features:
    - Global intensity map with highlighted patch regions
    - 2D intensity patches for each PSF in a group
    - Propagation slice view (horizontal slice, z propagation)
    - Log scale and range controls
    - Adjustable patch size
    """
    import matplotlib.patches as mpatches

    M, N = optimizer.M, optimizer.N
    N_up = int(N * upsampling)
    pixel_size_up = optimizer.pixel_size / upsampling
    randomness = getattr(optimizer, 'randomness', 0.0)
    random_seed = getattr(optimizer, 'random_seed', None)

    center_info = compute_psf_centers(
        M=M, overlap_ratio=optimizer.overlap_ratio, center_blend=optimizer.center_blend,
        z_ratio=1.0, N=N_up, output_size=N_up, device=optimizer.device,
        randomness=randomness, random_seed=random_seed
    )
    centers = center_info['centers_pixel'].cpu().numpy()  # [M*M, 2] format: [x, y]

    # Precompute focal plane intensity
    with torch.no_grad():
        I_focal = optimizer.forward(upsampling=upsampling).cpu().numpy()

    pitch = N_up / M
    num_psfs = M * M
    psfs_per_group = 4  # Fewer per group since we show more info
    num_groups = (num_psfs + psfs_per_group - 1) // psfs_per_group

    # Create widgets
    style = {'description_width': '60px'}

    group_selector = widgets.IntSlider(
        value=0, min=0, max=num_groups - 1, description='Group:',
        style=style, layout=widgets.Layout(width='280px'), continuous_update=False
    )
    w_patch_size = widgets.FloatSlider(
        value=0.6, min=0.3, max=1.0, step=0.1,
        description='Patch:', style=style, layout=widgets.Layout(width='180px'),
        continuous_update=False, readout_format='.1f'
    )
    w_global_log = widgets.FloatSlider(
        value=4, min=1, max=6, step=0.5,
        description='Global:', style=style, layout=widgets.Layout(width='180px'),
        continuous_update=False, readout_format='.1f'
    )
    w_patch_log = widgets.Checkbox(value=True, indent=False)
    w_patch_range = widgets.FloatSlider(
        value=4, min=1, max=6, step=0.5,
        layout=widgets.Layout(width='120px'),
        continuous_update=False, readout_format='.1f'
    )
    w_z_range = widgets.FloatSlider(
        value=2.0, min=0.5, max=5.0, step=0.5,
        description='Z(DOF):', style=style, layout=widgets.Layout(width='180px'),
        continuous_update=False, readout_format='.1f'
    )
    prev_btn = widgets.Button(description='◀', layout=widgets.Layout(width='40px'))
    next_btn = widgets.Button(description='▶', layout=widgets.Layout(width='40px'))

    plot_out = widgets.Output()

    def compute_propagation_for_psf(cx, cy, z_values, half_width):
        """Compute propagation map for a single PSF location."""
        n_z = len(z_values)
        width = 2 * half_width
        intensity_map = np.zeros((n_z, width))

        with torch.no_grad():
            for i, z in enumerate(z_values):
                I = optimizer.forward(z=z, upsampling=upsampling).cpu().numpy()
                if 0 <= cy < I.shape[0]:
                    x_start = max(0, cx - half_width)
                    x_end = min(I.shape[1], cx + half_width)
                    slice_data = I[cy, x_start:x_end]
                    # Pad if needed
                    if len(slice_data) < width:
                        padded = np.zeros(width)
                        offset = (width - len(slice_data)) // 2
                        padded[offset:offset+len(slice_data)] = slice_data
                        slice_data = padded
                    intensity_map[i, :len(slice_data)] = slice_data[:width]
        return intensity_map

    def plot_group(group_idx, global_log, patch_log, patch_range, z_range_dof, patch_ratio):
        plt.close('all')

        patch_half = int(pitch * patch_ratio)
        start_idx = group_idx * psfs_per_group
        end_idx = min((group_idx + 1) * psfs_per_group, num_psfs)
        n_cols = end_idx - start_idx

        # Figure layout: Global view + n_cols columns for PSFs
        fig, axes = plt.subplots(2, n_cols + 1, figsize=(3.5 + 3 * n_cols, 7),
                                gridspec_kw={'width_ratios': [1.3] + [1] * n_cols})

        # Global intensity map (spans both rows visually by making row 1 empty)
        ax_global = axes[0, 0]
        axes[1, 0].axis('off')  # Hide bottom-left
        vmin_global = I_focal.max() * (10 ** (-global_log))
        ax_global.imshow(I_focal, cmap='hot',
                        norm=mcolors.LogNorm(vmin=vmin_global, vmax=I_focal.max()))
        ax_global.set_title(f'Global ({global_log:.0f} dec)', fontsize=9)
        ax_global.axis('off')

        # Draw patch boxes for current group
        colors = plt.cm.tab10(np.linspace(0, 1, n_cols))
        group_psfs = []
        for j in range(n_cols):
            idx = start_idx + j
            cx, cy = int(centers[idx, 0]), int(centers[idx, 1])
            x_start = max(0, cx - patch_half)
            y_start = max(0, cy - patch_half)
            x_end = min(N_up, cx + patch_half)
            y_end = min(N_up, cy + patch_half)

            group_psfs.append({
                'idx': idx, 'cx': cx, 'cy': cy,
                'x_start': x_start, 'x_end': x_end,
                'y_start': y_start, 'y_end': y_end,
                'row': idx // M, 'col': idx % M
            })

            # Draw dashed rectangle
            rect = mpatches.Rectangle(
                (x_start, y_start), x_end - x_start, y_end - y_start,
                linewidth=1.5, linestyle='--', edgecolor=colors[j], facecolor='none'
            )
            ax_global.add_patch(rect)
            ax_global.text(x_start, y_start - 3, f"#{idx}",
                          color=colors[j], fontsize=9, fontweight='bold')

        # Prepare z values for propagation
        focal_dist = optimizer.focal_length
        dof = optimizer.depth_of_focus
        z_min = focal_dist - z_range_dof * dof
        z_max = focal_dist + z_range_dof * dof
        n_z = 30
        z_values = torch.linspace(z_min, z_max, n_z)
        z_relative_mm = (z_values - focal_dist).numpy() * 1e3

        # Plot each PSF
        for j, psf in enumerate(group_psfs):
            # Row 0: 2D patch
            ax_patch = axes[0, j + 1]
            patch_data = I_focal[psf['y_start']:psf['y_end'], psf['x_start']:psf['x_end']]
            if patch_data.size > 0:
                if patch_log:
                    vmin_patch = I_focal.max() * (10 ** (-patch_range))
                    ax_patch.imshow(patch_data, cmap='hot',
                                   norm=mcolors.LogNorm(vmin=vmin_patch, vmax=I_focal.max()))
                else:
                    ax_patch.imshow(patch_data, cmap='hot')
            scale_text = f"log{patch_range:.0f}" if patch_log else "lin"
            ax_patch.set_title(f"#{psf['idx']} [{psf['row']},{psf['col']}]", fontsize=9, color=colors[j])
            ax_patch.axis('off')

            # Row 1: Propagation slice (always log for visibility)
            ax_prop = axes[1, j + 1]
            prop_map = compute_propagation_for_psf(psf['cx'], psf['cy'], z_values, patch_half)

            # Normalize and apply log
            prop_max = prop_map.max() if prop_map.max() > 0 else 1
            prop_norm = prop_map / prop_max
            prop_log_data = np.log10(prop_norm + 10**(-patch_range))
            prop_log_data = np.clip(prop_log_data, -patch_range, 0)

            extent = [-patch_half * pixel_size_up * 1e3, patch_half * pixel_size_up * 1e3,
                     z_relative_mm[-1], z_relative_mm[0]]
            ax_prop.imshow(prop_log_data, cmap='hot', aspect='auto', extent=extent)
            ax_prop.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.8)
            # DOF markers
            for mult in [0.5, 1.0]:
                ax_prop.axhline(y=mult * dof * 1e3, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
                ax_prop.axhline(y=-mult * dof * 1e3, color='white', linestyle='--', linewidth=0.5, alpha=0.5)

            ax_prop.set_xlabel('X (mm)', fontsize=8)
            if j == 0:
                ax_prop.set_ylabel('Δz (mm)', fontsize=8)
            ax_prop.tick_params(labelsize=7)

        scale_label = f"log{patch_range:.0f}" if patch_log else "linear"
        fig.suptitle(f'PSF Patches - Group {group_idx + 1}/{num_groups} (patch={patch_ratio:.1f}, {scale_label})',
                    fontweight='bold', fontsize=11)
        plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.15)
        plt.show()

    def update_all(change=None):
        with plot_out:
            clear_output(wait=True)
            plot_group(group_selector.value, w_global_log.value, w_patch_log.value,
                      w_patch_range.value, w_z_range.value, w_patch_size.value)

    def on_prev(b):
        if group_selector.value > 0:
            group_selector.value -= 1

    def on_next(b):
        if group_selector.value < num_groups - 1:
            group_selector.value += 1

    # Setup callbacks
    group_selector.observe(update_all, names='value')
    w_global_log.observe(update_all, names='value')
    w_patch_log.observe(update_all, names='value')
    w_patch_range.observe(update_all, names='value')
    w_z_range.observe(update_all, names='value')
    w_patch_size.observe(update_all, names='value')
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    # Display controls with HTML labels
    controls = widgets.VBox([
        widgets.HTML('<h4>PSF Patches Viewer</h4>'),
        widgets.HBox([group_selector, prev_btn, next_btn, w_patch_size]),
        widgets.HBox([
            widgets.HTML('<span style="margin-right:3px">Global:</span>'), w_global_log,
            widgets.HTML('<span style="margin:0 8px">Patch Log:</span>'), w_patch_log, w_patch_range,
            w_z_range
        ]),
    ])
    display(controls)
    display(plot_out)

    # Initial plot
    update_all()


def plot_atf_interactive(optimizer, atf: np.ndarray, pixel_size: float):
    """Interactive ATF visualization with lenslet selector."""
    M = optimizer.M
    extent_value = 1 / (pixel_size * 1e3)

    def plot_atf_single(row=0, col=0):
        plt.close('all')
        extent_range = [0, extent_value, 0, extent_value]
        idx = row * M + col
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        intensity = np.abs(atf[idx]).T
        ax1.imshow(np.angle(atf[idx].T), cmap='hsv', alpha=intensity / intensity.max(), extent=extent_range, origin='lower')
        ax1.set_title(f'Phase [{row},{col}]'); ax1.set_xlabel('X (mm^-1)'); ax1.set_ylabel('Y (mm^-1)')
        ax2.imshow(np.abs(atf[idx].T), extent=extent_range, origin='lower', cmap='viridis')
        ax2.set_title(f'Amplitude [{row},{col}]'); ax2.set_xlabel('X (mm^-1)'); ax2.set_ylabel('Y (mm^-1)')
        plt.tight_layout(); plt.show()

    from ipywidgets import interact, IntSlider
    interact(plot_atf_single, row=IntSlider(min=0, max=M-1, value=0, description='Row:'),
             col=IntSlider(min=0, max=M-1, value=0, description='Col:'))


def compute_and_save_atf(optimizer, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute ATF from optimizer and optionally save to .mat file."""
    N = optimizer.N * 5
    pixel_size = optimizer.pixel_size / 5
    atf = optimizer.compute_transfer_function(N=N, pixel_size=pixel_size, verbose=False)
    result = {
        'atf': atf, 'pixel_size': pixel_size, 'wavelength': optimizer.wavelength,
        'focal_length': optimizer.focal_length, 'M': optimizer.M, 'spot_radius': optimizer.airy_radius
    }
    if output_path:
        import scipy.io as sio
        sio.savemat(output_path, result)
        print(f"ATF saved: {output_path}")
    return result
