#!/usr/bin/env python3
"""
Vorticity plot generator for a single Athena++ file.


Two display modes:

- `vort_type='simulation'`: raw simulation vorticity ω = ∂v_y/∂x − ∂v_x/∂y
- `vort_type='physical'`: rescaled physical vorticity using collapse
    parameters (S, α) and rescaled coordinates.

Optional cropping can be applied to zoom in on a specific region.
vmin and vmax control the color scale limits.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files,
    open_hdf_files_with_collapse,
    get_collapse_profile,
)

from athena_collapse_analysis.utils import collapse_param_decomposition


def plot_vorticity(path_file, file, show=True, save_path=None,
                       vmin=0.0, vmax=None, cmap='RdBu_r',
                       nz_slice=0, vort_type="simulation", crop=None):
    """
    Plot vorticity or physical (rescaled) vorticity from one Athena++ file.

    vort_type : {"simulation", "physical"}
        "simulation" → ω = dv_y/dx − dv_x/dy
        "physical"   → ω̃ using metric coefficients (S, α) and rescaled coords
    """

    # --- Load data (primitive variables) ---
    data = open_hdf_files_with_collapse(path_file, files=[file], read_every=1)

    x = data["x1"]
    y = data["x2"]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    time = data["time"]

    vx = data["v1"][0, :, :, nz_slice]
    vy = data["v2"][0, :, :, nz_slice]

    # --- Compute vorticity ---
    dvydx = np.gradient(vy, dx, axis=0)
    dvxdy = np.gradient(vx, dy, axis=1)

    if vort_type == "simulation":
        # ω = ∂v_y/∂x − ∂v_x/∂y
        vort = dvydx - dvxdy
        Xplot = x
        Yplot = y

    elif vort_type == "physical":
        # --- compute collapse parameters (global) ---
        R = data["Rglobal"][0]
        Lz = data["Lzglobal"][0]

        S, alpha = collapse_param_decomposition(R, Lz)

        # metric coefficients
        gxx = S**2 * alpha**(-4)
        gyy = S**2 * alpha**2

        # physical vorticity
        vort = alpha * (gyy * dvydx - gxx * dvxdy)

        # rescaled coordinates
        Xplot = x * alpha**(-3/2)
        Yplot = y * alpha**(3/2)

    else:
        raise ValueError("vort_type must be 'simulation' or 'physical'")

    # ============================================================
    # --- Optional cropping ---
    # crop should be a 4-tuple: (xmin, xmax, ymin, ymax) in the same
    # coordinate units as Xplot and Yplot. If provided, subset the
    # coordinate arrays and the vorticity field accordingly.
    if crop is not None:
        try:
            xmin, xmax, ymin, ymax = crop
        except Exception:
            raise ValueError("crop must be a 4-tuple: (xmin, xmax, ymin, ymax)")

        # find index ranges on the 1D coordinate arrays
        ix = np.where((Xplot >= xmin) & (Xplot <= xmax))[0]
        iy = np.where((Yplot >= ymin) & (Yplot <= ymax))[0]

        if ix.size == 0 or iy.size == 0:
            raise ValueError(
                f"crop region yields empty selection: x in [{xmin},{xmax}], y in [{ymin},{ymax}]"
            )

        # subset coordinates and data; vort is indexed as (nx, ny)
        Xplot = Xplot[ix]
        Yplot = Yplot[iy]
        vort = vort[np.ix_(ix, iy)]

    # ============================================================
    # --- Plot ---
    # ============================================================

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    # color range
    vmin_plot = 0.0 if vmin is None else vmin
    vmax_plot = np.max(vort) if vmax is None else vmax
    if vmax_plot <= vmin_plot:
        vmax_plot = np.max(np.abs(vort))
        vmin_plot = -vmax_plot

    pc = ax.pcolormesh(Xplot, Yplot, vort.T,
                       cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                       shading='auto')

    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label('vorticity')

    if vort_type == "simulation":
        ax.set_title(f'vorticity (simulation) at t = {time[0]:.3f}')
    else:
        ax.set_title(f'vorticity (physical) at t = {time[0]:.3f}, '
                     f'S={S:.3f}, α={alpha:.3f}')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved vorticity plot to: {save_path}")

    if show:
        plt.show()

    return fig, ax

    

if __name__ == "__main__":

    from athena_collapse_analysis.config import RAW_DIR
    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    files = get_hdf_files(path_simu)

    plot_vorticity(path_simu, files[-1], vort_type="physical", crop=(-1, 1, -1, 1))
