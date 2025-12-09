#!/usr/bin/env python3
"""
Plot initial 1D profiles (vorticity, vy, rotation period) from Athena++ data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files,
)


def plot_initial_profiles(path_file, show=True, save_path=None):
    """
    Load a .athdf file and plot 1D profiles along x at mid-y:
      - vorticity_z
      - vy
      - rotation period T = 2π r / vy

    Parameters
    ----------
    path_file : str
        Path to a single Athena++ .athdf file.
    
    Returns
    -------
    None    
    """

    # --- Load data (primitive variables) ---
    data = open_hdf_files([path_file], read_every=1)

    x = data["x1"]
    y = data["x2"]
    time = data["time"]

    vx = data["v1"]
    vy = data["v2"]

    # --- Compute vorticity ---
    # vort_z = ∂v_y/∂x - ∂v_x/∂y
    dvydx = np.gradient(vy, x, axis=1)
    dvxdy = np.gradient(vx, y, axis=2)
    vortz = dvydx - dvxdy  # shape (Nt, Nx, Ny, 1)

    time_idx = 0
    mid_y = len(y) // 2

    # --- Prepare figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # ============================================================
    # 1) Vorticity profile
    # ============================================================
    axes[0].plot(x, vortz[0, :, mid_y, 0], label='initial')
    axes[0].plot(x, vortz[time_idx, :, mid_y, 0], label='time')
    axes[0].set_title(f'vorticity profile\ntime={time[time_idx]:.1f}')
    axes[0].set_xlabel('x')
    axes[0].legend(loc='upper left', prop={'size': 8})
    axes[0].grid()

    # ============================================================
    # 2) vy profile
    # ============================================================
    axes[1].plot(x, vy[0, :, mid_y, 0], label='initial')
    axes[1].set_title(f'vy profile\ntime={time[time_idx]:.1f}')
    axes[1].set_xlabel('x')
    axes[1].grid()
    axes[1].axhline(0, color='black')
    axes[1].axvline(0, color='black')
    axes[1].legend(loc='upper left', prop={'size': 8})

    # ============================================================
    # 3) Rotation period T = 2π r / v_y
    # ============================================================
    # secure division
    vy_safe = np.where(np.abs(vy[0, :, mid_y, 0]) > 1e-8,
                       vy[0, :, mid_y, 0], np.nan)
    T = 2 * np.pi * x / vy_safe

    axes[2].plot(x, T, label='initial')
    axes[2].set_title(f'Initial Rotation period\nT0={T[len(x)//2]:.1f}')
    axes[2].set_xlabel('x')
    axes[2].grid()
    axes[2].axhline(0, color='black')
    axes[2].axvline(0, color='black')
    axes[2].legend(loc='upper left', prop={'size': 8})

    if save_path is not None:
        plt.savefig(save_path+'initial_profiles.pdf')
    if show:
        plt.show()



if __name__ == "__main__":

    from athena_collapse_analysis.config import RAW_DIR
    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    files = get_hdf_files(path_simu)

    plot_initial_profiles(files[0])
