#!/usr/bin/env python3
"""
Plot collapse global profiles R(t) and Lz(t) from a simulation folder output.

- plot line: profiles from the .hst file
- plot scatter:    values interpolated at each .athdf snapshot

Should be the same if the interpolation as been well done.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files_with_collapse,
    get_collapse_profile,
)


def plot_collapse_profiles(path_simu, plot=True, save_path=None):
    """
    Open Athena++ files one-by-one and plot:
      - R(t), Lz(t) from .hst (continuous)
      - interpolated R_snap, Lz_snap from each .athdf file (scatter)

    Parameters
    ----------
    path_simu : str
        Path to the simulation directory.
    plot : bool, default True
        If True, display the figure.
    save_path : str or None, default None
        If not None, save the figure to this file path.
    """

    # ----------------------------------------------------
    # Load .athdf and .hst files
    # ----------------------------------------------------
    files = get_hdf_files(path_simu)
    N = len(files)
    print(f"Found {N} files.")
    
    t_hst, R_hst, Lz_hst = get_collapse_profile(path_simu)

    t_snap = np.zeros(N)
    R_snap = np.zeros(N)
    Lz_snap = np.zeros(N)

    # ----------------------------------------------------
    # Extract collapse data in files
    # ----------------------------------------------------
    for i, f in enumerate(files):
        print(f"[{i+1}/{N}] Loading {f}")

        dic = open_hdf_files_with_collapse(path_simu, [f], read_every=1, adia=True)
        t0 = dic["time"][0]  # scalar
        t_snap[i] = t0

        R_snap[i] = dic['Rglobal'][0] # Interpolate collapse diagnostics at t0
        Lz_snap[i] = dic['Lzglobal'][0]

    # ----------------------------------------------------
    # Plot result
    # ----------------------------------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(t_hst, R_hst, "k-", lw=2, label="R (hst)")
        ax.scatter(t_snap, R_snap, s=20, color="tab:blue", label="R (hdf dump)")

        ax.plot(t_hst, Lz_hst, "k--", lw=2, label="Lz (hst)")
        ax.scatter(t_snap, Lz_snap, s=20, color="tab:red", label="Lz (hdf dump)")

        ax.set_xlabel("time")
        ax.set_ylabel("R, Lz")
        ax.set_title("Collapse Profiles")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()

    # Save if required
    if save_path is not None:
        print(f"Saving figure to {save_path}")
        fig.savefig(save_path, dpi=200)

    # Show if required
    if plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    from athena_collapse_analysis.config import RAW_DIR
    path = os.path.join(RAW_DIR, "typical_simu_20251311/")
    plot_collapse_profiles(path)
