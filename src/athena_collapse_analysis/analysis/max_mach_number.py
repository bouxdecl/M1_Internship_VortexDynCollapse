#!/usr/bin/env python3
"""
Maximum Mach number diagnostics for Athena++ collapse simulations.

For each output file:
- Compute local sound speed c_s with adiabatic index gamma
- Computes Mach number M = |v| / c_s
- Extracts max(M) over the domain

Provides:
- compute_max_mach_number()
- plot_max_mach_number()
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from athena_collapse_analysis import utils
from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files_with_collapse,
)


# ============================================================
# Core routine
# ============================================================

def compute_max_mach_number(path_simu, files, gamma=5/3, cs_0=1.0, nz_slice=None, verbose=False):
    """
    Compute maximum Mach number file by file.

    Parameters
    ----------
    path_simu : str
        Path to simulation directory
    files : list[str]
        Athena++ HDF5 files ordered in time
    nz_slice : int or None
        If None → full 3D domain
        If int  → single z slice
    verbose : bool
        Print diagnostics for each file

    Returns
    -------
    time : (Nt,) ndarray
    Mach_max : (Nt,) ndarray
    """

    Nfiles = len(files)
    time = np.zeros(Nfiles)
    Mach_max = np.zeros(Nfiles)

    for i, f in enumerate(files):
        data = open_hdf_files_with_collapse(
            path_simu, files=[f], read_every=1, adia=True
        )

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------
        time[i] = data["time"][0]

        # ----------------------------------------------------
        # Primitive fields
        # ----------------------------------------------------
        rho = data["rho"][0]

        vx = data["v1"][0]
        vy = data["v2"][0]
        vz = data["v3"][0]

        Etot = data["Etot"][0]
        press = utils.pressure_from_conservatives(
            rho, Etot, vx, vy, vz, gamma = gamma
        )

        if nz_slice is not None:
            rho = rho[:, :, nz_slice]
            press = press[:, :, nz_slice]
            vx = vx[:, :, nz_slice]
            vy = vy[:, :, nz_slice]
            vz = vz[:, :, nz_slice]

        # ----------------------------------------------------
        # Mach number
        # ----------------------------------------------------

        vmag = np.sqrt(vx**2 + vy**2 + vz**2)
        cs = np.sqrt(gamma * press / rho)

        mach = vmag / cs
        Mach_max[i] = np.nanmax(mach)

        Mach_x = np.nanmax(np.abs(vx) / (cs_0/data["Lzglobal"][0]))
        Mach_y = np.nanmax(np.abs(vy) / (cs_0/data["Rglobal"][0] ))

        if verbose:
            print(
                f"[{i+1}/{Nfiles}] "
                f"t={time[i]:.5e}  "
                f"Mach_max={Mach_max[i]:.6e}"
            )

    return time, Mach_x, Mach_y, Mach_max


# ============================================================
# Plotting
# ============================================================

def plot_max_mach_number(time, Mach_max, Mach_x, Mach_y, show=True, save_path=None):
    """
    Plot maximum Mach number vs time.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(time, Mach_max, label='Mach_max')
    ax.plot(time, Mach_x, label='Mach_x')
    ax.plot(time, Mach_y, label='Mach_y')
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("Maximum Mach number")
    ax.set_title("Maximum Mach number evolution")
    ax.grid(True, which="both")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Mach plot to: {save_path}")

    if show:
        plt.show()

    return fig, ax


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":

    from athena_collapse_analysis.config import RAW_DIR

    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    files = get_hdf_files(path_simu)

    time, Mach_x, Mach_y, Mach_max = compute_max_mach_number(
        path_simu,
        files,
        nz_slice=None,   # or e.g. nz_slice=0
        verbose=True
    )

    plot_max_mach_number(time, Mach_max, Mach_x, Mach_y)
