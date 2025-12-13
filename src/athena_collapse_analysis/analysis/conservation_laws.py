#!/usr/bin/env python3
"""
Conservation laws checker for Athena++ collapse simulations.
Rescale and integrate fields to compute conserved quantities during collapse.

Computes file-by-file:
- Total mass   M(t) = ∫ S^3 rho dV
- Total physical vorticity Omega(t) = ∫ ω̃ dV where ω̃ = gyy ∂v_y/∂x − gxx ∂v_x/∂y

Plots relative conservation errors and returns arrays.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files_with_collapse,
)

from athena_collapse_analysis.utils import collapse_param_decomposition


def print_field_dtypes(data):
    print("Field dtypes:")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:12s}: dtype={v.dtype}, shape={v.shape}")


# ============================================================
# Core routine
# ============================================================

def compute_conservation_laws(path_simu, files, nz_slice=None, verbose=False):
    """
    Compute conserved quantities file by file.

    Parameters
    ----------
    path_simu : str
        Path to simulation directory
    files : list[str]
        Athena++ HDF5 files (ordered in time)
    nz_slice : int or None
        If None → full 3D sums
        If int  → single z-slice diagnostics

    Returns
    -------
    time : (Nt,) array
    Mtot : (Nt,) array
        Total mass (with collapse rescaling)
    Omega_phys : (Nt,) array
        Total physical vorticity
    """

    Nfiles = len(files)

    time = np.zeros(Nfiles)
    Mtot = np.zeros(Nfiles)
    Omega_phys = np.zeros(Nfiles)

    for i, f in enumerate(files):
        data = open_hdf_files_with_collapse(
            path_simu, files=[f], read_every=1
        )

        if verbose and i == Nfiles - 1:
            print_field_dtypes(data)

        # --- time ---
        time[i] = data["time"][0]

        # --- collapse parameters ---
        R = data["Rglobal"][0]
        Lz = data["Lzglobal"][0]
        S, alpha = collapse_param_decomposition(R, Lz)

        # ====================================================
        # Total density (mass)
        # ====================================================
        rho = data["rho"][0]  # (Nx, Ny, Nz)

        if nz_slice is not None:
            rho = rho[:, :, nz_slice]

        # collapse-rescaled volume factor
        Mtot[i] = np.sum(rho) * S**3

        # ====================================================
        # Total physical vorticity
        # ====================================================
        x = data["x1"]
        y = data["x2"]
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        vx = data["v1"][0]
        vy = data["v2"][0]

        if nz_slice is not None:
            vx = vx[:, :, nz_slice]
            vy = vy[:, :, nz_slice]

        dvydx = np.gradient(vy, dx, axis=0)
        dvxdy = np.gradient(vx, dy, axis=1)

        # metric coefficients
        gxx = S**2 * alpha**(-4)
        gyy = S**2 * alpha**2

        omega_tilde = (gyy * dvydx - gxx * dvxdy) # physical vorticity field without alpha factor to have the conserved quantity

        Omega_phys[i] = np.sum(omega_tilde)

        print(
            f"[{i+1}/{Nt}] t={time[i]:.4e}  "
            f"M={Mtot[i]:.6e}  Ω~={Omega_phys[i]:.6e}"
        )

    return time, Mtot, Omega_phys


# ============================================================
# Plotting routine
# ============================================================
def plot_conservation_laws(time, Mtot, Omega_phys, show=True, save_path=None):
    """
    Plot relative conservation errors.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --------------------------------------------------------
    # Total density conservation
    # --------------------------------------------------------
    rel_mass = (Mtot - Mtot[0]) / Mtot[0]

    axes[0].plot(time, rel_mass)
    axes[0].set_yscale("symlog", linthresh=1e-12)
    axes[0].set_title("Total density conservation")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("Relative error")
    axes[0].grid(True, which="both")

    # --------------------------------------------------------
    # Total physical vorticity conservation
    # --------------------------------------------------------
    rel_vort = (Omega_phys - Omega_phys[0]) / Omega_phys[0]

    axes[1].plot(time, rel_vort)
    axes[1].set_yscale("symlog", linthresh=1e-7)
    axes[1].set_title("Total physical vorticity conservation")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("Relative error")
    axes[1].grid(True, which="both")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig, axes




if __name__ == "__main__":

    from athena_collapse_analysis.config import RAW_DIR

    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    files = get_hdf_files(path_simu)

    time, Mtot, Omega_phys = compute_conservation_laws(
        path_simu,
        files,
        nz_slice=None,   # or e.g. nz_slice=0
        verbose=True
    )

    plot_conservation_laws(time, Mtot, Omega_phys)

