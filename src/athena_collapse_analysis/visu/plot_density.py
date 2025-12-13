#!/usr/bin/env python3
"""
Load Athena++ fields using ath_io and plot the first rho snapshot.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from athena_collapse_analysis.config import RAW_DIR
from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files,
)


def plot_density(path_file, slice_index=None):
    """
    Load first rho field from a directory of .athdf files and plot it.

    Parameters
    ----------
    data_path : str
        Directory containing Athena++ .athdf files.
    slice_index : int or None
        If None: use central slice along x3.
        Otherwise: use user-specified index.
    """

    # 1. open file
    data = open_hdf_files([path_file], read_every=1, adia=False)

    rho = data["rho"]               # shape: (Nt, Nx1, Nx2, Nx3)
    x1  = data["x1"]
    x2  = data["x2"]
    x3  = data["x3"]

    rho0 = rho[0]

    # 4. Choose a slice along x3 (third axis)
    if slice_index is None:
        slice_index = len(x3) // 2  # middle slice

    rho_slice = rho0[:, :, slice_index]  # 2D array (Nx1, Nx2)

    # 5. Make the plot
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(x2, x1, rho_slice, shading='auto', cmap='RdBu_r')
    plt.colorbar(label="rho")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"density (simulation) at t = {data['time'][0]:.3f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    files = get_hdf_files(path_simu)

    plot_density(files[0])
