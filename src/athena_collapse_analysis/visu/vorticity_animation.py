#!/usr/bin/env python3
"""
Generate a vorticity movie from a folder of Athena++ .athdf files.
Loads files one by one (memory efficient).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files,
)


def compute_vorticity(vx, vy, x, y):
    """
    Compute vertical vorticity ω_z = ∂v_y/∂x – ∂v_x/∂y.

    vx, vy : (Nx, Ny) arrays
    x, y   : 1D arrays
    """
    dvydx = np.gradient(vy, x, axis=0)
    dvxdy = np.gradient(vx, y, axis=1)
    return dvydx - dvxdy


def make_vorticity_movie(path_simu, outname="vorticity_movie.mp4"):
    # --- Collect files ---
    files = get_hdf_files(path_simu)
    Nframes = len(files)

    print(f"Found {Nframes} frames.")
    print("Preparing animation...")

    # --- Load spatial coords from first file ---
    data0 = open_hdf_files([files[0]], read_every=1)
    x = data0["x1"]
    y = data0["x2"]

    # Midplane: take z index = middle
    Nz = len(data0["x3"])
    midz = Nz // 2

    # --- Prepare figure ---
    fig, ax = plt.subplots(figsize=(6,5))
    img = ax.imshow(np.zeros((len(x), len(y))), 
                    origin='lower', 
                    extent=[y.min(), y.max(), x.min(), x.max()],
                    cmap='RdBu_r')
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("vorticity")

    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

    def init():
        img.set_data(np.zeros((len(x), len(y))))
        text.set_text("")
        return img, text

    # --- Frame update function ---
    def update(i):
        file_i = files[i]
        print(f"Loading {file_i}")

        data = open_hdf_files([file_i], read_every=1)

        vx = data["v1"][0, :, :, midz]
        vy = data["v2"][0, :, :, midz]
        t  = data["time"][0]

        vort = compute_vorticity(vx, vy, x, y)

        img.set_data(vort.T)
        img.set_clim(vort.min(), vort.max())
        text.set_text(f"t = {t:.3f}")

        return img, text

    # --- Create animation ---
    ani = animation.FuncAnimation(fig, update,
                                  frames=Nframes,
                                  init_func=init,
                                  blit=False)

    print("Saving MP4...")
    ani.save(outname, writer='ffmpeg', dpi=150)
    print(f"Movie saved: {outname}")


if __name__ == "__main__":
    from athena_collapse_analysis.config import RAW_DIR

    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    make_vorticity_movie(path_simu, outname="omegaTilda_movie.mp4")
