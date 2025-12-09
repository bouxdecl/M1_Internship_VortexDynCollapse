#!/usr/bin/env python3
"""
Generate a vorticity movie from Athena++ .athdf snapshots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from athena_collapse_analysis.utils import collapse_param_decomposition
from athena_collapse_analysis.io.ath_io import get_hdf_files, open_hdf_files_with_collapse


def ddx_4th(f, d, axis=0):
    """
    4th-order centered finite difference along a given axis.
    """
    return (-np.roll(f, -2, axis=axis)
            + 8*np.roll(f, -1, axis=axis)
            - 8*np.roll(f,  1, axis=axis)
            +   np.roll(f,  2, axis=axis)) / (12*d)


def make_vorticity_movie(path_simu, outname=None,
                         vort_type="simulation",
                         crop=None):
    """
    Create a vorticity movie from a folder of Athena++ snapshots.

    Parameters
    ----------
    vort_type : str
        "simulation" → ω = dv_y/dx - dv_x/dy
        "physical"   → ω_phys using metric coefficients and rescaled coords
    crop : tuple or None
        (xmin, xmax, ymin, ymax) in rescaled coordinates. If None → full domain.
    """

    if outname is None:
        outname = f"vorticity_movie_{vort_type}.mp4"

    # -------------------------------
    # 1. Collect files
    # -------------------------------
    files = get_hdf_files(path_simu)
    Nframes = len(files)
    print(f"Found {Nframes} frames.")

    # -------------------------------
    # 2. Load geometry + first collapse snapshot
    # -------------------------------
    data0 = open_hdf_files_with_collapse(path_simu, [files[0]], read_every=1)
    x = data0["x1"]
    y = data0["x2"]
    Nz = len(data0["x3"])
    midz = Nz // 2

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    # -------------------------------
    # 3. Prepare figure
    # -------------------------------
    fig, ax = plt.subplots(figsize=(6,5))
    dummy = np.zeros((len(x), len(y)))

    # Initialize pcolormesh
    Xgrid, Ygrid = np.meshgrid(y, x)
    pc = ax.pcolormesh(Xgrid, Ygrid, dummy, cmap='RdBu_r', shading='auto')
    cbar = plt.colorbar(pc, ax=ax)
    cbar.set_label("vorticity")

    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # -------------------------------
    # 4. Init function
    # -------------------------------
    def init():
        pc.set_array(dummy.ravel())
        text.set_text("")
        return pc, text

    # -------------------------------
    # 5. Update function
    # -------------------------------
    def update(i):
        file_i = files[i]
        print(f"[{i+1}/{Nframes}] Loading {file_i}")
        data = open_hdf_files_with_collapse(path_simu, [file_i], read_every=1)

        vx = data["v1"][0, :, :, midz]
        vy = data["v2"][0, :, :, midz]
        t  = data["time"][0]

        dvy_dx = ddx_4th(vy, dx, axis=0)
        dvx_dy = ddx_4th(vx, dy, axis=1)

        # -------------------------------
        # Compute vorticity
        # -------------------------------
        if vort_type == "simulation":
            vort = dvy_dx - dvx_dy
            S, alpha = 1.0, 1.0
            Xplot, Yplot = x, y

        elif vort_type == "physical":
            R = data['Rglobal'][0]
            Lz = data['Lzglobal'][0]
            S, alpha = collapse_param_decomposition(np.array([R]), np.array([Lz]))
            S, alpha = S[0], alpha[0]

            gxx = S**2 * alpha**-4
            gyy = S**2 * alpha**2
            omega_tilda = gyy * dvy_dx - gxx * dvx_dy
            vort = omega_tilda * alpha

            # Rescaled coordinates
            Xplot = x * alpha**(-3/2)
            Yplot = y * alpha**( 3/2)

        else:
            raise ValueError("vort_type must be 'simulation' or 'physical'.")

        # -------------------------------
        # Crop if requested
        # -------------------------------
        if crop is not None:
            xmin, xmax, ymin, ymax = crop
            mask_x = (Xplot >= xmin) & (Xplot <= xmax)
            mask_y = (Yplot >= ymin) & (Yplot <= ymax)
            vort_plot = vort[mask_x][:, mask_y]
            Xplot = Xplot[mask_x]
            Yplot = Yplot[mask_y]
        else:
            vort_plot = vort

        # -------------------------------
        # Update pcolormesh
        # -------------------------------
        pc.set_array(vort_plot.ravel())
        vmax = np.max(np.abs(vort_plot))
        pc.set_clim(-vmax, vmax)

        # Update meshgrid for axes scaling if rescaled
        pc.set_offsets(np.c_[Yplot.repeat(len(Xplot)), np.tile(Xplot, len(Yplot))])
        ax.set_xlim(Yplot.min(), Yplot.max())
        ax.set_ylim(Xplot.min(), Xplot.max())

        if vort_type == "simulation":
            ax.set_title(f"Simulation Vorticity")
            text.set_text(f"t = {t:.3f}")
        if vort_type == "physical":
            ax.set_title(f"Physical Vorticity (rescaled coords)")
            text.set_text(f"t = {t:.3f}, S={S:.3f}, α={alpha:.3f}")
        return pc, text

    # -------------------------------
    # 6. Create animation
    # -------------------------------
    ani = animation.FuncAnimation(fig, update,
                                  frames=Nframes,
                                  init_func=init,
                                  blit=False)

    print("Saving movie...")
    ani.save(outname, writer='ffmpeg', dpi=150)
    print(f"Movie saved: {outname}")







if __name__ == "__main__":
    from athena_collapse_analysis.config import RAW_DIR
    # Example simulation path
    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")

    # --- Test that folder exists and contains files ---
    assert os.path.exists(path_simu), f"Simulation path does not exist: {path_simu}"
    files = get_hdf_files(path_simu)
    assert len(files) > 0, f"No .athdf files found in: {path_simu}"
    print(f"Found {len(files)} snapshot files in {path_simu}")

    # --- Make the vorticity movie ---
    make_vorticity_movie(path_simu, outname="omegaTilda_movie.mp4",
                         vort_type="physical", crop=None)


