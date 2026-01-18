#!/usr/bin/env python3
"""
Vorticity movie generator from a folder with Athena++ outputs (one .hst and the .athdf dump)

This module produces an MP4 movie showing the z-component of the
vorticity on a chosen midplane (or any `nz_slice`). 

Two display modes:

- `vort_type='simulation'`: raw simulation vorticity ω = ∂v_y/∂x − ∂v_x/∂y
- `vort_type='physical'`: rescaled physical vorticity using collapse
    parameters (S, α) and rescaled coordinates.

Notes
-----
A crop region can be specified for zoomed-in views. In that case the movie shows only
the selected area.
vmin and vmax control the color scale limits.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from athena_collapse_analysis.utils import collapse_param_decomposition, compute_physical_vorticity
from athena_collapse_analysis.io.ath_io import get_hdf_files, open_hdf_files_with_collapse



def make_vorticity_movie(path_simu, outname=None,
                         vort_type="simulation",
                         crop=None,
                         vmin=0.0, vmax=None, cmap='RdBu_r',
                         nz_slice=0, fps=10, dpi=150):
    """
    Create a vorticity movie from a folder of Athena++ snapshots.

    Parameters
    ----------
    path_simu : str
        Path to the simulation raw directory containing .athdf files.
    outname : str or None
        Output movie filename. If None a default name is used.
    vort_type : {"simulation", "physical"}
        Calculation mode: 'simulation' uses raw derivatives; 'physical'
        applies metric factors and rescaled coordinates.
    crop : tuple or None
        (xmin, xmax, ymin, ymax) in plotting coordinates. If None the
        full domain is shown (for physical mode a global rescaled extent
        is computed and used for a consistent view across frames).
    vmin, vmax : float or None
        Color scale limits. Default vmin=0.0, vmax=None (uses data-driven
        maximum); if vmax <= vmin the code falls back to a symmetric
        [-maxabs, maxabs] range.
    cmap : str
        Matplotlib colormap for the pcolormesh.
    nz_slice : int
        Index of the z-slice to plot (midplane by default).
    fps : int
        Frames per second for the output movie.
    dpi : int
        Output resolution (used for the saved movie frames).

    Returns
    -------
    None
    """

    if outname is None:
        outname = f"vorticity_movie_{vort_type}.mp4"

    files = get_hdf_files(path_simu)
    Nframes = len(files)
    print(f"Found {Nframes} frames.")

    # load geometry from first file
    data0 = open_hdf_files_with_collapse(path_simu, [files[0]], read_every=1)
    x = data0["x1"]
    y = data0["x2"]
    Nz = len(data0["x3"])
    midz = nz_slice if nz_slice is not None else (Nz // 2) # choose the z-index to slice (use provided or use center)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Prepare figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    cbar = None

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    # For physical mode without cropping, compute global rescaled extent 
    # for consistent plot limits across frames
    if crop is None and vort_type == "physical":
        Xmin_global = +np.inf
        Xmax_global = -np.inf
        Ymin_global = +np.inf
        Ymax_global = -np.inf

        for f in files:
            dat = open_hdf_files_with_collapse(path_simu, [f])
            R = dat['Rglobal'][0]
            Lz = dat['Lzglobal'][0]
            Sarr, aarr = collapse_param_decomposition(np.array([R]), np.array([Lz]))
            S, alpha = Sarr[0], aarr[0]

            Xtmp = x * alpha**(-3/2)
            Ytmp = y * alpha**(+3/2)

            Xmin_global = min(Xmin_global, Xtmp.min())
            Xmax_global = max(Xmax_global, Xtmp.max())
            Ymin_global = min(Ymin_global, Ytmp.min())
            Ymax_global = max(Ymax_global, Ytmp.max())

        print("Global rescaled extent:")
        print("X:", Xmin_global, Xmax_global)
        print("Y:", Ymin_global, Ymax_global)



    def init():
        # nothing to initialise — each frame clears and redraws
        return []

    def update(i):
        file_i = files[i]
        print(f"[{i+1}/{Nframes}] Loading {file_i}")
        data = open_hdf_files_with_collapse(path_simu, [file_i])

        vx = data["v1"][0, :, :, midz]
        vy = data["v2"][0, :, :, midz]
        t = data["time"][0]

        # compute vorticity based on selected mode
        dvy_dx = np.gradient(vy, dx, axis=0)
        dvx_dy = np.gradient(vx, dy, axis=1)
        
        if vort_type == "simulation":
            vort = dvy_dx - dvx_dy
            S, alpha = 1.0, 1.0
            Xplot, Yplot = x, y

        elif vort_type == "physical":
            vort, S, alpha = compute_physical_vorticity(data)
            # Rescaled physical coordinates
            Xplot = x * alpha**(-3/2)
            Yplot = y * alpha**(3/2)

        else:
            raise ValueError("vort_type must be 'simulation' or 'physical'.")


        # -------------------------------
        # Plotting parameters
        # -------------------------------

        if crop is not None:
            try:
                xmin, xmax, ymin, ymax = crop
            except Exception:
                raise ValueError("crop must be a 4-tuple: (xmin, xmax, ymin, ymax)")

            ix = np.where((Xplot >= xmin) & (Xplot <= xmax))[0]
            iy = np.where((Yplot >= ymin) & (Yplot <= ymax))[0]
            if ix.size == 0 or iy.size == 0:
                raise ValueError(
                    f"crop region yields empty selection: x in [{xmin},{xmax}], y in [{ymin},{ymax}]"
                )
            Xplot = Xplot[ix]
            Yplot = Yplot[iy]
            vort_plot = vort[np.ix_(ix, iy)]
        else:
            vort_plot = vort

        # Determine color range
        vmin_plot = 0.0 if vmin is None else vmin
        vmax_plot = np.max(vort_plot) if vmax is None else vmax
        if vmax_plot <= vmin_plot:
            vmax_plot = np.max(np.abs(vort_plot))
            vmin_plot = -vmax_plot

        nonlocal cbar

        ax.clear()  # remove previous artists

        # set a fixed viewing window for physical coordinates if required
        if vort_type == "physical":
            if crop is not None:
                xmin, xmax, ymin, ymax = crop
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_xlim(Xmin_global, Xmax_global)
                ax.set_ylim(Ymin_global, Ymax_global)

        # draw vorticity
        pc = ax.pcolormesh(Xplot, Yplot, vort_plot.T,
                           cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                           shading='auto')

        # replace colorbar at each frame
        if cbar is not None:
            try:
                cbar.remove()
            except Exception:
                pass
        cbar = fig.colorbar(pc, ax=ax)
        cbar.set_label('vorticity')

        # Title and labels
        if vort_type == "simulation":
            ax.set_title(f'vorticity (simulation) at t = {t:.3f}')
        else:
            ax.set_title(f'vorticity (physical) \n at t = {t:.3f}, '
                        f'S={S:.3f}, α={alpha:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if crop is not None and vort_type == "physical":
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_aspect('equal')



    # -------------------------------
    # Animation generation
    # -------------------------------
    ani = animation.FuncAnimation(fig, update,
                                  frames=Nframes,
                                  init_func=init,
                                  blit=False)

    print("Saving movie...")
    ani.save(outname, writer='ffmpeg', dpi=dpi, fps=fps)
    print(f"Movie saved: {outname}")



if __name__ == "__main__":
    from athena_collapse_analysis.config import RAW_DIR
    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")

    # --- Make the vorticity movie ---

    make_vorticity_movie(path_simu, outname="omegaTilda_movie_simulation.mp4",
                         vort_type="simulation")
    
    make_vorticity_movie(path_simu, outname="omegaTilda_movie_physical.mp4",
                         vort_type="physical")
    make_vorticity_movie(path_simu, outname="omegaTilda_movie_physical_crop.mp4",
                         vort_type="physical", crop=(-1.5, 1.5, -1.5, 1.5))