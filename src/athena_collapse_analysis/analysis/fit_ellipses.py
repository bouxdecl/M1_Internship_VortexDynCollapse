#!/usr/bin/env python3
"""
Ellipse fitting diagnostics from physical vorticity ω̃ (omegaTilda).

Pipeline:
1. Compute physical vorticity ω̃ from Athena++ output.
2. Solve Poisson equation Δψ = -ω̃ for the streamfunction ψ.
3. Extract closed streamfunction contours.
4. Fit centered ellipses to the contours.
5. Plot streamfunction contours with fitted ellipses.

6. Plot ellipse aspect-ratio and orientation profiles over multiple times.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq

from athena_collapse_analysis.utils import collapse_param_decomposition, compute_physical_vorticity
from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files_with_collapse,
)

# ============================================================
# Fitting one ellipse from points
# ============================================================

def fit_centered_ellipse(x, y):
    """
    Fit a centered ellipse to a set of (x, y) points using least squares.

    Returns
    -------
    a : float
        Semi-major axis
    b : float
        Semi-minor axis
    theta_major : float
        Orientation angle of major axis in radians (0 ≤ θ < π)
    e : float
        Eccentricity
    """

    D = np.vstack([x**2, x*y, y**2]).T
    d = np.ones_like(x)

    # quadratic form coefficients
    A, B, C = np.linalg.lstsq(D, d, rcond=None)[0]

    # Angle with x-axis from coefficients 
    theta = 0.5 * np.arctan2(B, A - C)
    c, s = np.cos(theta), np.sin(theta)

    # Rotate quadratic form
    Ap = A*c**2 + B*c*s + C*s**2
    Cp = A*s**2 - B*c*s + C*c**2

    r1, r2 = 1 / np.sqrt(Ap), 1 / np.sqrt(Cp)

    # Major/minor axis assignment and angle adjustment
    if r1 >= r2:
        a, b = r1, r2
        theta_major = theta
    else:
        a, b = r2, r1
        theta_major = theta + np.pi / 2

    # eccentricity computation
    e = np.sqrt(1 - (b / a)**2)
    theta_major = (theta_major + np.pi) % np.pi

    return a, b, theta_major, e


# ============================================================
# Physics : vorticity an streamfunction
# ============================================================
def solve_streamfunction(omega, dx, dy, alpha):
    """
    Solve the Poisson equation Δψ = -ω̃ using FFT in 2D.

    Parameters
    ----------
    omega : ndarray
        Physical vorticity ω̃
    dx, dy : float
        Grid spacing in x and y
    alpha : float
        Collapse anisotropy parameter

    Returns
    -------
    psi : ndarray
        Streamfunction ψ
    """
    Nx, Ny = omega.shape
    a32 = alpha**(3 / 2)

    kx = fftfreq(Nx, d=dx * a32**-1) * 2 * np.pi
    ky = fftfreq(Ny, d=dy * a32) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # avoid division by zero

    psi_hat = -fft2(omega) / K2
    psi_hat[0, 0] = 0.0  # zero-mean condition

    return np.real(ifft2(psi_hat))


# ============================================================
# Contours → ellipses
# ============================================================

def extract_contours(X, Y, psi, levels=40):
    """Extract contour paths of streamfunction (Matplotlib-robust)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.contour(X, Y, psi.T, levels=levels, colors="k")
    plt.close(fig)

    contours = []
    for path in cs.get_paths():
        contours.append(path.vertices)

    return contours



def fit_ellipses_from_contours(
    contours,
    min_points=25,
    max_radius=1.0,
):
    """Fit ellipses to selected contours."""
    ellipses = []

    for verts in contours:
        if verts.shape[0] < min_points:
            continue

        xvals, yvals = verts[:, 0], verts[:, 1]

        if np.hypot(xvals.mean(), yvals.mean()) > max_radius:
            continue

        try:
            ellipses.append(fit_centered_ellipse(xvals, yvals))
        except Exception:
            pass

    return ellipses


# ============================================================
# Plotting
# ============================================================
import numpy as np
import matplotlib.pyplot as plt


def plot_streamfunction_and_ellipses(
    X, Y, psi, ellipses,
    levels=40,
    xlim=None,
    ylim=None,
):
    """
    Plot streamfunction contours and fitted ellipses (consistent geometry).
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    # --- Streamfunction contours ---
    ax.contour(
        X, Y, psi.T,
        levels=levels,
        colors="k",
        linewidths=0.8
    )

    # --- Ellipses (parametric form) ---
    t = np.linspace(0, 2*np.pi, 400)

    for a, b, theta, e in ellipses:
        ct, st = np.cos(theta), np.sin(theta)

        x = a * np.cos(t)
        y = b * np.sin(t)

        Xell = ct * x - st * y
        Yell = st * x + ct * y

        ax.plot(Xell, Yell, "r--", lw=2)

    if xlim is None:
        xlim = np.max(np.abs(X))
    if ylim is None:    
        ylim = np.max(np.abs(Y))
    
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(r"$\tilde{x}$")
    ax.set_ylabel(r"$\tilde{y}$")
    ax.set_title("Streamfunction contours and fitted ellipses")

    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return fig, ax



def plot_multi_time_ellipse_profiles(
    path_simu,
    files,
    time_indices,
    nz_slice=0,
    rmin_skip=1,
    xmin_plot=0,
    xmax_plot=None,
    ylim_tuple_eta=None,
    ylim_tuple_theta=None,
):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for i in time_indices:
        # Read simulation time and ellipses
        data = open_hdf_files_with_collapse(path_simu, files=[files[i]])
        time = data["time"][0]

        ellipses = process_single_file(
            path_simu,
            files[i],
            nz_slice=nz_slice,
            n_ellipses=50,
            plot=False,
        )

        # extract profiles
        if len(ellipses) < rmin_skip + 1:
            continue

        a = np.array([e[0] for e in ellipses])
        b = np.array([e[1] for e in ellipses])
        theta = np.array([e[2] for e in ellipses])

        r = np.sqrt(a * b)
        eta = a / b

        r = r[rmin_skip:]
        eta = eta[rmin_skip:]
        theta = theta[rmin_skip:]

        axs[0].plot(r, eta, "o-", label=f"t={np.round(time)}")
        axs[1].plot(r, np.degrees(theta), "o-", label=f"t={np.round(time)}")

    if xmax_plot is not None:
        axs[0].set_xlim(xmin_plot, xmax_plot)
    if ylim_tuple_eta is not None:
        axs[0].set_ylim(*ylim_tuple_eta)
    axs[0].set_xlabel(r"$\sqrt{ab}$")
    axs[0].set_ylabel(r"$a/b$")
    axs[0].set_title("Aspect-ratio profile")
    axs[0].grid()
    axs[0].legend()


    if xmax_plot is not None:
        axs[1].set_xlim(xmin_plot, xmax_plot)
    if ylim_tuple_theta is not None:
        axs[1].set_ylim(*ylim_tuple_theta)
    axs[1].set_xlabel(r"$\sqrt{ab}$")
    axs[1].set_ylabel(r"$\theta$ (deg)")
    axs[1].set_title("Orientation angle profile")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return fig, axs




# ============================================================
# High-level driver
# ============================================================

def process_single_file(path_simu, file, nz_slice=0, plot=True, n_ellipses=40, xlim=None, ylim=None):
    """Full pipeline for a single Athena++ output."""
    data = open_hdf_files_with_collapse(path_simu, files=[file])

    x, y = data["x1"], data["x2"]
    dx, dy = x[1] - x[0], y[1] - y[0]

    omega, S, alpha = compute_physical_vorticity(data, nz_slice)
    psi = solve_streamfunction(omega, dx, dy, alpha)

    a32 = alpha**(3 / 2)
    X = x * a32**-1
    Y = y * a32

    contours = extract_contours(X, Y, psi, levels=n_ellipses)
    ellipses = fit_ellipses_from_contours(contours)

    if plot:
        plot_streamfunction_and_ellipses(X, Y, psi, ellipses, xlim=xlim, ylim=ylim)
    
    return ellipses


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":

    from athena_collapse_analysis.config import RAW_DIR

    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    files = get_hdf_files(path_simu)

    ellipses = process_single_file(
        path_simu,
        files[-1],
        nz_slice=0,
        plot=True,
        n_ellipses=50,
        xlim=1.5,
        ylim=1.5,
    )


    print(f"Found {len(ellipses)} ellipses")

    plot_multi_time_ellipse_profiles(
    path_simu,
    files,
    time_indices=[5, 20, 40, 60],
    rmin_skip=1,
    xmax_plot=3.0,
    ylim_tuple_eta=(0.99, 1.1),
    ylim_tuple_theta=(0, 92),
    )

