"""
Functions to compute derived quantities and decompositions (pressure, collapse parameters, metric_transform) from Athena++ data.
"""

import numpy as np

# --- Derived quantities

def pressure_from_conservatives(rho, Etot, v1, v2, v3, gamma = 5/3):
    """
    Compute pressure from conservative variables.

    Parameters
    ----------
    rho : ndarray
        Density.
    Etot : ndarray
        Total energy density.
    v1, v2, v3 : ndarray
        Velocity components.
    gamma : float, optional
        Adiabatic index. Default is 5/3.

    Returns
    -------
    p : ndarray
        Thermodynamic pressure.
    """
    KE = 0.5 * rho * (v1**2 + v2**2 + v3**2)
    e = Etot - KE
    p = (gamma - 1) * e
    return p



def compute_physical_vorticity(data, nz_slice=0):
    """
    Compute physical vorticity ω̃ from velocity fields.

    Parameters
    ----------
    data : dict
        Athena++ HDF5 output as returned by open_hdf_files_with_collapse
    nz_slice : int
        z-index slice to extract

    Returns
    -------
    omega : ndarray
        Physical vorticity ω̃ at the given slice
    S : float
        Collapse scale factor
    alpha : float
        Collapse anisotropy parameter
    """
    x, y = data["x1"], data["x2"]
    dx, dy = x[1] - x[0], y[1] - y[0]

    vx = data["v1"][0, :, :, nz_slice]
    vy = data["v2"][0, :, :, nz_slice]

    dvydx = np.gradient(vy, dx, axis=0)
    dvxdy = np.gradient(vx, dy, axis=1)

    R = data["Rglobal"][0]
    Lz = data["Lzglobal"][0]
    S, alpha = collapse_param_decomposition(R, Lz)

    gxx = S**2 * alpha**(-4)
    gyy = S**2 * alpha**2

    omega = alpha * (gyy * dvydx - gxx * dvxdy)

    return omega, S, alpha





def collapse_param_decomposition(R, Lz, R_0=1.0, Lz_0=1.0):
    """
    Decompose collapse diagnostics into scale and anisotropy parameters.

    From the collapse global param `R`and `Lz`, constructs

    * a scale parameter ``S`` defined as::

          S = ( R / R[0]**2 * Lz / Lz[0] )**(1/3)

    * an anisotropy parameter ``alpha`` defined as::

          alpha = (R / R[0]) / (Lz / Lz[0])

    Parameters
    ----------
    R : ndarray of shape (Nt,)
        Collapse global param R as a function of time.
    Lz : ndarray of shape (Nt,)
        Collapse global param Lz as a function of time.
    R_0 : float, optional
        Initial value of R. Default is 1.0.
    Lz_0 : float, optional
        Initial value of Lz. Default is 1.0.
        
    Returns
    -------
    S : ndarray of shape (Nt,)
        Scale parameter describing isotropic contraction.
    alpha : ndarray of shape (N,)
        Anisotropy parameter describing deformation of the collapse.

    Raises
    ------
    ValueError
        If ``R`` and ``Lz`` are not positive.
    """
    if np.any(R <= 0) or np.any(Lz <= 0):
        raise ValueError("R and Lz must be positive")
    S = ( (R/R_0)**2 * Lz/Lz_0)**(1/3)
    alpha = ( (R/R_0) / (Lz/Lz_0) )**(1/3)
    return S, alpha


