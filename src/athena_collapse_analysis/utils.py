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



def collapse_param_decomposition(R, Lz):
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

    Returns
    -------
    S : ndarray of shape (Nt,)
        Scale parameter describing isotropic contraction.
    alpha : ndarray of shape (N,)
        Anisotropy parameter describing deformation of the collapse.

    Raises
    ------
    ValueError
        If ``R`` and ``Lz`` do not have the same shape.
    """
    if R.shape != Lz.shape:
        raise ValueError("R and Lz must have the same shape.")

    S = (R/R[0]**2 * Lz/Lz[0])**(1/3)
    alpha = (R/R[0]) / (Lz/Lz[0])
    return S, alpha


