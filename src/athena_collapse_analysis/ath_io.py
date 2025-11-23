"""
Input/Output utilities for Athena++ simulation data.
====================================================

This module provides tools for reading, organizing, and post-processing
Athena++ simulation output. It includes functions to

* discover and sort ``.athdf`` files,
* load conservative fields with optional spatial and temporal downsampling,
* compute derived hydrodynamical quantities (e.g. pressure),
* read collapse diagnostics from ``.hst`` history files,
* interpolate global collapse parameters (``Rglobal``, ``Lzglobal``) to match
  the time stamps of the ``.athdf`` outputs,
* return a unified dictionary suitable for analysis or visualization.

Notes
-----
Athena++ stores multidimensional fields in ``(z, x, y)`` order.
This module reorders them to ``(x1, x2, x3)`` for consistency with analytical
derivations and plotting conventions.

Collapse diagnostics read from the ``.hst`` file are normalized by the number
of meshblocks (``NumMeshBlocks``) to obtain global values.
"""

import glob
import os
import numpy as np

import sys
sys.path.insert(0, '../../../athena_collapsingbox/vis/python')
import athena_read



def get_hdf_files(data_path, verbose=True):
    """
    Return a sorted list of all `.athdf` files in a directory.

    Parameters
    ----------
    data_path : str
        Path to the directory containing Athena++ output files.
    verbose : bool, optional
        If True, prints the number of detected files. Default is True.

    Returns
    -------
    list of str
        Sorted list of full paths to `.athdf` files found in `data_path`.
    """
    files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.athdf')])
    if verbose:
        print( 'There are {} files.'.format(len(files)) )
    return files



def open_hdf_files_cons(files, read_every=1, resol=(1, 1, 1), adia=True):
    """
    Load conservative Athena++ `.athdf` output into a dictionary.

    Parameters
    ----------
    files : list of str
        List of paths to `.athdf` files.
    read_every : int, optional
        Read every Nth file to subsample the time series. Default is 1.
    resol : tuple of int, optional
        Downsampling factors in (x1, x2, x3). Default is (1, 1, 1).
    adia : bool, optional
        If True, simulation is adiabatic, reads also total energy `Etot`. If False, skips it.

    Returns
    -------
    dict
        Dictionary containing:
        - 'x1', 'x2', 'x3' : ndarray
            Cell-centered coordinates.
        - 'time' : ndarray of shape (Nt,)
            Simulation times.
        - 'rho' : ndarray of shape (Nt, Nx1, Nx2, Nx3)
            Density field.
        - 'v1', 'v2', 'v3' : ndarray of shape (Nt, Nx1, Nx2, Nx3)
            Velocity components.
        - 'Etot' : ndarray of shape (Nt, Nx1, Nx2, Nx3), optional
                    Total energy (if `adia=True`).

    Notes
    -----
    Athena++ stores arrays in (z, x, y) order; this routine transposes to (x1, x2, x3).
    """

    factor1, factor2, factor3 = resol

    files_read = files[::read_every]
    Nt = len(files_read)

    # Coordinates (z, x, y)
    _dic = athena_read.athdf(files_read[0])

    x1, x2, x3 = _dic['x1v'], _dic['x2v'], _dic['x3v']

    x1 = x1[::factor1]
    x2 = x2[::factor2]
    x3 = x3[::factor3]

    Nx1, Nx2, Nx3 = len(x1), len(x2), len(x3)

    # Initialize arrays with zeros (better semantics)
    time = np.zeros(Nt)
    rho = np.zeros((Nt, Nx1, Nx2, Nx3))
    if adia:
        Etot = np.zeros((Nt, Nx1, Nx2, Nx3))
    v1 = np.zeros((Nt, Nx1, Nx2, Nx3))
    v2 = np.zeros((Nt, Nx1, Nx2, Nx3))
    v3 = np.zeros((Nt, Nx1, Nx2, Nx3))

    for k, f in enumerate(files_read):
        print(f)
        _dic = athena_read.athdf(f)

        time[k] = _dic['Time']

        # Precompute downsampled transposed density once
        dens = _dic['dens'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3] #Athena++ stores array data as (z, x, y)
        rho[k] = dens

        if adia:
            Etot[k] = _dic['Etot'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]

        # Precompute mom arrays
        mom1 = _dic['mom1'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]
        mom2 = _dic['mom2'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]
        mom3 = _dic['mom3'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]

        # Compute velocity with elementwise division
        v1[k] = mom1 / dens
        v2[k] = mom2 / dens
        v3[k] = mom3 / dens

    del _dic

    dic_data = {'x1': x1, 'x2': x2, 'x3': x3, 'time': time, 'rho': rho, 'v1': v1, 'v2': v2, 'v3': v3}
    if adia:
        dic_data['Etot'] = Etot

    return dic_data


# ------------------------------------------------------------
# Add interpolated collapse parameters to a single dic_data
# ------------------------------------------------------------
def get_collapse_profile(data_path):
    """
    Load collapse diagnostics (Rglobal, Lzglobal) from a single `.hst` file.

    Parameters
    ----------
    data_path : str
        Directory containing both `.athdf` and `.hst` files.

    Returns
    -------
    time_hst : ndarray of shape (Nhst,)
        Times stored in the `.hst` file.
    Rglobal_hst : ndarray of shape (Nhst,)
        Radius diagnostic per meshblock.
    Lzglobal_hst : ndarray of shape (Nhst,)
        Angular momentum diagnostic per meshblock.

    Raises
    ------
    FileNotFoundError
        If zero or more than one `.hst` file is found.
    """

    # determine number of meshblocks
    first_hdf = athena_read.athdf(get_hdf_files(data_path)[0])
    ncores = first_hdf['NumMeshBlocks']
    del first_hdf

    # find exactly one .hst file
    hst_files = glob.glob(os.path.join(data_path, "*.hst"))
    if len(hst_files) == 0:
        raise FileNotFoundError("No .hst file found in data_path")
    if len(hst_files) > 1:
        raise FileNotFoundError("Several .hst files found in data_path")

    # load
    data_hst = athena_read.hst(hst_files[0])

    # extract profile
    time_hst     = data_hst['time']
    Rglobal_hst  = data_hst['Rglobal'] / ncores
    Lzglobal_hst = data_hst['Lz']      / ncores

    return time_hst, Rglobal_hst, Lzglobal_hst



# ------------------------------------------------------------
# Add interpolated collapse parameters to a single dic_data
# ------------------------------------------------------------
def add_collapse_param(dic_data, time_hst, Rglobal_hst, Lzglobal_hst):
    """
    Attach interpolated collapse parameters to a data dictionary.

    Parameters
    ----------
    dic_data : dict
        Output dictionary from `open_hdf_files_cons`
    time_hst : ndarray
        Time array from the `.hst` file.
    Rglobal_hst : ndarray
        Collapse global variable R from `.hst`.
    Lzglobal_hst : ndarray
        Collapse global variable Lz from `.hst`.

    Returns
    -------
    dict
        The input dictionary, augmented with:
        - 'Rglobal' : ndarray of shape (Nt,)
        - 'Lzglobal' : ndarray of shape (Nt,)

    Raises
    ------
    ValueError
        If interpolated arrays do not match the shape of `dic_data['time']`.
    """
    t = dic_data['time']
    t_min, t_max = time_hst[0], time_hst[-1]

    # Check for strictly out-of-bounds times
    if np.any(t < t_min) or np.any(t > t_max):
        raise ValueError(
            f"Simulation times in dic_data are outside the .hst time range: "
            f"[{t_min}, {t_max}]"
        )

    dic_data['Rglobal'] = np.interp(t, time_hst, Rglobal_hst)
    dic_data['Lzglobal'] = np.interp(t, time_hst, Lzglobal_hst)

    return dic_data




# ------------------------------------------------------------
# Full wrapper: load HDF files + add collapse parameters
# ------------------------------------------------------------
def load_cons_with_collapse(data_path, files, read_every=1, resol=(1,1,1), adia=True):
    """
    Load conservative fields and attach interpolated collapse diagnostics.

    This wrapper:
    1. Loads conservative variables from `.athdf` files using
       `open_hdf_files_cons`.
    2. Loads collapse diagnostics from the `.hst` file via
       `get_collapse_profile`.
    3. Interpolates the collapse diagnostics to the times of the
       conservative outputs.

    Parameters
    ----------
    data_path : str
        Path containing the simulation output.
    files : list of str
        List of `.athdf` files to load.
    read_every : int, optional
        Read every Nth `.athdf` file. Default is 1.
    resol : tuple of int, optional
        Downsampling factors along (x1, x2, x3). Default is (1, 1, 1).
    adia : bool, optional
        If True, load the total energy field `Etot`.

    Returns
    -------
    dict
        Conservative variables with added collapse profiles:
        - 'Rglobal' : ndarray
        - 'Lzglobal' : ndarray
    """
    dic_list = open_hdf_files_cons(files, read_every=read_every, resol=resol, adia=adia)
    time_hst, Rglobal_hst, Lzglobal_hst = get_collapse_profile(data_path)

    return add_collapse_param(dic_list, time_hst, Rglobal_hst, Lzglobal_hst)




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


