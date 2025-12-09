"""
Input/Output utilities for Athena++ simulation data.
====================================================

This module provides tools for reading Athena++ simulation output.
It includes functions to return a unified dictionary suitable for analysis or visualization.

* read collapse diagnostics from ``.hst`` history files,
* load conservative fields with optional spatial and temporal downsampling,
* compute derived quantities (e.g. pressure, collapse decomposition),

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

from . import athena_read


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


def open_hdf_files(
    files, 
    read_every=1, 
    resol=(1, 1, 1), 
    adia=True, 
    confirm_memory=False
):
    """
    Load conservative Athena++ `.athdf` output into a dictionary.
    Print memory usage estimation before loading if `confirm_memory` is `True`.

    Parameters
    ----------
    files : list of str
        Paths to athdf files.
    read_every : int
        Load every Nth file.
    resol : tuple of int
        Downsampling factors.
    adia : bool
        Whether to load total energy Etot if the simulation is adiabatic.
    confirm_memory : bool, default False
        If True, ask the user to confirm after showing RAM estimate.

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

    # --- Read header of first file for coordinate sizes ---
    _dic0 = athena_read.athdf(files_read[0])

    # Apply downsampling to coordinate vectors
    Nx1 = len(_dic0['x1v'][::factor1])
    Nx2 = len(_dic0['x2v'][::factor2])
    Nx3 = len(_dic0['x3v'][::factor3])

    # Number of 3D fields to store
    n_fields = 4  # rho, v1, v2, v3
    if adia:
        n_fields += 1  # Etot

    # --- Compute memory estimate ---
    n_cells = Nt * Nx1 * Nx2 * Nx3
    bytes_total_est = n_cells * n_fields * 8  # float64 = 8 bytes
    gb_est = bytes_total_est / (1024**3)

    print(
        f"\nEstimated RAM required: {gb_est:.3f} GB\n"
        f"  snapshots: {Nt}\n"
        f"  resolution: {Nx1} × {Nx2} × {Nx3}\n"
        f"  fields: {n_fields}\n"
        f"  downsampling resol = {resol}\n"
    )

    # --- Ask user for confirmation ---
    if confirm_memory:
        ans = input("Proceed with loading? (y/n): ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted by user.")
            return None

    # --- Proceed with actual loading ---
    x1 = _dic0['x1v'][::factor1]
    x2 = _dic0['x2v'][::factor2]
    x3 = _dic0['x3v'][::factor3]

    time = np.zeros(Nt)
    rho = np.zeros((Nt, Nx1, Nx2, Nx3))
    if adia:
        Etot = np.zeros((Nt, Nx1, Nx2, Nx3))
    v1 = np.zeros((Nt, Nx1, Nx2, Nx3))
    v2 = np.zeros((Nt, Nx1, Nx2, Nx3))
    v3 = np.zeros((Nt, Nx1, Nx2, Nx3))

    for k, f in enumerate(files_read):
        print("opening file", f)
        _dic = athena_read.athdf(f)

        time[k] = _dic['Time']

        dens = _dic['dens'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]
        rho[k] = dens

        if adia:
            Etot[k] = _dic['Etot'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]

        mom1 = _dic['mom1'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]
        mom2 = _dic['mom2'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]
        mom3 = _dic['mom3'].transpose(2, 1, 0)[::factor1, ::factor2, ::factor3]

        v1[k] = mom1 / dens
        v2[k] = mom2 / dens
        v3[k] = mom3 / dens

    dic_out = {'x1': x1, 'x2': x2, 'x3': x3, 'time': time,
               'rho': rho, 'v1': v1, 'v2': v2, 'v3': v3}

    if adia:
        dic_out['Etot'] = Etot

    return dic_out



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
        Output dictionary from `open_hdf_files`
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
def open_hdf_files_with_collapse(data_path, files, read_every=1, resol=(1,1,1), adia=True, confirm_memory=False):
    """
    Load conservative fields and attach interpolated collapse diagnostics.
    Print memory usage estimation before loading if `confirm_memory` is `True`.

    This wrapper:
    1. Loads conservative variables from `.athdf` files using
       `open_hdf_files`.
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
    confirm_memory : bool, default False
        If True, ask the user to confirm after showing RAM estimate.

    Returns
    -------
    dict
        dict variables of open_hdf_files with added collapse profiles:
        - 'Rglobal' : ndarray
        - 'Lzglobal' : ndarray
    """
    dic_list = open_hdf_files(files, read_every=read_every, resol=resol, adia=adia, confirm_memory=confirm_memory)
    time_hst, Rglobal_hst, Lzglobal_hst = get_collapse_profile(data_path)

    return add_collapse_param(dic_list, time_hst, Rglobal_hst, Lzglobal_hst)

