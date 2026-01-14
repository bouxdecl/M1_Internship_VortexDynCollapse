#!/usr/bin/env python3
"""
Extract simulation data for Hamiltonian comparison
==================================================

This standalone script extracts and caches all necessary data from
Athena++ simulation HDF5 files for later comparison with analytical models.

Extracted data:
- Time array
- Scale factor α(t)
- S parameter S(t)
- Reference vorticity ω₀
- Fitted ellipse parameters
- Derived (p, q) coordinates

The data is cached to avoid reprocessing large simulation datasets.

Usage:
    python extract_simulation_data.py
    
    # Or from another script:
    from extract_simulation_data import extract_and_cache_simulation_data
    data = extract_and_cache_simulation_data(path_simu)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Import required analysis tools
from athena_collapse_analysis.analysis.fit_ellipses import process_single_file
from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files_with_collapse,
)
from athena_collapse_analysis.utils import (
    collapse_param_decomposition,
    compute_physical_vorticity,
)


# ============================================================
# Coordinate transformation utilities
# ============================================================

def r_pol(eta):
    """
    Convert aspect ratio to polar radius coordinate.
    
    r = sqrt(2*(η - 1)² / η)
    
    Parameters
    ----------
    eta : float or array_like
        Aspect ratio (a/b)
    
    Returns
    -------
    r : float or array_like
        Polar radius coordinate
    """
    return np.sqrt(2 * (eta - 1)**2 / eta)


def extract_pq_from_ellipses(ellipse_params_arr, ellipse_index=1):
    """
    Extract (p, q) coordinates from ellipse parameters.
    
    Parameters
    ----------
    ellipse_params_arr : ndarray
        Array of ellipse parameters with shape (n_times, n_ellipses, 4)
        where the 4 parameters are [a, b, theta, e]
    ellipse_index : int, optional
        Which ellipse to use (default: 1, the second innermost)
    
    Returns
    -------
    p : ndarray
        p coordinate array
    q : ndarray
        q coordinate array
    """
    # Extract parameters for selected ellipse
    a = ellipse_params_arr[:, ellipse_index, 0]
    b = ellipse_params_arr[:, ellipse_index, 1]
    theta = ellipse_params_arr[:, ellipse_index, 2]
    
    # Compute angle (factor of 2)
    angle = 2 * theta
    
    # Compute r coordinate from aspect ratio
    eta = a / b
    r_coord = r_pol(eta)
    
    # Convert to (p, q)
    p = r_coord * np.cos(angle)
    q = r_coord * np.sin(angle)
    
    return p, q


# ============================================================
# Cache management utilities
# ============================================================

def get_cache_filename(path_simu, n_ellipses, ellipse_index):
    """
    Generate cache filename for simulation data.
    
    Parameters
    ----------
    path_simu : str
        Path to simulation directory
    n_ellipses : int
        Number of ellipse contours
    ellipse_index : int
        Which ellipse to track
    
    Returns
    -------
    cache_file : str
        Full path to cache file
    """
    # Create cache directory
    cache_dir = os.path.join(path_simu, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate filename based on parameters
    cache_filename = f"simulation_data_n{n_ellipses}_idx{ellipse_index}.pkl"
    cache_file = os.path.join(cache_dir, cache_filename)
    
    return cache_file


def save_simulation_data(data_dict, cache_file):
    """
    Save simulation data to pickle file.
    
    Parameters
    ----------
    data_dict : dict
        Simulation data dictionary
    cache_file : str
        Path to cache file
    """
    with open(cache_file, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved data to cache: {cache_file}")


def load_simulation_data(cache_file):
    """
    Load simulation data from pickle file.
    
    Parameters
    ----------
    cache_file : str
        Path to cache file
    
    Returns
    -------
    data_dict : dict
        Simulation data dictionary
    """
    with open(cache_file, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"✓ Loaded data from cache: {cache_file}")
    return data_dict


# ============================================================
# Main extraction function
# ============================================================

def extract_simulation_data(path_simu, files, nz_slice=0, 
                           n_ellipses=40, ellipse_index=1,
                           use_cache=True, force_recompute=False,
                           verbose=True):
    """
    Extract all necessary data from simulation files (with caching).
    
    Parameters
    ----------
    path_simu : str
        Path to simulation directory
    files : list
        List of HDF5 files
    nz_slice : int, optional
        z-slice to analyze (default: 0)
    n_ellipses : int, optional
        Number of ellipse contours to fit (default: 40)
    ellipse_index : int, optional
        Which ellipse to track (default: 1, the second innermost)
    use_cache : bool, optional
        Whether to use cached data if available (default: True)
    force_recompute : bool, optional
        Force recomputation even if cache exists (default: False)
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'time': time array
        - 'alpha': α(t) values
        - 'S': S(t) values
        - 'w0': reference vorticity
        - 'p_sim': simulated p values
        - 'q_sim': simulated q values
        - 'ellipse_params': full ellipse parameter array (n_times, n_ellipses, 4)
        - 'metadata': dict with extraction parameters
    """
    # Check for cached data
    cache_file = get_cache_filename(path_simu, n_ellipses, ellipse_index)
    
    if use_cache and not force_recompute and os.path.exists(cache_file):
        if verbose:
            print("Cache file found! Loading from cache...")
        try:
            data_dict = load_simulation_data(cache_file)
            
            # Verify cache is compatible with current files
            if len(data_dict['time']) == len(files):
                if verbose:
                    print("✓ Cache is valid and up-to-date")
                return data_dict
            else:
                if verbose:
                    print(f"⚠ Cache has {len(data_dict['time'])} files, but found {len(files)} files")
                    print("  Recomputing data...")
        except Exception as e:
            if verbose:
                print(f"⚠ Error loading cache: {e}")
                print("  Recomputing data...")
    
    # Compute data from scratch
    if verbose:
        print("Computing simulation data from HDF5 files...")
    
    n_files = len(files)
    
    # Storage arrays
    time_arr = np.zeros(n_files)
    alpha_arr = np.zeros(n_files)
    S_arr = np.zeros(n_files)
    w_arr = np.zeros(n_files)
    ellipse_params_arr = []
    
    if verbose:
        print(f"Processing {n_files} simulation files...")
    
    for i, file in enumerate(files):
        # Load data
        data = open_hdf_files_with_collapse(path_simu, files=[file])
        
        # Extract time and collapse parameters
        time_arr[i] = data["time"][0]
        
        # Compute physical vorticity and parameters
        omega, S, alpha = compute_physical_vorticity(data, nz_slice)
        
        # Get center vorticity for w0
        mid_x, mid_y = omega.shape[0] // 2, omega.shape[1] // 2
        if i == 0:
            w0 = omega[mid_x, mid_y]
        w_arr[i] = omega[mid_x, mid_y]
        
        # Store parameters
        alpha_arr[i] = alpha
        S_arr[i] = S
        
        # Fit ellipses
        ellipses = process_single_file(
            path_simu,
            file,
            nz_slice=nz_slice,
            n_ellipses=n_ellipses,
            plot=False,
        )
        
        ellipse_params_arr.append(ellipses)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_files} files")
    
    # Convert ellipse list to array
    max_ellipses = max(len(e) for e in ellipse_params_arr)
    
    # Create padded array
    ellipse_array = np.zeros((n_files, max_ellipses, 4))
    for i, ellipses in enumerate(ellipse_params_arr):
        n_ell = len(ellipses)
        if n_ell > 0:
            ellipse_array[i, :n_ell, :] = np.array(ellipses)
    
    # Extract (p, q) from ellipses
    p_sim, q_sim = extract_pq_from_ellipses(ellipse_array, ellipse_index)
    
    if verbose:
        print(f"\n✓ Extraction complete!")
        print(f"  Time range: [{time_arr[0]:.2f}, {time_arr[-1]:.2f}]")
        print(f"  ω₀ = {w0:.6f}")
        print(f"  α range: [{alpha_arr.min():.4f}, {alpha_arr.max():.4f}]")
        print(f"  S range: [{S_arr.min():.4f}, {S_arr.max():.4f}]")
        print(f"  Fitted {ellipse_array.shape[1]} ellipses at each timestep")
    
    # Create data dictionary with metadata
    data_dict = {
        'time': time_arr,
        'alpha': alpha_arr,
        'S': S_arr,
        'w0': w0,
        'w': w_arr,
        'p_sim': p_sim,
        'q_sim': q_sim,
        'ellipse_params': ellipse_array,
        'metadata': {
            'n_files': n_files,
            'nz_slice': nz_slice,
            'n_ellipses': n_ellipses,
            'ellipse_index': ellipse_index,
            'max_ellipses_fitted': ellipse_array.shape[1],
            'path_simu': path_simu,
        }
    }
    
    # Save to cache
    if use_cache:
        save_simulation_data(data_dict, cache_file)
    
    return data_dict


# ============================================================
# Convenience wrapper
# ============================================================

def extract_and_cache_simulation_data(path_simu, nz_slice=0, n_ellipses=40,
                                     ellipse_index=1, use_cache=True,
                                     force_recompute=False, verbose=True):
    """
    Extract simulation data with automatic file detection.
    
    This is a convenience wrapper that automatically finds HDF5 files
    in the simulation directory.
    
    Parameters
    ----------
    path_simu : str
        Path to simulation directory
    nz_slice : int, optional
        z-slice to analyze (default: 0)
    n_ellipses : int, optional
        Number of ellipse contours (default: 40)
    ellipse_index : int, optional
        Which ellipse to track (default: 1)
    use_cache : bool, optional
        Use cached data if available (default: True)
    force_recompute : bool, optional
        Force recomputation (default: False)
    verbose : bool, optional
        Print progress (default: True)
    
    Returns
    -------
    data_dict : dict
        Extracted simulation data
    """
    # Find HDF5 files
    files = get_hdf_files(path_simu)
    
    if verbose:
        print("=" * 70)
        print("Simulation Data Extraction")
        print("=" * 70)
        print(f"\nSimulation directory: {path_simu}")
        print(f"Found {len(files)} HDF5 files")
        print(f"Parameters: nz_slice={nz_slice}, n_ellipses={n_ellipses}, ellipse_index={ellipse_index}")
        print()
    
    # Extract data
    data = extract_simulation_data(
        path_simu, files,
        nz_slice=nz_slice,
        n_ellipses=n_ellipses,
        ellipse_index=ellipse_index,
        use_cache=use_cache,
        force_recompute=force_recompute,
        verbose=verbose,
    )
    
    if verbose:
        print("\n" + "=" * 70)
    
    return data


# ============================================================
# Visualization utilities
# ============================================================

def plot_extracted_data(data_dict, save_path=None):
    """
    Create diagnostic plots of extracted data.
    
    Parameters
    ----------
    data_dict : dict
        Extracted simulation data
    save_path : str, optional
        Path to save figure (if None, displays interactively)
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    
    # α(t)
    axs[0, 0].plot(data_dict['time'], data_dict['alpha'], lw=2, color='C0')
    axs[0, 0].set_xlabel('Time', fontsize=11)
    axs[0, 0].set_ylabel('α(t)', fontsize=11)
    axs[0, 0].set_title('Scale factor α(t)', fontsize=12)
    axs[0, 0].grid(alpha=0.3)
    
    # S(t)
    axs[0, 1].plot(data_dict['time'], data_dict['S'], lw=2, color='C1')
    axs[0, 1].set_xlabel('Time', fontsize=11)
    axs[0, 1].set_ylabel('S(t)', fontsize=11)
    axs[0, 1].set_title('S parameter S(t)', fontsize=12)
    axs[0, 1].grid(alpha=0.3)
    
    # ω(t)
    axs[0, 2].plot(data_dict['time'], data_dict['w'], lw=2, color='C2')
    axs[0, 2].axhline(data_dict['w0'], color='gray', ls='--', lw=1)
    axs[0, 2].set_xlabel('Time', fontsize=11)
    axs[0, 2].set_ylabel('ω(t)', fontsize=11)
    axs[0, 2].set_title('Central vorticity ω(t)', fontsize=12)
    
    # Phase space (p, q)
    axs[1, 0].plot(data_dict['q_sim'], data_dict['p_sim'], lw=2, color='C3')
    axs[1, 0].scatter(0, 0, color='orange', marker='+', s=100, zorder=5)
    axs[1, 0].axhline(0, color='gray', lw=0.5)
    axs[1, 0].axvline(0, color='gray', lw=0.5)
    axs[1, 0].set_xlabel('q', fontsize=11)
    axs[1, 0].set_ylabel('p', fontsize=11)
    axs[1, 0].set_title('Phase space (p, q)', fontsize=12)
    axs[1, 0].grid(alpha=0.3)
    
    # q(t)
    axs[1, 1].plot(data_dict['time'], data_dict['q_sim'], lw=2, color='C4')
    axs[1, 1].set_xlabel('Time', fontsize=11)
    axs[1, 1].set_ylabel('q', fontsize=11)
    axs[1, 1].set_title('q coordinate evolution', fontsize=12)
    axs[1, 1].grid(alpha=0.3)
    
    # p(t)
    axs[1, 2].plot(data_dict['time'], data_dict['p_sim'], lw=2, color='C5')
    axs[1, 2].axhline(0, color='gray', lw=0.5, ls='--')
    axs[1, 2].set_xlabel('Time', fontsize=11)
    axs[1, 2].set_ylabel('p', fontsize=11)
    axs[1, 2].set_title('p coordinate evolution', fontsize=12)
    axs[1, 2].grid(alpha=0.3)
    
    # Add metadata as title
    meta = data_dict['metadata']
    fig.suptitle(f"Extracted Simulation Data | ω₀={data_dict['w0']:.6f} | "
                 f"Files: {meta['n_files']} | Ellipse index: {meta['ellipse_index']}",
                 fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    from athena_collapse_analysis.config import RAW_DIR
    
    # Simulation path
    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    
    # Extract data (will use cache if available)
    data = extract_and_cache_simulation_data(
        path_simu,
        nz_slice=0,
        n_ellipses=40,
        ellipse_index=1,
        use_cache=True,
        force_recompute=False,
    )
    
    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    fig = plot_extracted_data(data)