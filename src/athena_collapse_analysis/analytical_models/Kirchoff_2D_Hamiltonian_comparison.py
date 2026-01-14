#!/usr/bin/env python3
"""
Compare Hamiltonian dynamics with simulation results
====================================================

This script:
1. Loads simulation data and extracts ellipse parameters
2. Computes (p, q) coordinates from fitted ellipses
3. Extracts α(t) and S(t) from simulation
4. Runs Hamiltonian integration
5. Compares Hamiltonian prediction with simulation results

Usage:
    python compare_hamiltonian_simulation.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import ellipse fitting utilities
from athena_collapse_analysis.analysis.fit_ellipses import process_single_file
from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files_with_collapse,
)
from athena_collapse_analysis.utils import (
    collapse_param_decomposition,
    compute_physical_vorticity,
)

# Import Hamiltonian dynamics
from athena_collapse_analysis.analytical_models.Kirchoff_2D_Hamiltonian import (
    run_hamiltonian_simulation,
    plot_phase_space_simple,
    q0_from_eps,
    compute_stretched_time, 
    compute_eps_from_strain
)


# ============================================================
# Coordinate transformation utilities
# ============================================================

def r_pol(eta):
    """
    Convert aspect ratio squared to polar radius coordinate.
    
    r = sqrt(2*(η - 1)**2 /η)
    
    Parameters
    ----------
    eta : float or array_like
        Aspect ratio (a/b)
    
    Returns
    -------
    r : float or array_like
        Polar radius coordinate
    """
    return np.sqrt(2*(eta-1)**2/eta)


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
    r_coord = r_pol(eta**2) # correct factor 2
    
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
    # Create cache filename based on parameters
    cache_dir = os.path.join(path_simu, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = f"hamiltonian_data_n{n_ellipses}_idx{ellipse_index}.pkl"
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
    print(f"  Saved data to cache: {cache_file}")


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
    print(f"  Loaded data from cache: {cache_file}")
    return data_dict


# ============================================================
# Simulation data extraction (with caching)
# ============================================================

def extract_simulation_data(path_simu, files, nz_slice=0, 
                           n_ellipses=40, ellipse_index=1,
                           use_cache=True, force_recompute=False):
    """
    Extract all necessary data from simulation files (with caching).
    
    Parameters
    ----------
    path_simu : str
        Path to simulation directory
    files : list
        List of HDF5 files
    nz_slice : int, optional
        z-slice to analyze
    n_ellipses : int, optional
        Number of ellipse contours to fit
    ellipse_index : int, optional
        Which ellipse to track
    use_cache : bool, optional
        Whether to use cached data if available (default: True)
    force_recompute : bool, optional
        Force recomputation even if cache exists (default: False)
    
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
        - 'ellipse_params': full ellipse parameter array
    """
    # Check for cached data
    cache_file = get_cache_filename(path_simu, n_ellipses, ellipse_index)
    
    if use_cache and not force_recompute and os.path.exists(cache_file):
        print("Cache file found! Loading from cache...")
        try:
            data_dict = load_simulation_data(cache_file)
            
            # Verify cache is compatible with current files
            if len(data_dict['time']) == len(files):
                print("  Cache is valid and up-to-date")
                return data_dict
            else:
                print(f"  Cache has {len(data_dict['time'])} files, but found {len(files)} files")
                print("  Recomputing data...")
        except Exception as e:
            print(f"  Error loading cache: {e}")
            print("  Recomputing data...")
    
    # Compute data from scratch
    print("Computing simulation data from HDF5 files...")
    
    n_files = len(files)
    
    # Storage arrays
    time_arr = np.zeros(n_files)
    alpha_arr = np.zeros(n_files)
    S_arr = np.zeros(n_files)
    ellipse_params_arr = []
    
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
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_files} files")
    
    # Convert ellipse list to array
    # Find maximum number of ellipses across all times
    max_ellipses = max(len(e) for e in ellipse_params_arr)
    
    # Create padded array
    ellipse_array = np.zeros((n_files, max_ellipses, 4))
    for i, ellipses in enumerate(ellipse_params_arr):
        n_ell = len(ellipses)
        if n_ell > 0:
            ellipse_array[i, :n_ell, :] = np.array(ellipses)
    
    # Extract (p, q) from ellipses
    p_sim, q_sim = extract_pq_from_ellipses(ellipse_array, ellipse_index)
    
    print(f"Extraction complete!")
    print(f"  Time range: [{time_arr[0]:.2f}, {time_arr[-1]:.2f}]")
    print(f"  w0 = {w0:.6f}")
    print(f"  α range: [{alpha_arr.min():.4f}, {alpha_arr.max():.4f}]")
    print(f"  S range: [{S_arr.min():.4f}, {S_arr.max():.4f}]")
    
    # Create data dictionary
    data_dict = {
        'time': time_arr,
        'alpha': alpha_arr,
        'S': S_arr,
        'w0': w0,
        'p_sim': p_sim,
        'q_sim': q_sim,
        'ellipse_params': ellipse_array,
    }
    
    # Save to cache
    if use_cache:
        save_simulation_data(data_dict, cache_file)
    
    return data_dict


# ============================================================
# Comparison plotting
# ============================================================

def plot_hamiltonian_vs_simulation(sol, sim_data, w0, t_from_tau, eps_func, 
                                   tmax_plot=None):
    """
    Create comprehensive comparison plot.
    
    Parameters
    ----------
    sol : OdeResult
        Hamiltonian integration result
    sim_data : dict
        Simulation data dictionary
    w0 : float
        Reference vorticity
    t_from_tau : callable
        Stretched to physical time mapping
    eps_func : callable
        Strain parameter function
    tmax_plot : float, optional
        Maximum time for plotting
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    # Convert Hamiltonian time
    t_ham = t_from_tau(sol.t / w0)
    
    fig = plt.figure(figsize=(15, 5))
    
    # --- Phase space ---
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(sol.y[1], sol.y[0], alpha=0.6, lw=2, label="Hamiltonian")
    ax1.plot(sim_data['q_sim'], sim_data['p_sim'], lw=2, label="Simulation")
    
    # Stable point
    eps_final = eps_func(t_ham[-1])

    ax1.scatter(q0_from_eps(eps_final), 0, color='green', s=100, 
                label='Stable point', zorder=5)
    ax1.scatter(0, 0, color='orange', marker='+', s=100)
    
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.axvline(0, color='gray', lw=0.5)
    ax1.legend(loc='upper left', facecolor='white', framealpha=1)
    ax1.set_xlabel('q')
    ax1.set_ylabel('p')
    ax1.set_title('Phase-space trajectory')
    ax1.grid(alpha=0.3)
    
    # --- q(t) ---
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(t_ham, sol.y[1], label="Hamiltonian", lw=2)
    ax2.plot(sim_data['time'], sim_data['q_sim'], label="Simulation", lw=2)
    ax2.plot(sim_data['time'], q0_from_eps(eps_func(sim_data['time'])), 
             '--', label='Stable point', lw=1.5, color='green')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('q')
    ax2.set_title('q(t)')
    ax2.legend(loc='upper right', facecolor='white', framealpha=1)
    ax2.grid(alpha=0.3)
    if tmax_plot:
        ax2.set_xlim(0, tmax_plot)
    
    # --- p(t) ---
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(t_ham, sol.y[0], label="Hamiltonian", lw=2)
    ax3.plot(sim_data['time'], sim_data['p_sim'], label="Simulation", lw=2)
    ax3.axhline(0, label='Stable point', color='C2', lw=1.5, ls='--')
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('p')
    ax3.set_title('p(t)')
    ax3.legend(loc='upper right', facecolor='white', framealpha=1)
    ax3.grid(alpha=0.3)
    if tmax_plot:
        ax3.set_xlim(0, tmax_plot)
    
    plt.tight_layout()
    
    return fig


def plot_input_parameters(sim_data):
    """
    Plot α(t) and S(t) from simulation.
    
    Parameters
    ----------
    sim_data : dict
        Simulation data dictionary
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # α(t)
    axs[0].plot(sim_data['time'], sim_data['alpha'], lw=2, color='C0')
    axs[0].set_xlabel('Time', fontsize=12)
    axs[0].set_ylabel('α(t)', fontsize=12)
    axs[0].set_title('Scale factor α(t)', fontsize=13)
    axs[0].grid(alpha=0.3)
    
    # S(t)
    axs[1].plot(sim_data['time'], sim_data['S'], lw=2, color='C1')
    axs[1].set_xlabel('Time', fontsize=12)
    axs[1].set_ylabel('S(t)', fontsize=12)
    axs[1].set_title('S parameter S(t)', fontsize=13)
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig


# ============================================================
# Main comparison workflow
# ============================================================

def run_comparison(path_simu, files=None, nz_slice=0, n_ellipses=40, 
                  ellipse_index=1, z0=(0.0, 0.0), nsteps=2000,
                  tmax_plot=None, use_cache=True, force_recompute=False):
    """
    Complete workflow: extract simulation data and compare with Hamiltonian.
    
    Parameters
    ----------
    path_simu : str
        Path to simulation directory
    files : list, optional
        List of files (if None, loads all HDF5 files)
    nz_slice : int, optional
        z-slice to analyze
    n_ellipses : int, optional
        Number of ellipse contours
    ellipse_index : int, optional
        Which ellipse to track
    z0 : tuple, optional
        Initial condition for Hamiltonian
    nsteps : int, optional
        Number of integration steps
    tmax_plot : float, optional
        Maximum time for plotting
    use_cache : bool, optional
        Whether to use cached data if available (default: True)
    force_recompute : bool, optional
        Force recomputation even if cache exists (default: False)
    
    Returns
    -------
    sol : OdeResult
        Hamiltonian solution
    sim_data : dict
        Simulation data
    figs : tuple
        (comparison_fig, input_params_fig, detailed_fig, simple_fig)
    """
    print("=" * 70)
    print("Hamiltonian vs Simulation Comparison")
    print("=" * 70)
    
    # Load files if not provided
    if files is None:
        files = get_hdf_files(path_simu)
        print(f"Found {len(files)} HDF5 files")
    
    # Extract simulation data (with caching)
    print("\n1. Extracting simulation data...")
    sim_data = extract_simulation_data(
        path_simu, files, nz_slice=nz_slice, 
        n_ellipses=n_ellipses, ellipse_index=ellipse_index,
        use_cache=use_cache, force_recompute=force_recompute
    )
    
    # Plot input parameters
    print("\n2. Plotting input parameters...")
    fig_inputs = plot_input_parameters(sim_data)
    
    # Run Hamiltonian simulation
    print("\n3. Running Hamiltonian integration...")
    sol, diagnostics = run_hamiltonian_simulation(
        sim_data['time'],
        sim_data['alpha'],
        sim_data['S'],
        sim_data['w0'],
        z0=z0,
        nsteps=nsteps,
    )
    
    # Create comparison plot
    print("\n4. Creating comparison plots...")
    
    # Get necessary functions from diagnostics
    tau, t_from_tau = compute_stretched_time(
        sim_data['time'], sim_data['alpha'], sim_data['S']
    )
    eps_func = compute_eps_from_strain(
        sim_data['time'], sim_data['alpha'], sim_data['S'], sim_data['w0']
    )
    
    fig_comparison = plot_hamiltonian_vs_simulation(
        sol, sim_data, sim_data['w0'], t_from_tau, eps_func,
        tmax_plot=tmax_plot
    )
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    
    plt.show()
    
    return sol, sim_data, (fig_comparison, fig_inputs, fig_detailed, fig_simple)


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    from athena_collapse_analysis.config import RAW_DIR
    
    # Simulation path
    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    
    # Run comparison (will use cache if available)
    sol, sim_data, figs = run_comparison(
        path_simu,
        nz_slice=0,
        n_ellipses=40,
        ellipse_index=1,  # Use second innermost ellipse
        z0=(0.0, 0.0),    # Start from origin
        nsteps=2000,
        tmax_plot=None,   # Plot full time range
        use_cache=True,   # Use cached data if available
        force_recompute=False,  # Set to True to force recomputation
    )
    
    # Access results
    fig_comparison, fig_inputs, fig_detailed, fig_simple = figs
    
    print(f"\nFinal Hamiltonian state:")
    print(f"  p = {sol.y[0][-1]:.6f}")
    print(f"  q = {sol.y[1][-1]:.6f}")
    print(f"\nFinal simulation state:")
    print(f"  p = {sim_data['p_sim'][-1]:.6f}")
    print(f"  q = {sim_data['q_sim'][-1]:.6f}")
    
    # To force recomputation (e.g., after changing parameters):
    # sol, sim_data, figs = run_comparison(
    #     path_simu,
    #     force_recompute=True,
    # )