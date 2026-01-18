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

# Import ellipse fitting utilities
from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
)

from athena_collapse_analysis.analysis.extract_2D_hamiltonian_model_parameters import (
    extract_simulation_data,
)

# Import Hamiltonian dynamics
from athena_collapse_analysis.analytical_models.Kirchoff_2D_Hamiltonian import (
    run_hamiltonian_simulation,
    q0_from_eps,
    compute_stretched_time, 
    compute_eps_from_strain
)



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
    
    return sol, sim_data, fig_comparison










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
    
    # To force recomputation (e.g., after changing parameters):
    # sol, sim_data, figs = run_comparison(
    #     path_simu,
    #     force_recompute=True,
    # )