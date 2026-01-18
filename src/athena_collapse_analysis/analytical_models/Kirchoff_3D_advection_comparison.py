#!/usr/bin/env python3
"""
Compare 3D vortex tube theory with simulation results
=====================================================

This script:
1. Loads simulation data (from cache or HDF5 files)
2. Extracts α(t), S(t), and w0 from simulation
3. Runs 3D vortex tube integration
4. Extracts (p, q) from 3D deformation
5. Compares with simulation (p, q)

Usage:
    python compare_3d_simulation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import simulation data extraction (with caching)
from athena_collapse_analysis.analysis.extract_2D_hamiltonian_model_parameters import (
    extract_simulation_data,
)


# Import 3D vortex tube model
from athena_collapse_analysis.analytical_models.Kirchoff_3D_advection import (
    integrate_system,
    extract_pq_from_3d
)

# Import 2D Hamiltonian utilities for plotting
from athena_collapse_analysis.analytical_models.Kirchoff_2D_Hamiltonian import (
    compute_stretched_time,
    compute_eps_from_strain,
    q0_from_eps,
)



# ============================================================
# Extract (p, q) from 3D vortex tube deformation
# ============================================================


from athena_collapse_analysis.io.ath_io import get_hdf_files


# ============================================================
# Comparison plotting
# ============================================================
def plot_3d_vs_simulation(t_3d, p_3d, q_3d, sim_data, eps_func, tmax_plot=None):
    """
    Create comprehensive comparison plot between 3D theory and simulation.
    
    Parameters
    ----------
    t_3d : ndarray
        3D theory time array
    p_3d : ndarray
        3D theory p values
    q_3d : ndarray
        3D theory q values
    sim_data : dict
        Simulation data dictionary
    eps_func : callable
        Strain parameter function
    tmax_plot : float, optional
        Maximum time for plotting
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig = plt.figure(figsize=(15, 5))
    q_3d, p_3d = q_3d/2, -p_3d/2

    # --- Phase space ---
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(q_3d, p_3d, lw=2, label="3D Vortex Tube", alpha=0.8)
    ax1.plot(sim_data['q_sim'], sim_data['p_sim'], lw=2, 
             label="Simulation", alpha=0.8, ls='-')
    
    # Stable point
    eps_final = eps_func(t_3d[-1])
    ax1.scatter(q0_from_eps(eps_final), 0, color='green', s=100, 
                label='Stable point', zorder=5)
    ax1.scatter(0, 0, color='orange', marker='+', s=100)
    
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.axvline(0, color='gray', lw=0.5)
    ax1.legend(loc='best', facecolor='white', framealpha=1)
    ax1.set_xlabel('q', fontsize=12)
    ax1.set_ylabel('p', fontsize=12)
    ax1.set_title('Phase-space trajectory', fontsize=13)
    ax1.grid(alpha=0.3)
    
    # --- q(t) ---
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(t_3d, q_3d, label="3D Vortex Tube", lw=2, alpha=0.8)
    ax2.plot(sim_data['time'], sim_data['q_sim'], label="Simulation", 
             lw=2, alpha=0.8, ls='-')
    ax2.plot(sim_data['time'], q0_from_eps(eps_func(sim_data['time'])), ':', 
             label='Stable point', lw=1.5, color='green')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('q', fontsize=12)
    ax2.set_title('q(t)', fontsize=13)
    ax2.legend(loc='best', facecolor='white', framealpha=1)
    ax2.grid(alpha=0.3)
    if tmax_plot:
        ax2.set_xlim(0, tmax_plot)
    
    # --- p(t) ---
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(t_3d, p_3d, label="3D Vortex Tube", lw=2, alpha=0.8)
    ax3.plot(sim_data['time'], sim_data['p_sim'], label="Simulation", 
             lw=2, alpha=0.8, ls='-')
    ax3.axhline(0, label='Stable point', color='green', lw=1.5, ls=':')
    
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('p', fontsize=12)
    ax3.set_title('p(t)', fontsize=13)
    ax3.legend(loc='best', facecolor='white', framealpha=1)
    ax3.grid(alpha=0.3)
    if tmax_plot:
        ax3.set_xlim(0, tmax_plot)
    
    plt.tight_layout()
    
    return fig


# ============================================================
# Main comparison workflow
# ============================================================
def run_3d_simulation_comparison(path_simu, files=None, nz_slice=0, 
                                 n_ellipses=40, ellipse_index=1,
                                 nsteps=500, tmax_plot=None,
                                 use_cache=True, force_recompute=False):
    """
    Complete workflow: load simulation data and compare with 3D vortex tube.
    
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
    nsteps : int, optional
        Number of time steps for 3D integration
    tmax_plot : float, optional
        Maximum time for plotting
    use_cache : bool, optional
        Whether to use cached simulation data (default: True)
    force_recompute : bool, optional
        Force recomputation of simulation data (default: False)
    
    Returns
    -------
    results : dict
        Dictionary containing all results
    figs : tuple
        (comparison_fig, difference_fig, input_params_fig)
    """
    print("=" * 70)
    print("3D Vortex Tube vs Simulation Comparison")
    print("=" * 70)
    
    # ==========================================
    # 1. Load simulation data
    # ==========================================
    print("\n1. Loading simulation data...")
    
    if files is None:
        files = get_hdf_files(path_simu)
        print(f"  Found {len(files)} HDF5 files")
    
    sim_data = extract_simulation_data(
        path_simu, files, nz_slice=nz_slice,
        n_ellipses=n_ellipses, ellipse_index=ellipse_index,
        use_cache=use_cache, force_recompute=force_recompute
    )
    
    # ==========================================
    # 3. Run 3D vortex tube integration
    # ==========================================
    print("\n3. Integrating 3D vortex tube model...")
    print(f"  Configuration: ω₀ = (0, {sim_data['w0']}, 0)")
    print("  Cross-section: zx-plane perpendicular to vorticity")
    
    # Create vorticity vector (y-component only)
    omega0 = np.array([0.0, sim_data['w0'], 0.0])
    
    # Create interpolated functions for S(t) and α(t)
    S_func = interp1d(sim_data['time'], sim_data['S']**-1, 
                      kind='linear', fill_value='extrapolate')
    alpha_func = interp1d(sim_data['time'], sim_data['alpha']**-1, 
                          kind='linear', fill_value='extrapolate')
    
    # Time span from simulation
    t_span = (sim_data['time'][0], sim_data['time'][-1])
    
    # Integrate 3D system
    t_3d, omega_t, F_t = integrate_system(
        omega0/2, S_func, alpha_func,
        t_span=t_span, n_steps=nsteps
    )
    
    print(f"  3D integration complete: {len(t_3d)} time steps")
    
    # ==========================================
    # 4. Extract (p, q) from 3D solution
    # ==========================================
    print("\n4. Extracting (p, q) from 3D deformation...")
    p_3d, q_3d = extract_pq_from_3d(F_t, omega0)
    
    # ==========================================
    # 5. Create comparison plots
    # ==========================================
    print("\n5. Creating comparison plots...")
    
    # Compute strain parameter for plotting
    tau, t_from_tau = compute_stretched_time(
        sim_data['time'], sim_data['alpha'], sim_data['S']
    )
    eps_func = compute_eps_from_strain(
        sim_data['time'], sim_data['alpha'], sim_data['S'], sim_data['w0']
    )
    
    # Comparison plot
    fig_comparison = plot_3d_vs_simulation(
        t_3d, p_3d, q_3d, sim_data, eps_func, tmax_plot=tmax_plot
    )
    
    # Difference metrics
    #fig_difference = plot_difference_metrics(
    #    t_3d, p_3d, q_3d, sim_data
    #)
    
    # ==========================================
    # 6. Summary statistics
    # ==========================================
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    
    # Interpolate for comparison at same times
    p_sim_interp = interp1d(sim_data['time'], sim_data['p_sim'], kind='linear')
    q_sim_interp = interp1d(sim_data['time'], sim_data['q_sim'], kind='linear')
    
    # Common time points
    t_common = t_3d[(t_3d >= sim_data['time'][0]) & (t_3d <= sim_data['time'][-1])]
    mask = (t_3d >= sim_data['time'][0]) & (t_3d <= sim_data['time'][-1])
    p_3d_common = p_3d[mask]
    q_3d_common = q_3d[mask]
    p_sim_common = p_sim_interp(t_common)
    q_sim_common = q_sim_interp(t_common)
    
    # RMS differences
    rms_p = np.sqrt(np.mean((p_3d_common - p_sim_common)**2))
    rms_q = np.sqrt(np.mean((q_3d_common - q_sim_common)**2))
    rms_total = np.sqrt(np.mean((p_3d_common - p_sim_common)**2 + 
                                 (q_3d_common - q_sim_common)**2))
    
    print(f"\nRMS differences:")
    print(f"  RMS(Δp) = {rms_p:.6f}")
    print(f"  RMS(Δq) = {rms_q:.6f}")
    print(f"  RMS(total) = {rms_total:.6f}")
    
    print(f"\nFinal states:")
    print(f"  3D:         p = {p_3d[-1]:.6f}, q = {q_3d[-1]:.6f}")
    print(f"  Simulation: p = {sim_data['p_sim'][-1]:.6f}, q = {sim_data['q_sim'][-1]:.6f}")
    
    print("\n" + "=" * 70)
    
    plt.show()
    
    # Package results
    results = {
        't_3d': t_3d,
        'p_3d': p_3d,
        'q_3d': q_3d,
        't_sim': sim_data['time'],
        'p_sim': sim_data['p_sim'],
        'q_sim': sim_data['q_sim'],
        'omega_t': omega_t,
        'F_t': F_t,
        'rms_p': rms_p,
        'rms_q': rms_q,
        'rms_total': rms_total,
    }
    
    return results, (fig_comparison, fig_difference, fig_inputs)


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    from athena_collapse_analysis.config import RAW_DIR
    
    # Simulation path
    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")
    
    print("\n" + "=" * 70)
    print("EXAMPLE: 3D Vortex Tube Theory vs Simulation")
    print("=" * 70)
    print("\nPhysical setup:")
    print("  - Vortex tube aligned with y-axis: ω₀ = (0, ω₀y, 0)")
    print("  - Cross-section in zx-plane (2D dynamics)")
    print("  - 3D theory includes full deformation gradient F(t)")
    
    # Run comparison (will use cache if available)
    results, figs = run_3d_simulation_comparison(
        path_simu,
        nz_slice=0,
        n_ellipses=40,
        ellipse_index=1,  # Use second innermost ellipse
        nsteps=500,       # Number of 3D integration steps
        tmax_plot=None,   # Plot full time range
        use_cache=True,   # Use cached simulation data
        force_recompute=False,  # Set to True to recompute from HDF5
    )
    
    fig_comparison, fig_difference, fig_inputs = figs
    
    print("\nComparison complete!")
    print(f"The 3D theory {'agrees well with' if results['rms_total'] < 0.01 else 'shows differences from'} simulation")
    print(f"(RMS distance = {results['rms_total']:.6f})")
    
    # To force recomputation of simulation data:
    # results, figs = run_3d_simulation_comparison(
    #     path_simu,
    #     force_recompute=True,
    # )