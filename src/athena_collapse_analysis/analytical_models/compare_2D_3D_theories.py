#!/usr/bin/env python3
"""
Compare 2D Hamiltonian dynamics with 3D vortex tube theory
===========================================================

This script compares two analytical theories for vortex deformation:
1. 2D Hamiltonian model (Kirchhoff dynamics in zx-plane)
2. 3D vortex tube model (advection of uniform vorticity patch)

For the special case ω₀ = (0, ω₀y, 0), the vortex tube is aligned with y-axis,
and its cross-section in the zx-plane corresponds to the 2D problem.

Usage:
    python compare_2d_3d_theories.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import 3D vortex tube model
import sys
import os
sys.path.append(os.path.dirname(__file__))
from Kirchoff_3D_advection import integrate_system, orthonormal_basis_perp

# Import 2D Hamiltonian model
from Kirchoff_2D_Hamiltonian import (
    run_hamiltonian_simulation,
    compute_stretched_time,
    compute_eps_from_strain,
    q0_from_eps,
)


# ============================================================
# Extract (p, q) from 3D vortex tube deformation
# ============================================================

def compute_aspect_ratio_3d(F, omega0):
    """
    Compute aspect ratio of vortex tube cross-section.
    
    Parameters
    ----------
    F : ndarray (3, 3)
        Deformation gradient tensor
    omega0 : ndarray (3,)
        Initial vorticity vector
    
    Returns
    -------
    eta : float
        Aspect ratio (major/minor axis)
    """
    # Get basis for plane perpendicular to omega0
    e1, e2 = orthonormal_basis_perp(omega0)
    P = np.vstack([e1, e2])  # 2×3 projector
    
    # Right Cauchy-Green tensor
    C = F.T @ F
    
    # Restrict to cross-section
    C2 = P @ C @ P.T
    
    # Eigenvalues give squared semi-axes
    eigs = np.linalg.eigvalsh(C2)
    
    # Aspect ratio
    eta = np.sqrt(eigs.max() / eigs.min())
    
    return eta


def compute_orientation_angle_3d(F, omega0):
    """
    Compute orientation angle of vortex tube cross-section.
    
    For omega0 = (0, w0y, 0), this measures the angle in the zx-plane.
    
    Parameters
    ----------
    F : ndarray (3, 3)
        Deformation gradient tensor
    omega0 : ndarray (3,)
        Initial vorticity vector (should be (0, w0y, 0))
    
    Returns
    -------
    theta : float
        Orientation angle in radians (measured from z-axis towards x-axis)
    """
    # Get basis for plane perpendicular to omega0 (y-axis)
    # e1, e2 span the zx-plane
    e1, e2 = orthonormal_basis_perp(omega0)
    P = np.vstack([e1, e2])  # 2×3 projector
    
    # Right Cauchy-Green tensor
    C = F.T @ F
    
    # Restrict to cross-section
    C2 = P @ C @ P.T
    
    # Eigenvectors give principal axes
    eigvals, eigvecs = np.linalg.eigh(C2)
    
    # Principal direction (major axis) in (e1, e2) coordinates
    idx_max = np.argmax(eigvals)
    major_axis_2d = eigvecs[:, idx_max]
    
    # Back to 3D
    major_axis_3d = major_axis_2d[0] * e1 + major_axis_2d[1] * e2
    
    # For omega0 = (0, w0y, 0), the cross-section is in zx-plane
    # Angle is measured from z-axis (index 2) towards x-axis (index 0)
    theta = np.arctan2(major_axis_3d[0], major_axis_3d[2])
    
    return theta


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


def extract_pq_from_3d(F_t, omega0):
    """
    Extract (p, q) coordinates from 3D vortex tube evolution.
    
    For omega0 = (0, w0y, 0), extracts deformation in the zx-plane.
    
    Parameters
    ----------
    F_t : ndarray (n_times, 3, 3)
        Deformation gradient tensors at each time
    omega0 : ndarray (3,)
        Initial vorticity vector (should be (0, w0y, 0))
    
    Returns
    -------
    p : ndarray
        p coordinate array
    q : ndarray
        q coordinate array
    """
    n_times = len(F_t)
    p = np.zeros(n_times)
    q = np.zeros(n_times)
    
    for i in range(n_times):
        # Compute aspect ratio and orientation
        eta = compute_aspect_ratio_3d(F_t[i], omega0)
        theta = compute_orientation_angle_3d(F_t[i], omega0)
        
        # Convert to (p, q) using same transformation as simulation
        angle = 2 * -theta
        r_coord = r_pol(eta**2)
        
        p[i] = r_coord * np.cos(angle)
        q[i] = r_coord * np.sin(angle)
    
    return p, q


# ============================================================
# Comparison plotting
# ============================================================

def plot_2d_vs_3d_comparison(sol_2d, t_2d, p_3d, q_3d, t_3d, 
                             w0, eps_func, tmax_plot=None):
    """
    Create comparison plot between 2D Hamiltonian and 3D theory.
    
    Parameters
    ----------
    sol_2d : OdeResult
        2D Hamiltonian solution
    t_2d : ndarray
        2D Hamiltonian time array
    p_3d : ndarray
        3D theory p values
    q_3d : ndarray
        3D theory q values
    t_3d : ndarray
        3D theory time array
    w0 : float
        Reference vorticity
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
    
    # --- Phase space ---
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(sol_2d.y[1], sol_2d.y[0], lw=2, label="2D Hamiltonian", alpha=0.8)
    ax1.plot(q_3d, p_3d, lw=2, label="3D Vortex Tube", alpha=0.8, ls='--')
    
    # Stable point
    eps_final = eps_func(t_2d[-1])
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
    ax2.plot(t_2d, sol_2d.y[1], label="2D Hamiltonian", lw=2, alpha=0.8)
    ax2.plot(t_3d, q_3d, label="3D Vortex Tube", lw=2, alpha=0.8, ls='--')
    ax2.plot(t_3d, q0_from_eps(eps_func(t_3d)), ':', 
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
    ax3.plot(t_2d, sol_2d.y[0], label="2D Hamiltonian", lw=2, alpha=0.8)
    ax3.plot(t_3d, p_3d, label="3D Vortex Tube", lw=2, alpha=0.8, ls='--')
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


def plot_difference_metrics(t_2d, sol_2d, t_3d, p_3d, q_3d):
    """
    Plot difference metrics between 2D and 3D theories.
    
    Parameters
    ----------
    t_2d : ndarray
        2D time array
    sol_2d : OdeResult
        2D solution
    t_3d : ndarray
        3D time array
    p_3d : ndarray
        3D p values
    q_3d : ndarray
        3D q values
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    # Interpolate 3D solution to 2D time points
    p_3d_interp = interp1d(t_3d, p_3d, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    q_3d_interp = interp1d(t_3d, q_3d, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    
    p_3d_at_2d = p_3d_interp(t_2d)
    q_3d_at_2d = q_3d_interp(t_2d)
    
    # Compute differences
    dp = sol_2d.y[0] - p_3d_at_2d
    dq = sol_2d.y[1] - q_3d_at_2d
    
    # Euclidean distance
    distance = np.sqrt(dp**2 + dq**2)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # Δp
    axs[0].plot(t_2d, dp, lw=2, color='C0')
    axs[0].axhline(0, color='gray', lw=0.5, ls='--')
    axs[0].set_xlabel('Time', fontsize=12)
    axs[0].set_ylabel('Δp', fontsize=12)
    axs[0].set_title('p difference (2D - 3D)', fontsize=13)
    axs[0].grid(alpha=0.3)
    
    # Δq
    axs[1].plot(t_2d, dq, lw=2, color='C1')
    axs[1].axhline(0, color='gray', lw=0.5, ls='--')
    axs[1].set_xlabel('Time', fontsize=12)
    axs[1].set_ylabel('Δq', fontsize=12)
    axs[1].set_title('q difference (2D - 3D)', fontsize=13)
    axs[1].grid(alpha=0.3)
    
    # Euclidean distance
    axs[2].plot(t_2d, distance, lw=2, color='C2')
    axs[2].set_xlabel('Time', fontsize=12)
    axs[2].set_ylabel('|Δ(p,q)|', fontsize=12)
    axs[2].set_title('Euclidean distance', fontsize=13)
    axs[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig


# ============================================================
# Main comparison workflow
# ============================================================

def run_2d_3d_comparison(w0y, S_func, alpha_func, t_span=(0.0, 10.0), 
                        nsteps=500, z0_2d=(0.0, 0.0)):
    """
    Run complete comparison between 2D Hamiltonian and 3D vortex tube.
    
    Parameters
    ----------
    w0y : float
        Vorticity magnitude (y-component only, aligned with y-axis)
    S_func : callable
        Background flow scale factor S(t)
    alpha_func : callable
        Background flow anisotropy α(t)
    t_span : tuple, optional
        Time span (t_start, t_end)
    nsteps : int, optional
        Number of time steps
    z0_2d : tuple, optional
        Initial condition for 2D Hamiltonian
    
    Returns
    -------
    results : dict
        Dictionary containing all results
    figs : tuple
        (comparison_fig, difference_fig)
    """
    print("=" * 70)
    print("2D Hamiltonian vs 3D Vortex Tube Comparison")
    print("=" * 70)
    print(f"\nConfiguration: ω₀ = (0, {w0y}, 0)")
    print("2D problem: zx-plane perpendicular to vorticity")
    
    # ==========================================
    # 1. Run 3D vortex tube integration
    # ==========================================
    print("\n1. Integrating 3D vortex tube model...")
    omega0 = np.array([0.0, w0y, 0.0])
    
    t_3d, omega_t, F_t = integrate_system(
        omega0, S_func, alpha_func, 
        t_span=t_span, n_steps=nsteps
    )
    
    print(f"  3D integration complete: {len(t_3d)} time steps")
    
    # Extract (p, q) from 3D solution
    print("  Extracting (p, q) from zx-plane cross-section...")
    p_3d, q_3d = extract_pq_from_3d(F_t, omega0)
    
    # ==========================================
    # 2. Run 2D Hamiltonian integration
    # ==========================================
    print("\n2. Integrating 2D Hamiltonian model...")
    
    # Create time and parameter arrays for 2D model
    time = t_3d
    alpha_vals = np.array([alpha_func(t) for t in time])
    S_vals = np.array([S_func(t) for t in time])
    
    # Run Hamiltonian integration
    sol_2d, diagnostics = run_hamiltonian_simulation(
        time, alpha_vals, S_vals, w0y,
        z0=z0_2d,
        nsteps=nsteps,
    )
    
    print(f"  2D integration complete")
    
    # ==========================================
    # 3. Create comparison plots
    # ==========================================
    print("\n3. Creating comparison plots...")
    
    # Get strain parameter for plotting
    tau, t_from_tau = compute_stretched_time(time, alpha_vals, S_vals)
    eps_func = compute_eps_from_strain(time, alpha_vals, S_vals, w0y)
    
    # Convert 2D Hamiltonian time
    t_2d = t_from_tau(sol_2d.t / w0y)
    
    # Comparison plot
    fig_comparison = plot_2d_vs_3d_comparison(
        sol_2d, t_2d, p_3d, q_3d, t_3d, w0y, eps_func,
        tmax_plot=t_span[1]
    )
    
    # Difference metrics
    fig_difference = plot_difference_metrics(
        t_2d, sol_2d, t_3d, p_3d, q_3d
    )
    
    # ==========================================
    # 4. Summary statistics
    # ==========================================
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    
    # Interpolate for comparison at same times
    p_3d_interp = interp1d(t_3d, p_3d, kind='linear')
    q_3d_interp = interp1d(t_3d, q_3d, kind='linear')
    
    # Common time points
    t_common = t_2d[t_2d <= t_3d[-1]]
    p_2d_common = sol_2d.y[0][t_2d <= t_3d[-1]]
    q_2d_common = sol_2d.y[1][t_2d <= t_3d[-1]]
    p_3d_common = p_3d_interp(t_common)
    q_3d_common = q_3d_interp(t_common)
    
    # RMS differences
    rms_p = np.sqrt(np.mean((p_2d_common - p_3d_common)**2))
    rms_q = np.sqrt(np.mean((q_2d_common - q_3d_common)**2))
    rms_total = np.sqrt(np.mean((p_2d_common - p_3d_common)**2 + 
                                 (q_2d_common - q_3d_common)**2))
    
    print(f"\nRMS differences:")
    print(f"  RMS(Δp) = {rms_p:.6f}")
    print(f"  RMS(Δq) = {rms_q:.6f}")
    print(f"  RMS(total) = {rms_total:.6f}")
    
    print(f"\nFinal states:")
    print(f"  2D: p = {sol_2d.y[0][-1]:.6f}, q = {sol_2d.y[1][-1]:.6f}")
    print(f"  3D: p = {p_3d[-1]:.6f}, q = {q_3d[-1]:.6f}")
    
    print("\n" + "=" * 70)
    
    plt.show()
    
    # Package results
    results = {
        't_2d': t_2d,
        'p_2d': sol_2d.y[0],
        'q_2d': sol_2d.y[1],
        't_3d': t_3d,
        'p_3d': p_3d,
        'q_3d': q_3d,
        'omega_t': omega_t,
        'F_t': F_t,
        'rms_p': rms_p,
        'rms_q': rms_q,
        'rms_total': rms_total,
    }
    
    return results, (fig_comparison, fig_difference)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    
    # Parameters
    w0 = 2 * np.pi / 100  # Vorticity magnitude (z-component only)

    # Time parameters
    tc = 2000.0
    # Background flow functions
    def S(t):
        """Scale factor - exponential decay."""
        return  np.exp(-t / (2 * tc))
    
    def alpha(t):
        """Anisotropy - oscillating."""
        return np.exp(-t / tc)

    # Time span
    t_span = (0.0, 1000.0)
    
    # Run comparison
    results, figs = run_2d_3d_comparison(
        w0y=w0,
        S_func=S,
        alpha_func=alpha,
        t_span=t_span,
        nsteps=500,
        z0_2d=(0.0, 0.0),  # Start at origin
    )
    
    fig_comparison, fig_difference = figs
    
    print("\nComparison complete!")
    print(f"The theories {'agree well' if results['rms_total'] < 0.01 else 'show some differences'}")
    print(f"(RMS distance = {results['rms_total']:.6f})")