#!/usr/bin/env python3
"""
Hamiltonian dynamics for elliptic vortex deformation
====================================================

This module implements the reduced Hamiltonian model describing the
dynamics of elliptic vortices under time-dependent strain.

Main features
-------------
- Hamiltonian equations of motion in (p, q) phase space.
- Numerical integration of the Hamiltonian system.
- Interactive input for simulation parameters.

References
----------
[1] Kayo Ide, Stephen Wiggins, 1995, The dynamics of elliptically shaped 
    regions of uniform vorticity in time-periodic, linear external velocity 
    fields, Fluid Dynamics Research 
    https://doi.org/10.1016/0169-5983(95)94956-T

[2] N. C. Hurst et al., 2021, Hamiltonian Form from Adiabatic behavior of 
    an elliptical vortex in a time-dependent external strain flow
    Phys. Rev. Fluids 
    https://doi.org/10.1103/PhysRevFluids.6.054703
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import interp1d


def hamiltonian_rhs(tau, z, w0, eps_func, t_from_tau):
    """
    Hamiltonian equations Right Side  in (p, q) variables.
    
    From Hurst et al., 2021 https://doi.org/10.1103/PhysRevFluids.6.054703

    Hamiltonian:
        H = 2 ln((p² + q² + 8)/2) − ε q sqrt(p² + q² + 8)

    Equations:
        dq/dτ = ∂H/∂p
        dp/dτ = −∂H/∂q

    Parameters
    ----------
    tau : float
        Stretched time τ
    z : array_like
        State vector [p, q]
    w0 : float
        Reference vorticity ω₀
    eps_func : callable
        ε(t) as a function of physical time
    t_from_tau : callable
        Mapping τ → t

    Returns
    -------
    dzdtau : list
        [dp/dτ, dq/dτ]
    """
    p, q = z
    R2 = p**2 + q**2 + 8
    sqrtR = np.sqrt(R2)

    t = t_from_tau(tau / w0)
    eps = eps_func(t)

    # dq/dτ = ∂H/∂p
    dqdt = 4 * p / R2 - eps * q * p / sqrtR

    # dp/dτ = −∂H/∂q
    dpdt = -(4 * q / R2 - eps * (sqrtR + q**2 / sqrtR))

    return [dpdt, dqdt]


# ============================================================
# Integrator
# ============================================================

def integrate_hamiltonian(tau_max, w0, eps_func, t_from_tau, 
                         z0=(0.0, 0.0), nsteps=2000):
    """
    Integrate Hamiltonian dynamics up to τ_max.

    Parameters
    ----------
    tau_max : float
        Maximum stretched time
    w0 : float
        Reference vorticity
    eps_func : callable
        Strain parameter as function of time
    t_from_tau : callable
        Inverse mapping from stretched to physical time
    z0 : tuple, optional
        Initial condition (p0, q0)
    nsteps : int, optional
        Number of evaluation points

    Returns
    -------
    sol : OdeResult
        SciPy integration result with attributes:
        - t: stretched time τ array
        - y: [p, q] solution array
    """
    tau_span = (0.0, tau_max * w0)
    tau_eval = np.linspace(*tau_span, nsteps)

    sol = solve_ivp(
        hamiltonian_rhs,
        tau_span,
        z0,
        t_eval=tau_eval,
        args=(w0, eps_func, t_from_tau),
        method="RK45",
    )

    return sol


# Parameter conversions

def q0_from_eps(eps):
    """
    Stable fixed point q₀(ε) using small ε expansion (ε < 0.1).
    
    Parameters
    ----------
    eps : float or array_like
        Strain parameter
    
    Returns
    -------
    q0 : float or array_like
        Equilibrium value of q coordinate
    """
    eps = np.asarray(eps)
    return 4 * np.sqrt(2) * (eps + 10 * eps**3 + 214 * eps**5)


def compute_stretched_time(time, alpha_vals, S_vals):
    """
    Compute stretched time τ from physical time.
    
    τ = ∫₀ᵗ α(t') S(t')⁻² dt'
    
    Parameters
    ----------
    time : array_like
        Physical time array
    alpha_vals : array_like
        Scale factor α(t) values
    S_vals : array_like
        S parameter S(t) values
    
    Returns
    -------
    tau : array_like
        Stretched time array
    t_from_tau : callable
        Interpolation function mapping τ → t
    """
    integrand = alpha_vals * S_vals**(-2)
    tau = cumtrapz(integrand, time, initial=0)
    t_from_tau = interp1d(tau, time, kind='linear', fill_value='extrapolate')
    
    return tau, t_from_tau


def compute_eps_from_strain(time, alpha_vals, S_vals, w0):
    """
    Compute strain parameter ε(t) from simulation inputs.
    
    General definition:
        ε(t) = -α̇/α / (w0 * α(t) * S(t)⁻²)
    
    where α̇ is the time derivative of α(t).
    
    Parameters
    ----------
    time : array_like
        Physical time array
    alpha_vals : array_like
        Scale factor α(t)
    S_vals : array_like
        S parameter S(t)
    w0 : float
        Reference vorticity
    
    Returns
    -------
    eps_func : callable
        Strain parameter as function of time
    """
    alpha_dot = np.gradient(alpha_vals, time)
    
    eps_vals = -3/4 * (alpha_dot / alpha_vals) / (w0 * alpha_vals * S_vals**(-2))
    eps_func = interp1d(time, eps_vals, kind='linear', fill_value='extrapolate')
    
    return eps_func



# Visualization


def plot_phase_space_simple(sol, w0, t_from_tau, eps_func, tmax_plot=None):
    """
    Simple 3-panel phase space plot matching the reference style.

    Parameters
    ----------
    sol : OdeResult
        Hamiltonian integration result
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
        Matplotlib figure object
    """
    # Convert stretched time to physical time
    t_ham = t_from_tau(sol.t / w0)
    
    fig = plt.figure(figsize=(15, 5))
    
    # --- Phase space trajectory ---
    plt.subplot(1, 3, 1)
    plt.plot(sol.y[1], sol.y[0], alpha=0.5)
    
    # Stable point at final time
    eps_final = eps_func(t_ham[-1])
    plt.scatter(q0_from_eps(eps_final), 0, color='green', label='Stable point')
    plt.scatter(0, 0, color='orange', marker='+')
    
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.legend(loc='upper left', facecolor='white', framealpha=1)
    plt.xlabel('q')
    plt.ylabel('p')
    plt.title('Phase-space trajectory')
    
    # --- q(t) evolution ---
    plt.subplot(1, 3, 2)
    plt.plot(t_ham, sol.y[1])
    plt.plot(t_ham, q0_from_eps(eps_func(t_ham)), label='stable point')
    
    plt.xlabel('Time')
    plt.ylabel('q')
    plt.title('q')
    plt.legend(loc='upper right', facecolor='white', framealpha=1)
    if tmax_plot:
        plt.xlim(0, tmax_plot)
    
    # --- p(t) evolution ---
    plt.subplot(1, 3, 3)
    plt.plot(t_ham, sol.y[0])
    plt.axhline(0, label='stable point', color='C1')
    
    plt.xlabel('Time')
    plt.ylabel('p')
    plt.title('p')
    plt.legend(loc='upper right', facecolor='white', framealpha=1)
    if tmax_plot:
        plt.xlim(0, tmax_plot)
    
    plt.tight_layout()
    
    return fig


def plot_hamiltonian_solution(sol, w0, t_from_tau, eps_func, 
                              alpha_vals, S_vals, time_sim):
    """
    Plot Hamiltonian solution and diagnostics.

    Parameters
    ----------
    sol : OdeResult
        Hamiltonian integration result
    w0 : float
        Reference vorticity
    t_from_tau : callable
        Stretched to physical time mapping
    eps_func : callable
        Strain parameter function
    alpha_vals : array_like
        Scale factor values for plotting
    S_vals : array_like
        S parameter values for plotting
    time_sim : array_like
        Simulation time points

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    # Convert stretched time to physical time
    t_ham = t_from_tau(sol.t / w0)
    
    fig = plt.figure(figsize=(16, 10))
    
    # --- Input parameters ---
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time_sim, alpha_vals, lw=2, color='C0')
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("α(t)", fontsize=11)
    ax1.set_title("Scale factor α(t)", fontsize=12)
    ax1.grid(alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time_sim, S_vals, lw=2, color='C1')
    ax2.set_xlabel("Time", fontsize=11)
    ax2.set_ylabel("S(t)", fontsize=11)
    ax2.set_title("S parameter S(t)", fontsize=12)
    ax2.grid(alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    eps_vals_plot = eps_func(time_sim)
    ax3.plot(time_sim, eps_vals_plot, lw=2, color='C2')
    ax3.set_xlabel("Time", fontsize=11)
    ax3.set_ylabel("ε(t)", fontsize=11)
    ax3.set_title("Strain parameter ε(t)", fontsize=12)
    ax3.grid(alpha=0.3)
    
    # --- Phase space trajectory ---
    ax4 = plt.subplot(3, 3, 4)
    
    # Color by time
    points = np.array([sol.y[1], sol.y[0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='viridis', linewidth=2)
    lc.set_array(t_ham)
    line = ax4.add_collection(lc)
    
    eps_final = eps_func(t_ham[-1])
    ax4.scatter(q0_from_eps(eps_final), 0, c="red", s=150, 
                marker='*', label="Stable point", zorder=5, edgecolor='black')
    ax4.scatter(sol.y[1][0], sol.y[0][0], c="green", s=100, 
                marker='o', label="Initial", zorder=5, edgecolor='black')
    
    ax4.axhline(0, color="gray", lw=0.5, ls='--', alpha=0.7)
    ax4.axvline(0, color="gray", lw=0.5, ls='--', alpha=0.7)
    ax4.set_xlabel("q", fontsize=11)
    ax4.set_ylabel("p", fontsize=11)
    ax4.set_title("Phase space trajectory", fontsize=12)
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    ax4.autoscale()
    
    cbar = plt.colorbar(line, ax=ax4)
    cbar.set_label('Time', fontsize=10)
    
    # --- q(t) evolution ---
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(t_ham, sol.y[1], lw=2, label="q(t)", color='C3')
    ax5.plot(time_sim, q0_from_eps(eps_func(time_sim)), ":", 
             label="q₀(ε)", lw=2, color='red')
    ax5.set_xlabel("Time", fontsize=11)
    ax5.set_ylabel("q", fontsize=11)
    ax5.set_title("q coordinate evolution", fontsize=12)
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # --- p(t) evolution ---
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(t_ham, sol.y[0], lw=2, color='C4', label="p(t)")
    ax6.axhline(0, color="gray", lw=0.5, ls='--', alpha=0.7)
    ax6.set_xlabel("Time", fontsize=11)
    ax6.set_ylabel("p", fontsize=11)
    ax6.set_title("p coordinate evolution", fontsize=12)
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # --- Hamiltonian value ---
    ax7 = plt.subplot(3, 3, 7)
    H_vals = np.zeros_like(sol.y[0])
    for i in range(len(sol.y[0])):
        p, q = sol.y[0][i], sol.y[1][i]
        R2 = p**2 + q**2 + 8
        eps = eps_func(t_ham[i])
        H_vals[i] = 2 * np.log(R2/2) - eps * q * np.sqrt(R2)
    
    ax7.plot(t_ham, H_vals, lw=2, color='C5')
    ax7.set_xlabel("Time", fontsize=11)
    ax7.set_ylabel("H(p, q)", fontsize=11)
    ax7.set_title("Hamiltonian value", fontsize=12)
    ax7.grid(alpha=0.3)
    
    # --- R² = p² + q² + 8 ---
    ax8 = plt.subplot(3, 3, 8)
    R2_vals = sol.y[0]**2 + sol.y[1]**2 + 8
    ax8.plot(t_ham, R2_vals, lw=2, color='C6')
    ax8.set_xlabel("Time", fontsize=11)
    ax8.set_ylabel("R²", fontsize=11)
    ax8.set_title("R² = p² + q² + 8", fontsize=12)
    ax8.grid(alpha=0.3)
    
    # --- Phase portrait detail ---
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(sol.y[1], sol.y[0], lw=2, color='C7', alpha=0.7)
    ax9.scatter(sol.y[1][0], sol.y[0][0], c="green", s=100, 
                marker='o', label="Start", zorder=5, edgecolor='black')
    ax9.scatter(sol.y[1][-1], sol.y[0][-1], c="blue", s=100, 
                marker='s', label="End", zorder=5, edgecolor='black')
    ax9.axhline(0, color="gray", lw=0.5, ls='--', alpha=0.7)
    ax9.axvline(0, color="gray", lw=0.5, ls='--', alpha=0.7)
    ax9.set_xlabel("q", fontsize=11)
    ax9.set_ylabel("p", fontsize=11)
    ax9.set_title("Phase portrait (start → end)", fontsize=12)
    ax9.legend(loc='best')
    ax9.grid(alpha=0.3)
    
    plt.suptitle("Hamiltonian Vortex Dynamics", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig





# ============================================================
# Main simulation runner
# ============================================================

def run_hamiltonian_simulation(time, alpha_vals, S_vals, w0,
                               z0=(0.0, 0.0), nsteps=2000, plot=False, save_figs_path=None):
    """
    Run complete Hamiltonian simulation from input parameters.
    
    Parameters
    ----------
    time : array_like
        Physical time array
    alpha_vals : array_like
        Scale factor α(t) values at each time point
    S_vals : array_like
        S parameter S(t) values at each time point
    w0 : float
        Reference vorticity ω₀
    z0 : tuple, optional
        Initial condition (p0, q0) (default: (0.0, 0.0))
    nsteps : int, optional
        Number of integration steps (default: 2000)
    
    Returns
    -------
    sol : OdeResult
        Integration result
    figs : tuple of Figure
        (detailed_diagnostics_figure, simple_phase_space_figure)
    diagnostics : dict
        Dictionary containing:
        - 'time_physical': physical time array
        - 'tau': stretched time array
        - 'eps': strain parameter values
        - 'H': Hamiltonian values
    """
    # Compute stretched time
    tau, t_from_tau = compute_stretched_time(time, alpha_vals, S_vals)
    
    # Compute strain parameter (no longer needs tc)
    eps_func = compute_eps_from_strain(time, alpha_vals, S_vals, w0)
    
    # Maximum stretched time
    tau_max = tau[-1]
    
    # Integrate Hamiltonian system
    print(f"Integrating Hamiltonian dynamics...")
    print(f"  w0 = {w0:.4f}")
    print(f"  τ_max = {tau_max:.4f}")
    print(f"  Initial condition: p0 = {z0[0]:.4f}, q0 = {z0[1]:.4f}")
    
    sol = integrate_hamiltonian(
        tau_max,
        w0,
        eps_func,
        t_from_tau,
        z0=z0,
        nsteps=nsteps,
    )
    
    print(f"  Integration complete: {len(sol.t)} steps")
    
    # Create diagnostic plots
    if plot == True:
        fig_detailed = plot_hamiltonian_solution(
            sol, w0, t_from_tau, eps_func, alpha_vals, S_vals, time
        )
        
        fig_simple = plot_phase_space_simple(
            sol, w0, t_from_tau, eps_func, tmax_plot=time[-1]
        )
    if save_figs_path is not None:
        fig_detailed.savefig(save_figs_path + "hamiltonian_detailed_diagnostics.png", dpi=300)
        fig_simple.savefig(save_figs_path + "hamiltonian_phase_space.png", dpi=300)
        print("  Figures saved to disk.")

    # Compute diagnostics
    t_phys = t_from_tau(sol.t / w0)
    H_vals = np.zeros_like(sol.y[0])
    for i in range(len(sol.y[0])):
        p, q = sol.y[0][i], sol.y[1][i]
        R2 = p**2 + q**2 + 8
        eps = eps_func(t_phys[i])
        H_vals[i] = 2 * np.log(R2/2) - eps * q * np.sqrt(R2)
    
    diagnostics = {
        'time_physical': t_phys,
        'tau': sol.t / w0,
        'eps': eps_func(t_phys),
        'H': H_vals,
        'p': sol.y[0],
        'q': sol.y[1],
    }

    return sol, diagnostics


# Examples of usage

def example_oscillating_flow():
    """
    Example with oscillating α and S parameters.
    """
    print("Example: Oscillating flow parameters")
    print("=" * 60)
    
    # Create time array
    t_end = 100.0
    time = np.linspace(0, t_end, 500)
    
    # Define oscillating parameters
    alpha_vals = 1.0 + 0.2 * np.sin(0.1 * time)
    S_vals = 1.0 + 0.3 * np.cos(0.15 * time)
    
    # Simulation parameters
    w0 = 0.1
    z0 = (0.1, 0.1)  # Start slightly away from origin
    
    # Run simulation
    sol, diagnostics = run_hamiltonian_simulation(
        time, alpha_vals, S_vals, w0, z0=z0, nsteps=2000, plot=True
    )
    
    plt.show()
    
    return sol, diagnostics


def example_expansion_flow():
    """
    Example with expanding flow (increasing S).
    """
    print("\nExample: Expanding flow")
    print("=" * 60)
    
    # Create time array
    t_end = 50.0
    time = np.linspace(0, t_end, 500)
    
    # Expanding flow
    alpha_vals = np.ones_like(time)
    S_vals = 1.0 + 0.5 * time / t_end  # Linearly increasing
    
    # Simulation parameters
    w0 = 0.01
    z0 = (0.0, 0.0)
    
    # Run simulation
    sol, diagnostics = run_hamiltonian_simulation(
        time, alpha_vals, S_vals, w0, z0=z0, nsteps=1500, plot=True
    )
    
    plt.show()
    
    return sol, diagnostics


def example_exponential_decay_constant_product():
    """
    Example with S⁻²(t)α(t) = constant and α(t) = exp(-t/tc).
    
    This corresponds to a vortex in an exponentially decaying flow
    where the product S⁻²α remains constant.
    
    Parameters:
    - tc = 2000 (decay time scale)
    - ω₀ = 2π/100 (vortex frequency)
    - S⁻²(t)α(t) = constant = 1
    """
    print("\nExample: Exponential decay with constant S⁻²α")
    print("=" * 60)
    
    # Time parameters
    tc = 2000.0
    t_end = 3000.0  # Run for 1.5 decay times
    time = np.linspace(0, t_end, 1000)
    
    # Define α(t) = exp(-t/tc)
    alpha_vals = np.exp(-time / tc)
    
    # From constraint S⁻²(t)α(t) = C (constant)
    # We can choose C = 1 for simplicity
    # Then S⁻²(t) = 1/α(t) = exp(t/tc)
    # So S(t) = exp(-t/(2*tc))
    C = 1.0
    S_vals = np.exp(-time / (2 * tc))
    
    # Verify the constraint
    product = S_vals**(-2) * alpha_vals
    print(f"  S⁻²(t)α(t) = {product[0]:.6f} (should be constant ≈ {C:.1f})")
    print(f"  S⁻²(t)α(t) variation: {np.std(product):.2e}")
    
    # Vortex parameters
    w0 = 2 * np.pi / 100  # ω₀ = 2π/100
    z0 = (0.0, 0.0)  # Start at origin
    
    print(f"\n  tc = {tc:.1f}")
    print(f"  ω₀ = 2π/100 = {w0:.6f}")
    print(f"  α(0) = {alpha_vals[0]:.4f}, α(t_end) = {alpha_vals[-1]:.4f}")
    print(f"  S(0) = {S_vals[0]:.4f}, S(t_end) = {S_vals[-1]:.4f}")
    
    # Run simulation
    sol, diagnostics = run_hamiltonian_simulation(
        time, alpha_vals, S_vals, w0, z0=z0, nsteps=2000, plot=True
    )
    
    # Additional diagnostic: verify stretched time behavior
    print(f"\n  Physical time range: [0, {t_end:.1f}]")
    print(f"  Stretched time range: [0, {diagnostics['tau'][-1]:.4f}]")
    print(f"  τ_max / t_max = {diagnostics['tau'][-1] / t_end:.6f}")
    
    plt.show()
    
    return sol, diagnostics


if __name__ == "__main__":
    # Run examples
    print("Running example 1...")
    sol1, diag1 = example_oscillating_flow()
    
    print("\nRunning example 2...")
    sol2, diag2 = example_expansion_flow()
    
    print("\nRunning example 3...")
    sol3, diag3 = example_exponential_decay_constant_product()