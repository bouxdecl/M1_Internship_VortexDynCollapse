"""
Uniform-vorticity patch in time-dependent linear background flow.

Simulates the evolution of a vortex patch under a prescribed linear background
flow, tracking both vorticity orientation and material deformation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from athena_collapse_analysis.analysis.extract_2D_hamiltonian_model_parameters import (
    r_pol,
)

# =============================================================================
# Background Flow
# =============================================================================

def background_flow_matrix(t, S, alpha):
    """
    Compute linear background flow velocity gradient matrix.
    
    Parameters
    ----------
    t : float
        Time.
    S : callable
        Isotropic scale factor function S(t).
    alpha : callable
        Anisotropy factor function alpha(t).
    
    Returns
    -------
    ndarray, shape (3, 3)
        Velocity gradient tensor at time t.
    """
    eps = 1e-8
    Sdot = (S(t + eps) - S(t - eps)) / (2 * eps)
    alpha_dot = (alpha(t + eps) - alpha(t - eps)) / (2 * eps)
    
    return Sdot / S(t) * np.eye(3) + alpha_dot / alpha(t) * np.diag([1, 1, -2])


# =============================================================================
# ODE System
# =============================================================================

def antisymmetric_matrix(omega):
    """
    Construct antisymmetric matrix W such that W @ r = (1/2) omega × r.
    
    Parameters
    ----------
    omega : ndarray, shape (3,)
        Vorticity vector.
    
    Returns
    -------
    ndarray, shape (3, 3)
        Antisymmetric matrix.
    """
    wx, wy, wz = omega
    return np.array([[0.0, -wz, wy],
                     [wz, 0.0, -wx],
                     [-wy, wx, 0.0]])


def integrate_system(omega0, S, alpha, t_span=(0.0, 5.0), n_steps=500):
    """
    Integrate vorticity and deformation gradient evolution.
    
    Parameters
    ----------
    omega0 : ndarray, shape (3,)
        Initial vorticity vector.
    S : callable
        Scale factor function S(t).
    alpha : callable
        Anisotropy function alpha(t).
    t_span : tuple of float, optional
        Time interval (t_start, t_end).
    n_steps : int, optional
        Number of output time points.
    
    Returns
    -------
    t : ndarray, shape (n_steps,)
        Time points.
    omega_t : ndarray, shape (n_steps, 3)
        Vorticity evolution.
    F_t : ndarray, shape (n_steps, 3, 3)
        Deformation gradient evolution.
    """
    F0 = np.eye(3)
    y0 = np.concatenate([omega0, F0.ravel()])
    t_eval = np.linspace(*t_span, n_steps)
    
    def rhs(t, y, S, alpha):
        """
        Right-hand side of coupled ODE system for (omega, F).
        
        Parameters
        ----------
        t : float
            Time.
        y : ndarray, shape (12,)
            State vector [omega (3,), F.ravel() (9,)].
        S : callable
            Scale factor function.
        alpha : callable
            Anisotropy function.
        
        Returns
        -------
        ndarray, shape (12,)
            Time derivative [domega/dt, dF/dt.ravel()].
        """
        omega = y[:3]
        F = y[3:].reshape(3, 3)
        
        A = background_flow_matrix(t, S, alpha)
        
        domega = A @ omega
        dF = (A + 0.5 * antisymmetric_matrix(omega)) @ F
        
        return np.concatenate([domega, dF.ravel()])
    
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                    rtol=1e-9, atol=1e-12, args=(S, alpha))
    
    omega_t = sol.y[:3].T
    F_t = sol.y[3:].T.reshape(-1, 3, 3)
    
    return t_eval, omega_t, F_t


# =============================================================================
# Geometry Utilities
# =============================================================================

def orthonormal_basis_perp(n):
    """
    Construct orthonormal basis (e1, e2) spanning plane perpendicular to n.
    
    Parameters
    ----------
    n : ndarray, shape (3,)
        Normal vector.
    
    Returns
    -------
    e1, e2 : ndarray, shape (3,)
        Orthonormal basis vectors.
    """
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    
    return e1, e2


def tube_axis(F, omega0):
    """
    Compute unit vector along advected vortex tube axis.
    
    Parameters
    ----------
    F : ndarray, shape (3, 3)
        Deformation gradient.
    omega0 : ndarray, shape (3,)
        Initial vorticity direction.
    
    Returns
    -------
    ndarray, shape (3,)
        Unit vector along tube axis.
    """
    v = F @ omega0
    return v / np.linalg.norm(v)


# =============================================================================
# Diagnostic Quantities
# =============================================================================

def compute_aspect_ratio_3D(F, omega0):
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
    e1, e2 = orthonormal_basis_perp(omega0)
    P = np.vstack([e1, e2])
    
    C = F.T @ F
    C2 = P @ C @ P.T
    
    eigs = np.linalg.eigvalsh(C2)
    eta = np.sqrt(eigs.max() / eigs.min())
    
    return eta


def compute_orientation_angle_3d(F, omega0):
    """
    Compute orientation angle of vortex tube cross-section.
    
    Parameters
    ----------
    F : ndarray (3, 3)
        Deformation gradient tensor
    omega0 : ndarray (3,)
        Initial vorticity vector
    
    Returns
    -------
    angle : float
        Orientation angle (in radians) of the major axis of the cross-section.
    """
    e1, e2 = orthonormal_basis_perp(omega0)
    P = np.vstack([e1, e2])
    
    C = F.T @ F
    C2 = P @ C @ P.T
    
    eigs, evecs = np.linalg.eigh(C2)
    major_axis = evecs[:, np.argmax(eigs)]
    angle = np.arctan2(major_axis[1], major_axis[0])
    
    return angle


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
        eta = compute_aspect_ratio_3D(F_t[i], omega0)
        theta = compute_orientation_angle_3d(F_t[i], omega0)
        
        angle = 2 * theta
        r_coord = r_pol(eta)
        
        p[i] = r_coord * np.cos(angle)
        q[i] = r_coord * np.sin(angle)
    
    return p, q


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_omega_components(t, omega_t):
    """Plot vorticity components versus time."""
    plt.figure()
    plt.plot(t, omega_t[:, 0], label=r"$\omega_x$")
    plt.plot(t, omega_t[:, 1], label=r"$\omega_y$")
    plt.plot(t, omega_t[:, 2], label=r"$\omega_z$")
    plt.xlabel("t")
    plt.ylabel(r"$\omega$")
    plt.title("Vorticity components vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_phase_space_simple(t, p, q, tmax_plot=None):
    """
    Simple 3-panel phase space plot.
    
    Parameters
    ----------
    t : ndarray
        Time array.
    p : ndarray
        p coordinate array.
    q : ndarray
        q coordinate array.
    tmax_plot : float, optional
        Maximum time for plotting.
    
    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    """
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(q, p, alpha=0.5)
    plt.scatter(0, 0, color='orange', marker='+')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.xlabel('q')
    plt.ylabel('p')
    plt.title('Phase-space trajectory')
    
    plt.subplot(1, 3, 2)
    plt.plot(t, q)
    plt.xlabel('Time')
    plt.ylabel('q')
    plt.title('q')
    if tmax_plot:
        plt.xlim(0, tmax_plot)
    
    plt.subplot(1, 3, 3)
    plt.plot(t, p)
    plt.axhline(0, color='gray', lw=0.5)
    plt.xlabel('Time')
    plt.ylabel('p')
    plt.title('p')
    if tmax_plot:
        plt.xlim(0, tmax_plot)
    
    plt.tight_layout()
    return fig


def plot_pq_if_single_component(t, F_t, omega0, tmax_plot=None):
    """
    Plot (p, q) phase space if omega0 has only one non-zero component.
    
    Parameters
    ----------
    t : ndarray
        Time array.
    F_t : ndarray, shape (n, 3, 3)
        Deformation gradient evolution.
    omega0 : ndarray, shape (3,)
        Initial vorticity vector.
    tmax_plot : float, optional
        Maximum time for plotting.
    """
    non_zero = np.abs(omega0) > 1e-10
    if np.sum(non_zero) == 1:
        p, q = extract_pq_from_3d(F_t, omega0)
        fig = plot_phase_space_simple(t, p, q, tmax_plot=tmax_plot)
        return fig
    else:
        print("Skipping (p, q) plot: omega0 has more than one non-zero component")
        return None


# =============================================================================
# 3D Visualization
# =============================================================================

def plot_vortex_tube_snapshot(F, omega, omega0, n_circles=30, tube_length=1.5,
                              n_theta=120, stretch_factor=4.0, show_vorticity=True):
    """
    Plot vortex tube at single time with visual stretching.
    
    Parameters
    ----------
    F : ndarray, shape (3, 3)
        Deformation gradient.
    omega : ndarray, shape (3,)
        Vorticity vector.
    omega0 : ndarray, shape (3,)
        Initial vorticity.
    n_circles : int, optional
        Number of cross-sections.
    tube_length : float, optional
        Tube length.
    n_theta : int, optional
        Angular resolution.
    stretch_factor : float, optional
        Visual stretching factor along axis.
    show_vorticity : bool, optional
        Whether to show vorticity arrow.
    """
    omega_hat0 = omega0 / np.linalg.norm(omega0)
    e1, e2 = orthonormal_basis_perp(omega0)
    
    theta = np.linspace(0, 2 * np.pi, n_theta)
    s_vals = np.linspace(-tube_length / 2, tube_length / 2, n_circles)
    
    X0 = np.zeros((3, n_circles, n_theta))
    for i, s in enumerate(s_vals):
        X0[:, i, :] = (np.outer(e1, np.cos(theta)) +
                       np.outer(e2, np.sin(theta)) +
                       s * omega_hat0[:, None])
    
    X = F @ X0.reshape(3, -1)
    X = X.reshape(3, n_circles, n_theta)
    
    ell = tube_axis(F, omega0)
    for i in range(n_circles):
        for j in range(n_theta):
            x = X[:, i, j]
            x_par = np.dot(x, ell) * ell
            x_perp = x - x_par
            X[:, i, j] = x_perp + stretch_factor * x_par
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_surface(X[0], X[1], X[2], color="steelblue",
                   alpha=0.6, linewidth=0, shade=True)
    
    if show_vorticity:
        scale = 0.4 * np.max(np.abs(X))
        ax.quiver(0, 0, 0, omega[0], omega[1], omega[2],
                 length=scale, normalize=True, color="k")
    
    Xflat = X.reshape(3, -1)
    R = np.ptp(Xflat, axis=1).max()
    mid = Xflat.mean(axis=1)
    
    ax.set_xlim(mid[0] - R / 2, mid[0] + R / 2)
    ax.set_ylim(mid[1] - R / 2, mid[1] + R / 2)
    ax.set_zlim(mid[2] - R / 2, mid[2] + R / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Vortex tube (visual stretch ×{stretch_factor})")
    plt.tight_layout()
    plt.show()


def animate_vortex(t, omega_t, F_t, omega0, filename="vortex_evolution.mp4",
                   n_frames=240, elev=25, azim0=30, azim_rate=0.4):
    """
    Animate evolution of vortex cross-section and vorticity vector.
    
    Parameters
    ----------
    t : ndarray
        Time points.
    omega_t : ndarray, shape (n, 3)
        Vorticity evolution.
    F_t : ndarray, shape (n, 3, 3)
        Deformation gradient evolution.
    omega0 : ndarray, shape (3,)
        Initial vorticity.
    filename : str, optional
        Output filename.
    n_frames : int, optional
        Number of animation frames.
    elev, azim0, azim_rate : float, optional
        Camera parameters.
    
    Returns
    -------
    FuncAnimation
        Animation object.
    """
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    
    theta = np.linspace(0, 2 * np.pi, 300)
    e1, e2 = orthonormal_basis_perp(omega0)
    circle = np.outer(e1, np.cos(theta)) + np.outer(e2, np.sin(theta))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    line, = ax.plot([], [], [], lw=2, label="Advected cross-section")
    quiv = None
    
    ellipse_final = F_t[idx[-1]] @ circle
    R = np.ptp(ellipse_final, axis=1).max()
    mid = ellipse_final.mean(axis=1)
    
    ax.set_xlim(mid[0] - R / 2, mid[0] + R / 2)
    ax.set_ylim(mid[1] - R / 2, mid[1] + R / 2)
    ax.set_zlim(mid[2] - R / 2, mid[2] + R / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    def update(k):
        nonlocal quiv
        i = idx[k]
        
        ellipse = F_t[i] @ circle
        omega = omega_t[i]
        
        line.set_data(ellipse[0], ellipse[1])
        line.set_3d_properties(ellipse[2])
        
        if quiv is not None:
            quiv.remove()
        
        scale = 0.5 * np.max(np.abs(ellipse))
        quiv = ax.quiver(0, 0, 0, omega[0], omega[1], omega[2],
                        length=scale, normalize=True)
        
        ax.view_init(elev=elev, azim=azim0 + azim_rate * k)
        ax.set_title(f"t = {t[i]:.2f}")
        return line, quiv
    
    ani = animation.FuncAnimation(fig, update, frames=len(idx),
                                 interval=60, blit=False)
    ani.save(filename, writer="ffmpeg", dpi=200)
    return ani


def animate_vortex_tube_surface(t, omega_t, F_t, omega0, n_circles=25,
                                tube_length=1.5, n_theta=120,
                                filename="vortex_tube_surface.mp4",
                                n_frames=240, elev=25, azim0=30, azim_rate=0.4):
    """
    Animate vortex tube as a continuous surface.
    
    Parameters
    ----------
    t : ndarray
        Time points.
    omega_t : ndarray, shape (n, 3)
        Vorticity evolution.
    F_t : ndarray, shape (n, 3, 3)
        Deformation gradient evolution.
    omega0 : ndarray, shape (3,)
        Initial vorticity.
    n_circles : int, optional
        Number of cross-sections.
    tube_length : float, optional
        Tube length.
    n_theta : int, optional
        Angular resolution.
    filename : str, optional
        Output filename.
    n_frames : int, optional
        Number of animation frames.
    elev, azim0, azim_rate : float, optional
        Camera parameters.
    
    Returns
    -------
    FuncAnimation
        Animation object.
    """
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    
    omega_hat = omega0 / np.linalg.norm(omega0)
    e1, e2 = orthonormal_basis_perp(omega0)
    
    theta = np.linspace(0, 2 * np.pi, n_theta)
    s_vals = np.linspace(-tube_length / 2, tube_length / 2, n_circles)
    
    X0 = np.zeros((3, n_circles, n_theta))
    for i, s in enumerate(s_vals):
        X0[:, i, :] = (np.outer(e1, np.cos(theta)) +
                       np.outer(e2, np.sin(theta)) +
                       s * omega_hat[:, None])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    surf = None
    quiv = None
    
    Xf = F_t[idx[-1]] @ X0.reshape(3, -1)
    R = np.ptp(Xf, axis=1).max()
    mid = Xf.mean(axis=1)
    
    ax.set_xlim(mid[0] - R / 2, mid[0] + R / 2)
    ax.set_ylim(mid[1] - R / 2, mid[1] + R / 2)
    ax.set_zlim(mid[2] - R / 2, mid[2] + R / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    def update(k):
        nonlocal surf, quiv
        i = idx[k]
        
        F = F_t[i]
        omega = omega_t[i]
        
        X = F @ X0.reshape(3, -1)
        X = X.reshape(3, n_circles, n_theta)
        
        if surf is not None:
            surf.remove()
        
        surf = ax.plot_surface(X[0], X[1], X[2], color="steelblue",
                              alpha=0.6, linewidth=0, shade=True)
        
        if quiv is not None:
            quiv.remove()
        
        scale = 0.5 * np.max(np.abs(X))
        quiv = ax.quiver(0, 0, 0, omega[0], omega[1], omega[2],
                        length=scale, normalize=True, color="k")
        
        ax.view_init(elev=elev, azim=azim0 + azim_rate * k)
        ax.set_title(f"t = {t[i]:.2f}")
        
        return surf, quiv
    
    ani = animation.FuncAnimation(fig, update, frames=len(idx),
                                 interval=60, blit=False)
    
    ani.save(filename, writer="ffmpeg", dpi=200)
    return ani


# =============================================================================
# Example Usage
# =============================================================================

def example_exponential_decay_constant_product():
    """
    Example with S⁻²(t)α(t) = constant and α(t) = exp(-t/tc).
    
    This corresponds to a vortex in an exponentially decaying flow
    where the product S⁻²α remains constant.
    
    Parameters
    ----------
    tc : float
        Decay time scale (default: 2000)
    w0 : float
        Vortex frequency (default: 2π/100)
    
    Returns
    -------
    t : ndarray
        Time points.
    omega_t : ndarray
        Vorticity evolution.
    F_t : ndarray
        Deformation gradient evolution.
    """
    print("\nExample: Exponential decay with constant S⁻²α")
    print("=" * 60)
    
    tc = 2000.0
    t_end = 3000.0
    
    alpha = lambda t: np.exp(-t / tc)
    S = lambda t: np.exp(-t / (2 * tc))
    
    t_test = np.linspace(0, t_end, 100)
    product = S(t_test)**(-2) * alpha(t_test)
    print(f"  S⁻²(t)α(t) = {product[0]:.6f} (should be constant)")
    print(f"  S⁻²(t)α(t) variation: {np.std(product):.2e}")
    
    w0 = 2 * np.pi / 100
    omega0 = np.array([0.0, w0, 0.0])
    
    print(f"\n  tc = {tc:.1f}")
    print(f"  ω₀ = 2π/100 = {w0:.6f}")
    print(f"  α(0) = {alpha(0):.4f}, α(t_end) = {alpha(t_end):.4f}")
    print(f"  S(0) = {S(0):.4f}, S(t_end) = {S(t_end):.4f}")
    
    t_span = (0.0, t_end)
    t, omega_t, F_t = integrate_system(omega0, S, alpha, t_span=t_span, n_steps=1000)
    
    plot_omega_components(t, omega_t)
    plot_pq_if_single_component(t, F_t, omega0)
    
    
    animate_vortex_tube_surface(t, omega_t, F_t, omega0,
                               filename="vortex_tube_exponential_decay.mp4",
                               n_frames=240)
    
    plt.show()
    
    return t, omega_t, F_t


def example_oscillating_flow():
    """
    Example with oscillating α and S parameters.
    
    Returns
    -------
    t : ndarray
        Time points.
    omega_t : ndarray
        Vorticity evolution.
    F_t : ndarray
        Deformation gradient evolution.
    """
    print("\nExample: Oscillating flow parameters")
    print("=" * 60)
    
    t_end = 100.0
    
    alpha = lambda t: 1.0 + 0.2 * np.sin(0.1 * t)
    S = lambda t: 1.0 + 0.3 * np.cos(0.15 * t)
    
    w0 = 0.1
    omega0 = np.array([0.0, w0, 0.0])
    
    print(f"  t_end = {t_end:.1f}")
    print(f"  ω₀ = {w0:.3f}")
    print(f"  α(t) = 1.0 + 0.2 sin(0.1t)")
    print(f"  S(t) = 1.0 + 0.3 cos(0.15t)")
    
    t_span = (0.0, t_end)
    t, omega_t, F_t = integrate_system(omega0, S, alpha, t_span=t_span, n_steps=500)
    
    plot_omega_components(t, omega_t)
    plot_pq_if_single_component(t, F_t, omega0)
    
    animate_vortex_tube_surface(t, omega_t, F_t, omega0,
                  filename="vortex_oscillating.mp4",
                  n_frames=240)
    
    plt.show()
    
    return t, omega_t, F_t


if __name__ == "__main__":
    
    # Run example: exponential decay with constant S⁻²α
    t, omega_t, F_t = example_exponential_decay_constant_product()
    
    # Uncomment to run oscillating flow example
    t, omega_t, F_t = example_oscillating_flow()