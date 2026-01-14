"""
Uniform-vorticity patch in a time-dependent linear background flow
=================================================================

Splits:
  - integration of (omega, F)
  - plotting of omega(t)
  - 3D animation of advected cross-section
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


# ======================================================
# Geometry utilities
# ======================================================

def orthonormal_basis_perp(n):
    """Return (e1, e2) spanning the plane orthogonal to n."""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def W(omega):
    """Antisymmetric matrix for (1/2) omega x r."""
    wx, wy, wz = omega
    return np.array([[0.0, -wz,  wy],
                     [wz,  0.0, -wx],
                     [-wy, wx,  0.0]])


def Background_flow_matrix(t, S, alpha):
    """Linear background flow matrix."""
    eps = 1e-8
    Sdot = (S(t+eps) - S(t-eps)) / (2*eps)
    alpha_dot = (alpha(t+eps) - alpha(t-eps)) / (2*eps)
    return Sdot/S(t) * np.eye(3) + alpha_dot/alpha(t) * np.diag([1, 1, -2])

# ======================================================
# Coupled ODE system
# ======================================================
def rhs(t, y, S, alpha):
    omega = y[:3]
    F = y[3:].reshape(3, 3)
    domega = Background_flow_matrix(t, S, alpha) @ omega
    M = Background_flow_matrix(t, S, alpha) + 0.5 * W(omega)
    dF = M @ F
    return np.concatenate([domega, dF.ravel()])


# ======================================================
# Integration
# ======================================================

def integrate_system(omega0, S, alpha, t_span=(0.0, 5.0), n_steps=500):
    """Integrate omega(t) and F(t)."""
    F0 = np.eye(3)
    y0 = np.concatenate([omega0, F0.ravel()])
    t_eval = np.linspace(*t_span, n_steps)

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
        args=(S, alpha)
    )

    omega_t = sol.y[:3].T
    F_t = sol.y[3:].T.reshape(-1, 3, 3)

    return t_eval, omega_t, F_t


# ======================================================
# Plot omega components
# ======================================================

def plot_omega_components(t, omega_t):
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



def make_vortex_tube(omega0, n_circles=7, tube_length=1.0, n_theta=200):
    """
    Return a list of circles forming a vortex tube initially
    aligned with omega0.
    """
    omega_hat = omega0 / np.linalg.norm(omega0)
    e1, e2 = orthonormal_basis_perp(omega0)

    theta = np.linspace(0, 2*np.pi, n_theta)
    s_vals = np.linspace(-tube_length/2, tube_length/2, n_circles)

    circles = []
    for s in s_vals:
        circle = (
            np.outer(e1, np.cos(theta))
          + np.outer(e2, np.sin(theta))
          + s * omega_hat[:, None]
        )
        circles.append(circle)

    return circles


def plot_angle_with_z(t, omega_t):
    omega_norm = np.linalg.norm(omega_t, axis=1)
    cos_theta = omega_t[:, 2] / omega_norm
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    plt.figure()
    plt.plot(t, 180/np.pi*theta)
    plt.xlabel("t")
    plt.ylabel(r"$\theta_z$")
    plt.title("Angle between vorticity and z-axis")
    plt.tight_layout()
    plt.show()


def plot_xy_angle(t, omega_t):
    phi = np.arctan2(omega_t[:, 1], omega_t[:, 0])
    phi = np.unwrap(phi)

    plt.figure()
    plt.plot(t, phi)
    plt.xlabel("t")
    plt.ylabel(r"$\phi$")
    plt.title("Vorticity angle in the xy-plane")
    plt.tight_layout()
    plt.show()


def compute_aspect_ratio(F_t, omega0):
    e1, e2 = orthonormal_basis_perp(omega0)
    P = np.vstack([e1, e2])  # 2×3 projector

    ratios = []

    for F in F_t:
        C = F.T @ F
        C2 = P @ C @ P.T     # restriction to cross-section
        eigs = np.linalg.eigvalsh(C2)
        ratio = np.sqrt(eigs.max() / eigs.min())
        ratios.append(ratio)

    return np.array(ratios)


def plot_aspect_ratio(t, F_t, omega0):
    ar = compute_aspect_ratio(F_t, omega0)

    plt.figure()
    plt.plot(t, ar)
    plt.xlabel("t")
    plt.ylabel("Aspect ratio")
    plt.title("Vortex cross-section aspect ratio")
    plt.tight_layout()
    plt.show()


def tube_axis(F, omega0):
    v = F @ omega0
    return v / np.linalg.norm(v)


def tube_aspect_ratio(F_t, omega0):
    ratios = []

    for F in F_t:
        ell = tube_axis(F, omega0)
        P = np.eye(3) - np.outer(ell, ell)

        C = F @ F.T
        Cperp = P @ C @ P

        # restrict to 2D subspace
        eigs = np.linalg.eigvalsh(Cperp)
        eigs = eigs[eigs > 1e-12]  # remove null direction

        ratio = np.sqrt(eigs.max() / eigs.min())
        ratios.append(ratio)

    return np.array(ratios)


def tube_angle_with_z(F_t, omega0):
    angles = []
    for F in F_t:
        ell = tube_axis(F, omega0)
        angles.append(np.arccos(np.clip(ell[2], -1, 1)))
    return np.array(angles)

def tube_xy_angle(F_t, omega0):
    angles = []
    for F in F_t:
        ell = tube_axis(F, omega0)
        angles.append(np.arctan2(ell[1], ell[0]))
    return np.unwrap(np.array(angles))


# ======================================================
# 3D animation
# ======================================================

def animate_vortex(t, omega_t, F_t, omega0,
                   filename="vortex_patch_evolution.mp4",
                   n_frames=240,
                   elev=25,
                   azim0=30,
                   azim_rate=0.4):

    # subsample for movie
    idx = np.linspace(0, len(t)-1, n_frames).astype(int)

    # reference circle orthogonal to omega0
    theta = np.linspace(0, 2*np.pi, 300)
    e1, e2 = orthonormal_basis_perp(omega0)
    circle = (np.outer(e1, np.cos(theta))
            + np.outer(e2, np.sin(theta)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], lw=2, label="Advected unit circle")
    quiv = None

    # fixed bounds
    ellipse_final = F_t[idx[-1]] @ circle
    R = np.ptp(ellipse_final, axis=1).max()
    mid = ellipse_final.mean(axis=1)

    ax.set_xlim(mid[0]-R/2, mid[0]+R/2)
    ax.set_ylim(mid[1]-R/2, mid[1]+R/2)
    ax.set_zlim(mid[2]-R/2, mid[2]+R/2)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Final vortex patch geometry and vorticity")

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
        quiv = ax.quiver(0, 0, 0,
                         omega[0], omega[1], omega[2],
                         length=scale, normalize=True)

        ax.view_init(elev=elev, azim=azim0 + azim_rate * k)
        
        ax.set_title(f"t = {t[i]:.2f}")
        return line, quiv

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(idx),
                                  interval=60,
                                  blit=False)

    ani.save(filename, writer="ffmpeg", dpi=200)
    return ani



def animate_vortex_tube_surface(t, omega_t, F_t, omega0,
                                n_circles=25,
                                tube_length=1.5,
                                n_theta=120,
                                filename="vortex_tube_surface.mp4",
                                n_frames=240,
                                elev=25,
                                azim0=30,
                                azim_rate=0.4):
    """
    Animate a vortex tube as a continuous surface.
    """

    # ---------- subsample time ----------
    idx = np.linspace(0, len(t)-1, n_frames).astype(int)

    # ---------- initial tube geometry ----------
    omega_hat = omega0 / np.linalg.norm(omega0)
    e1, e2 = orthonormal_basis_perp(omega0)

    theta = np.linspace(0, 2*np.pi, n_theta)
    s_vals = np.linspace(-tube_length/2, tube_length/2, n_circles)

    # build reference tube grid (3, Ns, Nθ)
    X0 = np.zeros((3, n_circles, n_theta))
    for i, s in enumerate(s_vals):
        X0[:, i, :] = (
            np.outer(e1, np.cos(theta))
          + np.outer(e2, np.sin(theta))
          + s * omega_hat[:, None]
        )

    # ---------- figure ----------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = None
    quiv = None

    # ---------- bounds ----------
    Xf = F_t[idx[-1]] @ X0.reshape(3, -1)
    R = np.ptp(Xf, axis=1).max()
    mid = Xf.mean(axis=1)

    ax.set_xlim(mid[0]-R/2, mid[0]+R/2)
    ax.set_ylim(mid[1]-R/2, mid[1]+R/2)
    ax.set_zlim(mid[2]-R/2, mid[2]+R/2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # ---------- update ----------
    def update(k):
        nonlocal surf, quiv
        i = idx[k]

        F = F_t[i]
        omega = omega_t[i]

        X = F @ X0.reshape(3, -1)
        X = X.reshape(3, n_circles, n_theta)

        if surf is not None:
            surf.remove()

        surf = ax.plot_surface(
            X[0], X[1], X[2],
            rstride=1, cstride=1,
            color="steelblue",
            alpha=0.6,
            linewidth=0,
            shade=True
        )

        if quiv is not None:
            quiv.remove()

        scale = 0.5 * np.max(np.abs(X))
        quiv = ax.quiver(0, 0, 0,
                         omega[0], omega[1], omega[2],
                         length=scale, normalize=True,
                         color="k")

        ax.view_init(elev=elev, azim=azim0 + azim_rate * k)
        ax.set_title(f"t = {t[i]:.2f}")

        return surf, quiv

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(idx),
        interval=60,
        blit=False
    )

    ani.save(filename, writer="ffmpeg", dpi=200)
    return ani



def plot_vortex_tube_snapshot(F, omega, omega0,
                              n_circles=30,
                              tube_length=1.5,
                              n_theta=120,
                              stretch_factor=4.0,
                              show_vorticity=True):
    """
    Plot the vortex tube at a single time, with optional
    visual stretching along the tube axis.
    """

    # ----- initial tube -----
    omega_hat0 = omega0 / np.linalg.norm(omega0)
    e1, e2 = orthonormal_basis_perp(omega0)

    theta = np.linspace(0, 2*np.pi, n_theta)
    s_vals = np.linspace(-tube_length/2, tube_length/2, n_circles)

    X0 = np.zeros((3, n_circles, n_theta))
    for i, s in enumerate(s_vals):
        X0[:, i, :] = (
            np.outer(e1, np.cos(theta))
          + np.outer(e2, np.sin(theta))
          + s * omega_hat0[:, None]
        )

    # ----- advect (physical tube) -----
    X = F @ X0.reshape(3, -1)
    X = X.reshape(3, n_circles, n_theta)

    # ----- visual stretch along tube axis -----
    ell = tube_axis(F, omega0)   # unit vector
    for i in range(n_circles):
        for j in range(n_theta):
            x = X[:, i, j]
            x_par = np.dot(x, ell) * ell
            x_perp = x - x_par
            X[:, i, j] = x_perp + stretch_factor * x_par

    # ----- plot -----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X[0], X[1], X[2],
        color="steelblue",
        alpha=0.6,
        linewidth=0,
        shade=True
    )

    if show_vorticity:
        scale = 0.4 * np.max(np.abs(X))
        ax.quiver(0, 0, 0,
                  omega[0], omega[1], omega[2],
                  length=scale, normalize=True,
                  color="k")

    # ----- bounds -----
    Xflat = X.reshape(3, -1)
    R = np.ptp(Xflat, axis=1).max()
    mid = Xflat.mean(axis=1)

    ax.set_xlim(mid[0]-R/2, mid[0]+R/2)
    ax.set_ylim(mid[1]-R/2, mid[1]+R/2)
    ax.set_zlim(mid[2]-R/2, mid[2]+R/2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(f"Vortex tube (visual stretch ×{stretch_factor})")
    plt.tight_layout()
    plt.show()



# ======================================================
# Example run
# ======================================================

if __name__ == "__main__":

    
    # Parameters
    w0 = 2 * np.pi / 100  # Vorticity magnitude (z-component only)
    omega0 = np.array([0.0, w0, 0.0])  # Initial vorticity vector

    # Time parameters
    tc = 2000.0
    # Background flow functions
    def S(t):
        """Scale factor - exponential decay."""
        return  np.exp(-t / (2 * tc))
    
    def alpha(t):
        """Anisotropy - oscillating."""
        return np.exp(-t / tc)
    

    # INTEGRATION
    t_span = (0.0, 1000.0)



    t, omega_t, F_t = integrate_system(omega0, S, alpha, t_span=t_span)

    # ----------------------
    # PLOTS
    # ----------------------

    # string labels for plots
    omega0_str = rf"$\omega_0 = ({omega0[0]:.1f},{omega0[1]:.1f},{omega0[2]:.1f})$"


    plot_omega_components(t, omega_t)
    
    plot_angle_with_z(t, omega_t)
    plot_xy_angle(t, omega_t)
    plot_aspect_ratio(t, F_t, omega0)

    # ---- TRUE tube geometry diagnostics ----
    tube_ar = tube_aspect_ratio(F_t, omega0)
    tube_theta_z = tube_angle_with_z(F_t, omega0)
    tube_phi = tube_xy_angle(F_t, omega0)

    plt.figure()
    plt.plot(t, tube_ar)
    plt.xlabel("t")
    plt.ylabel("Aspect ratio")
    plt.title("True vortex tube aspect ratio\n")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, 180/np.pi * tube_theta_z)
    plt.xlabel("t")
    plt.ylabel(r"$\theta_z$ (deg)")
    plt.title("Tube axis angle with z-axis\n")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, tube_phi)
    plt.xlabel("t")
    plt.ylabel(r"$\phi$")
    plt.title("Tube axis angle in xy-plane\n")
    plt.tight_layout()
    plt.show()
    
    # ----------------------
    # ANIMATION
    # ----------------------
    animate_vortex(t, omega_t, F_t, omega0,
                   filename="vortex_patch_evolution2.mp4",
                   n_frames=240,
                   elev=25,
                   azim0=30,
                   azim_rate=0.4
                   )
    
    '''
    i = -1  # choose time index

    plot_vortex_tube_snapshot(
        F_t[i],
        omega_t[i],
        omega0,
        n_circles=2,
        tube_length=1.5
    )

    
    animate_vortex_tube_surface(
        t, omega_t, F_t, omega0,
        n_circles=2,
        tube_length=1.5,
        filename="vortex_tube_surface.mp4",
        elev=25,
        azim0=30,
        azim_rate=0.5
    )
    '''