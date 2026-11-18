"""Steady conduction: resistances of walls and shells, the critical radius, a wall with k(T)
(Kirchhoff transform), and a rectangular plate by series and by finite differences."""

import numpy as np


def r_plane_wall(L: float, k: float, A: float = 1.0) -> float:
    """Thermal resistance R_th = L / (k * A) [K/W] of a plane wall."""
    if L <= 0.0 or k <= 0.0 or A <= 0.0:
        raise ValueError("L, k, and A must be positive.")
    return float(L / (k * A))


def r_cylindrical_shell(r1: float, r2: float, L: float, k: float) -> float:
    """Thermal resistance R_th = ln(r2/r1) / (2 * pi * L * k) [K/W] of a cylindrical shell."""
    if r1 <= 0.0 or r2 <= r1 or L <= 0.0 or k <= 0.0:
        raise ValueError("Require 0 < r1 < r2, L > 0, k > 0.")
    return float(np.log(r2 / r1) / (2.0 * np.pi * L * k))


def r_spherical_shell(r1: float, r2: float, k: float) -> float:
    """Thermal resistance R_th = (1/r1 - 1/r2) / (4 * pi * k) [K/W] of a spherical shell."""
    if r1 <= 0.0 or r2 <= r1 or k <= 0.0:
        raise ValueError("Require 0 < r1 < r2 and k > 0.")
    return float((1.0 / r1 - 1.0 / r2) / (4.0 * np.pi * k))


def critical_insulation_radius(k_ins: float, h_ext: float, geometry: str = "cylinder") -> float:
    """Critical insulation outer radius r_crit = k_ins / h_ext (cylinder) or 2 * k_ins / h_ext (sphere) [m]."""
    if k_ins <= 0.0 or h_ext <= 0.0:
        raise ValueError("k_ins and h_ext must be positive.")
    if geometry == "cylinder":
        return float(k_ins / h_ext)
    if geometry == "sphere":
        return float(2.0 * k_ins / h_ext)
    raise ValueError("geometry must be 'cylinder' or 'sphere'.")


def kirchhoff_linear_k_wall(
    x: np.ndarray, L: float, T1: float, T2: float, k0: float, beta: float, T0: float | None = None
) -> tuple[float, np.ndarray]:
    """
    Exact heat flux q'' [W/m^2] and temperature profile T(x) across a plane wall of thickness L
    with temperature-dependent thermal conductivity k(T) = k0 * (1 + beta * (T - T0)).
    Derived via the Kirchhoff integral transform.
    """
    if L <= 0.0 or k0 <= 0.0:
        raise ValueError("L and k0 must be positive.")
    if T0 is None:
        T0 = T1
    T_mean = 0.5 * (T1 + T2)
    k_mean = k0 * (1.0 + beta * (T_mean - T0))
    if k_mean <= 0.0:
        raise ValueError("Mean thermal conductivity must be positive.")
    q_flux = k_mean * (T1 - T2) / L

    xi = x / L
    if abs(beta) < 1e-12:
        T_x = T1 - (T1 - T2) * xi
    else:
        # Solve k0 * [ (T - T1) + 0.5 * beta * ((T - T0)^2 - (T1 - T0)^2) ] = -q_flux * x
        u1 = 1.0 + beta * (T1 - T0)
        u2 = 1.0 + beta * (T2 - T0)
        u_sq = (1.0 - xi) * (u1 ** 2) + xi * (u2 ** 2)
        T_x = T0 + (np.sqrt(u_sq) - 1.0) / beta
    return float(q_flux), T_x


def analytical_2d_rectangular_plate(
    x: np.ndarray, y: np.ndarray, W: float, H: float, T_base: float, T_top: float, n_terms: int = 80
) -> np.ndarray:
    """
    Exact separation-of-variables Fourier series solution for 2-D steady Laplace equation nabla^2 T = 0
    on [0, W] x [0, H] with T=T_base on x=0, x=W, y=0 and T=T_top on y=H:
      theta(x, y) = (T - T_base)/(T_top - T_base) = (4/pi) sum_{m=1,3,5...} (1/m) sin(m pi x / W) sinh(m pi y / W) / sinh(m pi H / W)
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    theta = np.zeros_like(X, dtype=float)
    for k in range(n_terms):
        m = 2 * k + 1
        arg_H = m * np.pi * H / W
        # Use stable exponential ratio for sinh(m*pi*Y/W) / sinh(m*pi*H/W) when arg_H is large
        if arg_H < 50.0:
            ratio = np.sinh(m * np.pi * Y / W) / np.sinh(arg_H)
        else:
            ratio = np.exp(m * np.pi * (Y - H) / W) * (1.0 - np.exp(-2.0 * m * np.pi * Y / W)) / (1.0 - np.exp(-2.0 * arg_H))
        theta += (4.0 / (np.pi * m)) * np.sin(m * np.pi * X / W) * ratio
    return T_base + (T_top - T_base) * theta


def solve_2d_steady_rect(
    Nx: int, Ny: int, W: float, H: float, T_left: float, T_right: float, T_bottom: float, T_top: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From-scratch 2-D five-point finite-difference Laplacian solver on a uniform Nx x Ny grid.
    Assembles the linear system for interior nodes and solves directly via np.linalg.solve.
    """
    x = np.linspace(0.0, W, Nx)
    y = np.linspace(0.0, H, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    rx = 1.0 / (dx ** 2)
    ry = 1.0 / (dy ** 2)
    diag_val = -2.0 * (rx + ry)

    T = np.zeros((Nx, Ny), dtype=float)
    T[0, :] = T_left
    T[-1, :] = T_right
    T[:, 0] = T_bottom
    T[:, -1] = T_top

    nx_in = Nx - 2
    ny_in = Ny - 2
    n_unknowns = nx_in * ny_in
    A = np.zeros((n_unknowns, n_unknowns), dtype=float)
    b = np.zeros(n_unknowns, dtype=float)

    def idx(i_in: int, j_in: int) -> int:
        return i_in * ny_in + j_in

    for i_in in range(nx_in):
        for j_in in range(ny_in):
            row = idx(i_in, j_in)
            A[row, row] = diag_val

            # Left neighbor (i_in - 1)
            if i_in > 0:
                A[row, idx(i_in - 1, j_in)] = rx
            else:
                b[row] -= rx * T_left

            # Right neighbor (i_in + 1)
            if i_in < nx_in - 1:
                A[row, idx(i_in + 1, j_in)] = rx
            else:
                b[row] -= rx * T_right

            # Bottom neighbor (j_in - 1)
            if j_in > 0:
                A[row, idx(i_in, j_in - 1)] = ry
            else:
                b[row] -= ry * T_bottom

            # Top neighbor (j_in + 1)
            if j_in < ny_in - 1:
                A[row, idx(i_in, j_in + 1)] = ry
            else:
                b[row] -= ry * T_top

    T_in = np.linalg.solve(A, b).reshape((nx_in, ny_in))
    T[1:-1, 1:-1] = T_in
    return x, y, T
