"""Reference solutions used to check the simple models of the book.

Each function solves a problem more completely than the model in the text, by a
different method, so that the two can be compared.  These functions use SciPy
(``pip install -e ".[book]"``); the rest of the package needs NumPy alone.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import SIGMA_SB

__all__ = ["insulated_pipe_fv", "steady_conduction_fv", "plate_fin_2d_fv", "plate_fin_2d_series", "laplace_2d_fd", "transient_conduction_fv", "enthalpy_1d_fv", "similarity_bvp", "boundary_layer_march", "graetz_march"]


def insulated_pipe_fv(
    T_water,
    r1: float,
    r2: float,
    k_ins: float,
    h: float,
    eps: float,
    T_inf: float,
    T_sur: float | None = None,
    n_cells: int = 200,
    t_eval=None,
    rho_c_ins: float = 0.0,
    C_cladding: float = 0.0,
    C_pipe=None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
):
    """Finite-volume model of an insulated pipe, per metre of length.

    The insulation (radii ``r1`` to ``r2``, conductivity ``k_ins``) is divided into
    ``n_cells`` annular cells.  Its inner face is at the temperature of the pipe contents;
    its outer face carries a thin cladding that loses heat by convection
    (``h``, to air at ``T_inf``) and by radiation (emissivity ``eps``, to surroundings at
    ``T_sur``, by default ``T_inf``).

    Steady state (``t_eval`` is None): ``T_water`` is a fixed temperature [K].  Returns a
    dict with the cell centres ``r``, the cell temperatures ``T``, the surface temperature
    ``T_s`` and the heat loss per metre ``q``.

    Cooling (``t_eval`` given, in s): ``T_water`` is the initial temperature, and the
    contents cool with heat capacity per metre ``C_pipe(T)`` [J/(m K)] (a callable or a
    number).  The insulation (volumetric heat capacity ``rho_c_ins``) and the cladding (heat
    capacity per metre ``C_cladding``) store heat too.  The initial state is the steady
    profile.  Returns a dict with ``t``, the contents temperature ``T_water``, the surface
    temperature ``T_s``, the heat loss ``q`` and the energy-balance residual
    ``energy_error`` (stored-energy change plus heat lost, relative to the heat lost).
    """
    from scipy.integrate import cumulative_trapezoid, solve_ivp
    from scipy.linalg import solve_banded
    from scipy.sparse import diags

    T_sur = T_inf if T_sur is None else T_sur
    # Standard second-order finite volumes on a uniform grid: the conductance between two
    # points is k times the area of the face between them, divided by their distance.
    faces = np.linspace(r1, r2, n_cells + 1)
    centres = 0.5 * (faces[:-1] + faces[1:])
    two_pi_k = 2.0 * np.pi * k_ins
    G_in = two_pi_k * 0.5 * (r1 + centres[0]) / (centres[0] - r1)        # contents -> first cell centre
    G = two_pi_k * faces[1:-1] / np.diff(centres)                        # between cell centres
    G_out = two_pi_k * 0.5 * (r2 + centres[-1]) / (r2 - centres[-1])     # last cell centre -> cladding
    perimeter = 2.0 * np.pi * r2

    def loss(Ts):
        return perimeter * (h * (Ts - T_inf) + eps * SIGMA_SB * (Ts ** 4 - T_sur ** 4))

    def dloss(Ts):
        return perimeter * (h + 4.0 * eps * SIGMA_SB * Ts ** 3)

    def net_heat(Tw, y):
        """Net heat flow into each insulation cell and into the cladding (last entry)."""
        T, Ts = y[:-1], y[-1]
        flow = np.empty(n_cells + 1)
        inner = np.concatenate(([G_in * (Tw - T[0])], G * (T[:-1] - T[1:])))  # into cells from inside
        outer = np.concatenate((G * (T[:-1] - T[1:]), [G_out * (T[-1] - Ts)]))  # out of cells
        flow[:-1] = inner - outer
        flow[-1] = G_out * (T[-1] - Ts) - loss(Ts)
        return flow

    def jac_y(y):
        main = np.empty(n_cells + 1)
        main[:-1] = -(np.concatenate(([G_in], G)) + np.concatenate((G, [G_out])))
        main[-1] = -G_out - dloss(y[-1])
        off = np.concatenate((G, [G_out]))
        return main, off

    def steady(Tw):
        """Newton's method on the tridiagonal system of steady cell balances."""
        y = np.concatenate((np.linspace(Tw, T_inf, n_cells + 1)[:-1], [T_inf]))
        for _ in range(100):
            main, off = jac_y(y)
            bands = np.zeros((3, n_cells + 1))
            bands[0, 1:], bands[1], bands[2, :-1] = off, main, off
            step = solve_banded((1, 1), bands, net_heat(Tw, y))
            y = y - step
            if np.max(np.abs(step)) < 1e-11 * max(1.0, abs(Tw)):
                return y
        raise RuntimeError("steady insulated_pipe_fv did not converge.")

    y_steady = steady(float(T_water))
    if t_eval is None:
        return {"r": centres, "T": y_steady[:-1], "T_s": y_steady[-1], "q": loss(y_steady[-1]),
                "faces": faces}

    C_pipe_fn = C_pipe if callable(C_pipe) else (lambda T, c=C_pipe: c + 0.0 * T)
    cap = np.concatenate((rho_c_ins * np.pi * (faces[1:] ** 2 - faces[:-1] ** 2), [C_cladding]))
    if np.any(cap <= 0.0):
        raise ValueError("cooling needs rho_c_ins > 0 and C_cladding > 0.")

    def rhs(t, z):
        Tw, y = z[0], z[1:]
        dTw = -G_in * (Tw - y[0]) / C_pipe_fn(Tw)
        return np.concatenate(([dTw], net_heat(Tw, y) / cap))

    def jac(t, z):
        Tw, y = z[0], z[1:]
        main, off = jac_y(y)
        n = n_cells + 2
        d0 = np.concatenate(([-G_in / C_pipe_fn(Tw)], main / cap))
        up = np.concatenate(([G_in / C_pipe_fn(Tw)], off / cap[:-1]))
        lo = np.concatenate(([G_in / cap[0]], off / cap[1:]))
        return diags([lo, d0, up], [-1, 0, 1], shape=(n, n), format="csc")

    t_eval = np.asarray(t_eval, dtype=float)
    z0 = np.concatenate(([float(T_water)], y_steady))
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), z0, method="BDF", t_eval=t_eval, jac=jac,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    Tw, Y = sol.y[0], sol.y[1:]
    Ts = Y[-1]
    q = loss(Ts)
    # energy balance: heat stored in contents, insulation and cladding, against heat lost
    grid = np.linspace(Tw.min(), Tw.max(), 4001)
    F = np.interp(Tw, grid, cumulative_trapezoid(C_pipe_fn(grid), grid, initial=0.0))
    stored = (F - F[0]) + (cap[:, None] * (Y - Y[:, :1])).sum(axis=0)
    lost = cumulative_trapezoid(q, sol.t, initial=0.0)
    energy_error = np.max(np.abs(stored + lost)) / max(lost[-1], 1e-300)
    return {"t": sol.t, "T_water": Tw, "T_s": Ts, "q": q, "profiles": Y[:-1], "r": centres,
            "energy_error": energy_error}


def _geometry_terms(faces, geometry):
    """Cell volumes per unit (area, length or sphere) and the area function A(r)."""
    if geometry == "plane":
        return np.diff(faces), (lambda r: np.ones_like(np.asarray(r, dtype=float)))
    if geometry == "cylinder":
        return np.pi * np.diff(faces ** 2), (lambda r: 2.0 * np.pi * np.asarray(r, dtype=float))
    if geometry == "sphere":
        return 4.0 / 3.0 * np.pi * np.diff(faces ** 3), (lambda r: 4.0 * np.pi * np.asarray(r, dtype=float) ** 2)
    raise ValueError("geometry must be 'plane', 'cylinder' or 'sphere'.")


def _half_resistance(r_from, r_to, k, geometry):
    """Resistance of a layer between r_from and r_to with constant k (per unit area, per metre, or total)."""
    if geometry == "plane":
        return np.abs(r_to - r_from) / k
    if geometry == "cylinder":
        return np.abs(np.log(r_to / r_from)) / (2.0 * np.pi * k)
    return np.abs(1.0 / r_from - 1.0 / r_to) / (4.0 * np.pi * k)


def steady_conduction_fv(faces, k, geometry="plane", q_gen=0.0, left=("symmetry",), right=("temperature", 300.0),
                         tol=1e-11, maxiter=100):
    """Steady one-dimensional conduction by finite volumes, solved by Newton's method.

    ``faces`` are the cell boundaries (a wall from faces[0] to faces[-1]; for a cylinder or a
    sphere, radii).  ``k`` is either an array with one conductivity per cell, or a callable
    ``k(T, cell_index)`` for a conductivity that depends on temperature; ``q_gen`` is the heat
    generated per unit volume in each cell (W/m^3, a number or an array).

    Each boundary condition is one of
      ("temperature", T)                          a fixed surface temperature,
      ("convection", h, T_inf)                    Newton's law of cooling,
      ("convection", h, T_inf, eps, T_sur)        convection and grey-body radiation,
      ("symmetry",)                               no heat flow (the centre of a cylinder or sphere).

    The conductance between two cell centres is the series sum of the two half-cell resistances,
    each with the conductivity of its own cell, so layered walls are represented exactly when k
    is constant in each layer.  Returns a dict with the cell centres ``r``, temperatures ``T``,
    the surface temperatures ``T_left`` and ``T_right``, the heat flows out through each surface
    ``Q_left`` and ``Q_right`` (per unit area, per metre or total, for plane, cylinder or sphere)
    and the energy-balance residual ``balance`` (generation minus the two outflows).
    """
    faces = np.asarray(faces, dtype=float)
    n = len(faces) - 1
    centres = 0.5 * (faces[:-1] + faces[1:])
    volumes, area = _geometry_terms(faces, geometry)
    q_gen = np.broadcast_to(np.asarray(q_gen, dtype=float), (n,))
    k_of = k if callable(k) else (lambda T, i, kk=np.broadcast_to(np.asarray(k, dtype=float), (n,)): kk[i] + 0.0 * T)
    idx = np.arange(n)
    has_left = left[0] != "symmetry"
    has_right = right[0] != "symmetry"
    if geometry != "plane" and faces[0] == 0.0 and has_left:
        raise ValueError("a cylinder or sphere that reaches r = 0 needs left=('symmetry',).")

    def surface_flux(bc, Ts, r_face):
        """Heat leaving the surface into the surroundings, and its derivative in Ts."""
        kind = bc[0]
        if kind == "convection":
            h, T_inf = bc[1], bc[2]
            eps, T_sur = (bc[3], bc[4]) if len(bc) > 3 else (0.0, T_inf)
            A = float(area(r_face))
            return (A * (h * (Ts - T_inf) + eps * SIGMA_SB * (Ts ** 4 - T_sur ** 4)),
                    A * (h + 4.0 * eps * SIGMA_SB * Ts ** 3))
        raise ValueError(f"unknown boundary condition {kind!r}")

    def residual(y):
        T = y[:n]
        kc = k_of(T, idx)
        with np.errstate(divide="ignore"):                             # r = 0 at a centre of symmetry
            R_in = _half_resistance(centres, faces[:-1], kc, geometry)  # centre -> left face
        R_out = _half_resistance(centres, faces[1:], kc, geometry)    # centre -> right face
        G = 1.0 / (R_out[:-1] + R_in[1:])
        flow = G * (T[:-1] - T[1:])                                    # left to right between centres
        res = q_gen * volumes
        res[:-1] -= flow
        res[1:] += flow
        extra = []
        Q_left = Q_right = 0.0
        pos = n
        if has_left:
            T_l = left[1] if left[0] == "temperature" else y[pos]
            Q_left = (T[0] - T_l) / R_in[0]                            # leaves through the left face
            res[0] -= Q_left
            if left[0] != "temperature":
                extra.append(Q_left - surface_flux(left, y[pos], faces[0])[0])
                pos += 1
        if has_right:
            T_r = right[1] if right[0] == "temperature" else y[pos]
            Q_right = (T[-1] - T_r) / R_out[-1]
            res[-1] -= Q_right
            if right[0] != "temperature":
                extra.append(Q_right - surface_flux(right, y[pos], faces[-1])[0])
        return np.concatenate((res, extra)), Q_left, Q_right

    n_extra = int(has_left and left[0] != "temperature") + int(has_right and right[0] != "temperature")
    guess = [bc[1] if bc[0] == "temperature" else bc[2] for bc in (left, right) if bc[0] != "symmetry"]
    T0 = float(np.mean(guess)) if guess else 300.0
    y = np.full(n + n_extra, T0)
    m = n + n_extra
    for _ in range(maxiter):
        F, _, _ = residual(y)
        # Jacobian by finite differences (dense: the grids of the book have at most a few hundred cells)
        J = np.empty((m, m))
        step = 1e-6 * np.maximum(1.0, np.abs(y))
        for j in range(m):
            yp = y.copy()
            yp[j] += step[j]
            J[:, j] = (residual(yp)[0] - F) / step[j]
        dy = np.linalg.solve(J, -F)
        y = y + dy
        if np.max(np.abs(dy)) < tol * max(1.0, np.max(np.abs(y))):
            break
    else:
        raise RuntimeError("steady_conduction_fv did not converge.")
    F, Q_left, Q_right = residual(y)
    T = y[:n]
    pos = n
    out = {"r": centres, "T": T, "faces": faces}
    if has_left:
        out["T_left"] = left[1] if left[0] == "temperature" else y[pos]
        pos += left[0] != "temperature"
    if has_right:
        out["T_right"] = right[1] if right[0] == "temperature" else y[pos]
    out["Q_left"], out["Q_right"] = Q_left, Q_right
    generated = float(np.sum(q_gen * volumes))
    out["balance"] = generated - Q_left - Q_right
    return out


def plate_fin_2d_fv(L: float, b: float, k: float, h: float, T_b: float, T_inf: float,
                    nx: int = 200, ny: int = 20, tip: str = "adiabatic", h_tip: float | None = None):
    """Steady two-dimensional conduction in half of a straight plate fin, by finite volumes.

    The domain is 0 <= x <= L along the fin and 0 <= y <= b across half its thickness, with
    the base x = 0 held at ``T_b``, symmetry at the mid-plane y = 0, convection to ``T_inf``
    with coefficient ``h`` on the face y = b, and an adiabatic (``tip="adiabatic"``) or
    convective (``tip="convection"``, coefficient ``h_tip``, default ``h``) tip at x = L.
    The grid is uniform with ``nx`` by ``ny`` cells, and the linear system is solved directly.

    Returns a dict with the cell centres ``x`` and ``y``, the temperatures ``T`` (shape ny by
    nx), the heat entering through the base ``Q_base`` and leaving through the face and tip
    ``Q_out``, both in W per metre of fin width for the half fin, and their relative
    difference ``balance``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    if tip not in ("adiabatic", "convection"):
        raise ValueError("tip must be 'adiabatic' or 'convection'")
    h_tip = h if h_tip is None else h_tip
    dx, dy = L / nx, b / ny
    n = nx * ny
    idx = np.arange(n).reshape(ny, nx)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    diag = np.zeros(n)
    gx = k * dy / dx                      # conductance between neighbours along x (per unit width)
    gy = k * dx / dy                      # conductance between neighbours across
    # neighbours along x
    a, c = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    rows += [a, c]; cols += [c, a]; vals += [np.full(a.size, -gx)] * 2
    np.add.at(diag, a, gx); np.add.at(diag, c, gx)
    # neighbours across
    a, c = idx[:-1, :].ravel(), idx[1:, :].ravel()
    rows += [a, c]; cols += [c, a]; vals += [np.full(a.size, -gy)] * 2
    np.add.at(diag, a, gy); np.add.at(diag, c, gy)
    # base: half-cell conduction to the wall at T_b
    g_base = k * dy / (0.5 * dx)
    base = idx[:, 0]
    diag[base] += g_base
    rhs[base] += g_base * T_b
    # face y = b: half-cell conduction in series with convection
    g_face = dx / (0.5 * dy / k + 1.0 / h)
    face = idx[-1, :]
    diag[face] += g_face
    rhs[face] += g_face * T_inf
    # tip
    g_tip = 0.0
    if tip == "convection":
        g_tip = dy / (0.5 * dx / k + 1.0 / h_tip)
        diag[idx[:, -1]] += g_tip
        rhs[idx[:, -1]] += g_tip * T_inf
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    T = spsolve(A, rhs).reshape(ny, nx)
    Q_base = float(np.sum(g_base * (T_b - T[:, 0])))
    Q_out = float(np.sum(g_face * (T[-1, :] - T_inf)) + np.sum(g_tip * (T[:, -1] - T_inf)))
    return {
        "x": (np.arange(nx) + 0.5) * dx,
        "y": (np.arange(ny) + 0.5) * dy,
        "T": T,
        "Q_base": Q_base,
        "Q_out": Q_out,
        "balance": (Q_base - Q_out) / Q_base,
    }


def plate_fin_2d_series(L: float, b: float, k: float, h: float, n_terms: int = 400):
    """Exact two-dimensional solution for half of a straight plate fin with an adiabatic tip.

    Same problem as :func:`plate_fin_2d_fv` with ``tip="adiabatic"``.  Separation of
    variables gives

        theta / theta_b = sum_n a_n cos(lam_n y) cosh(lam_n (L - x)) / cosh(lam_n L),

    with lam_n b tan(lam_n b) = Bi = h b / k and
    a_n = 2 sin(lam_n b) / (lam_n b + sin(lam_n b) cos(lam_n b)).

    Returns a dict with the eigenvalues ``lam``, the coefficients ``a``, the heat entering
    the base per unit temperature difference and per metre of width, ``Q_per_K`` (for the
    half fin), and a function ``theta(x, y)`` giving theta / theta_b.
    """
    from scipy.optimize import brentq

    Bi = h * b / k
    z = np.empty(n_terms)
    for j in range(n_terms):
        # root of z sin z - Bi cos z = 0 in (j pi, j pi + pi/2)
        f = lambda s: s * np.sin(s) - Bi * np.cos(s)
        z[j] = brentq(f, j * np.pi + 1e-14, j * np.pi + 0.5 * np.pi, xtol=1e-15)
    lam = z / b
    a = 2.0 * np.sin(z) / (z + np.sin(z) * np.cos(z))
    Q_per_K = float(k * np.sum(a * np.sin(z) * np.tanh(lam * L)))

    def theta(x, y):
        x = np.asarray(x, dtype=float)[..., None]
        y = np.asarray(y, dtype=float)[..., None]
        # cosh(lam (L - x)) / cosh(lam L), written to avoid overflow
        ratio = np.exp(-lam * x) * (1 + np.exp(-2 * lam * (L - x))) / (1 + np.exp(-2 * lam * L))
        return np.sum(a * np.cos(lam * y) * ratio, axis=-1)

    return {"lam": lam, "a": a, "Q_per_K": Q_per_K, "theta": theta}


def laplace_2d_fd(width: float, depth: float, h: float, k: float = 1.0, circles=(),
                  top=("temperature", 0.0), bottom=("insulated",), left=("insulated",), right=("insulated",)):
    """Steady two-dimensional conduction without generation, by finite differences.

    The domain is the rectangle 0 <= x <= ``width``, 0 <= y <= ``depth``, with y measured
    downwards from the top edge (a ground surface, say).  Nodes are ``h`` apart.  Circular
    holes, given as ``circles = [(xc, yc, r, T), ...]``, are held at temperature T (pipes);
    where a grid link crosses a circle the Shortley–Weller stencil uses the exact distance
    to the circle, which keeps the scheme second order on curved boundaries.

    Each edge takes one of: ``("temperature", T)`` with T a number or a function of the
    position along the edge; ``("insulated",)`` (also a plane of symmetry); or
    ``("convection", H, T_inf)``.

    Returns a dict with node coordinates ``x`` and ``y``, temperatures ``T`` (shape ny by nx,
    NaN inside the circles), and ``Q_circles``, the heat leaving each circle per metre of
    length, from a second-order sum of the fluxes across a square of cell faces around it.
    A circle cut by an insulated left or right edge (a pipe on a plane of symmetry) is
    counted as the half that lies in the domain.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    nx, ny = int(round(width / h)) + 1, int(round(depth / h)) + 1
    x = np.arange(nx) * h
    y = np.arange(ny) * h
    X, Y = np.meshgrid(x, y)
    inside = np.zeros((ny, nx), dtype=bool)
    for (xc, yc, r, _) in circles:
        inside |= (X - xc) ** 2 + (Y - yc) ** 2 <= r * r * (1 + 1e-12)
    fixed = np.full((ny, nx), np.nan)
    edges = {"top": top, "bottom": bottom, "left": left, "right": right}
    for name, bc in edges.items():
        if bc[0] == "temperature":
            val = bc[1]
            if name in ("top", "bottom"):
                row = 0 if name == "top" else ny - 1
                fixed[row, :] = val(x) if callable(val) else val
            else:
                col = 0 if name == "left" else nx - 1
                fixed[:, col] = val(y) if callable(val) else val
        elif bc[0] not in ("insulated", "convection"):
            raise ValueError(f"unknown boundary condition {bc!r}")
    unknown = ~inside & np.isnan(fixed)
    number = -np.ones((ny, nx), dtype=int)
    number[unknown] = np.arange(unknown.sum())
    n = int(unknown.sum())
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)

    def crossing(i, j, di, dj):
        """Fraction of the link from node (j, i) towards (j+dj, i+di) before a circle, and its T."""
        best, T_c = 1.0, None
        px, py = x[i], y[j]
        for (xc, yc, r, Tcirc) in circles:
            # solve |p + s (d h) - c| = r for the smallest s in (0, 1]
            dx, dy = di * h, dj * h
            fx, fy = px - xc, py - yc
            a = dx * dx + dy * dy
            b = 2 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - r * r
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            s = (-b - np.sqrt(disc)) / (2 * a)
            if 0 < s <= 1.0 and s < best:
                best, T_c = s, Tcirc
        return best, T_c

    jj, ii = np.nonzero(unknown)
    for p, (j, i) in enumerate(zip(jj, ii)):
        diag = 0.0
        for axis in (0, 1):                         # 0: x direction, 1: y direction
            arms = []
            for sgn in (+1, -1):
                di, dj = (sgn, 0) if axis == 0 else (0, sgn)
                i2, j2 = i + di, j + dj
                if 0 <= i2 < nx and 0 <= j2 < ny:
                    if inside[j2, i2]:
                        theta, Tc = crossing(i, j, di, dj)
                        arms.append(("value", theta, Tc))
                    elif not np.isnan(fixed[j2, i2]):
                        arms.append(("value", 1.0, fixed[j2, i2]))
                    else:
                        arms.append(("node", 1.0, number[j2, i2]))
                else:                                # outside: mirror, with convection if asked
                    name = {(1, 0): "right", (-1, 0): "left", (0, 1): "bottom", (0, -1): "top"}[(di, dj)]
                    arms.append(("ghost", 1.0, edges[name]))
            # Shortley-Weller weights for the two arms
            tE, tW = arms[0][1], arms[1][1]
            wE, wW = 2.0 / (tE * (tE + tW)), 2.0 / (tW * (tE + tW))
            for (kind, theta, data), w, other in ((arms[0], wE, arms[1]), (arms[1], wW, arms[0])):
                diag -= w
                if kind == "value":
                    rhs[p] -= w * data
                elif kind == "node":
                    rows.append(p); cols.append(data); vals.append(w)
                else:                                # ghost = mirror of the opposite node
                    if other[0] == "node":
                        rows.append(p); cols.append(other[2]); vals.append(w)
                    elif other[0] == "value":
                        rhs[p] -= w * other[2]
                    else:
                        raise ValueError("domain must be at least two nodes wide")
                    if data[0] == "convection":    # ghost = mirror - 2 h H/k (T_P - T_inf)
                        H, T_inf = data[1], data[2]
                        diag -= w * 2 * h * H / k
                        rhs[p] -= w * 2 * h * H / k * T_inf
        rows.append(p); cols.append(p); vals.append(diag)
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    T = fixed.copy()
    T[unknown] = spsolve(A, rhs)
    T[inside] = np.nan

    Q = []
    for (xc, yc, r, _) in circles:
        # square of cell faces around the circle, one node clear of it
        i0 = int(np.floor((xc - r) / h)) - 1
        i1 = int(np.ceil((xc + r) / h)) + 1
        j0 = int(np.floor((yc - r) / h)) - 1
        j1 = int(np.ceil((yc + r) / h)) + 1
        if j0 < 0 or j1 >= ny - 1 or i1 >= nx - 1:
            raise ValueError("circle too close to the edge of the domain for the flux contour")
        cut_left = i0 < 0
        i0 = max(i0, 0)
        total = 0.0
        for i in range(i0, i1 + 1):                  # faces above and below
            wgt = 0.5 if (cut_left and i == 0) else 1.0
            total += wgt * k * (T[j0, i] - T[j0 - 1, i])
            total += wgt * k * (T[j1, i] - T[j1 + 1, i])
        for j in range(j0, j1 + 1):                  # faces at the sides
            total += k * (T[j, i1] - T[j, i1 + 1])
            if not cut_left:
                total += k * (T[j, i0] - T[j, i0 - 1])
        Q.append(float(total))
    return {"x": x, "y": y, "T": T, "Q_circles": Q}


def transient_conduction_fv(R: float, k: float, rho_c: float, h, T_inf: float, T_init, geometry: str = "slab",
                            n_cells: int = 100, t_end: float = 1.0, n_steps: int = 1000, theta: float = 0.5,
                            save_every: int = 1):
    """Transient one-dimensional conduction by finite volumes and the theta method in time.

    A slab of half-thickness ``R`` (symmetric about x = 0), a long cylinder or a sphere of
    radius ``R``, initially at ``T_init`` (a number or an array of cell values), is cooled
    or heated at its surface by a fluid at ``T_inf`` with coefficient ``h``.  ``h`` may be a
    function of the surface temperature, in which case it is evaluated at the start of each
    step.  ``theta = 0.5`` is the Crank–Nicolson scheme, ``theta = 1`` backward Euler.

    Returns a dict with the cell centres ``r``, the saved times ``t``, the cell
    temperatures ``T`` (one row per saved time), the surface temperatures ``T_surface``,
    the heat lost through the surface per unit area of the mid-plane (slab), per metre
    (cylinder) or in total (sphere), ``Q_lost``, and the change of stored energy ``Q_stored``
    over the same time; for a constant ``h`` the two agree to rounding error.
    """
    from scipy.linalg import solve_banded

    if geometry not in ("slab", "cylinder", "sphere"):
        raise ValueError("geometry must be 'slab', 'cylinder' or 'sphere'")
    faces = np.linspace(0.0, R, n_cells + 1)
    r = 0.5 * (faces[:-1] + faces[1:])
    dr = R / n_cells
    if geometry == "slab":
        area = np.ones_like(faces)
        vol = np.full(n_cells, dr)
    elif geometry == "cylinder":
        area = 2 * np.pi * faces
        vol = np.pi * (faces[1:] ** 2 - faces[:-1] ** 2)
    else:
        area = 4 * np.pi * faces**2
        vol = 4.0 / 3.0 * np.pi * (faces[1:] ** 3 - faces[:-1] ** 3)
    G = k * area[1:-1] / dr                       # conductances between neighbouring cells
    C = rho_c * vol                               # heat capacities
    h_of_T = h if callable(h) else (lambda Ts, h0=h: h0)

    def surface_conductance(T_last):
        hs = h_of_T(T_last)
        # half cell of conduction in series with the surface film
        return area[-1] / (0.5 * dr / k + 1.0 / hs) if hs > 0 else 0.0

    def operator(Gs):
        """Banded matrix L (3 x n) with dT/dt = (L T + b)/C."""
        ab = np.zeros((3, n_cells))
        diag = np.zeros(n_cells)
        diag[:-1] -= G
        diag[1:] -= G
        diag[-1] -= Gs
        ab[1] = diag
        ab[0, 1:] = G                             # super-diagonal
        ab[2, :-1] = G                            # sub-diagonal
        return ab

    def apply(ab, T):
        out = ab[1] * T
        out[:-1] += ab[0, 1:] * T[1:]
        out[1:] += ab[2, :-1] * T[:-1]
        return out

    dt = t_end / n_steps
    T = np.full(n_cells, float(T_init)) if np.ndim(T_init) == 0 else np.array(T_init, dtype=float)
    E0 = float(np.sum(C * T))
    Q_lost = 0.0
    times, history, surface, lost = [0.0], [T.copy()], [], [0.0]

    def T_surf(T):
        """Surface temperature from the last cell, the half cell and the film in series."""
        hs = h_of_T(T_now_surface[0])
        g_half = k / (0.5 * dr)
        return (g_half * T[-1] + hs * T_inf) / (g_half + hs)

    T_now_surface = [T[-1]]                     # surface temperature used for h(T); starts at the cell value
    T_now_surface[0] = T_surf(T)
    surface.append(T_now_surface[0])
    for step in range(1, n_steps + 1):
        Gs = surface_conductance(T_now_surface[0])
        ab = operator(Gs)
        b = np.zeros(n_cells)
        b[-1] = Gs * T_inf
        rhs = C * T + (1 - theta) * dt * (apply(ab, T) + b) + theta * dt * b
        lhs = -theta * dt * ab
        lhs[1] += C
        T_new = solve_banded((1, 1), lhs, rhs)
        # heat lost through the surface during the step, consistent with the scheme
        Q_lost += dt * Gs * ((1 - theta) * (T[-1] - T_inf) + theta * (T_new[-1] - T_inf))
        T = T_new
        T_now_surface[0] = T_surf(T)
        if step % save_every == 0 or step == n_steps:
            times.append(step * dt)
            history.append(T.copy())
            surface.append(T_now_surface[0])
            lost.append(Q_lost)
    E = np.array([np.sum(C * Th) for Th in history])
    return {
        "r": r,
        "t": np.array(times),
        "T": np.array(history),
        "T_surface": np.array(surface),
        "Q_lost": np.array(lost),
        "Q_stored": E0 - E,
    }


def enthalpy_1d_fv(faces, k_f, C_f, k_u, C_u, latent, T_freeze: float, T_init, t_end: float,
                   left=("insulated",), right=("insulated",), t_save=None, safety: float = 0.4):
    """One-dimensional conduction with freezing and thawing, by the enthalpy method.

    Each cell carries its enthalpy per unit volume H, measured from the frozen state at
    ``T_freeze``: H < 0 is frozen (T = T_freeze + H/C_f), 0 <= H <= latent is at the freezing
    temperature with liquid fraction H/latent, and H > latent is unfrozen
    (T = T_freeze + (H - latent)/C_u).  The conductivity is k_f + (k_u - k_f) times the liquid
    fraction.  Properties may be numbers or arrays (one value per cell), so layers of
    different materials are allowed; ``latent = 0`` gives plain conduction.  Time is
    explicit (forward Euler) with the largest stable step times ``safety``.

    Edges: ``("insulated",)`` or ``("temperature", T)``, with T a number or a function of
    time.  Returns a dict with the cell centres ``x``, the saved times ``t`` and
    temperatures ``T``, the frozen depth ``front`` (the frozen volume per unit area,
    counted from the left edge), and the energy balance ``balance`` (change of enthalpy
    minus the heat that crossed the edges, relative to the largest of the two).
    """
    faces = np.asarray(faces, dtype=float)
    dx = np.diff(faces)
    n = dx.size
    as_cells = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()
    k_f, C_f, k_u, C_u, latent = map(as_cells, (k_f, C_f, k_u, C_u, latent))
    T0 = as_cells(T_init)
    H = np.where(T0 < T_freeze, C_f * (T0 - T_freeze), latent + C_u * (T0 - T_freeze))

    def temperature(H):
        return np.where(H < 0, T_freeze + H / C_f,
                        np.where(H > latent, T_freeze + (H - latent) / C_u, T_freeze))

    def liquid(H):
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.where(latent > 0, H / np.where(latent > 0, latent, 1.0), (H > 0).astype(float))
        return np.clip(phi, 0.0, 1.0)

    edge_value = lambda bc, t: (bc[1](t) if callable(bc[1]) else bc[1])
    dt = safety * np.min(np.minimum(C_f, C_u) * dx**2 / np.maximum(k_f, k_u)) / 2.0
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    t_save = np.array([t_end]) if t_save is None else np.asarray(t_save, dtype=float)
    save_steps = np.clip(np.round(t_save / dt).astype(int), 0, n_steps)
    H_start = float(np.sum(H * dx))
    Q_in = 0.0
    saved_T, saved_front, saved_t = [], [], []
    xc = 0.5 * (faces[:-1] + faces[1:])

    def record(step, H):
        saved_t.append(step * dt)
        saved_T.append(temperature(H))
        saved_front.append(float(np.sum((1.0 - liquid(H)) * dx)))

    if 0 in save_steps:
        record(0, H)
    for step in range(1, n_steps + 1):
        T = temperature(H)
        k = k_f + (k_u - k_f) * liquid(H)
        G = 1.0 / (0.5 * dx[:-1] / k[:-1] + 0.5 * dx[1:] / k[1:])     # between cells
        flux = G * (T[:-1] - T[1:])                                     # left to right
        dH = np.zeros(n)
        dH[:-1] -= flux
        dH[1:] += flux
        t_mid = (step - 0.5) * dt
        q_left = q_right = 0.0
        if left[0] == "temperature":
            q_left = k[0] / (0.5 * dx[0]) * (edge_value(left, t_mid) - T[0])
        if right[0] == "temperature":
            q_right = k[-1] / (0.5 * dx[-1]) * (edge_value(right, t_mid) - T[-1])
        dH[0] += q_left
        dH[-1] += q_right
        H = H + dt * dH / dx
        Q_in += dt * (q_left + q_right)
        if step in save_steps:
            record(step, H)
    change = float(np.sum(H * dx)) - H_start
    scale = max(abs(change), abs(Q_in), 1e-300)
    return {"x": xc, "t": np.array(saved_t), "T": np.array(saved_T), "front": np.array(saved_front),
            "balance": (change - Q_in) / scale}


def similarity_bvp(Pr: float, eta_max: float = 12.0, tol: float = 1e-10):
    """Blasius and Pohlhausen similarity equations solved together by collocation.

    f''' + f f''/2 = 0 and theta'' + (Pr/2) f theta' = 0, with f(0) = f'(0) = 0, f'(inf) = 1,
    theta(0) = 0 and theta(inf) = 1, on 0 <= eta <= ``eta_max`` (enlarged for small Pr).
    Returns a dict with ``fpp0`` = f''(0), ``dtheta0`` = theta'(0) = Nu_x / Re_x^(1/2), and the
    solution object ``sol``.
    """
    from scipy.integrate import solve_bvp

    eta_end = eta_max * max(1.0, 1.0 / np.sqrt(Pr))
    eta = np.linspace(0.0, eta_end, 400)

    def rhs(e, y):
        f, fp, fpp, th, thp = y
        return np.vstack((fp, fpp, -0.5 * f * fpp, thp, -0.5 * Pr * f * thp))

    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[1] - 1.0, ya[3], yb[3] - 1.0])

    guess = np.vstack((eta - 1.72 * (1 - np.exp(-eta)), 1 - np.exp(-eta), np.exp(-eta),
                       1 - np.exp(-eta * min(1.0, np.sqrt(Pr))), np.exp(-eta)))
    sol = solve_bvp(rhs, bc, eta, guess, tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return {"fpp0": float(sol.sol(0.0)[2]), "dtheta0": float(sol.sol(0.0)[4]), "sol": sol}


def boundary_layer_march(Pr: float, x_end: float = 1.0, n_x: int = 400, n_y: int = 200, y_max: float = 12.0,
                         n_iter: int = 3, x_heat_start: float = 0.0):
    """Laminar flat-plate boundary layer by marching finite differences, without similarity.

    The boundary-layer equations, in units of the plate length L and the free-stream
    velocity U, with y stretched by Re_L^(1/2): u u_x + v u_y = u_yy, u_x + v_y = 0 and
    u T_x + v T_y = T_yy / Pr, where T = (T - T_w)/(T_inf - T_w).  The flow starts uniform at
    a small x and is marched downstream with implicit (backward) steps in x, geometrically
    spaced, solving a tridiagonal system across the layer and iterating ``n_iter`` times on
    the nonlinear terms; v follows from continuity.  ``y_max`` is in units of
    sqrt(x_end) (that is, in eta at x_end).  With ``x_heat_start`` > 0 the wall is at the
    stream temperature (T = 1) upstream of that point and heated (T = 0) beyond it.

    Returns a dict with ``x``, the skin friction ``Cf_sqrtRe`` = Cf_x Re_x^(1/2) and the heat
    transfer ``Nu_sqrtRe`` = Nu_x / Re_x^(1/2) at each x, and the profiles at x_end.
    """
    from scipy.linalg import solve_banded

    y = np.linspace(0.0, y_max * np.sqrt(x_end), n_y + 1)
    dy = y[1] - y[0]
    x = np.geomspace(1e-4 * x_end, x_end, n_x + 1)
    u = np.ones(n_y + 1); u[0] = 0.0
    T = np.ones(n_y + 1); T[0] = 0.0 if x[0] >= x_heat_start else 1.0
    v = np.zeros(n_y + 1)
    Cf, Nu = [], []

    def solve_tri(a_lower, a_diag, a_upper, rhs):
        ab = np.zeros((3, a_diag.size))
        ab[0, 1:] = a_upper[:-1]
        ab[1] = a_diag
        ab[2, :-1] = a_lower[1:]
        return solve_banded((1, 1), ab, rhs)

    for k in range(1, n_x + 1):
        dx = x[k] - x[k - 1]
        u_old, T_old = u.copy(), T.copy()
        u_new, v_new = u.copy(), v.copy()
        for _ in range(n_iter):
            # interior nodes 1..n_y-1; u(0) = 0, u(y_max) = 1
            uc, vc = np.maximum(u_new[1:-1], 1e-12), v_new[1:-1]
            lower = -vc / (2 * dy) - 1 / dy**2
            upper = vc / (2 * dy) - 1 / dy**2
            diag = uc / dx + 2 / dy**2
            rhs = uc * u_old[1:-1] / dx
            rhs[-1] -= upper[-1] * 1.0
            u_new[1:-1] = solve_tri(lower, diag, upper, rhs)
            # continuity: v_y = -u_x, integrated from the wall with the trapezoidal rule
            dudx = (u_new - u_old) / dx
            v_new = -np.concatenate(([0.0], np.cumsum(0.5 * (dudx[1:] + dudx[:-1]) * dy)))
        u, v = u_new, v_new
        uc, vc = np.maximum(u[1:-1], 1e-12), v[1:-1]
        lower = -vc / (2 * dy) - 1 / (Pr * dy**2)
        upper = vc / (2 * dy) - 1 / (Pr * dy**2)
        diag = uc / dx + 2 / (Pr * dy**2)
        T[0] = 0.0 if x[k] >= x_heat_start else 1.0
        rhs = uc * T_old[1:-1] / dx
        rhs[-1] -= upper[-1] * 1.0
        rhs[0] -= lower[0] * T[0]
        T[1:-1] = solve_tri(lower, diag, upper, rhs)
        # second-order one-sided wall derivatives
        dudy0 = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * dy)
        dTdy0 = (-3 * T[0] + 4 * T[1] - T[2]) / (2 * dy)
        Cf.append(2 * dudy0 * np.sqrt(x[k]))        # Cf_x Re_x^(1/2)
        Nu.append(dTdy0 * np.sqrt(x[k]))            # Nu_x / Re_x^(1/2)
    return {"x": x[1:], "Cf_sqrtRe": np.array(Cf), "Nu_sqrtRe": np.array(Nu), "eta": y / np.sqrt(x_end), "u": u, "T": T}


def graetz_march(zeta_end: float = 1.0, n_r: int = 100, n_x: int = 400, n: float = 1.0, wall: str = "temperature",
                 zeta_start: float = 1e-6):
    """Thermal entry region of laminar, fully developed tube flow, by marching finite volumes.

    The velocity is that of a power-law fluid, u/U = (3n+1)/(n+1) (1 - rho^((n+1)/n)), with
    n = 1 the parabola.  In terms of zeta = (x/D)/(Re Pr) the energy equation is
    u theta_zeta = (4/rho) d/drho(rho d theta/drho).  The inlet is at theta = 1.  With
    ``wall="temperature"`` the wall is at theta = 0; with ``wall="flux"`` a uniform flux enters
    (d theta/drho = 1 at rho = 1).  Cells are uniform in rho; steps in zeta are geometric and
    Crank-Nicolson.

    Returns a dict with ``zeta``, the mean (mixing-cup) temperature ``theta_m``, the local
    Nusselt number ``Nu_x`` (based on D), and the final profile.
    """
    from scipy.linalg import solve_banded

    faces = np.linspace(0.0, 1.0, n_r + 1)
    rho = 0.5 * (faces[:-1] + faces[1:])
    drho = 1.0 / n_r
    vol = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)            # rho drho per cell
    u = (3 * n + 1) / (n + 1) * (1 - rho ** ((n + 1) / n))
    cap = u * vol                                              # "capacity" per cell
    G = 4.0 * faces[1:-1] / drho                               # conductances between cells
    if wall == "temperature":
        G_wall = 4.0 * 1.0 / (0.5 * drho)
    elif wall != "flux":
        raise ValueError("wall must be 'temperature' or 'flux'")
    ab = np.zeros((3, n_r))
    ab[1, :-1] -= G
    ab[1, 1:] -= G
    ab[0, 1:] = G
    ab[2, :-1] = G
    source = np.zeros(n_r)
    if wall == "temperature":
        ab[1, -1] -= G_wall                                    # wall at theta = 0
    else:
        source[-1] = 4.0 * 1.0                                 # 4 rho dtheta/drho at rho = 1

    def apply(T):
        out = ab[1] * T
        out[:-1] += ab[0, 1:] * T[1:]
        out[1:] += ab[2, :-1] * T[:-1]
        return out + source

    zeta = np.concatenate(([0.0], np.geomspace(zeta_start, zeta_end, n_x)))
    T = np.ones(n_r)
    Tm, Nu = [1.0], [np.nan]
    for k in range(1, zeta.size):
        dz = zeta[k] - zeta[k - 1]
        lhs = -0.5 * dz * ab
        lhs[1] += cap
        T = solve_banded((1, 1), lhs, cap * T + 0.5 * dz * apply(T) + 0.5 * dz * source)
        T_mean = float(np.sum(u * T * vol) / np.sum(u * vol))
        if wall == "temperature":
            grad = -T[-1] / (0.5 * drho)                       # d theta/drho at the wall
            Nu.append(-2.0 * grad / T_mean)
        else:
            T_wall = T[-1] + 0.5 * drho * 1.0
            Nu.append(2.0 * 1.0 / (T_wall - T_mean))
        Tm.append(T_mean)
    return {"zeta": zeta, "theta_m": np.array(Tm), "Nu_x": np.array(Nu), "rho": rho, "theta": T}


def free_convection_bvp(Pr: float, eta_max: float | None = None, tol: float = 1e-10):
    """Ostrach's free-convection similarity equations solved by collocation.

    F''' + 3 F F'' - 2 F'^2 + theta = 0 and theta'' + 3 Pr F theta' = 0 with F(0) = F'(0) = 0,
    theta(0) = 1 and F'(eta_max) = theta(eta_max) = 0, on a domain of length ``eta_max``
    (by default 40 Pr^(-1/2) for Pr < 1 and 40 Pr^(1/4) above).  Returns F''(0), G = -theta'(0)
    and the solution object.  Outside 0.01 <= Pr <= 100 the solution is continued from the
    nearer end of that range.
    """
    from scipy.integrate import solve_bvp

    def solve(P, guess_sol=None):
        E = eta_max if eta_max is not None else 40.0 * max(P ** -0.5, P ** 0.25)
        eta = np.linspace(0.0, E, 600)

        def rhs(e, y):
            F, Fp, Fpp, th, thp = y
            return np.vstack((Fp, Fpp, -3 * F * Fpp + 2 * Fp**2 - th, thp, -3 * P * F * thp))

        def bc(ya, yb):
            return np.array([ya[0], ya[1], ya[3] - 1.0, yb[1], yb[3]])

        if guess_sol is None:
            guess = np.vstack((0.3 * (1 - np.exp(-eta)) ** 2, 0.6 * eta * np.exp(-eta), 0.6 * (1 - eta) * np.exp(-eta),
                               np.exp(-eta), -np.exp(-eta)))
        else:
            guess = guess_sol.sol(np.minimum(eta, guess_sol.x[-1]))
        sol = solve_bvp(rhs, bc, eta, guess, tol=tol, max_nodes=300000)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    if 0.01 <= Pr <= 100.0:
        sol = solve(Pr)
    elif Pr < 0.01:
        sol = solve(0.01)
        for P in np.geomspace(0.01, Pr, 6)[1:]:
            sol = solve(P, sol)
    else:
        sol = solve(100.0)
        for P in np.geomspace(100.0, Pr, 6)[1:]:
            sol = solve(P, sol)
    return {"fpp0": float(sol.sol(0.0)[2]), "G": float(-sol.sol(0.0)[4]), "sol": sol}


def free_convection_limits(tol: float = 1e-10):
    """Large-Prandtl-number limit of the free-convection similarity solution.

    For Pr -> infinity the thermal layer is thin and inertia drops out of it: with
    eta = Pr^(-1/4) z and F = Pr^(-3/4) g(z), g''' + theta = 0 and theta'' + 3 g theta' = 0,
    with g(0) = g'(0) = 0, theta(0) = 1, and g''(inf) = theta(inf) = 0 (the shear stress of the
    thin layer vanishes at its edge).  Returns the constant c in -theta'(0) = c Pr^(1/4).
    """
    from scipy.integrate import solve_bvp

    z = np.linspace(0.0, 20.0, 400)

    def rhs(e, y):
        g, gp, gpp, th, thp = y
        return np.vstack((gp, gpp, -th, thp, -3 * g * thp))

    def bc(ya, yb):
        return np.array([ya[0], ya[1], ya[3] - 1.0, yb[2], yb[3]])

    guess = np.vstack((0.5 * z, 0.5 * np.ones_like(z), np.exp(-z), np.exp(-z), -np.exp(-z)))
    sol = solve_bvp(rhs, bc, z, guess, tol=tol, max_nodes=300000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return {"large_Pr": float(-sol.sol(0.0)[4])}


def free_convection_march(Pr: float, xi_start: float = 1e-8, n_s: int = 400, n_eta: int = 300,
                          eta_max: float | None = None, n_iter: int = 10):
    """Laminar free convection on an isothermal vertical plate by marching, without similarity.

    The boundary-layer equations are written in the coordinates xi = x/L and
    eta = (y/x)(Gr_x/4)^(1/4), with psi = 4 nu (Gr_x/4)^(1/4) F(xi, eta) and U = dF/d eta:

        U'' + 3 F U' - 2 U^2 + theta = 4 xi (U dU/dxi - U' dF/dxi),
        theta''/Pr + 3 F theta' = 4 xi (U dtheta/dxi - theta' dF/dxi).

    Nothing is assumed about the xi dependence.  The march starts at ``xi_start`` from rough
    polynomial profiles (not the similarity solution), steps implicitly in ln xi, solves a
    tridiagonal system across the layer for U and for theta, iterates ``n_iter`` times on the
    nonlinear terms, and integrates F from U.  Returns xi, the local G = -dtheta/deta at the
    wall (Nu_x = G (Gr_x/4)^(1/4)), and the profiles at xi = 1.  Tested for air (Pr near 0.7);
    at Pr of order 10 the fixed-point iteration converges poorly.
    """
    from scipy.linalg import solve_banded

    if eta_max is None:
        eta_max = 12.0 * max(Pr ** -0.5, Pr ** 0.25)
    eta = np.linspace(0.0, eta_max, n_eta + 1)
    h = eta[1] - eta[0]
    s = np.linspace(np.log(xi_start), 0.0, n_s + 1)
    z = np.clip(eta / (6.0 * max(1.0, Pr ** -0.5)), 0.0, 1.0)
    U = 0.9 * z * (1 - z) ** 2
    T = (1 - z) ** 2

    def integrate(U):
        return np.concatenate(([0.0], np.cumsum(0.5 * (U[1:] + U[:-1]) * h)))

    def solve_tri(lo, di, up, rhs):
        ab = np.zeros((3, di.size))
        ab[0, 1:] = up[:-1]
        ab[1] = di
        ab[2, :-1] = lo[1:]
        return solve_banded((1, 1), ab, rhs)

    F = integrate(U)
    G = []
    for k in range(1, n_s + 1):
        ds = s[k] - s[k - 1]
        U_prev, T_prev, F_prev = U.copy(), T.copy(), F.copy()
        for _ in range(n_iter):
            c = (3 * F + 4 * (F - F_prev) / ds)[1:-1]
            Ub = np.maximum(U[1:-1], 0.0)
            lo, up = 1 / h**2 - c / (2 * h), 1 / h**2 + c / (2 * h)
            di = -2 / h**2 - 2 * Ub - 4 * Ub / ds
            U = U.copy()
            U[1:-1] = solve_tri(lo, di, up, -T[1:-1] - 4 * Ub * U_prev[1:-1] / ds)
            lo, up = 1 / (Pr * h**2) - c / (2 * h), 1 / (Pr * h**2) + c / (2 * h)
            di = -2 / (Pr * h**2) - 4 * Ub / ds
            rhs = -4 * Ub * T_prev[1:-1] / ds
            rhs[0] -= lo[0] * 1.0
            T = T.copy()
            T[1:-1] = solve_tri(lo, di, up, rhs)
            F = integrate(U)
        G.append((3 * T[0] - 4 * T[1] + T[2]) / (2 * h))
    return {"xi": np.exp(s[1:]), "G": np.array(G), "eta": eta, "U": U, "theta": T}


def film_condensation_bvp(Ja: float, tol: float = 1e-10):
    """Subcooling of a laminar condensate film by collocation, with the eigenvalue as an unknown.

    theta'' + lam f theta' = 0, f = eta^2/2 - eta^3/6, with theta(0) = 0, theta(1) = 1 and
    theta'(1) = lam/(3 Ja), solved by ``solve_bvp`` with lam as a free parameter.  Returns
    lam, the wall gradient c = theta'(0) and h_fg'/h_fg = c^4 (3 Ja/lam).
    """
    from scipy.integrate import solve_bvp

    eta = np.linspace(0.0, 1.0, 50)

    def rhs(e, y, p):
        return np.vstack((y[1], -p[0] * (e**2 / 2 - e**3 / 6) * y[1]))

    def bc(ya, yb, p):
        return np.array([ya[0], yb[0] - 1.0, yb[1] - p[0] / (3.0 * Ja)])

    lam0 = 3.0 * Ja / (1.0 + 0.4 * Ja)            # a rough start; the exact eigenvalue is a little smaller than 3 Ja
    sol = solve_bvp(rhs, bc, eta, np.vstack((eta, np.ones_like(eta))), p=[lam0], tol=tol, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(sol.message)
    lam, c = float(sol.p[0]), float(sol.sol(0.0)[1])
    return {"lam": lam, "c": c, "ratio": c**4 * 3.0 * Ja / lam}


def film_condensation_march(Ja: float, s_span: float = 15.0, n_s: int = 300, n_eta: int = 100, n_iter: int = 6):
    """Laminar film condensation on a vertical plate by marching down the plate, without similarity.

    With Nusselt's velocity profile across a film of local thickness delta(x), the energy
    equation of the film becomes, in eta = y/delta and s = ln x,

        theta'' + 3 Ja g f theta' = 4 Ja d f' d theta/ds,    d(d)/ds + d = g,   g = theta'(1),

    where d = delta^4 / (x 4 mu k dT/(rho g drho h_fg)) is the scaled fourth power of the film
    thickness (d = 1 is Nusselt's film) and the interface condition k dT/delta = h_fg dGamma/dx
    gives the second equation.  The march starts from Nusselt's linear profile and d = 1 and
    steps implicitly in s over ``s_span`` (a factor e^s_span in distance down the plate).
    Returns s, the local ratio h_fg'/h_fg = theta'(0)^4 / d at each step, and the final profile.
    """
    from scipy.linalg import solve_banded

    eta = np.linspace(0.0, 1.0, n_eta + 1)
    h = eta[1] - eta[0]
    f, fp = eta**2 / 2 - eta**3 / 6, eta - eta**2 / 2
    theta, d = eta.copy(), 1.0
    s = np.linspace(0.0, s_span, n_s + 1)
    ds = s[1] - s[0]
    ratio = []
    for _ in range(n_s):
        theta_prev, d_prev = theta.copy(), d
        for _ in range(n_iter):
            g = (3 * theta[-1] - 4 * theta[-2] + theta[-3]) / (2 * h)
            d = (d_prev + ds * g) / (1.0 + ds)
            a = 3 * Ja * g * f[1:-1]
            lo, up = 1 / h**2 - a / (2 * h), 1 / h**2 + a / (2 * h)
            di = -2 / h**2 - 4 * Ja * d * fp[1:-1] / ds
            rhs = -4 * Ja * d * fp[1:-1] * theta_prev[1:-1] / ds
            rhs[-1] -= up[-1] * 1.0
            ab = np.zeros((3, n_eta - 1))
            ab[0, 1:], ab[1], ab[2, :-1] = up[:-1], di, lo[1:]
            theta = theta.copy()
            theta[1:-1] = solve_banded((1, 1), ab, rhs)
        c = (-3 * theta[0] + 4 * theta[1] - theta[2]) / (2 * h)
        ratio.append(c**4 / d)
    return {"s": s[1:], "ratio": np.array(ratio), "eta": eta, "theta": theta, "d": d}


def crossflow_cells(NTU: float, Cr: float, n: int = 100, C_min: str = "x"):
    """Single-pass crossflow exchanger, both fluids unmixed, by a grid of n x n small exchangers.

    The stream that flows along x (rows) enters at x = 0 with temperature 1, the stream that
    flows along y (columns) enters at y = 0 with temperature 0.  ``C_min`` says which stream
    has the smaller capacity rate; NTU = UA/C_min and Cr = C_min/C_max.  Each cell is a small
    exchanger whose heat flow uses the means of its inlet and outlet temperatures (second
    order); the cells are swept column by column.  Returns the effectiveness and the
    temperature fields Tx (stream along x) and Ty (stream along y) at the cell outlets, indexed
    [row (y), column (x)].
    """
    if C_min not in ("x", "y"):
        raise ValueError("C_min must be 'x' or 'y'.")
    Cx, Cy = (1.0, 1.0 / Cr) if C_min == "x" else (1.0 / Cr, 1.0)
    ua = NTU / n**2                      # in units of C_min
    cx, cy = Cx / n, Cy / n              # capacity of one row and of one column
    Tx = np.ones(n)
    Tx_field, Ty_field = np.zeros((n, n)), np.zeros((n, n))
    factor = ua / (1.0 + ua / (2 * cx) + ua / (2 * cy))
    for j in range(n):
        ty = 0.0
        for i in range(n):
            q = factor * (Tx[i] - ty)
            Tx[i] -= q / cx
            ty += q / cy
            Tx_field[i, j], Ty_field[i, j] = Tx[i], ty
    Q = Cx * (1.0 - Tx.mean())
    return {"eps": Q / 1.0, "Tx": Tx_field, "Ty": Ty_field}


def shell_and_tube_1_2_bvp(NTU: float, Cr: float, shell: str = "min", tol: float = 1e-10):
    """One shell pass and two tube passes, by collocation of the three-stream equations.

    The shell fluid, mixed across the shell, flows from x = 0 to 1; tube pass 1 flows from
    x = 0 to 1 and pass 2 back, each with half of the area.  The hot stream enters at 1 and the
    cold at 0; ``shell`` says whether the shell carries C_min ('min') or C_max ('max').
    Returns the effectiveness and the solution.
    """
    from scipy.integrate import solve_bvp

    C_s, C_t = (1.0, 1.0 / Cr) if shell == "min" else (1.0 / Cr, 1.0)
    UA = NTU                                     # in units of C_min
    x = np.linspace(0.0, 1.0, 50)

    def rhs(x, y):
        Ts, T1, T2 = y
        return np.vstack((-(UA / 2) * ((Ts - T1) + (Ts - T2)) / C_s, (UA / 2) * (Ts - T1) / C_t, -(UA / 2) * (Ts - T2) / C_t))

    def bc(ya, yb):          # shell hot, entering at x = 0; tubes cold, entering pass 1 at x = 0
        return np.array([ya[0] - 1.0, ya[1], yb[2] - yb[1]])

    guess = np.vstack((1 - 0.5 * x, 0.3 * x, 0.3 + 0.0 * x))
    sol = solve_bvp(rhs, bc, x, guess, tol=tol, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(sol.message)
    Ts_out = float(sol.sol(1.0)[0])
    Q = C_s * (1.0 - Ts_out)
    return {"eps": Q / 1.0, "sol": sol}


# ---------------------------------------------------------------- Chapter 13: view factors and enclosures

def _cosine_directions(rng, n):
    """Diffuse (cosine-weighted) directions about the local +z axis: (sin t cos p, sin t sin p, cos t)."""
    u1, u2 = rng.random(n), rng.random(n)
    st = np.sqrt(u1)
    ph = 2.0 * np.pi * u2
    return st * np.cos(ph), st * np.sin(ph), np.sqrt(1.0 - u1)


def mc_box_exchange(Lx, Ly, Lz, eps, emit_face, n_bundles=1_000_000, seed=0, max_bounces=10_000):
    """Monte Carlo ray tracing inside a rectangular box with diffuse gray faces.

    Bundles leave face ``emit_face`` (faces ordered as ``radiation.BOX_FACES``) from
    uniformly distributed points in diffuse directions.  At each face they hit they are
    absorbed with probability eps[face] and otherwise reflected diffusely from the hit
    point; eps = 0 is a perfect diffuse reflector (a reradiating wall), eps = 1 a black
    face.  Returns the fraction of the bundles absorbed by each face and the standard
    error of each fraction.  With all eps = 1 the fractions are the view factors.
    """
    rng = np.random.default_rng(seed)
    L = np.array([Lx, Ly, Lz], dtype=float)
    eps = np.asarray(eps, dtype=float)
    # face -> (normal axis, position 0 or 1, inward sign)
    face_axis = np.array([2, 2, 0, 0, 1, 1])
    face_side = np.array([0, 1, 0, 1, 0, 1])

    def emit(face, n):
        ax, side = face_axis[face], face_side[face]
        p = rng.random((n, 3)) * L
        p[:, ax] = side * L[ax]
        a, b, c = _cosine_directions(rng, n)
        d = np.empty((n, 3))
        t1, t2 = [k for k in range(3) if k != ax]
        d[:, t1], d[:, t2] = a, b
        d[:, ax] = c if side == 0 else -c
        return p, d

    counts = np.zeros(6)
    p, d = emit(emit_face, n_bundles)
    for _ in range(max_bounces):
        if len(p) == 0:
            break
        # distance to the planes x=0/Lx, y=0/Ly, z=0/Lz along d
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hi = np.where(d > 0, (L - p) / d, np.inf)
            t_lo = np.where(d < 0, -p / d, np.inf)
        t_all = np.minimum(t_hi, t_lo)
        ax = np.argmin(t_all, axis=1)
        t = t_all[np.arange(len(p)), ax]
        side = (d[np.arange(len(p)), ax] > 0).astype(int)
        hit = np.where(ax == 2, 0, np.where(ax == 0, 2, 4)) + side
        p = p + t[:, None] * d
        absorbed = rng.random(len(p)) < eps[hit]
        counts += np.bincount(hit[absorbed], minlength=6)
        keep = ~absorbed
        p, hit = p[keep], hit[keep]
        # diffuse reflection from the hit point
        n = len(p)
        a, b, c = _cosine_directions(rng, n)
        d = np.empty((n, 3))
        axh = face_axis[hit]
        for k in range(3):
            others = [m for m in range(3) if m != k]
            sel = axh == k
            d[sel, others[0]] = a[sel]
            d[sel, others[1]] = b[sel]
            d[sel, k] = np.where(face_side[hit[sel]] == 0, c[sel], -c[sel])
        p[np.arange(n), axh] = face_side[hit] * L[axh]
    frac = counts / n_bundles
    return frac, np.sqrt(frac * (1.0 - frac) / n_bundles)


def view_factor_gauss_rectangles(rect1, rect2, n=16):
    """View factor F_12 between two plane rectangles by product Gauss-Legendre quadrature.

    Each rectangle is (origin, u, v): the corner point and its two edge vectors in 3-D.
    The four-fold integral of cos(t1) cos(t2) / (pi r^2) over both areas, Eq. (13.1), is
    evaluated with n nodes along each of the four edge directions.  The rectangles must
    see each other completely (no blocking, both facing each other).
    """
    x, w = np.polynomial.legendre.leggauss(n)
    s, ws = 0.5 * (x + 1.0), 0.5 * w

    def points(rect):
        o, u, v = (np.asarray(q, dtype=float) for q in rect)
        P = o + s[:, None, None] * u + s[None, :, None] * v
        W = np.outer(ws, ws) * np.linalg.norm(np.cross(u, v))
        nrm = np.cross(u, v) / np.linalg.norm(np.cross(u, v))
        return P.reshape(-1, 3), W.ravel(), nrm, np.linalg.norm(np.cross(u, v))

    P1, W1, n1, A1 = points(rect1)
    P2, W2, n2, _ = points(rect2)
    D = P2[None, :, :] - P1[:, None, :]
    r2 = np.einsum("ijk,ijk->ij", D, D)
    c1 = np.abs(D @ n1)
    c2 = np.abs(D @ n2)
    K = c1 * c2 / (np.pi * r2 * r2)
    return float(W1 @ K @ W2 / A1)


def view_factor_gauss_strips(a1, b1, a2, b2, n=16):
    """2-D view factor between two long straight strips by Gauss-Legendre quadrature.

    F_12 = (1/L_1) int int cos(t1) cos(t2) / (2 r) ds_1 ds_2, the two-dimensional form of
    Eq. (13.1); a1, b1 and a2, b2 are the end points of the strips.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    s, ws = 0.5 * (x + 1.0), 0.5 * w
    a1, b1, a2, b2 = (np.asarray(p, dtype=float) for p in (a1, b1, a2, b2))
    L1, L2 = np.hypot(*(b1 - a1)), np.hypot(*(b2 - a2))
    t1, t2 = (b1 - a1) / L1, (b2 - a2) / L2
    n1, n2 = np.array([-t1[1], t1[0]]), np.array([-t2[1], t2[0]])
    P1 = a1 + np.outer(s, b1 - a1)
    P2 = a2 + np.outer(s, b2 - a2)
    D = P2[None, :, :] - P1[:, None, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", D, D))
    K = np.abs(D @ n1) * np.abs(D @ n2) / (2.0 * r ** 3)
    return float((ws * L1) @ K @ (ws * L2) / L1)


def mc_cylinder_cavity(depth, radius, eps, n_bundles=1_000_000, seed=0, max_bounces=10_000):
    """Apparent emissivity of an isothermal cylindrical cavity by Monte Carlo ray tracing.

    The cavity has a closed flat bottom at z = 0, a side wall r = radius and an open
    mouth at z = depth.  Bundles enter through the mouth from a diffuse outside source,
    are absorbed at each wall hit with probability eps and are otherwise reflected
    diffusely; a bundle that reaches the mouth again escapes.  By reciprocity the
    absorbed fraction equals the apparent emissivity of the mouth.  Returns the fraction
    and its standard error.
    """
    rng = np.random.default_rng(seed)
    R, H = float(radius), float(depth)
    n0 = n_bundles
    rr = R * np.sqrt(rng.random(n0))
    th = 2.0 * np.pi * rng.random(n0)
    p = np.column_stack([rr * np.cos(th), rr * np.sin(th), np.full(n0, H)])
    a, b, c = _cosine_directions(rng, n0)
    d = np.column_stack([a, b, -c])
    absorbed_total = 0
    for _ in range(max_bounces):
        if len(p) == 0:
            break
        # side wall: |p_xy + t d_xy| = R
        A2 = d[:, 0] ** 2 + d[:, 1] ** 2
        B = p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1]
        C = p[:, 0] ** 2 + p[:, 1] ** 2 - R * R
        with np.errstate(divide="ignore", invalid="ignore"):
            t_side = np.where(A2 > 1e-300, (-B + np.sqrt(np.maximum(B * B - A2 * C, 0.0))) / A2, np.inf)
            t_bot = np.where(d[:, 2] < 0, -p[:, 2] / d[:, 2], np.inf)
            t_top = np.where(d[:, 2] > 0, (H - p[:, 2]) / d[:, 2], np.inf)
        t = np.minimum(np.minimum(t_side, t_bot), t_top)
        which = np.where(t == t_top, 2, np.where(t == t_bot, 1, 0))     # 0 side, 1 bottom, 2 mouth
        p = p + t[:, None] * d
        stay = which != 2
        p, which = p[stay], which[stay]
        absorbed = rng.random(len(p)) < eps
        absorbed_total += int(absorbed.sum())
        p, which = p[~absorbed], which[~absorbed]
        n = len(p)
        a, b, c = _cosine_directions(rng, n)
        d = np.empty((n, 3))
        bot = which == 1
        d[bot] = np.column_stack([a[bot], b[bot], c[bot]])
        p[bot, 2] = 0.0
        sd = ~bot
        # local frame on the side wall: inward normal -e_r, tangents e_phi and e_z
        phi = np.arctan2(p[sd, 1], p[sd, 0])
        er = np.column_stack([np.cos(phi), np.sin(phi), np.zeros(sd.sum())])
        ep = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(sd.sum())])
        ez = np.array([0.0, 0.0, 1.0])
        d[sd] = -c[sd, None] * er + a[sd, None] * ep + b[sd, None] * ez
        p[sd, 0:2] = R * (1.0 - 1e-12) * er[:, 0:2]
    frac = absorbed_total / n0
    return frac, float(np.sqrt(frac * (1.0 - frac) / n0))


# ---------------------------------------------------------------- Chapter 14: molecular diffusion

def stefan_tube_exact(x_s, x_top, r):
    """Exact similarity solution of the unsteady Stefan tube with a falling liquid surface.

    The gas (constant total concentration c) fills 0 < z < s(t) above a liquid of molar
    density r c; the liquid level starts at the mouth.  With eta = z / (2 sqrt(D t)) the
    profile is x = x_top + C [erf(eta + a) - erf(a)], a = (r - 1) beta, and the
    level is s = 2 beta sqrt(D t), where beta solves
    x_s - x_top = r beta (1 - x_s) sqrt(pi) [erfcx(a) exp((2 r - 1) beta^2) - erfcx(r beta)].
    Returns beta and the profile as a function of eta.
    """
    from scipy.optimize import brentq
    from scipy.special import erfcx

    def g(b):
        a = (r - 1.0) * b
        return r * b * (1.0 - x_s) * math.sqrt(math.pi) * (erfcx(a) * math.exp((2.0 * r - 1.0) * b * b) - erfcx(r * b)) - (x_s - x_top)

    b_qs = math.sqrt(math.log((1.0 - x_top) / (1.0 - x_s)) / (2.0 * r))
    beta = brentq(g, 0.2 * b_qs, 5.0 * b_qs, xtol=1e-17, rtol=1e-15)
    a = (r - 1.0) * beta
    A = r * beta * (1.0 - x_s) * math.sqrt(math.pi)

    def profile(eta):              # written with erfcx to avoid overflow when r is large
        u = np.asarray(eta, dtype=float) + a
        return x_top + A * (erfcx(a) * np.exp((2.0 * r - 1.0) * beta * beta)
                            - erfcx(u) * np.exp(r * r * beta * beta - u * u))
    return beta, profile


def stefan_tube_march(x_s, x_top, r, D, s0, t_span, n=100, x0=None, t_eval=None, rtol=1e-10):
    """Unsteady Stefan tube by the method of lines on a grid fixed to the moving level.

    With xi = z/s(t), the vapour equation x_t + v x_z = D x_zz (v the molar-average velocity,
    v = -(r - 1) ds/dt) becomes x_t = D x_xixi / s^2 + (xi s'/s - v/s) x_xi on 0 < xi < 1, with
    x = x_top at the mouth, x = x_s at the liquid, and the level moving as
    D x_z = r (1 - x_s) ds/dt.  Central differences (second order) and a BDF integrator.
    ``x0`` is the initial profile on the n + 1 nodes (default: air, x = x_top, above the
    liquid).  Returns the scipy solution with y[-1] = s(t).
    """
    from scipy.integrate import solve_ivp

    xi = np.linspace(0.0, 1.0, n + 1)
    h = 1.0 / n
    if x0 is None:
        x0 = np.full(n + 1, x_top)
    y0 = np.concatenate([np.asarray(x0, dtype=float)[1:-1], [s0]])

    def rhs(t, y):
        s = y[-1]
        x = np.concatenate([[x_top], y[:-1], [x_s]])
        dx_end = (3.0 * x[-1] - 4.0 * x[-2] + x[-3]) / (2.0 * h)
        sdot = D * dx_end / (s * r * (1.0 - x_s))
        v = -(r - 1.0) * sdot
        dxi = (x[2:] - x[:-2]) / (2.0 * h)
        d2 = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / h ** 2
        return np.concatenate([D / s ** 2 * d2 + (xi[1:-1] * sdot / s - v / s) * dxi, [sdot]])

    return solve_ivp(rhs, t_span, y0, method="BDF", rtol=rtol, atol=1e-14, t_eval=t_eval, dense_output=True)


# ---------------------------------------------------------------- Chapter 15: diffusion with reaction

def pellet_bvp(phi, shape="slab", order=1.0, tol=1e-10):
    """Effectiveness factor of an n-th order reaction in a pellet by collocation (solve_bvp).

    Same problem as ``diffusion.pellet_fd``: u'' + (s/x) u' = Phi^2 u^order, u'(0) = 0, u(1) = 1,
    Phi = (s + 1) phi.  The singular term at x = 0 is handled by solve_bvp's S matrix.  Returns
    eta = (s + 1) u'(1)/Phi^2 and the solution.
    """
    from scipy.integrate import solve_bvp

    s = {"slab": 0, "cylinder": 1, "sphere": 2}[shape]
    Phi = (s + 1.0) * float(phi)
    S = np.array([[0.0, 0.0], [0.0, -float(s)]])

    def f(x, y):
        return np.vstack([y[1], Phi ** 2 * np.maximum(y[0], 0.0) ** order])

    def bc(ya, yb):
        return np.array([ya[1], yb[0] - 1.0])

    x = np.linspace(0.0, 1.0, 101)
    guess = np.vstack([np.exp(-Phi * (1.0 - x)), Phi * np.exp(-Phi * (1.0 - x))])
    sol = solve_bvp(f, bc, x, guess, S=S if s > 0 else None, tol=tol, max_nodes=200000)
    return {"eta": float((s + 1.0) * sol.sol(1.0)[1] / Phi ** 2), "sol": sol, "success": bool(sol.success)}


def one_phase_front_exact(St):
    """Sharp reacting front with storage of the diffusing gas in the reacted layer (a one-phase
    Stefan problem): front x = 2 lam sqrt(D t / eps), with lam e^(lam^2) erf(lam) = St/sqrt(pi),
    St = eps c0 / a.  Returns lam."""
    from scipy.optimize import brentq
    from scipy.special import erf

    g = lambda lam: lam * math.exp(lam * lam) * erf(lam) - St / math.sqrt(math.pi)
    return brentq(g, 1e-12, 10.0, xtol=1e-15)


def reacting_front_march(Da, St, x_max=1.0, t_end=1.0, n=400, t_eval=None):
    """Gas diffusing into a porous solid and reacting with it at a finite rate, method of lines.

    Dimensionless: eps dc/dt = d2c/dx2 - Da c s, ds/dt = -St Da c s / eps ... written as
        dc/dt = c_xx - Da c s,   ds/dt = -St Da c s,
    with time in units of eps L^2/D (L the length scale of x), c = 1 at x = 0, dc/dx = 0 at
    x_max, c = 0 and s = 1 initially.  Da = k a L^2/D compares reaction with diffusion; for
    large Da the reaction zone is thin and the front follows the one-phase Stefan problem with
    St = eps c0/a.  Returns the scipy solution; y[:n] is c at the nodes 1..n, y[n:] is s.
    """
    from scipy.integrate import solve_ivp
    from scipy.sparse import diags, bmat

    x = np.linspace(0.0, x_max, n + 1)
    h = x[1] - x[0]
    D2 = diags([np.ones(n - 1), -2.0 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]).tolil() / h ** 2
    D2[n - 1, n - 2] = 2.0 / h ** 2                   # mirror node at x_max: zero flux
    D2 = D2.tocsr()
    bnd = np.zeros(n)
    bnd[0] = 1.0 / h ** 2                             # c = 1 at x = 0

    def rhs(t, y):
        c, s = y[:n], y[n:]
        r = Da * c * s
        return np.concatenate([D2 @ c + bnd - r, -St * r])

    def jac(t, y):
        c, s = y[:n], y[n:]
        return bmat([[D2 - diags(Da * s), -diags(Da * c)], [diags(-St * Da * s), diags(-St * Da * c)]]).tocsc()

    y0 = np.concatenate([np.zeros(n), np.ones(n)])
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="BDF", jac=jac, rtol=1e-7, atol=1e-10, t_eval=t_eval)
    sol.x = x[1:]
    return sol


def species_similarity(Sc: float, B: float = 0.0, eta_max: float = 14.0, tol: float = 1e-10):
    """Laminar flat-plate boundary layer with a species leaving (or entering) the wall, solved
    by collocation without a film model.

    f''' + f f''/2 = 0 and phi'' + (Sc/2) f phi' = 0, phi = (m - m_e)/(m_s - m_e), with
    f'(0) = 0, f'(inf) = 1, phi(0) = 1, phi(inf) = 0, and the wall velocity set by the species
    balance at an interface that only the transferred species crosses:
    f(0) = (2 B / Sc) phi'(0), with B = (m_s - m_e)/(1 - m_s) (blowing for B > 0, suction for
    B < 0).  The case B = 0 is the low-rate analogy of the Pohlhausen problem.  Large |B| is
    reached by continuation from B = 0.  Returns ``fpp0`` = f''(0), ``dphi0`` = phi'(0)
    (= -Sh_x/Re_x^(1/2) at the same B), ``f0`` and the solution ``sol``.
    """
    from scipy.integrate import solve_bvp

    eta_end = eta_max * max(1.0, 1.0 / np.sqrt(Sc)) * (1.0 + max(B, 0.0) ** 0.5)
    eta = np.linspace(0.0, eta_end, 500)

    def rhs(e, y):
        f, fp, fpp, ph, php = y
        return np.vstack((fp, fpp, -0.5 * f * fpp, php, -0.5 * Sc * f * php))

    sol = None
    steps = [0.0] if B == 0.0 else np.concatenate(([0.0], B * np.linspace(0.05, 1.0, 20) ** 2))
    for b in steps:
        def bc(ya, yb, b=b):
            return np.array([ya[0] - 2.0 * b / Sc * ya[4], ya[1], yb[1] - 1.0, ya[3] - 1.0, yb[3]])
        if sol is None:
            guess = np.vstack((eta - 1.72 * (1 - np.exp(-eta)), 1 - np.exp(-eta), np.exp(-eta),
                               np.exp(-eta * min(1.0, np.sqrt(Sc)) * 0.5), -0.5 * np.exp(-eta * 0.5)))
            sol = solve_bvp(rhs, bc, eta, guess, tol=tol, max_nodes=300000)
        else:
            sol = solve_bvp(rhs, bc, sol.x, sol.y, tol=tol, max_nodes=300000)
        if not sol.success:
            raise RuntimeError(sol.message)
    y0 = sol.sol(0.0)
    return {"fpp0": float(y0[2]), "dphi0": float(y0[4]), "f0": float(y0[0]), "sol": sol}


def drop_evaporation_transient(B: float, eps: float, s_max: float = math.log(1000.0), n: int = 400,
                               R2_end: float = 0.01, rtol: float = 1e-8):
    """Evaporation of a drop at rest by unsteady diffusion and Stefan flow in the gas, without
    the quasi-steady assumption.

    The drop of radius R(t) (initially R0) is put into vapour-free gas at t = 0; the vapour
    mass fraction is m_s at its surface, and the gas density rho_g and diffusivity D are
    constant; ``eps`` = rho_g / rho_l.  With s = ln(r/R) and the stretched time
    d tau = D dt / R^2, the vapour equation is

        m_tau = e^(-2s) (m_ss + m_s) - G [eps + (1 - eps) e^(-3s)] m_s,

    where G = -m_s(0)/(1 - m_s) gives the evaporation rate mdot = 4 pi R rho_g D G, and the
    radius follows d ln R^2 / d tau = -2 eps G.  The far boundary s_max stays at m = m_e = 0;
    B = m_s/(1 - m_s).  The method of lines (second-order differences, n cells) is integrated
    with BDF until R^2/R0^2 = ``R2_end``.

    Returns a dict with ``t`` (in units of R0^2/D), ``R2`` = R^2/R0^2, ``G`` and the
    quasi-steady value ``G_qs`` = ln(1+B)/(1 - e^(-s_max)) for the same truncated domain.
    """
    from scipy.integrate import solve_ivp
    from scipy.sparse import lil_matrix

    m_s = B / (1.0 + B)
    s = np.linspace(0.0, s_max, n + 1)
    h = s[1] - s[0]
    e2, e3 = np.exp(-2.0 * s[1:-1]), np.exp(-3.0 * s[1:-1])

    def grad0(m_in):
        return (-3.0 * m_s + 4.0 * m_in[0] - m_in[1]) / (2.0 * h)

    def rhs(tau, y):
        m_in = y[:-1]
        m = np.concatenate(([m_s], m_in, [0.0]))
        G = -grad0(m_in) / (1.0 - m_s)
        ms = (m[2:] - m[:-2]) / (2.0 * h)
        mss = (m[2:] - 2.0 * m[1:-1] + m[:-2]) / h ** 2
        dm = e2 * (mss + ms) - G * (eps + (1.0 - eps) * e3) * ms
        return np.concatenate((dm, [-2.0 * eps * G]))

    N = n - 1
    sp = lil_matrix((N + 1, N + 1))
    for i in range(N):
        for j in (i - 1, i, i + 1):
            if 0 <= j < N:
                sp[i, j] = 1
        sp[i, 0] = sp[i, 1] = 1
    sp[N, 0] = sp[N, 1] = 1

    def stop(tau, y):
        return y[-1] - math.log(R2_end)
    stop.terminal = True

    y0 = np.concatenate((np.zeros(N), [0.0]))
    tau_end = 10.0 * (-math.log(R2_end)) / (2.0 * eps * max(math.log1p(B), 1e-12))
    sol = solve_ivp(rhs, (0.0, tau_end), y0, method="BDF", jac_sparsity=sp.tocsr(), events=stop,
                    rtol=rtol, atol=1e-12, dense_output=False, max_step=np.inf)
    tau = sol.t
    R2 = np.exp(sol.y[-1])
    G = np.array([-grad0(sol.y[:-1, k]) / (1.0 - m_s) for k in range(tau.size)])
    # physical time: dt = R^2 dtau (units R0^2/D), trapezoidal
    t = np.concatenate(([0.0], np.cumsum(0.5 * (R2[1:] + R2[:-1]) * np.diff(tau))))
    return {"t": t, "R2": R2, "G": G, "tau": tau, "G_qs": math.log1p(B) / (1.0 - math.exp(-s_max))}


def wetted_pad_march(T_in: float, W_in: float, NTU: float, ratio: float = 1.0, P: float = 101325.0,
                     n_eval: int = 101, rtol: float = 1e-10):
    """Air flowing through a wetted pad, marched with the full nonlinear balances.

    With xi the fraction of the pad depth, NTU = h A / (m_a c_s) and the psychrometric ratio
    r = h / (g* c_s), the air temperature and humidity ratio obey

        dT/dxi = -NTU (T - T_s),    dW/dxi = (NTU / r) ln(1 + (W_s - W)/(1 + W)),

    where the water film is at the local wet-surface temperature T_s (no net heat to the
    water) and W_s = W_sat(T_s); c_s = c_pa + W c_pv changes as the air takes up vapour.
    Integrated with solve_ivp (Radau).  Returns a dict with ``xi``, ``T``, ``W``, ``T_s``.
    """
    from scipy.integrate import solve_ivp

    from .convective_mass import CP_DRY_AIR, CP_VAPOUR, humidity_ratio, wet_surface_temperature

    def rhs(xi, y):
        T, W = y
        T_s = wet_surface_temperature(T, W, ratio, P)
        W_s = humidity_ratio(T_s, 1.0, P)
        c_s0 = CP_DRY_AIR + W_in * CP_VAPOUR
        c_s = CP_DRY_AIR + W * CP_VAPOUR
        # NTU is defined with the inlet humid heat; h A / m_a is fixed along the pad
        return [-NTU * c_s0 / c_s * (T - T_s), NTU * c_s0 / (ratio * c_s) * math.log1p((W_s - W) / (1.0 + W))]

    xi = np.linspace(0.0, 1.0, n_eval)
    sol = solve_ivp(rhs, (0.0, 1.0), [T_in, W_in], method="Radau", t_eval=xi, rtol=rtol, atol=1e-12)
    T, W = sol.y
    T_s = np.array([wet_surface_temperature(a, b, ratio, P) for a, b in zip(T, W)])
    return {"xi": xi, "T": T, "W": W, "T_s": T_s}
