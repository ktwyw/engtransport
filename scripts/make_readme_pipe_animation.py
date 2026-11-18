"""Animation for the README: an insulated district-heating pipe cooling after the pumps stop.

The temperatures come from the finite-volume model of Chapter 1
(`engtransport.reference.insulated_pipe_fv`). Run from the repository root:
    python scripts/make_readme_pipe_animation.py
which writes docs/images/pipe_cooling.gif.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge

from engtransport import properties, reference
from engtransport.plotting import use_style

ROOT = Path(__file__).resolve().parents[1]

# the pipe of Chapter 1
r_i, r_1, r_2 = 0.05355, 0.05715, 0.10715
k_ins, h, eps = 0.045, 25.0, 0.9
T_w0, T_inf = 383.15, 238.15
A_w, A_s = math.pi * r_i**2, math.pi * (r_1**2 - r_i**2)


def heat_capacity(T):
    w = properties.water_properties(np.clip(T, 273.16, 473.15))
    return w["rho"] * w["cp"] * A_w + 7850.0 * 460.0 * A_s


def main():
    use_style()
    t = np.linspace(0.0, 60 * 3600.0, 61)
    out = reference.insulated_pipe_fv(T_w0, r_1, r_2, k_ins, h, eps, T_inf, n_cells=30, t_eval=t,
                                      rho_c_ins=100.0 * 840.0, C_cladding=2 * math.pi * r_2 * 0.0005 * 7850 * 460,
                                      C_pipe=heat_capacity)
    Tw, prof, r = out["T_water"] - 273.15, out["profiles"] - 273.15, out["r"]
    t_f = np.interp(0.0, Tw[::-1], t[::-1]) / 3600

    cmap = plt.get_cmap("coolwarm")
    norm = matplotlib.colors.Normalize(vmin=-35, vmax=110)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})

    mm = 1e3
    faces = np.linspace(r_1, r_2, len(r) + 1)
    rings = [Wedge((0, 0), faces[j + 1] * mm, 0, 360, width=(faces[j + 1] - faces[j]) * mm) for j in range(len(r))]
    ring_coll = PatchCollection(rings, cmap=cmap, norm=norm, edgecolor="none")
    ax1.add_collection(ring_coll)
    water = Circle((0, 0), r_1 * mm, edgecolor="0.35", lw=2.5)
    ax1.add_patch(water)
    ax1.add_patch(Circle((0, 0), r_2 * mm, fill=False, edgecolor="0.2", lw=1.5))
    label = ax1.text(0, 0, "", ha="center", va="center", fontsize=10)
    ax1.set_xlim(-120, 120)
    ax1.set_ylim(-120, 120)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("air −35 °C, wind", fontsize=10)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, orientation="horizontal",
                        fraction=0.06, pad=0.02, aspect=30)
    cbar.set_label("temperature (°C)")

    ax2.plot(t / 3600, Tw, color="0.75", lw=1.2)
    line, = ax2.plot([], [], "C0", lw=2)
    dot, = ax2.plot([], [], "C0o", ms=6)
    ax2.axhline(0, color="C3", lw=0.8, ls="--")
    ax2.text(1, 3, "0 °C", color="C3", fontsize=9)
    ax2.axvline(t_f, color="C3", lw=0.6, ls=":")
    ax2.text(t_f + 0.8, 60, f"0 °C after\n{t_f:.0f} h", color="C3", fontsize=9)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(-30, 120)
    ax2.set_xlabel("time after the pumps stop (h)")
    ax2.set_ylabel("water temperature (°C)")
    ax2.set_title("50 mm of mineral wool, water at 110 °C", fontsize=10)

    def update(n):
        ring_coll.set_array(prof[:, n])
        water.set_facecolor(cmap(norm(Tw[n])))
        label.set_text(f"{Tw[n]:.0f} °C".replace("-", "−"))
        line.set_data(t[: n + 1] / 3600, Tw[: n + 1])
        dot.set_data([t[n] / 3600], [Tw[n]])
        return ring_coll, water, label, line, dot

    anim = FuncAnimation(fig, update, frames=len(t), blit=False)
    out_path = ROOT / "docs" / "images" / "pipe_cooling.gif"
    anim.save(out_path, writer=PillowWriter(fps=8), dpi=72)
    print(f"wrote {out_path.relative_to(ROOT)} ({out_path.stat().st_size / 1e3:.0f} kB), t_f = {t_f:.1f} h")


if __name__ == "__main__":
    main()
