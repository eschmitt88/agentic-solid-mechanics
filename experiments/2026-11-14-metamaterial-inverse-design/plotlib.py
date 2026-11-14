"""Shared plotting helpers (palette validated with the dataviz skill's validator; no dual axes)."""
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import design
import fem

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
SURF = "#fcfcfb"
FRAME = "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def outline(p, u_nodes=None, S=None, n=161):
    """Closed polygon of the left half-beam (reference, or deformed if nodal displacements on solver S are given)."""
    if u_nodes is None:
        s = jnp.linspace(0, 1, n)
        bot = np.asarray(fem.surface_points(jnp.asarray(p), s, -0.5 * jnp.ones(n), design.LH))
        top = np.asarray(fem.surface_points(jnp.asarray(p), s, 0.5 * jnp.ones(n), design.LH))
    else:
        m = S.mesh
        X = np.asarray(S.coords(jnp.asarray(p))) + np.asarray(u_nodes).reshape(-1, 2)
        X = X.reshape(m["ni"], m["nj"], 2)
        bot, top = X[:, 0], X[:, -1]
    return np.vstack([bot, top[::-1]])


def draw_cell(ax, p, d=0.0, u_nodes=None, S=None, color=BLUE, alpha=1.0, frame=True, lw=0.0, label=None, fill=True):
    """Draw the unit cell: stiff walls + shuttle (grey) and the elastomer beam (colour)."""
    xm = design.LH + design.WS / 2
    if frame:
        for x0 in (-5.0, 2 * xm):
            ax.add_patch(plt.Rectangle((x0, -15.5), 5.0, 25.0, color=FRAME, lw=0))
        ax.add_patch(plt.Rectangle((design.LH, _yshut(p) - d - 2.0), design.WS, 11.5, color=FRAME, lw=0))
    poly = outline(p, u_nodes, S)
    for mirror in (False, True):
        P = poly.copy()
        if mirror:
            P[:, 0] = 2 * xm - P[:, 0]
        if fill:
            ax.fill(P[:, 0], P[:, 1], color=color, alpha=alpha, lw=0, label=None if mirror else label)
        else:
            ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color=color, lw=lw or 1.2, alpha=alpha, label=None if mirror else label)
    ax.set_aspect("equal")
    ax.set_xlim(-5.5, 2 * xm + 5.5)
    ax.set_ylim(-15.8, 10.0)
    ax.axis("off")


def _yshut(p):
    y, _ = fem.centerline(jnp.asarray(p), jnp.ones(1), design.LH)
    return float(y[0])


def deformed_states(p, d_list, S=None):
    S = S or fem.Solver(nx=40, ny=2, Lh=design.LH)
    ds = np.array(sorted(d_list))
    F, us, infos = S.path(p, ds)
    return S, {float(d): np.asarray(fem.full_u(u, float(d), S.mesh)) for d, u in zip(ds, us)}, F
