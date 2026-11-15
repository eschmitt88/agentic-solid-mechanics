"""Shared plotting helpers (trial 9's palette, validated there with the dataviz skill's validator; fixed hue order; no dual axes)."""
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import design
import fem

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
SURF, FRAME = "#fcfcfb", "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def outline(p, nk, n=161):
    s = jnp.linspace(0, 1, n)
    bot = np.asarray(fem.surface_points(jnp.asarray(p), s, -0.5 * jnp.ones(n), design.LH, nk))
    top = np.asarray(fem.surface_points(jnp.asarray(p), s, 0.5 * jnp.ones(n), design.LH, nk))
    return np.vstack([bot, top[::-1]])


def draw_cell(ax, p, nk, y0=0.0, color=BLUE, alpha=1.0, frame=True, fill=True, lw=1.2, label=None):
    """One unit cell: stiff walls and shuttle (grey) and the elastomer beam (colour), drawn at vertical offset y0."""
    xm = design.LH + design.WS / 2
    poly = outline(p, nk)
    ysh = float(fem.centerline(jnp.asarray(p), jnp.ones(1), design.LH, nk)[0][0])
    if frame:
        for x0 in (-5.0, 2 * xm):
            ax.add_patch(plt.Rectangle((x0, y0 - 15.5), 5.0, 25.0, color=FRAME, lw=0))
        ax.add_patch(plt.Rectangle((design.LH, y0 + ysh - 2.0), design.WS, 11.5, color=FRAME, lw=0))
    for mirror in (False, True):
        P = poly.copy()
        P[:, 1] += y0
        if mirror:
            P[:, 0] = 2 * xm - P[:, 0]
        if fill:
            ax.fill(P[:, 0], P[:, 1], color=color, alpha=alpha, lw=0, label=None if mirror else label)
        else:
            ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color=color, lw=lw, alpha=alpha, label=None if mirror else label)
    ax.set_aspect("equal")
    ax.set_xlim(-5.5, 2 * xm + 5.5)
    ax.axis("off")


def draw_design(ax, p, nk, ncell=1, **kw):
    P = np.reshape(np.asarray(p, float), (ncell, -1))
    for c, q in enumerate(P):
        draw_cell(ax, q, nk, y0=c * 25.0, **kw)
    ax.set_ylim(-15.8, 10.0 + 25.0 * (ncell - 1))
