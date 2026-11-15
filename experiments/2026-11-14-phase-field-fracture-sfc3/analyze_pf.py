"""Rung-1 post-processing: metrics table + figures from results/runs/*.npz."""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#6250d6", red="#e34948", ink="#0b0b0b", mute="#52514e")
ORDER = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f6fa", "#b9d2f2", "#5d9ae0", "#1f5fb3", "#0b2a5c"])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6,
                     "figure.dpi": 130, "savefig.bbox": "tight", "axes.titlesize": 11, "lines.linewidth": 1.8})
RUNS = "results/runs"


def load(tag):
    p = os.path.join(RUNS, tag + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    r = {k: z[k] for k in z.files}
    r["meta"] = json.loads(str(z["meta"]))
    r["tag"] = tag
    return r


def grid(r, field=None):
    nx, ny = r["meta"]["nx"], r["meta"]["ny"]
    co = r["coords"][: nx * ny].reshape(nx, ny, 2)
    d = (r["d"] if field is None else field)[: nx * ny].reshape(nx, ny)
    return co[:, 0, 0], co[0, :, 1], d


def metrics(r):
    u, F = r["u"], r["F"]
    i = int(np.argmax(F))
    m = dict(peak_F=float(F[i]), u_peak=float(u[i]), ne=r["meta"]["ne"], h=r["meta"]["h"], l=r["meta"]["l"],
             wall_s=r["meta"]["wall_s"], n_steps=int(len(u)))
    below = np.nonzero((F < 0.05 * F[i]) & (np.arange(len(F)) > i))[0]
    m["u_fail_5pct"] = float(u[below[0]]) if len(below) else None
    m["F_end"] = float(F[-1]); m["u_end"] = float(u[-1])
    m["gamma_end"] = float(r["gamma"][-1])
    x, y, d = grid(r)
    if r["meta"]["case"] == "tension":
        cols = np.nonzero(x > 0.52)[0]
        dev = [abs(y[np.argmax(d[c])] - 0.5) for c in cols if d[c].max() > 0.9]
        m["path_max_dev"] = float(max(dev)) if dev else None
        m["crack_reaches_right"] = bool(d[-1].max() > 0.9)
    else:
        m.update(shear_path(x, y, d))
    return m


def shear_path(x, y, d, thr=0.85):
    """Centreline below the notch: per row, x of max d. Angle from points 0.02-0.10 mm from the tip."""
    rows = np.nonzero(y < 0.5 - 1e-9)[0]
    pts = []
    for j in rows:
        i = int(np.argmax(d[:, j]))
        if d[i, j] > thr and x[i] > 0.45:
            pts.append((x[i], y[j]))
    out = dict(crack_angle_deg=None, crack_end_x=None, crack_lowest_y=None)
    if len(pts) < 3:
        return out
    pts = np.array(pts)
    rr = np.hypot(pts[:, 0] - 0.5, pts[:, 1] - 0.5)
    for name, lo, hi in (("crack_angle_deg", 0.02, 0.10), ("crack_angle_deg_w2", 0.03, 0.15), ("crack_angle_deg_w3", 0.01, 0.06)):
        s = (rr > lo) & (rr < hi)
        if s.sum() >= 3:
            a = np.polyfit(pts[s, 1], pts[s, 0], 1)[0]     # dx/dy
            out[name] = float(np.degrees(np.arctan2(1.0, -a)))   # angle below the x-axis
    out["crack_lowest_y"] = float(pts[:, 1].min())
    # where the centreline first comes within 0.02 mm of the bottom edge (after that it runs along the clamped edge)
    near = pts[pts[:, 1] <= 0.02]
    out["crack_end_x"] = float(near[np.argmax(near[:, 1]), 0]) if len(near) else None
    low = pts[pts[:, 1] <= pts[:, 1].min() + 1e-9]
    out["crack_x_on_bottom_row"] = float(low[0, 0]) if pts[:, 1].min() < 0.02 else None
    # second crack check: damage > thr in the upper half away from the notch tip
    X, Y = np.meshgrid(x, y, indexing="ij")
    out["second_crack_upper"] = bool(np.any((d > thr) & (Y > 0.5 + 0.06)))
    out["second_crack_left"] = bool(np.any((d > thr) & (X < 0.4)))
    return out


def field_img(ax, r, field=None, title=None):
    x, y, d = grid(r, field)
    ax.pcolormesh(x, y, np.clip(d, 0, 1).T, cmap=SEQ, vmin=0, vmax=1, shading="gouraud", rasterized=True)
    ax.plot([0, 0.5], [0.5, 0.5], color="#ffffff", lw=1.2)
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(False)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#888")
    if title:
        ax.set_title(title, fontsize=9.5)


if __name__ == "__main__":
    tags = sorted(os.path.basename(p)[:-4] for p in glob.glob(RUNS + "/*.npz"))
    allm = {}
    for t in tags:
        r = load(t)
        allm[t] = metrics(r)
        m = allm[t]
        print(f"{t:32s} ne={m['ne']:7d} peak={m['peak_F']:.4f} u_peak={m['u_peak']:.5f} "
              f"fail={m['u_fail_5pct']} end(u={m['u_end']:.5f},F={m['F_end']:.3f}) gam={m['gamma_end']:.3f} "
              + (f"ang={m.get('crack_angle_deg')} endx={m.get('crack_end_x')}" if t.startswith('s_') else f"dev={m.get('path_max_dev')}")
              + f" wall={m['wall_s']:.0f}s")
    json.dump(allm, open("results/pf_run_metrics.json", "w"), indent=1)
