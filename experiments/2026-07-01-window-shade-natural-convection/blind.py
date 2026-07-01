"""blind.py — effective window heat-transfer coefficient vs venetian-slat angle.

Standard fenestration natural-convection model: a tall differentially-heated 2D
cavity — window wall (isothermal, hot) on the right, room boundary (isothermal,
ambient) on the left, adiabatic top/bottom — with an interior venetian blind
(slats) inside. The blind slats pivot in place at fixed pitch; only the tilt
angle theta changes. Effective h(theta) ∝ the cavity Nusselt number.

Geometry (consistent across the sweep): slat chord 2.5", pitch 2.25" (chord minus
the ~1/4" overlap at near-vertical), angle theta in (-90, +90). Reuses the
de-Vahl-Davis-validated thermal LBM (jaxlbm.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxlbm as L

HERE = Path(__file__).parent
T0 = 0.5
Ra, Pr = 1e5, 0.71        # moderate, safely-laminar-steady (validated <=5% at 1e5)

# geometry in cells (cell = chord/CHORD)
CHORD = 40                # 2.5" -> 40 cells
PITCH = 36                # 2.25" (chord - 0.25" overlap at vertical)
THICK = 3                 # slat thickness (cells)
W = 84                    # cavity width (window<->room), ~5.25"
NSLAT = 6
XC = 58                   # slat-centre x from left (near the right/window wall)
MARGIN = 30               # top/bottom margin
H = NSLAT * PITCH + 2 * MARGIN


def _seg_dist(gx, gy, x0, y0, x1, y1):
    """distance from grid points (gx,gy) to segment (x0,y0)-(x1,y1)."""
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((gx - x0) * dx + (gy - y0) * dy) / L2, 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    return np.hypot(gx - px, gy - py)


def geometry(theta_deg):
    nx, ny = W, H
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    solid = np.zeros((nx, ny), bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True     # walls
    # slats
    th = np.radians(theta_deg)
    ex, ey = 0.5 * CHORD * np.cos(th), 0.5 * CHORD * np.sin(th)
    y_centres = (H - (NSLAT - 1) * PITCH) / 2 + np.arange(NSLAT) * PITCH
    slat = np.zeros((nx, ny), bool)
    for yc in y_centres:
        d = _seg_dist(gx, gy, XC - ex, yc - ey, XC + ex, yc + ey)
        slat |= d <= THICK / 2
    slat[0, :] = slat[-1, :] = slat[:, 0] = slat[:, -1] = False        # don't double-count walls
    solid |= slat
    # temperature BCs: window (right) hot, room (left) cold; top/bottom + slats adiabatic
    dirich = np.zeros((nx, ny), bool); Tval = np.zeros((nx, ny))
    dirich[-1, :] = True; Tval[-1, :] = 1.0        # window (hot)
    dirich[0, :] = True;  Tval[0, :] = 0.0         # room (cold)
    adia = np.zeros((nx, ny), bool)
    adia[:, 0] = adia[:, -1] = True                # top/bottom
    adia |= slat                                   # adiabatic slats
    adia[0, :] = adia[-1, :] = False               # walls take Dirichlet
    return (dict(solid=jnp.asarray(solid), dirich=jnp.asarray(dirich),
                 Tval=jnp.asarray(Tval), adia=jnp.asarray(adia)), slat)


def nusselt(f, g, alpha):
    rho = f.sum(-1); T = g.sum(-1)
    ux = jnp.tensordot(f, L.CX, (-1, 0)) / rho
    dTdx = (jnp.roll(T, -1, 0) - jnp.roll(T, 1, 0)) / 2.0
    flux = ux * T - alpha * dTdx
    # window (hot) is on the right -> heat flows in -x; report the transfer magnitude.
    # Measure the through-flux in the CLEAR room-side band (left of the blind), where
    # it equals the window<->room transfer by conservation and is uncorrupted by slats.
    planeNu = -jnp.mean(flux[:, 1:-1], axis=1) * W / alpha
    xa, xb = 3, XC - CHORD // 2 - 4          # clear region left of the blind
    band = planeNu[xa:xb]
    return float(jnp.mean(band)), float(jnp.min(band)), float(jnp.max(band))


def run_angle(theta, p):
    bc, slat = geometry(theta)
    f, g = L.init(W, H, T0)

    def maxspeed(f, g):
        rho = f.sum(-1)
        ux = jnp.tensordot(f, L.CX, (-1, 0)) / rho
        uy = jnp.tensordot(f, L.CY, (-1, 0)) / rho
        return jnp.max(jnp.sqrt(ux ** 2 + uy ** 2))

    f, g, blocks, _ = L.run_steady(f, g, bc, p, T0, block=2000, max_blocks=500,
                                   tol=1e-6, probe=maxspeed)
    nu, lo, hi = nusselt(f, g, p["alpha"])
    T = np.asarray(g.sum(-1))
    return nu, lo, hi, blocks, T, slat


def main():
    angles = [float(a) for a in sys.argv[1:]] or \
             [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]
    p = L.transport_from(Ra, Pr, W, alpha=0.02)
    print(f"cavity {W}x{H}, Ra_W={Ra:.0e}, Pr={Pr}, chord={CHORD} pitch={PITCH} cells")
    rows = []
    for th in angles:
        t0 = time.time()
        nu, lo, hi, blocks, T, slat = run_angle(th, p)
        dt = time.time() - t0
        rows.append({"theta": th, "Nu": round(nu, 4), "Nu_spread": [round(lo, 3), round(hi, 3)],
                     "blocks": blocks, "wall_s": round(dt, 1)})
        np.save(HERE / "results" / f"blind_T_{int(th):+03d}.npy", T)
        np.save(HERE / "results" / f"blind_slat_{int(th):+03d}.npy", slat)
        print(f"θ={th:+5.0f}°: Nu={nu:.3f}  Nu(x)∈[{lo:.2f},{hi:.2f}]  {blocks*2}k steps  {dt:.1f}s", flush=True)
    nu0 = next((r["Nu"] for r in rows if r["theta"] == 0), rows[len(rows) // 2]["Nu"])
    for r in rows:
        r["h_rel"] = round(r["Nu"] / nu0, 4)
    out = {"Ra": Ra, "Pr": Pr, "cavity": [W, H], "chord": CHORD, "pitch": PITCH,
           "h_ref_theta": 0, "rows": rows}
    (HERE / "results" / "blind_sweep.json").write_text(json.dumps(out, indent=2))
    print("\nθ, Nu, h/h(0):")
    for r in rows:
        print(f"  {r['theta']:+5.0f}°  Nu={r['Nu']:.3f}  h/h(0)={r['h_rel']:.3f}")
    print("wrote results/blind_sweep.json")


if __name__ == "__main__":
    main()
