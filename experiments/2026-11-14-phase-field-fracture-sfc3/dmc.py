"""Damage Mechanics Challenge (3-point bending of printed gypsum) with the pf3d hybrid phase-field solver.

Coordinates as in the challenge packet: x along the length (load rod at x=0), y through the width
(front y=-W/2, back y=+W/2), z up (top z=0, bottom z=-2W). W=12.7, H=25.4, L=76.2 mm, span 60.96 mm.
"""
import sys, os, time, json, argparse
import numpy as np
import pf3d

W, Hh, L = 12.7, 25.4, 76.2
XS = 0.4 * L            # support x = +-30.48 (10 % of length from each end)

# notch: x-centre on the front face, on the back face, depth on front, depth on back  [mm]
GEOMS = {
    "HC":  dict(xf=0.0,     xb=0.0,     df=5.08, db=5.08),      # centre notch, mode I
    "HB":  dict(xf=9.525,   xb=9.525,   df=5.08, db=5.08),      # notch at 3.75W from the end (0.75W off centre)
    "HA":  dict(xf=19.05,   xb=19.05,   df=5.08, db=5.08),      # notch at 4.5W from the end (1.5W off centre)
    "H45": dict(xf=-6.35,   xb=6.35,    df=5.08, db=5.08),      # centre notch rotated 45 deg in plan
    "CH":  dict(xf=-12.7,   xb=-6.35,   df=5.08, db=2.54),      # challenge: off centre, 63.4 deg skew, tapered depth
}
NOTCH_W = 1.0


def marched(a, b, hfun):
    pts = [a]
    while pts[-1] < b - 1e-9:
        pts.append(pts[-1] + hfun(pts[-1]))
    pts = np.array(pts)
    return a + (pts - a) * (b - a) / (pts[-1] - a)


def build(geom, h, ny, l, Gc, E=1400.0, nu=0.2, hmax=2.5, kres=1e-5, hz_coarse=None, shield_r=2.0, verbose=True):
    g = GEOMS[geom]
    xlo, xhi = min(g["xf"], g["xb"], 0.0) - 4.0, max(g["xf"], g["xb"], 0.0) + 4.0
    def hfun(x):
        dist = max(xlo - x, x - xhi, 0.0)
        return min(hmax, h + 0.25 * dist)
    marks = sorted(set([-L / 2, -XS, xlo, 0.0, xhi, XS, L / 2]))
    xs = [marks[0]]
    for a, b in zip(marks[:-1], marks[1:]):
        if a >= xlo and b <= xhi:
            n = int(round((b - a) / h)); seg = np.linspace(a, b, n + 1)
        else:
            seg = marched(a, b, hfun)
        xs += list(seg[1:])
    xs = np.array(xs)
    zs = np.linspace(-Hh, 0, int(round(Hh / h)) + 1)
    ys = np.linspace(-W / 2, W / 2, ny + 1)
    slope = (g["xb"] - g["xf"]) / W                      # dx/dy of the notch plane
    sin_t = 1.0 / np.sqrt(1 + slope ** 2)                # perpendicular width -> x-extent
    def notch(x, y, z):
        xn = g["xf"] + slope * (y + W / 2)
        dep = g["df"] + (g["db"] - g["df"]) * (y + W / 2) / W
        return (np.abs(x - xn) * sin_t < NOTCH_W / 2) & (z < -Hh + dep)
    def shield(x, y, z):
        r1 = np.hypot(x, z); r2 = np.hypot(x - XS, z + Hh); r3 = np.hypot(x + XS, z + Hh)
        return (r1 < shield_r) | (r2 < shield_r) | (r3 < shield_r)
    P = pf3d.PF3D(xs, ys, zs, E, nu, Gc, l, remove_fn=notch, shield_fn=shield, kres=kres, verbose=verbose)
    tol = 1e-6
    load = P.nodes_where(lambda x, y, z: (np.abs(x) < tol) & (z > -tol))
    sL = P.nodes_where(lambda x, y, z: (np.abs(x + XS) < tol) & (z < -Hh + tol))
    sR = P.nodes_where(lambda x, y, z: (np.abs(x - XS) < tol) & (z < -Hh + tol))
    assert len(load) == ny + 1 and len(sL) == ny + 1 and len(sR) == ny + 1, (len(load), len(sL), len(sR))
    P.fix(load, 2); P.fix(load, 0)
    P.fix(sL, 2); P.fix(sR, 2)
    c = P.coords
    P.fix(sL[np.argmin(c[sL, 1])][None], 1); P.fix(sR[np.argmin(c[sR, 1])][None], 1)
    P.load_nodes = load
    return P


def run(geom, h, ny, l, Gc, E=1400.0, nu=0.2, ddelta=None, max_stag=400, stag_tol=2e-3, tag=None, outdir="results/dmc",
        stop_frac=0.25, max_steps=400, verbose=True, **kw):
    os.makedirs(outdir, exist_ok=True)
    P = build(geom, h, ny, l, Gc, E, nu, verbose=verbose, **kw)
    tag = tag or f"{geom}_l{l}_h{h}_ny{ny}"
    # elastic stiffness first (tiny step), to size the load increments
    uD = np.zeros(3 * P.nn); uD[3 * P.load_nodes + 2] = -1e-6
    P.step(uD, max_stag=1)
    k_el = -P.reaction(P.load_nodes, 2) / 1e-6
    P.H = P.H * 0; P.d = P.d * 0
    if ddelta is None:
        # nominal strength scale sigma ~ sqrt(E Gc / l); bending load scale; aim for ~50 steps to peak
        sig = np.sqrt(E * Gc / l)
        P_est = sig * W * (Hh - 5.08) ** 2 / (1.5 * 2 * XS)
        ddelta = 1.5 * P_est / k_el / 50
    rec = []; t0 = time.time(); Pmax = 0.0; delta = 0.0; dd = ddelta; n_after = 0
    for n in range(max_steps):
        delta += dd
        uD = np.zeros(3 * P.nn); uD[3 * P.load_nodes + 2] = -delta
        info = P.step(uD, max_stag=max_stag, stag_tol=stag_tol)
        F = -P.reaction(P.load_nodes, 2)
        el, ga = P.energies()
        rec.append((delta, F, info["nstag"], ga, el))
        Pmax = max(Pmax, F)
        if verbose and (n % 10 == 0 or info["nstag"] > 20):
            print(f"[{tag}] step {n} delta={delta:.5f} P={F:.2f} nstag={info['nstag']} cg={info['cg_u']} "
                  f"dmax={float(P.d.max()):.3f} t={time.time()-t0:.0f}s", flush=True)
        if F < stop_frac * Pmax and n > 5:
            break
        if F < 0.5 * Pmax and n > 5:
            n_after += 1
            if n_after >= int(os.environ.get("DMC_POSTPEAK_STEPS", "100000")):
                break
    rec = np.array(rec); i = int(rec[:, 1].argmax())
    res = dict(geom=geom, h=h, ny=ny, l=l, Gc=Gc, E=E, nu=nu, k_el=k_el, P_peak=float(rec[i, 1]), delta_peak=float(rec[i, 0]),
               ne=P.ne, wall_s=time.time() - t0, ddelta=ddelta, stag_tol=stag_tol)
    np.savez_compressed(os.path.join(outdir, tag + ".npz"), rec=rec, d=P.d_grid().astype(np.float32), xs=P.xs, ys=P.ys, zs=P.zs,
                        meta=json.dumps(res))
    print(f"[{tag}] RESULT k_el={k_el:.1f} N/mm  P_peak={rec[i,1]:.2f} N at delta_FE={rec[i,0]:.5f} mm  wall={res['wall_s']:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True); ap.add_argument("--h", type=float, default=0.5); ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--l", type=float, default=1.0); ap.add_argument("--Gc", type=float, default=0.01)
    ap.add_argument("--E", type=float, default=1400.0); ap.add_argument("--nu", type=float, default=0.2)
    ap.add_argument("--tag", default=None); ap.add_argument("--stag-tol", type=float, default=2e-3)
    ap.add_argument("--ddelta", type=float, default=None)
    a = ap.parse_args()
    run(a.geom, a.h, a.ny, a.l, a.Gc, a.E, a.nu, tag=a.tag, stag_tol=a.stag_tol, ddelta=a.ddelta)
