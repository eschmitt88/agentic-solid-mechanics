"""Rung 1c-1e — geometrically exact planar beams: Lee frame and the 215 deg
clamped-hinged deep circular arch.

Element: 2-node Simo-Reissner (shear-deformable, extensible, finite rotation)
beam, linear interpolation, one-point reduced integration.  Only the strain
energy is coded; internal force and tangent stiffness come from JAX autodiff.

    Gamma = R(beta0 + theta_mid)^T (x2 - x1)/L0 - e1     (axial, shear strain)
    kappa = (theta2 - theta1)/L0
    W_e   = L0/2 (EA Gamma1^2 + GA Gamma2^2 + EI kappa^2)
"""
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pathfollow import Continuation, load_control_newton
from plotstyle import C, setup

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"


class Frame:
    def __init__(self, X, conn, EA, GA, EI, fixed, load_dof):
        """X (nn,2) nodes, conn (ne,2), fixed = list of global dofs (3 per node:
        ux, uy, theta), load_dof = (global dof, sign)."""
        self.X, self.conn = np.asarray(X, float), np.asarray(conn, int)
        nn = len(X)
        d = self.X[self.conn[:, 1]] - self.X[self.conn[:, 0]]
        L0 = np.linalg.norm(d, axis=1)
        beta0 = np.arctan2(d[:, 1], d[:, 0])
        self.free = np.setdiff1d(np.arange(3 * nn), np.asarray(fixed, int))
        self.n = len(self.free)
        Xj, cj, L0j, b0j = jnp.asarray(self.X), jnp.asarray(self.conn), jnp.asarray(L0), jnp.asarray(beta0)
        freej = jnp.asarray(self.free)

        def energy(uf):
            u = jnp.zeros(3 * nn).at[freej].set(uf).reshape(nn, 3)
            x = Xj + u[:, :2]
            th = u[:, 2]
            dx = x[cj[:, 1]] - x[cj[:, 0]]
            thm = 0.5 * (th[cj[:, 0]] + th[cj[:, 1]]) + b0j
            c, s = jnp.cos(thm), jnp.sin(thm)
            g1 = (c * dx[:, 0] + s * dx[:, 1]) / L0j - 1.0
            g2 = (-s * dx[:, 0] + c * dx[:, 1]) / L0j
            kap = (th[cj[:, 1]] - th[cj[:, 0]]) / L0j
            return jnp.sum(0.5 * L0j * (EA * g1 ** 2 + GA * g2 ** 2 + EI * kap ** 2))

        f = jax.jit(jax.grad(energy))
        H = jax.jit(jax.hessian(energy))
        q = np.zeros(3 * nn)
        q[load_dof[0]] = load_dof[1]
        self.q = q[self.free]
        self.G = lambda u, lam: np.asarray(f(jnp.asarray(u))) - lam * self.q
        self.K = lambda u, lam: np.asarray(H(jnp.asarray(u)))
        self.Glam = lambda u, lam: -self.q
        self.iload = int(np.where(self.free == load_dof[0])[0][0])
        self.sign = load_dof[1]
        self.nn = nn

    def full(self, uf):
        u = np.zeros(3 * self.nn)
        u[self.free] = uf
        return u.reshape(self.nn, 3)


# ---------------------------------------------------------------- Lee frame
LEE = dict(L=120.0, a=24.0, E=720.0, A=6.0, I=2.0, nu=0.3)


def lee_frame(m):
    """m elements per 24-unit segment: 5m in the column, 5m in the beam."""
    L, a = LEE["L"], LEE["a"]
    n = 5 * m
    col = np.column_stack([np.zeros(n + 1), np.linspace(0, L, n + 1)])
    beam = np.column_stack([np.linspace(0, L, n + 1)[1:], np.full(n, L)])
    X = np.vstack([col, beam])
    conn = np.column_stack([np.arange(2 * n), np.arange(1, 2 * n + 1)])
    iload = n + m                       # node at (24, 120)
    assert np.allclose(X[iload], [a, L])
    fixed = [0, 1, 3 * (2 * n), 3 * (2 * n) + 1]     # both ends pinned
    EA, EI = LEE["E"] * LEE["A"], LEE["E"] * LEE["I"]
    GA = LEE["E"] / (2 * (1 + LEE["nu"])) * LEE["A"]
    return Frame(X, conn, EA, GA, EI, fixed, (3 * iload + 1, -1.0)), EI / L ** 2, iload


def run_lee(m, ds=0.02, verbose=False):
    fr, Pref, iload = lee_frame(m)
    L = LEE["L"]
    cont = Continuation(fr.G, fr.K, fr.Glam, psi=1.0 / (20 * Pref), uscale=L * np.sqrt(fr.n / 20), tol=1e-9)
    p0 = cont.start(np.zeros(fr.n), 0.0, fscale=20 * Pref)
    v = lambda p: -p.u[fr.iload] / L
    path, crit = cont.run(p0, ds, 5000, ds_max=ds, stop=lambda q: v(q) > 0.80 and q.lam > 0)
    lam = np.array([p.lam for p in path]) / Pref
    vv = np.array([v(p) for p in path])
    uu = np.array([fr.full(p.u)[iload, 0] for p in path]) / L
    lims = [(c.lam / Pref, -c.u[fr.iload] / L) for c in crit if c.kind == "limit"]
    bifs = [(c.lam / Pref, -c.u[fr.iload] / L) for c in crit if c.kind == "bifurcation"]
    # load at uy/L = 0.784 on the final (re-stiffened) branch: last crossing
    k = [i for i in range(len(vv) - 1) if (vv[i] - 0.784) * (vv[i + 1] - 0.784) <= 0]
    P784 = None
    if k:
        i = k[-1]
        P784 = float(lam[i] + (lam[i + 1] - lam[i]) * (0.784 - vv[i]) / (vv[i + 1] - vv[i]))
    # displacement turning points (snap-back) from sign changes of dv
    dv = np.diff(vv)
    turn = [float(vv[i + 1]) for i in range(len(dv) - 1) if dv[i] * dv[i + 1] < 0]
    return dict(m=m, n_el=10 * m, lam=lam, v=vv, uh=uu, limits=lims, bifs=bifs, P784=P784, turns=turn,
                neg=np.array([p.mon[0] for p in path]), path=path, fr=fr, Pref=Pref, nsolve=cont.nsolve)


# ---------------------------------------------------------------- deep arch
ARCH = dict(R=100.0, EI=1.0e6, EA=1.0e8, GA=5.0e7, angle=215.0)


def deep_arch(ne):
    R, half = ARCH["R"], np.deg2rad(ARCH["angle"] / 2)
    phi = np.linspace(-half, half, ne + 1)
    X = np.column_stack([R * np.sin(phi), R * np.cos(phi)])
    conn = np.column_stack([np.arange(ne), np.arange(1, ne + 1)])
    assert ne % 2 == 0
    icrown = ne // 2
    fixed = [0, 1, 3 * ne, 3 * ne + 1, 3 * ne + 2]      # left hinged, right clamped
    return Frame(X, conn, ARCH["EA"], ARCH["GA"], ARCH["EI"], fixed, (3 * icrown + 1, -1.0)), ARCH["EI"] / R ** 2, icrown


def run_arch(ne, ds=0.02):
    fr, Pref, ic = deep_arch(ne)
    R = ARCH["R"]
    cont = Continuation(fr.G, fr.K, fr.Glam, psi=1.0 / (9 * Pref), uscale=R * np.sqrt(fr.n / 20), tol=1e-9)
    p0 = cont.start(np.zeros(fr.n), 0.0, fscale=9 * Pref)
    v = lambda p: -p.u[fr.iload] / R
    # follow past the maximum and down to zero load again
    path, crit = cont.run(p0, ds, 6000, ds_max=ds, stop=lambda q: q.lam < -0.2 * Pref)
    lam = np.array([p.lam for p in path]) / Pref
    vv = np.array([v(p) for p in path])
    uu = np.array([fr.full(p.u)[ic, 0] for p in path]) / R
    lims = [(c.lam / Pref, -c.u[fr.iload] / R) for c in crit if c.kind == "limit"]
    bifs = [(c.lam / Pref, -c.u[fr.iload] / R) for c in crit if c.kind == "bifurcation"]
    return dict(ne=ne, lam=lam, v=vv, uh=uu, limits=lims, bifs=bifs, path=path, fr=fr, Pref=Pref,
                neg=np.array([p.mon[0] for p in path]))


def richardson(hs, vals):
    """Extrapolate vals(h) = v0 + c h^p from the three finest levels."""
    (h1, h2, h3), (v1, v2, v3) = hs[-3:], vals[-3:]
    p = np.log(abs((v1 - v2) / (v2 - v3))) / np.log(h1 / h2)
    v0 = v3 + (v3 - v2) / ((h2 / h3) ** p - 1)
    return float(v0), float(p)


def main():
    plt = setup()
    out = {}
    # ---------------- Lee frame: mesh convergence
    lee = [run_lee(m) for m in (1, 2, 4, 8, 16)]
    out["lee"] = {"reference": {"P1": 18.552, "v1": 0.407, "P784": 31.887},
                  "mesh": [dict(n_el=r["n_el"], limits=r["limits"], bifurcations=r["bifs"], P784=r["P784"],
                                turns=r["turns"], n_points=len(r["lam"]), linear_solves=r["nsolve"]) for r in lee]}
    hs = [1.0 / r["n_el"] for r in lee]
    P1 = [r["limits"][0][0] for r in lee]
    P7 = [r["P784"] for r in lee]
    P1x, p1 = richardson(hs, P1)
    P7x, p7 = richardson(hs, P7)
    out["lee"].update({"P1_by_mesh": P1, "P784_by_mesh": P7, "P1_extrap": P1x, "P1_order": p1,
                       "P784_extrap": P7x, "P784_order": p7,
                       "P1_finest": P1[-1], "v1_finest": lee[-1]["limits"][0][1],
                       "P1_rel_err_finest": abs(P1[-1] / 18.552 - 1), "P784_rel_err_finest": abs(P7[-1] / 31.887 - 1),
                       "P2_finest": lee[-1]["limits"][1]})
    best = lee[-1]
    np.savetxt(RES / "r1_lee_path.csv", np.column_stack([best["v"], best["uh"], best["lam"], best["neg"]]),
               delimiter=",", header="uy_over_L,ux_over_L,P_L2_over_EI,neg_eigs", comments="")
    # load-controlled Newton on the finest mesh
    fr, Pref = best["fr"], best["Pref"]
    lc = load_control_newton(fr.G, fr.K, np.linspace(0, 26, 27)[1:] * Pref, np.zeros(fr.n), tol=1e-9, fscale=20 * Pref)
    out["lee"]["load_control"] = [dict(P=r["lam"] / Pref, converged=r["converged"], iters=r["iters"],
                                       v=None if r["u"] is None else float(-r["u"][fr.iload] / LEE["L"])) for r in lc]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(best["v"], best["lam"], color=C["blue"], label="arc-length solver (160 elements)")
    un = best["neg"] > 0
    ax.plot(np.where(un, best["v"], np.nan), np.where(un, best["lam"], np.nan), color=C["blue"], lw=5, alpha=0.25,
            label="unstable under load control")
    okp = [r for r in out["lee"]["load_control"] if r["converged"]]
    ax.plot([r["v"] for r in okp], [r["P"] for r in okp], ls="none", marker="o", mfc="none", mec=C["bad"], ms=6,
            label="load-controlled Newton (converged)")
    ax.plot([0.407, 0.784], [18.552, 31.887], ls="none", marker="*", ms=15, color=C["ref"], label="NAFEMS NL7 reference points")
    for (P, v), name in zip(best["limits"], ["limit point", "limit point"]):
        ax.plot(v, P, marker="s", color=C["ref"], ms=6)
    ccx_csv = RES / "r1_ccx_lee_b32r_8.csv"
    if ccx_csv.exists():
        cd = np.loadtxt(ccx_csv, delimiter=",", skiprows=1)
        ax.plot(cd[::6, 0], cd[::6, 1], ls="none", marker="x", color=C["orange"], ms=5, label="CalculiX beams, displacement control")
    ax.set_ylim(-15, 48)
    ax.set_xlabel("vertical deflection of load point / L"); ax.set_ylabel("load  P·L²/EI")
    ax.set_title("Lee frame: snap-through and snap-back", loc="left"); ax.legend(loc="upper left", fontsize=9.5)
    ax = axes[1]
    X = fr.X
    ax.plot(X[:, 0], X[:, 1], color=C["mute"], lw=1.5, label="undeformed")
    i1 = next(i for i in range(len(best["lam"]) - 1) if best["lam"][i + 1] < best["lam"][i])   # first limit point
    i2 = int(np.argmin(best["lam"]))                                     # bottom of the snap-back
    i3 = max(i for i in range(len(best["lam"])) if best["lam"][i] < 31.887)   # NAFEMS second point
    idx = [i1, i2, i3]
    for i, col in zip(idx, [C["blue"], C["orange"], C["aqua"]]):
        u = fr.full(best["path"][i].u)
        ax.plot(X[:, 0] + u[:, 0], X[:, 1] + u[:, 1], color=col, label=f"P·L²/EI = {best['lam'][i]:.1f}")
    ax.set_aspect("equal"); ax.set_title("Deformed shapes", loc="left"); ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(RES / "r1_lee_frame.png")

    # ---------------- deep arch
    arch = [run_arch(ne) for ne in (20, 40, 80, 160, 320)]
    Pm = [max(l[0] for l in r["limits"]) for r in arch]
    ha = [1.0 / r["ne"] for r in arch]
    Pax, pa = richardson(ha, Pm)
    out["arch"] = {"reference": 8.972922, "Pmax_by_mesh": Pm, "n_el": [r["ne"] for r in arch],
                   "Pmax_extrap": Pax, "order": pa, "Pmax_finest": Pm[-1],
                   "rel_err_finest": abs(Pm[-1] / 8.972922 - 1), "rel_err_extrap": abs(Pax / 8.972922 - 1),
                   "limits_finest": arch[-1]["limits"], "bifurcations_finest": arch[-1]["bifs"]}
    ba = arch[-1]
    np.savetxt(RES / "r1_arch_path.csv", np.column_stack([ba["v"], ba["uh"], ba["lam"], ba["neg"]]),
               delimiter=",", header="uy_over_R,ux_over_R,P_R2_over_EI,neg_eigs", comments="")
    fra, Prefa = ba["fr"], ba["Pref"]
    lca = load_control_newton(fra.G, fra.K, np.linspace(0, 12, 25)[1:] * Prefa, np.zeros(fra.n), tol=1e-9, fscale=9 * Prefa)
    out["arch"]["load_control"] = [dict(P=r["lam"] / Prefa, converged=r["converged"], iters=r["iters"],
                                        v=None if r["u"] is None else float(-r["u"][fra.iload] / ARCH["R"])) for r in lca]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(ba["v"], ba["lam"], color=C["blue"], label="arc-length solver (320 elements)")
    ax.axhline(8.972922, color=C["ref"], lw=1, ls="--", label="reference maximum 8.9729 (Horák et al. 2022)")
    okp = [r for r in out["arch"]["load_control"] if r["converged"]]
    ax.plot([r["v"] for r in okp], [r["P"] for r in okp], ls="none", marker="o", mfc="none", mec=C["bad"], ms=6,
            label="load-controlled Newton (converged)")
    ax.set_xlabel("vertical deflection of crown / R"); ax.set_ylabel("load  P·R²/EI")
    ax.set_title("215° clamped–hinged arch", loc="left"); ax.legend(loc="lower center", fontsize=9.5)
    ax = axes[1]
    X = fra.X
    ax.plot(X[:, 0], X[:, 1], color=C["mute"], lw=1.5, label="undeformed")
    imax = int(np.argmax(ba["lam"]))
    for i, col in [(imax // 2, C["blue"]), (imax, C["orange"])]:
        u = fra.full(ba["path"][i].u)
        ax.plot(X[:, 0] + u[:, 0], X[:, 1] + u[:, 1], color=col, label=f"P·R²/EI = {ba['lam'][i]:.2f}")
    ax.set_aspect("equal"); ax.legend(loc="lower center", fontsize=9); ax.set_title("Deformed shapes", loc="left")
    fig.savefig(RES / "r1_arch.png")

    # ---------------- convergence figure
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog([r["n_el"] for r in lee], [abs(p / 18.552 - 1) for p in P1], marker="o", color=C["blue"], label="Lee frame, first limit load vs 18.552")
    ax.loglog([r["n_el"] for r in lee], [abs(p / 31.887 - 1) for p in P7], marker="s", color=C["orange"], label="Lee frame, load at u/L = 0.784 vs 31.887")
    ax.loglog([r["ne"] for r in arch], [abs(p / 8.972922 - 1) for p in Pm], marker="^", color=C["aqua"], label="deep arch, maximum load vs 8.9729")
    ax.axhline(0.01, color=C["ref"], lw=1, ls="--"); ax.text(11, 0.0115, "1 % tolerance", fontsize=9.5)
    ax.set_xlabel("number of beam elements"); ax.set_ylabel("relative difference from reference")
    ax.set_title("Mesh convergence, rung 1 beams", loc="left"); ax.legend(fontsize=9)
    fig.savefig(RES / "r1_beam_convergence.png")

    (RES / "r1_beams.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("mesh",)} for k, v in out.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
