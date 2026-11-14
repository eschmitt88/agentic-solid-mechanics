"""Rung 1a/1b — von Mises (two-bar) truss.

Two bars from supports (-b, 0), (+b, 0) to an apex at (x, y), initial apex
(0, h), Green-Lagrange bar strain, dead load P pulling the apex down.

Closed form of this discrete model (derived in-session, checked below with
sympy):
    primary (symmetric) path   x = 0,  P = EA y (h^2 - y^2) / L0^3
    limit load                 P_lim = 2 EA h^3 / (3 sqrt(3) L0^3)   at y = h/sqrt(3)
    sideways bifurcation       y_b^2 = h^2 - 2 b^2,  P_b = 2 EA b^2 y_b / L0^3
                               (exists if h^2 > 2 b^2; precedes the limit point if h^2 > 3 b^2)
    secondary branch           x^2 + y^2 = y_b^2,   P = 2 EA b^2 y / L0^3   (a circle)
"""
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pathfollow import Continuation, load_control_newton
from plotstyle import C, setup

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"
EA = 1.0


def make(b, h):
    L0 = np.sqrt(b * b + h * h)

    def energy(u):
        x, y = u[0], h + u[1]
        e1 = ((x + b) ** 2 + y ** 2 - L0 ** 2) / (2 * L0 ** 2)
        e2 = ((x - b) ** 2 + y ** 2 - L0 ** 2) / (2 * L0 ** 2)
        return 0.5 * EA * L0 * (e1 ** 2 + e2 ** 2)

    fint = jax.jit(jax.grad(energy))
    hess = jax.jit(jax.hessian(energy))
    q = np.array([0.0, -1.0])           # P > 0 pushes the apex down
    G = lambda u, lam: np.asarray(fint(jnp.asarray(u))) - lam * q
    K = lambda u, lam: np.asarray(hess(jnp.asarray(u)))
    Glam = lambda u, lam: -q
    return G, K, Glam, L0


def closed_form(b, h):
    L0 = np.sqrt(b * b + h * h)
    out = {"P_lim": 2 * EA * h ** 3 / (3 * np.sqrt(3) * L0 ** 3), "y_lim": h / np.sqrt(3)}
    if h * h > 2 * b * b:
        yb = np.sqrt(h * h - 2 * b * b)
        out.update({"y_bif": yb, "P_bif": 2 * EA * b * b * yb / L0 ** 3})
    return out


def sympy_check():
    """Re-derive the closed form symbolically (guards against algebra slips)."""
    import sympy as sp
    x, y, b, h, P, EAs = sp.symbols("x y b h P EA", positive=True)
    L0 = sp.sqrt(b ** 2 + h ** 2)
    e1 = ((x + b) ** 2 + y ** 2 - L0 ** 2) / (2 * L0 ** 2)
    e2 = ((x - b) ** 2 + y ** 2 - L0 ** 2) / (2 * L0 ** 2)
    Pi = EAs * L0 / 2 * (e1 ** 2 + e2 ** 2) + P * (y - h)
    # primary path
    Pprim = sp.solve(sp.diff(Pi, y).subs(x, 0), P)[0]
    assert sp.simplify(Pprim - EAs * y * (h ** 2 - y ** 2) / L0 ** 3) == 0
    ylim = h / sp.sqrt(3)
    assert sp.simplify(sp.diff(Pprim, y).subs(y, ylim)) == 0
    assert sp.simplify(Pprim.subs(y, ylim) - 2 * EAs * h ** 3 / (3 * sp.sqrt(3) * L0 ** 3)) == 0
    # sideways stiffness on the primary path vanishes at y^2 = h^2 - 2 b^2
    Kxx = sp.diff(Pi, x, 2).subs(x, 0)
    assert sp.simplify(Kxx.subs(y, sp.sqrt(h ** 2 - 2 * b ** 2))) == 0
    # secondary branch: on the circle x^2+y^2 = h^2-2b^2 with P = 2 EA b^2 y / L0^3
    xs = sp.sqrt(h ** 2 - 2 * b ** 2 - y ** 2)
    Ps = 2 * EAs * b ** 2 * y / L0 ** 3
    assert sp.simplify(sp.diff(Pi, x).subs({x: xs, P: Ps})) == 0
    assert sp.simplify(sp.diff(Pi, y).subs({x: xs, P: Ps})) == 0
    return True


def run_case(b, h, ds, nsteps, tag):
    G, K, Glam, L0 = make(b, h)
    cf = closed_form(b, h)
    cont = Continuation(G, K, Glam, psi=1.0 / cf["P_lim"], uscale=h, tol=1e-12)
    p0 = cont.start(np.zeros(2), 0.0, fscale=cf["P_lim"])
    path, crit = cont.run(p0, ds, nsteps, ds_max=ds, stop=lambda q: q.u[1] < -2.3 * h)
    out = {"b": b, "h": h, "closed_form": cf, "critical": [], "n_points": len(path)}
    for c in crit:
        out["critical"].append({"kind": c.kind, "lam": c.lam, "x": float(c.u[0]), "y": float(h + c.u[1]),
                                "neg_before": int(c.mon_before[0]), "neg_after": int(c.mon_after[0])})
    # residual of path points against the closed-form primary path
    ys = np.array([h + p.u[1] for p in path])
    lam = np.array([p.lam for p in path])
    out["max_path_err"] = float(np.max(np.abs(lam - EA * ys * (h * h - ys * ys) / L0 ** 3)) / cf["P_lim"])
    branch = None
    bifs = [c for c in crit if c.kind == "bifurcation"]
    if bifs:
        c = bifs[0]
        pb = cont.switch_branch(c, eps=0.02 * h)
        assert pb is not None, "branch switch failed"
        # walk the secondary branch; do not locate critical points (it reconnects at -y_b)
        bpath, _ = cont.run(pb, ds, 4000, ds_max=ds, locate=False,
                            stop=lambda q: abs(q.u[0]) < 0.015 * h and h + q.u[1] < 0)
        xs = np.array([p.u[0] for p in bpath]); yb_ = np.array([h + p.u[1] for p in bpath])
        lb = np.array([p.lam for p in bpath])
        out["branch"] = {
            "n_points": len(bpath),
            "max_circle_err": float(np.max(np.abs(np.sqrt(xs ** 2 + yb_ ** 2) - cf["y_bif"])) / cf["y_bif"]),
            "max_load_err": float(np.max(np.abs(lb - 2 * EA * b * b * yb_ / L0 ** 3)) / cf["P_lim"]),
            "max_abs_x_over_h": float(np.max(np.abs(xs)) / h),
        }
        branch = (xs, yb_, lb)
    # load-controlled Newton on the same problem
    lc = load_control_newton(G, K, np.linspace(0, 1.3 * cf["P_lim"], 40)[1:], np.zeros(2),
                             tol=1e-12, fscale=cf["P_lim"])
    out["load_control"] = [{"lam": r["lam"], "converged": r["converged"], "iters": r["iters"],
                            "x": None if r["u"] is None else float(r["u"][0]),
                            "y": None if r["u"] is None else float(h + r["u"][1])} for r in lc]
    np.savetxt(RES / f"r1_truss_{tag}_path.csv",
               np.column_stack([[p.u[0] for p in path], ys, lam, [p.mon[0] for p in path]]),
               delimiter=",", header="x,y,P,neg_eigs", comments="")
    return out, (np.array([p.u[0] for p in path]), ys, lam, np.array([p.mon[0] for p in path])), branch, lc


def main():
    assert sympy_check()
    res = {"sympy_closed_form_verified": True}
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (b, h, tag, title) in zip(axes, [(1.0, 0.3, "shallow", "Shallow truss (h/b = 0.3): snap-through"),
                                             (1.0, 2.5, "steep", "Steep truss (h/b = 2.5): sideways buckling first")]):
        out, (xs, ys, lam, mon), branch, lc = run_case(b, h, ds=0.02, nsteps=4000, tag=tag)
        cf = out["closed_form"]
        res[tag] = out
        d = (h - ys) / h
        yy = np.linspace(ys.min(), h, 600)
        L0 = np.sqrt(b * b + h * h)
        ax.plot((h - yy) / h, EA * yy * (h * h - yy * yy) / L0 ** 3 / cf["P_lim"], color=C["mute"], lw=5, alpha=0.45,
                label="closed form (primary path)")
        stable = mon == 0
        ax.plot(d, lam / cf["P_lim"], color=C["blue"], lw=1.6, label="arc-length solver, primary path")
        ax.plot(d[~stable], lam[~stable] / cf["P_lim"], color=C["blue"], lw=0, marker=".", ms=2.5)
        if branch is not None:
            bx, by, bl = branch
            ax.plot((h - by) / h, bl / cf["P_lim"], color=C["orange"], lw=2, label="switched branch (apex moves sideways)")
        okp = [r for r in lc if r["converged"]]
        on_first = [r for r in okp if (h - r["u"][1] - h) is not None]
        ax.plot([-(r["u"][1]) / h for r in okp], [r["lam"] / cf["P_lim"] for r in okp], ls="none", marker="o",
                mfc="none", mec=C["bad"], ms=6, label="load-controlled Newton (converged steps)")
        for c in out["critical"]:
            ax.plot((h - c["y"]) / h, c["lam"] / cf["P_lim"], marker="D" if c["kind"] == "bifurcation" else "s",
                    color=C["ref"], ms=7, ls="none")
            ax.annotate("bifurcation" if c["kind"] == "bifurcation" else "limit point",
                        ((h - c["y"]) / h, c["lam"] / cf["P_lim"]), textcoords="offset points", xytext=(8, 6), fontsize=9.5)
        ax.set_xlabel("apex drop / initial height"); ax.set_ylabel("load / closed-form limit load")
        ax.set_title(title, loc="left"); ax.legend(loc="upper center", fontsize=9.5)
        ax.set_ylim(-1.3, 3.6)
    fig.savefig(RES / "r1_truss.png")
    # headline numbers
    s, t = res["shallow"], res["steep"]
    lim_s = [c for c in s["critical"] if c["kind"] == "limit"][0]
    bif_t = [c for c in t["critical"] if c["kind"] == "bifurcation"][0]
    res["summary"] = {
        "1a_limit_load": lim_s["lam"], "1a_ref": s["closed_form"]["P_lim"],
        "1a_rel_err": abs(lim_s["lam"] / s["closed_form"]["P_lim"] - 1),
        "1b_bif_load": bif_t["lam"], "1b_ref": t["closed_form"]["P_bif"],
        "1b_rel_err": abs(bif_t["lam"] / t["closed_form"]["P_bif"] - 1),
        "1b_branch_circle_err": t["branch"]["max_circle_err"], "1b_branch_load_err": t["branch"]["max_load_err"],
        "1b_branch_max_sideways_over_h": t["branch"]["max_abs_x_over_h"],
    }
    (RES / "r1_truss.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
    for tag in ("shallow", "steep"):
        print(tag, "critical:", res[tag]["critical"])
        lcs = res[tag]["load_control"]
        print(tag, "load control:", [(round(r["lam"] / res[tag]["closed_form"]["P_lim"], 3), r["converged"], r["iters"],
                                      None if r["y"] is None else round(r["y"], 3),
                                      None if r["x"] is None else round(r["x"], 3)) for r in lcs][::3])


if __name__ == "__main__":
    main()
