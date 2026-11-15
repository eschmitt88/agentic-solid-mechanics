"""Series stack of two cells as two single-cell solves coupled by one scalar equation (design solver for H2).

Exact for a series stack: the cells interact only through the common force F and the travels add up,
    F_A(d_A; p_A) = F_B(D - d_A; p_B) = F.
Each cell's displacement-controlled curve is tabulated once per design (fine grid, both directions); the stack path
is followed in D by letting d_A slide downhill in the stack energy (dE/dd_A = F_A - F_B) to the next minimum, which is
what a quasi-static test does when the stack snaps.  Every sampled point is then re-solved exactly (FE states of both
cells at d_A and D - d_A, one Newton correction on d_A) and differentiated:
    dF/dp_A = k_B/(k_A + k_B) dF_A/dp_A ,   dF/dp_B = k_A/(k_A + k_B) dF_B/dp_B ,   k = dF/dd at fixed design.
The monolithic two-cell solve (fem.Solver(ncell=2)) is the cross-check; it costs ~100x more near snaps.
"""
import json, sys, time
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

import design, fem

GP = np.round(np.arange(1, 69) * 0.25, 6)  # 0.25 .. 17 mm (only seeds the exact refinement)
GN = -np.round(np.arange(1, 9) * 0.25, 6)  # -0.25 .. -2 mm


class StackProblem(design.Problem):
    def __init__(self, tname="H2", offsets=None, log=None):
        sp = design.SPEC[tname]
        S = fem.Solver(nx=40, ny=2, Lh=design.LH, nk=sp["nk"], nm=sp["nm"], ncell=1)
        super().__init__(tname, S=S, offsets=offsets, log=log)
        self.pen, self.pen_jac, self.lb, self.ub = design.make_penalty(sp["nk"], sp["nm"], 2)
        self.offsets = [np.zeros(len(self.lb))] if offsets is None else [np.zeros(len(self.lb))] + list(offsets)

    def cell(self, p):
        S = self.S
        Fp, zp, ip = S.path(p, GP)
        Fn, zn, inn = S.path(p, GN)
        d = np.concatenate([GN[::-1], [0.0], GP])
        F = np.concatenate([Fn[::-1], [0.0], Fp])
        zs = zn[::-1] + [jnp.zeros(S.nz)] + zp
        ok = np.isfinite(F)
        jumps = int(sum("jump" in i for i in ip + inn))
        return dict(p=np.asarray(p), d=d, F=F, zs=zs, f=PchipInterpolator(d[ok], F[ok], extrapolate=True), X=S.coords(jnp.asarray(p)), jumps=jumps,
                    bad=int(np.sum(~ok)), dmin=float(d[ok].min()), dmax=float(d[ok].max()))

    def probe(self, c, d):
        S = self.S
        i = int(np.argmin(np.abs(c["d"] - d)))
        z, info = S.minimise(c["zs"][i], float(d), c["X"], tol=1e-10)
        K = S.stiffness(z, float(d), c["X"])
        F, (g, gD) = S._dF(jnp.asarray(c["p"]), float(d), z, K)
        return 2 * float(F), 2 * np.asarray(g), 2 * float(gD)

    def follow(self, A, B, Ds, dD=0.05, h=0.01):
        """d_A along the loading path (interpolated curves): slide downhill in E(d_A) = E_A(d_A) + E_B(D - d_A)."""
        G = lambda a, D: float(A["f"](a) - B["f"](D - a))
        out, a, n_snaps = [], 0.0, 0
        Dfine = np.arange(dD, Ds[-1] + 1e-9, dD)
        k = 0
        for D in Dfine:
            lo, hi = max(A["dmin"], D - B["dmax"]), min(A["dmax"], D - B["dmin"])
            a = min(max(a, lo), hi)
            g0 = G(a, D)
            step = -h if g0 > 0 else h
            a0, b = a, a
            moved = 0
            while True:
                b2 = b + step
                if b2 < lo or b2 > hi:
                    a_new = min(max(b2, lo), hi)
                    break
                g2 = G(b2, D)
                if g2 == 0 or (g2 > 0) != (g0 > 0):
                    a_new = brentq(G, min(b, b2), max(b, b2), args=(D,), xtol=1e-10)
                    break
                b = b2
                moved += 1
            if abs(a_new - a0) > 20 * dD:
                n_snaps += 1
            a = a_new
            if k < len(Ds) and abs(D - Ds[k]) < 1e-9:
                out.append(a)
                k += 1
        return np.array(out), n_snaps

    def stack_eval(self, p):
        npc = self.S.npc
        A, B = self.cell(p[:npc]), self.cell(p[npc:])
        dA, n_snaps = self.follow(A, B, self.DS)
        F, J = np.zeros(len(self.DS)), np.zeros((len(self.DS), 2 * npc))
        dAx = dA.copy()
        for k, (D, a) in enumerate(zip(self.DS, dA)):
            for _ in range(3):  # exact refinement of d_A on the FE states
                FA, gA, kA = self.probe(A, a)
                FB, gB, kB = self.probe(B, D - a)
                if abs(FA - FB) < 1e-9 or kA + kB <= 0:
                    break
                a = a - (FA - FB) / (kA + kB)
            dAx[k] = a
            ks = kA + kB
            if ks > 0:
                F[k] = (kB * FA + kA * FB) / ks
                J[k, :npc], J[k, npc:] = kB / ks * gA, kA / ks * gB
            else:  # unstable point caught by the sampling (should not happen after `follow`): no usable sensitivity
                F[k] = 0.5 * (FA + FB)
        return F, J, dict(dA=dAx, snaps=n_snaps, jumps=A["jumps"] + B["jumps"], bad=A["bad"] + B["bad"])

    # design.Problem interface --------------------------------------------------------------------------
    def evaluate(self, p):
        p = np.asarray(p, float)
        if self.cache is not None and np.array_equal(self.cache[0], p):
            return self.cache
        t0 = time.time()
        sigs, Js, metas = [], [], []
        for off in self.offsets:
            try:
                F, Jm, meta = self.stack_eval(p + off)
            except Exception as e:  # a cell whose path fails (folded geometry after a wild step): huge residual, step rejected
                F, Jm, meta = np.full(len(self.DS), 1e3 / design.KPA), np.zeros((len(self.DS), len(p))), dict(dA=None, snaps=-1, jumps=-1, bad=99, error=repr(e))
            sigs.append(F * design.KPA)
            Js.append(Jm * design.KPA)
            metas.append(meta)
        self.cache = (p.copy(), sigs, (Js, metas))
        errs = [design.nrmse(s, self.tgt) for s in sigs]
        rec = dict(p=p.tolist(), nrmse=errs[0], mean_sq=float(np.mean(np.square(errs))), bad=metas[0]["bad"], switched=metas[0]["snaps"], jumps=metas[0]["jumps"], sec=time.time() - t0)
        self.hist.append(rec)
        if self.log:
            print(f"  eval {len(self.hist):3d} nrmse={rec['nrmse']:.4f} stack_snaps={rec['switched']} cell_jumps={rec['jumps']} bad={rec['bad']} {rec['sec']:.1f}s", file=self.log, flush=True)
        return self.cache

    def jac(self, p):
        p = np.asarray(p, float)
        _, sigs, (Js, _) = self.evaluate(p)
        n, k = len(self.tgt), len(self.offsets)
        return np.vstack([Jm / self.rng / np.sqrt(n * k) for Jm in Js] + [np.asarray(self.pen_jac(jnp.asarray(p)))])


if __name__ == "__main__":
    if sys.argv[1] == "check":
        P = StackProblem(log=sys.stdout)
        p = design.starts("H2")["A4_B6"]
        _, sigs, (Js, metas) = P.evaluate(p)
        S2 = fem.Solver(nx=40, ny=2, Lh=design.LH, nk=6, nm=6, ncell=2)
        t0 = time.time()
        F2, z2, i2 = S2.path(p, P.DS)
        print("monolithic", time.time() - t0, "s; max |sigma_reduced - sigma_monolithic| kPa", np.max(np.abs(F2 * design.KPA - sigs[0])), "range", np.ptp(sigs[0]))
        print("d_A reduced vs monolithic max diff", np.max(np.abs(metas[0]["dA"] - np.array([S2.cell_travels(z, D)[0] for z, D in zip(z2, P.DS)]))))
        for j in (0, 6, 13, 19):
            h = 1e-5
            pp, pm = p.copy(), p.copy()
            pp[j] += h; pm[j] -= h
            fd = (P.stack_eval(pp)[0] - P.stack_eval(pm)[0]) / (2 * h) * design.KPA
            an = Js[0][:, j]
            print("param", j, "max |FD - analytic| / max|analytic|", np.max(np.abs(fd - an)) / np.max(np.abs(an)), "worst index", int(np.argmax(np.abs(fd - an))))
    else:
        sname = sys.argv[1]
        prob = StackProblem(log=sys.stdout)
        res = design.run("H2", sname, design.starts("H2")[sname], prob=prob)
        json.dump(res, open(f"results/opt/H2__{sname}.json", "w"), indent=1)
