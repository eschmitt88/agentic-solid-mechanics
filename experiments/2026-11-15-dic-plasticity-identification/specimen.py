"""A specimen = cached DIC data + mesh + boundary conditions + DIC/FE levelling operators.
simulate(p) returns forces at the 41 stations and levelled Green-Lagrange strain fields at the three
field stations, with optional exact sensitivities."""
import os, numpy as np, scipy.sparse as sp
from scipy.spatial import cKDTree
import fem, data
from meshutil import mesh_outline, clip_loop_y, clean_loop

HERE = os.path.dirname(os.path.abspath(__file__))
THICK = 1.5


def strain_operator(xy, radius=1.0, kmax=96):
    """Sparse Gx, Gy (n x n): local affine least-squares gradient over neighbours within radius."""
    n = len(xy)
    tree = cKDTree(xy)
    d, idx = tree.query(xy, k=kmax, distance_upper_bound=radius)
    mask = np.isfinite(d)
    idx = np.where(mask, idx, 0)
    dx = (xy[idx] - xy[:, None, :]) * mask[..., None]
    A = np.concatenate([mask[..., None].astype(float), dx], -1)  # n,k,3
    AtA = np.einsum("nki,nkj->nij", A, A)
    AtA += 1e-12 * np.eye(3)
    W = np.einsum("nij,nkj->nik", np.linalg.inv(AtA), A)  # n,3,k
    rows = np.repeat(np.arange(n), kmax)
    Gx = sp.csr_matrix((np.where(mask, W[:, 1, :], 0).ravel(), (rows, idx.ravel())), shape=(n, n))
    Gy = sp.csr_matrix((np.where(mask, W[:, 2, :], 0).ravel(), (rows, idx.ravel())), shape=(n, n))
    return Gx, Gy, mask.sum(1)


def gl_strain(Gx, Gy, U, V):
    F11 = 1 + Gx @ U; F12 = Gy @ U; F21 = Gx @ V; F22 = 1 + Gy @ V
    Exx = 0.5 * (F11 * F11 + F21 * F21 - 1); Eyy = 0.5 * (F12 * F12 + F22 * F22 - 1); Exy = 0.5 * (F11 * F12 + F21 * F22)
    return np.stack([Exx, Eyy, Exy], 0), (F11, F12, F21, F22)


def gl_strain_tangent(Gx, Gy, Fc, dU, dV):
    F11, F12, F21, F22 = Fc
    d11 = Gx @ dU; d12 = Gy @ dU; d21 = Gx @ dV; d22 = Gy @ dV
    b = (lambda a: a[:, None]) if dU.ndim == 2 else (lambda a: a)
    dExx = b(F11) * d11 + b(F21) * d21
    dEyy = b(F12) * d12 + b(F22) * d22
    dExy = 0.5 * (b(F11) * d12 + d11 * b(F12) + b(F21) * d22 + d21 * b(F22))
    return np.stack([dExx, dEyy, dExy], 0)


def interp_matrix(nodes, elems, pts):
    """Sparse (npts x nn) T6 interpolation; returns matrix and inside-mask."""
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems[:, :3])
    e = tri.get_trifinder()(pts[:, 0], pts[:, 1])
    inside = e >= 0
    ee = e[inside]; P = pts[inside]
    Xe = nodes[elems[ee]]  # m,6,2
    a = Xe[:, 0]; J = np.stack([Xe[:, 1] - a, Xe[:, 2] - a], -1)
    xi = np.linalg.solve(J, (P - a)[..., None])[..., 0]
    for _ in range(4):  # Newton on the (possibly curved) isoparametric map
        N, dN = _shape_batch(xi)
        x = np.einsum("ma,mai->mi", N, Xe)
        Jm = np.einsum("mai,maj->mij", Xe, dN)
        xi = xi - np.linalg.solve(Jm, (x - P)[..., None])[..., 0]
    N, _ = _shape_batch(xi)
    rows = np.repeat(np.where(inside)[0], 6)
    A = sp.csr_matrix((N.ravel(), (rows, elems[ee].ravel())), shape=(len(pts), len(nodes)))
    return A, inside


def _shape_batch(xi):
    x, e = xi[:, 0], xi[:, 1]; z = 1 - x - e
    N = np.stack([z * (2 * z - 1), x * (2 * x - 1), e * (2 * e - 1), 4 * z * x, 4 * x * e, 4 * e * z], 1)
    o = np.zeros_like(x)
    dN = np.stack([np.stack([-(4 * z - 1), -(4 * z - 1)], 1), np.stack([4 * x - 1, o], 1), np.stack([o, 4 * e - 1], 1),
                   np.stack([4 * (z - x), -4 * x], 1), np.stack([4 * e, 4 * x], 1), np.stack([-4 * e, 4 * (z - e)], 1)], 1)
    return N, dN


class Specimen:
    def __init__(self, spec, role="calibration", h=2.5, h_fine=1.0, tag="", nodes=None, elems=None, verbose=True):
        self.spec = spec
        self.d = d = dict(np.load(data.prepare(spec, role)))
        self.ylo, self.yhi = float(d["ylo"]), float(d["yhi"])
        bo, bh = data.load_geometry(data.geom_key(spec))
        self.outline_full, self.holes = bo, bh
        outer = clip_loop_y(clean_loop(bo), self.ylo, self.yhi)
        holes = [hh for hh in bh if hh[:, 1].min() > self.ylo and hh[:, 1].max() < self.yhi]
        if len(holes) != len(bh):
            raise ValueError("a hole crosses the cut line")
        self.outer = outer
        if nodes is None:
            nodes, elems = mesh_outline(outer, holes, h, h_fine, workdir=os.path.join(HERE, "results", "cache", "mesh"),
                                        name=f"{spec}_h{h}_{h_fine}{tag}")
        self.mesh = fem.Mesh(nodes, elems, THICK)
        self.h, self.h_fine = h, h_fine
        nd = self.mesh.nodes
        self.bot = np.where(np.abs(nd[:, 1] - self.ylo) < 1e-6)[0]
        self.top = np.where(np.abs(nd[:, 1] - self.yhi) < 1e-6)[0]
        self.presc = np.concatenate([2 * self.bot, 2 * self.bot + 1, 2 * self.top, 2 * self.top + 1])
        # boundary values at stations
        c = d["coef"]  # (NST, line, comp, 4)
        def vals(nodes_i, il):
            A = np.vander((nd[nodes_i, 0] - 30.0) / 30.0, 4)
            return np.einsum("na,jca->jcn", A, c[:, il])  # NST, comp, n
        vb, vt = vals(self.bot, 0), vals(self.top, 1)
        self.ubar_st = np.concatenate([vb[:, 0], vb[:, 1], vt[:, 0], vt[:, 1]], 1)  # NST, npresc
        # DIC points and operators
        pts = np.stack([d["X0"], d["Y0"]], 1)
        valid = d["valid"] & (pts[:, 1] > self.ylo + 0.2) & (pts[:, 1] < self.yhi - 0.2)
        A, inside = interp_matrix(self.mesh.nodes, self.mesh.elems, pts)
        self.keep = keep = valid & inside
        self.pts = pts[keep]
        self.A = A[keep]
        self.Gx, self.Gy, self.nnb = strain_operator(self.pts, 1.0)
        self.E_dic = np.stack([gl_strain(self.Gx, self.Gy, d["U"][i][keep], d["V"][i][keep])[0] for i in range(3)], 0)  # 3 st,3 comp,n
        self.F_meas = d["Fst"]
        self.force_ok = ~d["drop"]
        self.force_ok[: data.FORCE_ST0] = False
        self.topy = 2 * self.top + 1
        if verbose:
            print(f"[{spec}] mesh nn={self.mesh.nn} ne={self.mesh.ne} | DIC pts kept {keep.sum()}/{len(keep)} "
                  f"(valid {d['valid'].sum()}, inside {inside.sum()}) | force stations {self.force_ok.sum()}")

    def ubar_fn(self, nsub=1):
        ub = self.ubar_st
        def f(s):
            s = s / nsub
            i = min(int(np.floor(s)), len(ub) - 2); w = s - i
            return (1 - w) * ub[i] + w * ub[i + 1]
        return f

    def simulate(self, p, sens_idx=None, nsub=1, last_station=None, want_u=False, elastic=False, hook=None):
        nst = data.NST if last_station is None else last_station + 1
        S = fem.Solver(self.mesh, self.presc)
        nk = 0 if sens_idx is None else len(sens_idx)
        F = np.full(data.NST, np.nan); F[0] = 0.0
        dF = np.zeros((data.NST, nk))
        E = {}; dE = {}; us = {}
        fs = list(data.FIELD_ST)
        def obs(k, u, R, du, dR, state):
            if k % nsub:
                return
            j = k // nsub
            F[j] = R[self.topy].sum()
            if hook is not None:
                hook(j, u, R, state)
            if nk:
                dF[j] = dR[self.topy].sum(0)
            if j in fs:
                Uf = self.A @ u[0::2]; Vf = self.A @ u[1::2]
                E[j], Fc = gl_strain(self.Gx, self.Gy, Uf, Vf)
                if nk:
                    dE[j] = gl_strain_tangent(self.Gx, self.Gy, Fc, self.A @ du[0::2], self.A @ du[1::2])
                if want_u:
                    us[j] = (u.copy(), state)
        pp = np.array(p, float)
        if elastic:
            pp = pp.copy(); pp[0] = 1e9
        S.run(self.ubar_fn(nsub), (nst - 1) * nsub + 1, pp, sens_idx=sens_idx, observe=obs)
        out = {"F": F, "dF": dF, "E": E, "dE": dE, "stats": S.stats}
        if want_u:
            out["u"] = us
        return out

    # ---- pre-registered metrics --------------------------------------------------------------
    def metrics(self, sim):
        ok = self.force_ok & np.isfinite(sim["F"])
        Fm, Fs = self.F_meas[ok], sim["F"][ok]
        m = {"force_nrmse": float(np.sqrt(np.mean((Fs - Fm) ** 2)) / np.mean(Fm))}
        jl = np.where(ok)[0][-1]
        m["force_last_err"] = float((sim["F"][jl] - self.F_meas[jl]) / self.F_meas[jl]); m["force_last_station"] = int(jl)
        w = np.array([1, 1, 2.0])[:, None]
        for i, j in enumerate(data.FIELD_ST):
            if j in sim["E"]:
                Em, Es = self.E_dic[i], sim["E"][j]
                m[f"strain_relL2_st{j}"] = float(np.sqrt((w * (Es - Em) ** 2).sum() / (w * Em ** 2).sum()))
        if 40 in sim["E"]:
            def hot(E):
                e1 = 0.5 * (E[0] + E[1]) + np.sqrt(0.25 * (E[0] - E[1]) ** 2 + E[2] ** 2)
                k = max(1, int(0.01 * len(e1)))
                return float(np.sort(e1)[-k:].mean())
            hm, hs = hot(self.E_dic[2]), hot(sim["E"][40])
            m["hot_dic"], m["hot_fe"], m["hot_err"] = hm, hs, (hs - hm) / hm
        return m
