"""Unit checks of fem.py: uniaxial tension vs closed form, FD check of sensitivities."""
import numpy as np, jax, jax.numpy as jnp, time
import fem
from meshutil import rect_t6

def uniaxial_closed_form(lam, p, E=None, nu=None):
    """True (Cauchy=Kirchhoff/J) uniaxial response in log strain; returns nominal stress P = T/lam."""
    E = fem.ELASTIC['E']
    from scipy.optimize import brentq
    eps = np.log(lam)
    sy = lambda a: float(fem.sigma_y(a, jnp.asarray(p)))
    out = []
    for e in np.atleast_1d(eps):
        if E * e <= sy(0.0): T = E * e
        else:
            a = brentq(lambda a: E * (e - a) - sy(a), 0, e)
            T = sy(a)
        out.append(T / np.exp(e))
    return np.array(out)

def run_uniaxial(p, lam_end=1.5, nst=26, nx=2, ny=2):
    nodes, el = rect_t6(10.0, 20.0, nx, ny)
    m = fem.Mesh(nodes, el, 1.5)
    bot = np.where(nodes[:, 1] < 1e-9)[0]; top = np.where(nodes[:, 1] > 20 - 1e-9)[0]
    presc = np.concatenate([2 * bot + 1, 2 * top + 1, [2 * bot[0]]])
    lam = np.linspace(1, lam_end, nst)
    def ubar(s):
        l = np.interp(s, np.arange(nst), lam)
        return np.concatenate([np.zeros(len(bot)), np.full(len(top), 20 * (l - 1)), [0.0]])
    S = fem.Solver(m, presc)
    F = []
    S.run(ubar, nst, p, observe=lambda k, u, R, du, dR, st: F.append(R[2 * top + 1].sum()))
    return lam[1:], np.array(F) / (10 * 1.5), S

if __name__ == '__main__':
    p = np.array([300.0, 600.0, 4.0, 800.0, 100.0, 40.0, 1.0, 0.5, 3.0])
    lam, P, S = run_uniaxial(p)
    Pc = uniaxial_closed_form(lam, p)
    print('uniaxial: max rel err vs closed form %.2e' % np.max(np.abs(P / Pc - 1)), S.stats)
    # anisotropic: uniaxial along y must be independent of a,h,c in stress
    p2 = p.copy(); p2[6:] = [1.3, 0.35, 2.4]
    lam, P2, S = run_uniaxial(p2)
    print('Hill uniaxial-y: max rel err vs closed form %.2e' % np.max(np.abs(P2 / Pc - 1)))
    # sensitivities vs FD on a heterogeneous problem (shear + tension of a block with clamped ends)
    nodes, el = rect_t6(10.0, 10.0, 3, 3)
    m = fem.Mesh(nodes, el, 1.5)
    bot = np.where(nodes[:, 1] < 1e-9)[0]; top = np.where(nodes[:, 1] > 10 - 1e-9)[0]
    presc = np.concatenate([2 * bot, 2 * bot + 1, 2 * top, 2 * top + 1])
    nst = 9
    def ubar(s):
        f = s / (nst - 1)
        return np.concatenate([np.zeros(2 * len(bot)), np.full(len(top), 1.2 * f), np.full(len(top), 2.0 * f)])
    def solve(p, sens=None):
        S = fem.Solver(m, presc); obs = {}
        def ob(k, u, R, du, dR, st):
            obs[k] = (R[2 * top + 1].sum(), u.copy(), None if du is None else (dR[2 * top + 1].sum(0), du.copy()))
        S.run(ubar, nst, p, sens_idx=sens, observe=ob)
        return obs
    pt = p2.copy()
    idx = list(range(9))
    t0 = time.time(); o = solve(pt, idx); print('sens solve %.1fs' % (time.time() - t0))
    k = nst - 1
    for j, i in enumerate(idx):
        hstep = 1e-5 * max(abs(pt[i]), 1e-3)
        pp = pt.copy(); pp[i] += hstep; pm = pt.copy(); pm[i] -= hstep
        op, om = solve(pp), solve(pm)
        fdF = (op[k][0] - om[k][0]) / (2 * hstep); fdu = (op[k][1] - om[k][1]) / (2 * hstep)
        aF = o[k][2][0][j]; au = o[k][2][1][:, j]
        print(f'{fem.PNAMES[i]:3s} dF/dp AD {aF:+.6e} FD {fdF:+.6e} rel {abs(aF - fdF) / (abs(fdF) + 1e-30):.1e} | du/dp rel {np.linalg.norm(au - fdu) / (np.linalg.norm(fdu) + 1e-30):.1e}')
