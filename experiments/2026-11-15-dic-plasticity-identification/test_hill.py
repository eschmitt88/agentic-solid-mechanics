"""Closed-form checks of the anisotropic part (not covered by the CalculiX cross-check, which is von Mises):
uniaxial tension along y -> lateral log strain = -nu T/E - h eps_p ;  along x -> stress = sigma_y(alpha)/sqrt(a)."""
import numpy as np, jax.numpy as jnp
import fem
from meshutil import rect_t6
from scipy.optimize import brentq
p = np.array([330.6, 379.3, 5.147, 842.2, 0.0, 1.0, 1.0735, 0.4460, 2.928])
E, nu = fem.ELASTIC["E"], fem.ELASTIC["nu"]
sy = lambda a: float(fem.sigma_y(a, jnp.asarray(p)))
def run(axis, lam_end=1.4, nst=21):
    L = [10.0, 20.0]
    nodes, el = rect_t6(L[0], L[1], 2, 2); m = fem.Mesh(nodes, el, 1.5)
    c = axis; lo = np.where(nodes[:, c] < 1e-9)[0]; hi = np.where(nodes[:, c] > L[c] - 1e-9)[0]
    presc = np.concatenate([2 * lo + c, 2 * hi + c, [2 * lo[0] + (1 - c)]])
    lam = np.linspace(1, lam_end, nst)
    ub = lambda s: np.concatenate([np.zeros(len(lo)), np.full(len(hi), L[c] * (np.interp(s, np.arange(nst), lam) - 1)), [0.0]])
    S = fem.Solver(m, presc); rec = {}
    S.run(ub, nst, p, observe=lambda k, u, R, du, dR, st: rec.update(F=R[2 * hi + c].sum(), u=u.copy()))
    o = 1 - c
    lam_lat = 1 - (rec["u"][o::2].max() - rec["u"][o::2].min()) / L[o]  # lateral contraction
    return lam_end, rec["F"] / (L[o] * 1.5), lam_lat
# along y
lam, P, lam_lat = run(1)
e = np.log(lam); a = brentq(lambda a: E * (e - a) - sy(a), 0, e); T = sy(a)
print("y-tension: stress rel err %.2e ; lateral log strain FE %.6f closed form %.6f (rel err %.2e)" % (P * lam / T - 1, np.log(lam_lat), -nu * T / E - p[7] * a, np.log(lam_lat) / (-nu * T / E - p[7] * a) - 1))
# along x: phi = sqrt(a) T  ->  T = sy(alpha)/sqrt(a), plastic work T d eps_p = sy d alpha  ->  eps_p = alpha sqrt(a)
lam, P, lam_lat = run(0)
e = np.log(lam); sa = np.sqrt(p[6])
a = brentq(lambda a: E * (e - a * sa) - sy(a) / sa, 0, e); T = sy(a) / sa
print("x-tension: stress rel err %.2e ; lateral log strain FE %.6f closed form %.6f (rel err %.2e)" % (P * lam / T - 1, np.log(lam_lat), -nu * T / E - (p[7] / p[6]) * a * sa, np.log(lam_lat) / (-nu * T / E - (p[7] / p[6]) * a * sa) - 1))
