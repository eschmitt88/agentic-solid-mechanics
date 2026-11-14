"""Qiu (MIT PhD thesis 2003, ch. 2; Qiu, Lang & Slocum JMEMS 2004) cosine-beam formulas.

Source opened this session: https://pergatory.mit.edu/research/jqiu/thesisscreen.pdf
Equation numbers are the thesis'.  Normalisation (2.22): F = f l^3/(E I h), Delta = d/h,
N^2 = p l^2/(E I).  N_j = (j+1) pi for j = 1, 3, 5, ...
"""
import numpy as np

PI = np.pi
N1 = 2 * PI
N3 = 4 * PI


def F1_closed(Delta, Q):
    """Eq (2.38): mode-1 solution, higher modes neglected."""
    return 1.5 * PI**4 * Q**2 * Delta * (Delta**2 - 3 * Delta + 2 + 4 / (3 * Q**2))


def F3_closed(Delta):
    """Eq (2.41)."""
    return 8 * PI**4 - 6 * PI**4 * Delta


def approx_points(E, I, h, l):
    """Eqs (2.47)-(2.48), Q >= 6, second mode constrained."""
    c = E * I * h / l**3
    return dict(f_top=740 * c, f_bot=370 * c, d_top=0.16 * h, d_bot=1.92 * h, d_end=1.99 * h, d_mid=4 * h / 3)


def _modes(nmodes):
    return np.array([(j + 1) * PI for j in range(1, 4 * nmodes, 4)])  # j = 1,5,9,...


def series_curve(Q, nmodes=20, n=4001):
    """F-Delta with higher modes kept, from eqs (2.23), (2.25), (2.31), (2.32), (2.36).

    Branch 1 (N^2 < N3^2): parametrise by N^2; eq (2.25) is a quadratic in F.
    Branch 3 (N^2 = N3^2): A3 free, F from (2.46).
    Returns Delta, F sorted along the physical (mode-2-constrained) path.
    """
    Nj = _modes(nmodes)
    out_D, out_F = [], []
    # branch 1: for each N2 in (-large, N3^2) solve quadratic for F
    # A1 = -N1^2/(2 (N2 - N1^2)) + 4F/(N1^2 (N2 - N1^2));  Aj = 4F/(Nj^2 (N2 - Nj^2))
    # (2.25): N2/(12 Q^2) = N1^2/16 - sum_j Aj^2 Nj^2 / 4
    N2s = np.concatenate([np.linspace(-3000, N1**2 - 1e-6, n)[:-1], np.linspace(N1**2 + 1e-6, N3**2, n)])
    for N2 in N2s:
        c = 4 / (Nj**2 * (N2 - Nj**2))  # Aj = c_j F (+ a0 for j=1)
        a0 = np.zeros_like(Nj)
        a0[0] = -N1**2 / (2 * (N2 - N1**2))
        # sum (a0 + c F)^2 Nj^2/4 = N1^2/16 - N2/(12Q^2)
        qa = np.sum(c**2 * Nj**2) / 4
        qb = np.sum(2 * a0 * c * Nj**2) / 4
        qc = np.sum(a0**2 * Nj**2) / 4 - (N1**2 / 16 - N2 / (12 * Q**2))
        disc = qb**2 - 4 * qa * qc
        if disc < 0:
            continue
        for sg in (+1, -1):
            F = (-qb + sg * np.sqrt(disc)) / (2 * qa)
            A = a0 + c * F
            out_D.append(1 - 2 * np.sum(A))
            out_F.append(F)
    D1b = np.array(out_D)
    F1b = np.array(out_F)
    o = np.argsort(D1b)
    D1b, F1b = D1b[o], F1b[o]
    # branch 3: N2 = N3^2, Delta = 1 - 2 sum Aj, Aj as above (A3 takes up the slack, needs A3^2 >= 0)
    c = 4 / (Nj**2 * (N3**2 - Nj**2))
    a0 = np.zeros_like(Nj)
    a0[0] = -N1**2 / (2 * (N3**2 - N1**2))
    Fs = np.linspace(-2000, 2000, n)
    A = a0[None, :] + c[None, :] * Fs[:, None]
    D3 = 1 - 2 * A.sum(1)
    A3sq = (N1**2 / 16 - N3**2 / (12 * Q**2) - np.sum(A**2 * Nj**2, axis=1) / 4) * 4 / N3**2
    ok = A3sq >= 0
    D3b, F3b = D3[ok], Fs[ok]
    if ok.any():
        lo, hi = D3b.min(), D3b.max()
        keep = (D1b < lo) | (D1b > hi)
        D = np.concatenate([D1b[keep], D3b])
        F = np.concatenate([F1b[keep], F3b])
    else:
        D, F = D1b, F1b
    o = np.argsort(D)
    D, F = D[o], F[o]
    m = (D >= -0.05) & (D <= 2.3)
    return D[m], F[m]


def curve_points(D, F):
    """f_top (first local max), f_bot (min), zero crossings, from a sampled curve."""
    i_top = np.argmax(F[D < 1.0])
    i_bot = np.argmin(np.where(D > 1.0, F, np.inf))
    res = dict(F_top=F[i_top], D_top=D[i_top], F_bot=-F[i_bot], D_bot=D[i_bot])
    sgn = np.sign(F)
    zc = np.where((sgn[:-1] * sgn[1:] < 0) & (D[:-1] > 0.05))[0]
    zs = [D[i] - F[i] * (D[i + 1] - D[i]) / (F[i + 1] - F[i]) for i in zc]
    if len(zs) >= 2:
        res["D_mid"], res["D_end"] = zs[0], zs[1]
    return res


if __name__ == "__main__":
    for Q in (3, 6, 12):
        D, F = series_curve(Q)
        print(Q, curve_points(D, F))
