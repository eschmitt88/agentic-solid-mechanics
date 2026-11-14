import numpy as np


def pts(d, F, h):
    """f_top, f_bot, d_top, d_bot, d_mid, d_end from a sampled curve (parabolic refinement of extrema)."""
    def refine(i):
        if 0 < i < len(d) - 1:
            c = np.polyfit(d[i - 1 : i + 2], F[i - 1 : i + 2], 2)
            x = -c[1] / (2 * c[0])
            return x, np.polyval(c, x)
        return d[i], F[i]

    it = np.argmax(np.where(d < h, F, -np.inf))
    ib = np.argmin(np.where(d > h, F, np.inf))
    dt, ft = refine(it)
    db, fb = refine(ib)
    m = d > 0.5 * h
    dd, FF = d[m], F[m]
    s = np.sign(FF)
    zc = np.where(s[:-1] * s[1:] < 0)[0]
    z = [dd[i] - FF[i] * (dd[i + 1] - dd[i]) / (FF[i + 1] - FF[i]) for i in zc]
    return dict(f_top=ft, f_bot=-fb, d_top=dt, d_bot=db, d_mid=z[0] if z else np.nan, d_end=z[-1] if len(z) > 1 else np.nan)
