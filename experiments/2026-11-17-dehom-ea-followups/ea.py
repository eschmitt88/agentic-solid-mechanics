"""Evolutionary machinery for trial 13: encoding, PCA, VAE, NSGA-II, mutation, HV.

Paper: Xu et al. 2026, Sec. 3 (Eqs. 5-6, 21, 23-30). Choices the paper leaves open
are marked OURS (from the Fable technical review, DESIGN.md / INTERFACES.md).
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Encoding (OURS): (w1, w2, theta) -> 5 channels invariant under theta -> theta + pi
# and under the swap (w1, w2, theta) -> (w2, w1, theta + pi/2):
#   s = w1 + w2,  d = (w1 - w2) (cos 2t, sin 2t),  q = (cos 4t, sin 4t).
# The paper PCA-compresses raw theta in [-4 pi, 4 pi]; raw angles have branch
# jumps that PCA/VAE blending turns into garbage.
# ----------------------------------------------------------------------------
NCH = 5


def encode(w1, w2, th):
    s = w1 + w2
    dm = w1 - w2
    return np.stack([s, dm * np.cos(2 * th), dm * np.sin(2 * th), np.cos(4 * th), np.sin(4 * th)], -1)


def decode(E, fallback_theta=None):
    """E (..., ny, nx, 5) image(s) -> w1, w2, theta. Bar 1 = the thicker bar."""
    E = np.asarray(E, float)
    s = np.clip(E[..., 0], 0.0, 2.0)
    dx, dy = E[..., 1], E[..., 2]
    qx, qy = E[..., 3], E[..., 4]
    dn = np.hypot(dx, dy)
    dn = np.minimum(dn, np.minimum(s, 2.0 - s))          # keeps 0 <= w2 <= w1 <= 1
    w1 = np.clip((s + dn) / 2, 0, 1)
    w2 = np.clip((s - dn) / 2, 0, 1)
    qn = np.hypot(qx, qy)
    # degenerate q (orientation singularity after blending): use 3x3 neighbourhood mean
    if E.ndim >= 3:
        from scipy.ndimage import uniform_filter
        size = [1] * (E.ndim - 3) + [3, 3]
        qxs, qys = uniform_filter(qx, size=size, mode="nearest"), uniform_filter(qy, size=size, mode="nearest")
        bad = qn < 0.1
        qx, qy = np.where(bad, qxs, qx), np.where(bad, qys, qy)
    tq = np.arctan2(qy, qx) / 4.0                         # theta mod pi/2
    td = np.arctan2(dy, dx) / 2.0                         # thicker-bar direction mod pi
    # Trial 14 fix: candidates k = 0..3 (quarter turns) were compared to td mod pi, so k and k+2
    # always tied exactly and rounding chose theta or theta + pi. A pi flip is the same bar but
    # flips the phasor's sign convention -> a different realisation (1e-12 input changes moved
    # compliance ~9 %, scratch/diag2x2.py). Two candidates, then canonicalise to (-pi/2, pi/2].
    cand = tq[..., None] + np.arange(2) * (np.pi / 2)
    diff = np.angle(np.exp(2j * (cand - td[..., None])))  # distance mod pi
    th = np.take_along_axis(cand, np.abs(diff).argmin(-1)[..., None], -1)[..., 0]
    th = np.where(dn > 1e-6, th, tq)
    th = th - np.pi * np.round(th / np.pi)
    if fallback_theta is not None:
        th = np.where(np.hypot(qx, qy) < 1e-3, fallback_theta, th)
    return w1, w2, th


# ----------------------------------------------------------------------------
# PCA per channel over the spatial dimension (paper Eqs. 5-6; per-channel and
# standardised = OURS, the paper shares one spatial basis across channels).
# ----------------------------------------------------------------------------
class ChannelPCA:
    def __init__(self, var=0.99, cap=128):
        self.var, self.cap = var, cap

    def fit(self, F):
        """F: (n, Nd, NCH) design-element features."""
        self.mean = F.mean(0)                              # (Nd, NCH)
        self.sd = F.reshape(-1, F.shape[-1]).std(0) + 1e-9
        self.Phi, self.ev, self.nc = [], [], []
        for c in range(F.shape[-1]):
            A = (F[..., c] - self.mean[:, c]) / self.sd[c]
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            lam = S ** 2 / (len(A) - 1)
            k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), self.var) + 1)
            k = min(k, self.cap)
            self.Phi.append(Vt[:k].T); self.ev.append(lam[:k]); self.nc.append(k)
        return self

    @property
    def dim(self):
        return int(sum(self.nc))

    def transform(self, F):
        """-> whitened coordinates (n, dim)."""
        out = []
        for c in range(F.shape[-1]):
            A = (F[..., c] - self.mean[:, c]) / self.sd[c]
            out.append((A @ self.Phi[c]) / np.sqrt(self.ev[c]))
        return np.concatenate(out, -1)

    def inverse(self, Z):
        F = np.empty(Z.shape[:-1] + self.mean.shape)
        i = 0
        for c, k in enumerate(self.nc):
            zc = Z[..., i:i + k] * np.sqrt(self.ev[c]); i += k
            F[..., c] = (zc @ self.Phi[c].T) * self.sd[c] + self.mean[:, c]
        return F

    def recon_error(self, F):
        R = self.inverse(self.transform(F))
        return float(np.sqrt(((R - F) ** 2).mean() / ((F - self.mean) ** 2).mean()))


# ----------------------------------------------------------------------------
# VAE (paper Eqs. 23-25): fully connected, 2 hidden layers each side.
# Sizes/epochs/beta = OURS (DDTD-lineage defaults per the review).
# ----------------------------------------------------------------------------
class VAE:
    def __init__(self, dim, hidden=(256, 128), latent=8, beta=1.0, epochs=1000, lr=1e-3, seed=0):
        import jax, jax.numpy as jnp, optax
        self.jax, self.jnp, self.optax = jax, jnp, optax
        self.dim, self.hidden, self.latent, self.beta = dim, hidden, latent, beta
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def _init(self, key):
        jax, jnp = self.jax, self.jnp
        sizes_e = [self.dim, *self.hidden]
        sizes_d = [self.latent, *self.hidden[::-1], self.dim]
        params = {"enc": [], "dec": []}
        for a, b in zip(sizes_e[:-1], sizes_e[1:]):
            key, k = jax.random.split(key); params["enc"].append((jax.random.normal(k, (a, b)) * np.sqrt(2 / a), jnp.zeros(b)))
        key, k1, k2 = jax.random.split(key, 3)
        h = sizes_e[-1]
        params["mu"] = (jax.random.normal(k1, (h, self.latent)) * np.sqrt(1 / h), jnp.zeros(self.latent))
        params["lv"] = (jax.random.normal(k2, (h, self.latent)) * 1e-3, jnp.zeros(self.latent))
        for a, b in zip(sizes_d[:-1], sizes_d[1:]):
            key, k = jax.random.split(key); params["dec"].append((jax.random.normal(k, (a, b)) * np.sqrt(2 / a), jnp.zeros(b)))
        return params

    def _enc(self, p, x):
        jnp = self.jnp
        for W, b in p["enc"]:
            x = jnp.tanh(x @ W + b)
        return x @ p["mu"][0] + p["mu"][1], x @ p["lv"][0] + p["lv"][1]

    def _dec(self, p, z):
        jnp = self.jnp
        for W, b in p["dec"][:-1]:
            z = jnp.tanh(z @ W + b)
        W, b = p["dec"][-1]
        return z @ W + b

    def fit(self, Z):
        jax, jnp, optax = self.jax, self.jnp, self.optax
        X = jnp.asarray(Z, dtype=jnp.float32)
        key = jax.random.PRNGKey(self.seed)
        key, k0 = jax.random.split(key)
        params = self._init(k0)
        opt = optax.adam(self.lr)
        st = opt.init(params)
        beta = self.beta

        def loss(p, k):
            mu, lv = self._enc(p, X)
            z = mu + jnp.exp(0.5 * lv) * jax.random.normal(k, mu.shape)
            xr = self._dec(p, z)
            rec = jnp.mean(jnp.sum((xr - X) ** 2, -1))
            kl = -0.5 * jnp.mean(jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), -1))
            return rec + beta * kl, (rec, kl)

        @jax.jit
        def step(p, s, k):
            (l, aux), g = jax.value_and_grad(loss, has_aux=True)(p, k)
            up, s = opt.update(g, s, p)
            return optax.apply_updates(p, up), s, l, aux

        for ep in range(self.epochs):
            key, k = jax.random.split(key)
            params, st, l, aux = step(params, st, k)
        self.params = params
        self.final_loss = (float(l), float(aux[0]), float(aux[1]))
        return self

    def encode(self, Z):
        mu, lv = self._enc(self.params, self.jnp.asarray(Z, self.jnp.float32))
        return np.asarray(mu), np.asarray(np.exp(0.5 * lv))

    def decode(self, L):
        return np.asarray(self._dec(self.params, self.jnp.asarray(L, self.jnp.float32)), float)


# ----------------------------------------------------------------------------
# NSGA-II (Deb 2002) with constrained domination; 2+ objectives, all minimised.
# ----------------------------------------------------------------------------
def nondominated_sort(F, cv=None):
    n = len(F)
    cv = np.zeros(n) if cv is None else np.asarray(cv)
    feas = cv <= 0
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dom = le & lt                                       # i dominates j (objective sense)
    # constrained domination: feasible beats infeasible; among infeasible, smaller cv wins
    dom = np.where(feas[:, None] & feas[None, :], dom,
                   np.where(feas[:, None] & ~feas[None, :], True,
                            np.where(~feas[:, None] & feas[None, :], False, cv[:, None] < cv[None, :])))
    np.fill_diagonal(dom, False)
    ndom = dom.sum(0)
    rank = np.full(n, -1)
    r = 0
    cur = np.where(ndom == 0)[0]
    while len(cur):
        rank[cur] = r
        ndom = ndom - dom[cur].sum(0)
        ndom[rank >= 0] = -1
        cur = np.where(ndom == 0)[0]
        r += 1
    return rank


def crowding(F):
    n, m = F.shape
    d = np.zeros(n)
    for k in range(m):
        o = np.argsort(F[:, k])
        span = F[o[-1], k] - F[o[0], k]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0 and n > 2:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def nsga2_select(F, k, cv=None):
    """Indices of k survivors (rank, then crowding) + their rank/crowding."""
    rank = nondominated_sort(F, cv)
    sel = []
    crowd = np.zeros(len(F))
    for r in range(rank.max() + 1):
        idx = np.where(rank == r)[0]
        c = crowding(F[idx]); crowd[idx] = c
        if len(sel) + len(idx) <= k:
            sel.extend(idx.tolist())
        else:
            sel.extend(idx[np.argsort(-c)][: k - len(sel)].tolist())
            break
    sel = np.array(sel)
    return sel, rank, crowd


def tournament(rng, rank, crowd, n):
    a, b = rng.integers(0, len(rank), (2, n))
    better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    return np.where(better, a, b)


# ----------------------------------------------------------------------------
# Hypervolume (2 objectives, minimised, normalised by a fixed gen-0 box).
# ----------------------------------------------------------------------------
def hv2d(F, ref):
    F = np.asarray(F, float)
    F = F[(F < ref).all(1)]
    if not len(F):
        return 0.0
    F = F[np.argsort(F[:, 0])]
    hv, best = 0.0, ref[1]
    for f0, f1 in F:
        if f1 < best:
            hv += (ref[0] - f0) * (best - f1)
            best = f1
    return hv


class Normaliser:
    """Fixed from the gen-0 non-dominated set (shared by every arm)."""

    def __init__(self, F0):
        rank = nondominated_sort(F0)
        P = F0[rank == 0]
        self.lo, self.hi = P.min(0), P.max(0)

    def __call__(self, F):
        return (np.asarray(F) - self.lo) / np.maximum(self.hi - self.lo, 1e-12)


# ----------------------------------------------------------------------------
# Mutation (paper Eqs. 26-30): deform the control-field image along an LF
# displacement field with radial attenuation; bicubic Keys (a = -0.5) resampling.
# Applied to the encoded channels (OURS) so angles never need interpolating.
# ----------------------------------------------------------------------------
def keys_kernel(r, a=-0.5):
    r = np.abs(r)
    return np.where(r < 1, (a + 2) * r ** 3 - (a + 3) * r ** 2 + 1,
                    np.where(r < 2, a * r ** 3 - 5 * a * r ** 2 + 8 * a * r - 4 * a, 0.0))


def bicubic_sample(img, yq, xq):
    """img (ny, nx, C); sample at fractional pixel coords (yq, xq) (ny, nx); edge-clamped."""
    ny, nx = img.shape[:2]
    y0, x0 = np.floor(yq).astype(int), np.floor(xq).astype(int)
    out = np.zeros(yq.shape + img.shape[2:])
    for m in range(-1, 3):
        wy = keys_kernel(yq - (y0 + m))
        yi = np.clip(y0 + m, 0, ny - 1)
        for n in range(-1, 3):
            wx = keys_kernel(xq - (x0 + n))
            xi = np.clip(x0 + n, 0, nx - 1)
            out += (wy * wx)[..., None] * img[yi, xi]
    return out


def mutate_image(img, ux, uy, rng, n_centres=10, r_m=10.0, p0=2.0, amp=1.5):
    """img (ny, nx, C) encoded control field; (ux, uy) LF displacement (ny, nx),
    rescaled so max |u| = amp coarse pixels (OURS; the paper gives no scale).
    n_centres perturbations applied cumulatively (Sec. 4.2.1)."""
    ny, nx = img.shape[:2]
    umax = np.sqrt(ux ** 2 + uy ** 2).max()
    ux, uy = ux / max(umax, 1e-12) * amp, uy / max(umax, 1e-12) * amp
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    out = img.copy()
    for _ in range(n_centres):
        i0, j0 = rng.uniform(0, ny), rng.uniform(0, nx)
        alpha = np.maximum(0.0, 1 - (np.hypot(yy - i0, xx - j0) / r_m) ** p0)
        # y-up arrays: sample upstream so content moves along u (paper Eq. 29 is y-down)
        out = bicubic_sample(out, yy - alpha * uy, xx - alpha * ux)
    return out
