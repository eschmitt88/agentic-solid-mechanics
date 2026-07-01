"""jaxlbm.py — thermal (Boussinesq) lattice-Boltzmann in JAX.

D2Q9 for the incompressible flow, D2Q5 for the temperature field, coupled by a
Guo-forced buoyancy body force (Boussinesq). Grid + mask based so the SAME solver
runs the de Vahl Davis validation cavity and the window/blind geometry:
angled slats are simply solid (bounce-back) nodes. Differentiable, GPU-native.

Conventions: arrays are (nx, ny, q). x = horizontal, y = vertical (up). Walls and
slats are full-way bounce-back (no-slip); temperature walls use anti-bounce-back
(Dirichlet) or bounce-back (adiabatic / zero-flux).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# --- D2Q9 (flow) ---
CX = jnp.array([0., 1., 0., -1., 0., 1., -1., -1., 1.])
CY = jnp.array([0., 0., 1., 0., -1., 1., 1., -1., -1.])
W = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPP = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
SHIFTS = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
CS2 = 1.0 / 3.0

# --- D2Q5 (temperature) ---
TX = jnp.array([0., 1., 0., -1., 0.])
TY = jnp.array([0., 0., 1., 0., -1.])
WT = jnp.array([1/3, 1/6, 1/6, 1/6, 1/6])
OPPT = jnp.array([0, 3, 4, 1, 2])
SHIFTS_T = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
CS2T = 1.0 / 3.0


def _stream(f, shifts):
    return jnp.stack([jnp.roll(f[..., i], shift=shifts[i], axis=(0, 1))
                      for i in range(f.shape[-1])], axis=-1)


def feq(rho, ux, uy):
    cu = CX * ux[..., None] + CY * uy[..., None]
    u2 = (ux ** 2 + uy ** 2)[..., None]
    return rho[..., None] * W * (1 + cu / CS2 + cu ** 2 / (2 * CS2 ** 2) - u2 / (2 * CS2))


def geq(T, ux, uy):
    cu = TX * ux[..., None] + TY * uy[..., None]
    return T[..., None] * WT * (1 + cu / CS2T)


def _guo(ux, uy, Gy, tauf):
    cu = CX * ux[..., None] + CY * uy[..., None]
    ax = (CX - ux[..., None]) / CS2 + cu / CS2 ** 2 * CX
    ay = (CY - uy[..., None]) / CS2 + cu / CS2 ** 2 * CY
    return (1 - 1 / (2 * tauf)) * W * (ay * Gy[..., None])  # Gx = 0


@partial(jax.jit, static_argnums=())
def step(f, g, solid, dirich, Tval, adia, tauf, taug, gbeta, T0):
    """One LBM step. solid: bounce-back mask (walls+slats). dirich: Dirichlet-T
    mask with target Tval. adia: adiabatic-T mask (top/bottom + slats)."""
    rho = f.sum(-1)
    T = g.sum(-1)
    Gy = gbeta * (T - T0)                       # buoyancy (up), Boussinesq
    ux = (jnp.tensordot(f, CX, (-1, 0))) / rho
    uy = (jnp.tensordot(f, CY, (-1, 0)) + 0.5 * Gy) / rho
    # collide
    f = f - (f - feq(rho, ux, uy)) / tauf + _guo(ux, uy, Gy, tauf)
    g = g - (g - geq(T, ux, uy)) / taug
    # stream
    f = _stream(f, SHIFTS)
    g = _stream(g, SHIFTS_T)
    # boundary conditions
    f = jnp.where(solid[..., None], f[..., OPP], f)                       # no-slip (bounce-back)
    g = jnp.where(dirich[..., None], -g[..., OPPT] + 2 * WT * Tval[..., None], g)  # Dirichlet T (anti-bounce-back)
    g = jnp.where(adia[..., None], g[..., OPPT], g)                       # adiabatic (bounce-back, zero flux)
    return f, g


def macros(f, g):
    rho = f.sum(-1)
    T = g.sum(-1)
    ux = jnp.tensordot(f, CX, (-1, 0)) / rho
    uy = jnp.tensordot(f, CY, (-1, 0)) / rho
    return rho, ux, uy, T


def init(nx, ny, T0):
    rho = jnp.ones((nx, ny))
    z = jnp.zeros((nx, ny))
    f = feq(rho, z, z)
    g = geq(jnp.full((nx, ny), T0), z, z)
    return f, g


def transport_from(Ra, Pr, H, alpha=0.03):
    """Lattice viscosity/diffusivity/buoyancy for a target (Ra, Pr) at height H
    (in lattice cells), with a chosen thermal diffusivity alpha (stability knob)."""
    nu = Pr * alpha
    gbeta = Ra * nu * alpha / H ** 3            # buoyancy accel per unit ΔT
    tauf = 0.5 + nu / CS2
    taug = 0.5 + alpha / CS2T
    return dict(tauf=tauf, taug=taug, gbeta=gbeta, nu=nu, alpha=alpha)


def run_steady(f, g, bc, p, T0, block=2000, max_blocks=200, tol=1e-7, probe=None):
    """Run to steady state in jitted blocks; stop when the probe scalar settles."""
    @jax.jit
    def run_block(f, g):
        def body(_, c):
            f, g = c
            return step(f, g, bc["solid"], bc["dirich"], bc["Tval"], bc["adia"],
                        p["tauf"], p["taug"], p["gbeta"], T0)
        return lax.fori_loop(0, block, body, (f, g))

    prev = jnp.inf
    for b in range(max_blocks):
        f, g = run_block(f, g)
        val = float(probe(f, g)) if probe is not None else float(g.sum())
        if b > 2 and abs(val - prev) < tol * (abs(val) + 1e-12):
            return f, g, b, val
        prev = val
    return f, g, max_blocks, prev
