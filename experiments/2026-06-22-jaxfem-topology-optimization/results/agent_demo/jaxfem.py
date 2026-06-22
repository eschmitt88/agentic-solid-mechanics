"""Differentiable 2D plane-stress FEM for topology optimization (JAX).

A compact, self-contained differentiable forward model — bilinear Q4 elements,
SIMP material interpolation, density filter — exposing a `compliance(rho)` that
is end-to-end autodifferentiable. `jax.grad(compliance)` gives the sensitivities
that drive gradient-based topology optimization (the loop-2 mechanism).

This stands in for JAX-FEM (the project's production differentiable solver) so
trial 3 is self-contained and transparent; the agentic loop is identical. Runs
on GPU automatically when JAX sees one, else CPU (device-agnostic).

Problem: a 2D cantilever — left edge fully clamped, unit downward load at the
middle of the right edge — minimise compliance fᵀu subject to a volume fraction.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _KE(nu: float = 0.3) -> jnp.ndarray:
    """8x8 plane-stress stiffness for a unit square, E=1."""
    k = jnp.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                   -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    idx = jnp.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ])
    return k[idx] / (1 - nu**2)


class TopOptProblem:
    """Holds geometry/BCs/filter; provides differentiable compliance & volume."""

    def __init__(self, nelx, nely, volfrac, penal=3.0, rmin=1.5,
                 E0=1.0, Emin=1e-9, nu=0.3):
        self.nelx, self.nely = nelx, nely
        self.volfrac, self.penal = volfrac, penal
        self.E0, self.Emin = E0, Emin
        self.nelem = nelx * nely
        self.KE = _KE(nu)

        ndof = 2 * (nelx + 1) * (nely + 1)
        self.ndof = ndof

        # element -> 8 global dofs
        edof = []
        for elx in range(nelx):
            for ely in range(nely):
                n1 = elx * (nely + 1) + ely
                n2 = (elx + 1) * (nely + 1) + ely
                n3 = (elx + 1) * (nely + 1) + (ely + 1)
                n4 = elx * (nely + 1) + (ely + 1)
                ns = [n1, n2, n3, n4]
                edof.append([d for n in ns for d in (2 * n, 2 * n + 1)])
        self.edof = jnp.array(edof)                      # (nelem, 8)

        # cantilever BCs: clamp the left edge (column 0 nodes)
        fixed = []
        for ely in range(nely + 1):
            n = 0 * (nely + 1) + ely
            fixed += [2 * n, 2 * n + 1]
        free = jnp.array([d for d in range(ndof) if d not in set(fixed)])
        self.free = free

        # unit downward load at middle of right edge
        f = jnp.zeros(ndof)
        n_load = nelx * (nely + 1) + (nely // 2)
        f = f.at[2 * n_load + 1].set(-1.0)
        self.f = f

        # density filter weights H (normalised), built on element-centre grid
        self.H, self.Hs = self._filter_weights(rmin)

    def _filter_weights(self, rmin):
        nelx, nely = self.nelx, self.nely
        cx = jnp.repeat(jnp.arange(nelx), nely).astype(float)   # element col
        cy = jnp.tile(jnp.arange(nely), nelx).astype(float)     # element row
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dist = jnp.sqrt(dx**2 + dy**2)
        H = jnp.maximum(0.0, rmin - dist)
        return H, jnp.sum(H, axis=1)

    def filt(self, rho):
        """Density filter: physical density = H·rho / Hs."""
        return (self.H @ rho) / self.Hs

    def _assemble(self, E_elem):
        Ke = E_elem[:, None, None] * self.KE[None]           # (nelem,8,8)
        rows = jnp.broadcast_to(self.edof[:, :, None], (self.nelem, 8, 8))
        cols = jnp.broadcast_to(self.edof[:, None, :], (self.nelem, 8, 8))
        K = jnp.zeros((self.ndof, self.ndof))
        return K.at[rows, cols].add(Ke)

    @partial(jax.jit, static_argnums=0)
    def compliance(self, rho):
        """Differentiable compliance fᵀu for design field rho in [0,1]."""
        rho_phys = self.filt(rho)
        E = self.Emin + rho_phys**self.penal * (self.E0 - self.Emin)
        K = self._assemble(E)
        Kff = K[jnp.ix_(self.free, self.free)]
        ff = self.f[self.free]
        uf = jnp.linalg.solve(Kff, ff)
        return ff @ uf

    @partial(jax.jit, static_argnums=0)
    def volume(self, rho):
        """Mean physical density (the volume-fraction quantity)."""
        return jnp.mean(self.filt(rho))

    def value_and_grad_compliance(self, rho):
        return jax.value_and_grad(self.compliance)(rho)
