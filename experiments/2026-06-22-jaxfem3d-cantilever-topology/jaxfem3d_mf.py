"""Matrix-free (sparse) variant of the 3D differentiable FEM.

The dense solver in jaxfem3d.py assembles the full ndof×ndof stiffness matrix and
calls jnp.linalg.solve — O(ndof²) memory, O(ndof³) time, which caps the grid at a
few thousand degrees of freedom. This variant never forms K: it applies the
stiffness operator element-by-element (gather element displacements → multiply by
the local 24×24 element stiffness → scatter-add) and solves K·u = f with a
Jacobi-preconditioned conjugate-gradient (CG) iteration. Memory is O(ndof);
arbitrarily larger grids fit on the GPU.

compliance(rho) stays end-to-end differentiable: `jax.scipy.sparse.linalg.cg`
differentiates through the linear solve via the implicit-function theorem (K is
symmetric), so `jax.grad(compliance)` gives the same sensitivities as the dense
solver — only faster and more scalable.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.sparse.linalg import cg

import jaxfem3d as j3

jax.config.update("jax_enable_x64", True)


class TopOpt3DMatrixFree(j3.TopOpt3D):
    """Same problem as TopOpt3D, but BOTH the density filter and the stiffness
    solve are matrix-free, so memory is O(ndof) instead of O(nelem²) + O(ndof²):

      * density filter — a small-radius 3D convolution instead of the dense
        nelem×nelem weight matrix H (which is itself an O(nelem²) memory wall);
      * stiffness solve — Jacobi-preconditioned conjugate gradient that applies
        K element-by-element instead of assembling and factorising it.
    """

    def __init__(self, *args, cg_tol=1e-8, cg_maxiter=4000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        self._KE_diag = jnp.diagonal(self.KE)  # (24,) for the Jacobi preconditioner

    # --- matrix-free density filter (replaces the dense H from the parent) --- #
    def _filter_weights(self, rmin):
        """Build a small convolution kernel instead of the dense H matrix.
        filt(rho) = conv(rho, k) / conv(ones, k) reproduces H·rho / Hs exactly
        (zero padding at the domain boundary matches the truncated row sums)."""
        R = int(np.ceil(rmin))
        rng = range(-R, R + 1)
        k = np.zeros((2 * R + 1,) * 3)
        for a, di in enumerate(rng):
            for b, dj in enumerate(rng):
                for c, dk in enumerate(rng):
                    k[a, b, c] = max(0.0, rmin - np.sqrt(di * di + dj * dj + dk * dk))
        self._fkernel = jnp.asarray(k)
        self._fdenom = self._conv3d(jnp.ones((self.nelx, self.nely, self.nelz)))
        return None, None  # parent stores these as self.H / self.Hs (unused here)

    def _conv3d(self, g):
        lhs = g[None, None]
        rhs = self._fkernel[None, None]
        return lax.conv_general_dilated(
            lhs, rhs, (1, 1, 1), "SAME",
            dimension_numbers=("NCDHW", "OIDHW", "NCDHW"))[0, 0]

    def filt(self, rho):
        g = rho.reshape(self.nelx, self.nely, self.nelz)
        return (self._conv3d(g) / self._fdenom).reshape(-1)

    def _K_matvec(self, u, E):
        """Apply K to a full displacement vector without assembling K."""
        ue = u[self.edof]                                   # (nelem, 24) gather
        fe = jnp.einsum("eij,ej->ei", E[:, None, None] * self.KE, ue)
        return jnp.zeros(self.ndof).at[self.edof].add(fe)   # scatter-add

    def _K_diagonal(self, E):
        """Diagonal of K (for the Jacobi preconditioner) without assembling K."""
        d_elem = E[:, None] * self._KE_diag[None, :]        # (nelem, 24)
        return jnp.zeros(self.ndof).at[self.edof].add(d_elem)

    @partial(jax.jit, static_argnums=0)
    def compliance(self, rho):
        rho_phys = self.filt(rho)
        E = self.Emin + rho_phys ** self.penal * (self.E0 - self.Emin)
        free = self.free
        ff = self.f[free]
        diag = self._K_diagonal(E)[free]

        def A(uf):
            u = jnp.zeros(self.ndof).at[free].set(uf)
            return self._K_matvec(u, E)[free]

        def M(r):                       # Jacobi (diagonal) preconditioner
            return r / diag

        uf, _ = cg(A, ff, tol=self.cg_tol, maxiter=self.cg_maxiter, M=M)
        return ff @ uf
