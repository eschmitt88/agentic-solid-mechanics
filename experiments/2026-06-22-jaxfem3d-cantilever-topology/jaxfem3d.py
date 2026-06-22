"""Differentiable 3D FEM for topology optimization (JAX).

8-node trilinear hexahedral elements (B8), 3D isotropic elasticity, SIMP
material interpolation, 3D density filter — an end-to-end autodifferentiable
`compliance(rho)`. `jax.grad(compliance)` gives the sensitivities that drive
gradient-based 3D topology optimisation. The 3D analogue of the trial-3 2D
solver; stands in for production JAX-FEM, GPU-native via JAX.

Cantilever: the root face (x=0) is fully clamped; a unit downward (-y) load is
applied at the centre of the free-end face (x=L). Unit-cube elements, E0=1.

Element/node ordering (local hex, matches the Gauss-quadrature stiffness):
  0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
Global node index for grid node (i,j,k): i*(nely+1)*(nelz+1) + j*(nelz+1) + k.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# local node corner offsets (x,y,z) in {0,1}, standard hex ordering
_NODES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)


def _D(nu: float) -> np.ndarray:
    """3D isotropic constitutive matrix (E=1), Voigt order (xx,yy,zz,xy,yz,zx)."""
    c = 1.0 / ((1 + nu) * (1 - 2 * nu))
    g = (1 - 2 * nu) / 2
    return c * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, g, 0, 0],
        [0, 0, 0, 0, g, 0],
        [0, 0, 0, 0, 0, g],
    ])


def _hex_KE(nu: float = 0.3) -> np.ndarray:
    """24x24 stiffness of a unit-cube B8 hex (E=1) via 2x2x2 Gauss quadrature."""
    D = _D(nu)
    gp = 1.0 / np.sqrt(3.0)
    pts = [(-gp, gp)[s] for s in range(2)]  # [-1/sqrt3, +1/sqrt3]
    signs = _NODES * 2 - 1                    # corner (xi,eta,zeta) signs in {-1,+1}
    KE = np.zeros((24, 24))
    # reference [-1,1]^3 -> unit cube [0,1]^3: J = 0.5 I, detJ = 1/8, dN/dx = 2 dN/dxi
    detJ, dxi_to_dx = 0.125, 2.0
    for xi in pts:
        for eta in pts:
            for zeta in pts:
                # shape-function derivatives wrt (xi,eta,zeta) at this Gauss point
                dN = np.zeros((8, 3))
                for a in range(8):
                    sx, sy, sz = signs[a]
                    dN[a, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta)
                    dN[a, 1] = 0.125 * sy * (1 + sx * xi) * (1 + sz * zeta)
                    dN[a, 2] = 0.125 * sz * (1 + sx * xi) * (1 + sy * eta)
                dN *= dxi_to_dx  # -> wrt physical x,y,z
                B = np.zeros((6, 24))
                for a in range(8):
                    bx, by, bz = dN[a]
                    c = 3 * a
                    B[0, c] = bx
                    B[1, c + 1] = by
                    B[2, c + 2] = bz
                    B[3, c] = by; B[3, c + 1] = bx
                    B[4, c + 1] = bz; B[4, c + 2] = by
                    B[5, c] = bz; B[5, c + 2] = bx
                KE += (B.T @ D @ B) * detJ  # weights = 1 for 2-pt Gauss
    return KE


class TopOpt3D:
    """3D topology-optimization problem: differentiable compliance & volume."""

    def __init__(self, nelx, nely, nelz, volfrac, penal=3.0, rmin=1.5,
                 E0=1.0, Emin=1e-9, nu=0.3):
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        self.volfrac, self.penal = volfrac, penal
        self.E0, self.Emin = E0, Emin
        self.nelem = nelx * nely * nelz
        self.KE = jnp.asarray(_hex_KE(nu))

        ny1, nz1 = nely + 1, nelz + 1
        nnode = (nelx + 1) * ny1 * nz1
        self.ndof = 3 * nnode

        def nid(i, j, k):
            return i * ny1 * nz1 + j * nz1 + k

        edof = []
        for ei in range(nelx):
            for ej in range(nely):
                for ek in range(nelz):
                    dofs = []
                    for (ox, oy, oz) in _NODES.astype(int):
                        n = nid(ei + ox, ej + oy, ek + oz)
                        dofs += [3 * n, 3 * n + 1, 3 * n + 2]
                    edof.append(dofs)
        self.edof = jnp.array(edof)  # (nelem, 24), element order ei->ej->ek

        # clamp root face x=0 (all dofs of nodes with i=0)
        fixed = set()
        for j in range(ny1):
            for k in range(nz1):
                n = nid(0, j, k)
                fixed.update({3 * n, 3 * n + 1, 3 * n + 2})
        self.free = jnp.array([d for d in range(self.ndof) if d not in fixed])

        # unit downward (-y) load at centre of free-end face x=L
        f = np.zeros(self.ndof)
        n_load = nid(nelx, nely // 2, nelz // 2)
        f[3 * n_load + 1] = -1.0
        self.f = jnp.asarray(f)

        self.H, self.Hs = self._filter_weights(rmin)

    def _filter_weights(self, rmin):
        nelx, nely, nelz = self.nelx, self.nely, self.nelz
        # element-centre coords in the same ei->ej->ek order as edof
        idx = np.arange(self.nelem)
        ei = idx // (nely * nelz)
        ej = (idx // nelz) % nely
        ek = idx % nelz
        c = np.stack([ei, ej, ek], axis=1).astype(float)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        H = np.maximum(0.0, rmin - d)
        return jnp.asarray(H), jnp.asarray(H.sum(1))

    def filt(self, rho):
        return (self.H @ rho) / self.Hs

    def _assemble(self, E_elem):
        Ke = E_elem[:, None, None] * self.KE[None]
        rows = jnp.broadcast_to(self.edof[:, :, None], (self.nelem, 24, 24))
        cols = jnp.broadcast_to(self.edof[:, None, :], (self.nelem, 24, 24))
        K = jnp.zeros((self.ndof, self.ndof))
        return K.at[rows, cols].add(Ke)

    @partial(jax.jit, static_argnums=0)
    def compliance(self, rho):
        rho_phys = self.filt(rho)
        E = self.Emin + rho_phys ** self.penal * (self.E0 - self.Emin)
        K = self._assemble(E)
        Kff = K[jnp.ix_(self.free, self.free)]
        ff = self.f[self.free]
        uf = jnp.linalg.solve(Kff, ff)
        return ff @ uf

    @partial(jax.jit, static_argnums=0)
    def volume(self, rho):
        return jnp.mean(self.filt(rho))

    def value_and_grad_compliance(self, rho):
        return jax.value_and_grad(self.compliance)(rho)
