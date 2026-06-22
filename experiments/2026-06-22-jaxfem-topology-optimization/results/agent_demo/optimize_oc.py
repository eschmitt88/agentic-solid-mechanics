"""Minimum-compliance topology optimization via the Optimality Criteria method.

Uses the differentiable JAX-FEM solver in jaxfem.py for compliance and its
autodiff sensitivities. The OC update is the classic fixed-point scheme with a
bisection on the Lagrange multiplier to enforce the volume constraint.
"""
import numpy as np
import jax
import jax.numpy as jnp

from jaxfem import TopOptProblem

NELX, NELY, VOLFRAC, PENAL, RMIN = 48, 16, 0.40, 3.0, 1.5
MAX_ITER = 200
MOVE = 0.2          # OC move limit
ETA = 0.5           # OC damping exponent
TOL = 2e-4          # convergence on relative compliance change


def oc_update(rho, dc, volfrac, move=MOVE, eta=ETA):
    """Optimality Criteria update with bisection on the Lagrange multiplier.

    dc is d(compliance)/d(rho) (negative). The OC scaling uses -dc / lambda.
    Volume constraint here is mean(physical density) <= volfrac; since the
    filter is linear and volume-preserving in the mean, we bisect on raw mean.
    """
    l1, l2 = 1e-9, 1e9
    n = rho.size
    rho_new = rho
    while (l2 - l1) / (0.5 * (l1 + l2)) > 1e-6:
        lmid = 0.5 * (l1 + l2)
        # B = -dc / lmid ; rho * sqrt(B) is the OC heuristic
        Be = np.maximum(0.0, -dc) / lmid
        rho_cand = rho * np.sqrt(Be) ** eta
        rho_new = np.clip(
            np.clip(rho_cand, rho - move, rho + move), 0.0, 1.0
        )
        # physical-density mean drives the constraint
        if prob_filt(rho_new).mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


# bind a numpy filter once we have the problem
prob_filt = None


def main():
    global prob_filt
    prob = TopOptProblem(NELX, NELY, VOLFRAC, penal=PENAL, rmin=RMIN)

    H = np.asarray(prob.H)
    Hs = np.asarray(prob.Hs)
    prob_filt = lambda r: (H @ r) / Hs

    vg = jax.jit(jax.value_and_grad(prob.compliance))

    rho = np.full(prob.nelem, VOLFRAC)
    c_prev = None
    it = 0
    for it in range(1, MAX_ITER + 1):
        c, dc = vg(jnp.asarray(rho))
        c = float(c)
        dc = np.asarray(dc)

        rho_old = rho.copy()
        rho = oc_update(rho, dc, VOLFRAC)

        change = np.max(np.abs(rho - rho_old))
        vol = float(prob.volume(jnp.asarray(rho)))
        rel = abs(c - c_prev) / c if c_prev is not None else 1.0
        print(f"it {it:3d}  c={c:.6f}  vol={vol:.4f}  change={change:.4f}  rel={rel:.2e}")

        if rel < TOL and it > 10:
            break
        c_prev = c

    # final evaluation
    c_final = float(prob.compliance(jnp.asarray(rho)))
    v_final = float(prob.volume(jnp.asarray(rho)))
    np.save("agent_density.npy", rho)

    print("=== RESULT ===")
    print(f"final_compliance: {c_final}")
    print(f"final_volume: {v_final}")
    print(f"n_iterations: {it}")


if __name__ == "__main__":
    main()
