import numpy as np
import jax
import jax.numpy as jnp
import jaxfem3d_mf as mf

# ---- problem -------------------------------------------------------------
prob = mf.TopOpt3DMatrixFree(24, 8, 8, 0.3, penal=3.0, rmin=1.5)
n = prob.nelem
vfrac = prob.volfrac
print(f"nelem={n}  volfrac={vfrac}")

# exact volume sensitivity (linear filter -> constant grad, but compute it once)
dv = np.asarray(jax.grad(prob.volume)(jnp.zeros(n)))

# ---- optimality-criteria update -----------------------------------------
def oc_update(rho, dc, move=0.2, eta=0.5):
    # dc <= 0 (compliance decreases with material). B_e = -dc / (lam * dv)
    l1, l2 = 1e-9, 1e9
    rho_new = rho
    while (l2 - l1) / (0.5 * (l1 + l2)) > 1e-6:
        lmid = 0.5 * (l1 + l2)
        Be = np.maximum(0.0, -dc / (lmid * dv))
        cand = rho * Be ** eta
        rho_new = np.clip(
            np.clip(cand, rho - move, rho + move), 0.0, 1.0
        )
        # volume = mean(filt(rho_new)); enforce <= vfrac via lambda bisection
        vol = float(prob.volume(jnp.asarray(rho_new)))
        if vol > vfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new

# ---- main loop -----------------------------------------------------------
rho = np.full(n, vfrac)
c_hist = []
for it in range(1, 201):
    c, dc = prob.value_and_grad_compliance(jnp.asarray(rho))
    c = float(c)
    dc = np.asarray(dc)
    rho_new = oc_update(rho, dc)
    change = float(np.max(np.abs(rho_new - rho)))
    rho = rho_new
    vol = float(prob.volume(jnp.asarray(rho)))
    c_hist.append(c)
    if it % 5 == 0 or it == 1:
        print(f"it {it:3d}  c={c:.6e}  vol={vol:.4f}  change={change:.4f}")
    if change < 1e-3 and it > 10:
        print(f"converged at it {it} (change={change:.4e})")
        break

c_final = float(prob.compliance(jnp.asarray(rho)))
vol_final = float(prob.volume(jnp.asarray(rho)))
print(f"\nFINAL  c={c_final:.6e}  vol={vol_final:.4f}  iters={it}")
np.save('agent_density.npy', np.asarray(rho))
print("saved agent_density.npy")
