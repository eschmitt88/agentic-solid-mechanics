"""devahl.py — validate the thermal LBM on the de Vahl Davis differentially-heated
square cavity: hot left wall, cold right wall, adiabatic top/bottom, no-slip walls.
Compare the average hot-wall Nusselt number to the published benchmark.

Reference Nu (de Vahl Davis 1983, Pr=0.71): Ra 1e3->1.118, 1e4->2.243, 1e5->4.519.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxlbm as L

HERE = Path(__file__).parent
REF = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519}
T0 = 0.5  # mean temperature; hot=1, cold=0


def setup(N):
    solid = np.zeros((N, N), bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True   # all 4 walls no-slip
    dirich = np.zeros((N, N), bool); Tval = np.zeros((N, N))
    dirich[0, :] = True;  Tval[0, :] = 1.0        # hot LEFT wall (x=0)
    dirich[-1, :] = True; Tval[-1, :] = 0.0       # cold RIGHT wall (x=N-1)
    adia = np.zeros((N, N), bool)
    adia[:, 0] = adia[:, -1] = True               # adiabatic bottom/top
    adia[0, :] = adia[-1, :] = False              # hot/cold take precedence
    return dict(solid=jnp.asarray(solid), dirich=jnp.asarray(dirich),
                Tval=jnp.asarray(Tval), adia=jnp.asarray(adia))


def nusselt_flux(f, g, N, alpha):
    """Nu from the horizontal heat flux integrated over height at each x-plane
    (convective + conductive). At steady state Nu(x) is x-independent; we report the
    central-region mean plus the min/max spread as a consistency check. ΔT=1, H=N."""
    rho = f.sum(-1); T = g.sum(-1)
    ux = jnp.tensordot(f, L.CX, (-1, 0)) / rho
    dTdx = (jnp.roll(T, -1, 0) - jnp.roll(T, 1, 0)) / 2.0
    flux = ux * T - alpha * dTdx
    planeNu = jnp.mean(flux[:, 1:-1], axis=1) * N / alpha     # Nu(x), y-integrated
    core = planeNu[N // 4:3 * N // 4]
    return float(jnp.mean(core)), float(jnp.min(planeNu[2:-2])), float(jnp.max(planeNu[2:-2]))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    Ra_list = [float(x) for x in sys.argv[2:]] or [1e4]
    out = {"N": N, "cases": []}
    for Ra in Ra_list:
        p = L.transport_from(Ra, 0.71, N, alpha=0.03)
        bc = setup(N)
        f, g = L.init(N, N, T0)
        t0 = time.time()

        def maxspeed(f, g):
            rho = f.sum(-1)
            ux = jnp.tensordot(f, L.CX, (-1, 0)) / rho
            uy = jnp.tensordot(f, L.CY, (-1, 0)) / rho
            return jnp.max(jnp.sqrt(ux ** 2 + uy ** 2))

        f, g, blocks, _ = L.run_steady(f, g, bc, p, T0, block=2000, max_blocks=600,
                                       tol=1e-6, probe=maxspeed)
        nu, nu_lo, nu_hi = nusselt_flux(f, g, N, p["alpha"])
        dt = time.time() - t0
        _, ux, uy, T = L.macros(f, g)
        umax = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
        ref = REF.get(Ra)
        err = 100 * (nu - ref) / ref if ref else None
        row = {"Ra": Ra, "Nu": round(nu, 4), "Nu_ref": ref,
               "err_pct": round(err, 2) if err is not None else None,
               "Nu_x_spread": [round(nu_lo, 3), round(nu_hi, 3)],
               "blocks": blocks, "steps": blocks * 2000, "max_speed": round(umax, 4),
               "wall_s": round(dt, 1)}
        out["cases"].append(row)
        print(f"Ra={Ra:.0e} N={N}: Nu={row['Nu']} (ref {ref}, err {row['err_pct']}%) "
              f"Nu(x)∈[{nu_lo:.2f},{nu_hi:.2f}] steps={row['steps']} umax={umax:.3f} {dt:.1f}s")
        np.save(HERE / "results" / f"devahl_T_Ra{Ra:.0e}.npy", np.asarray(T))
    (HERE / "results" / "devahl.json").write_text(json.dumps(out, indent=2))
    print("wrote results/devahl.json")


if __name__ == "__main__":
    main()
