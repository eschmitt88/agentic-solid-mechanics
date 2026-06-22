"""Reference 3D topology optimization (no LLM): Optimality-Criteria minimisation
of compliance at a fixed volume fraction, using JAX autodiff sensitivities from
jaxfem3d.py. Establishes the reference 3D structure and compliance, saves the
density field (.npy, shape (nelx,nely,nelz)).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import yaml

import jaxfem3d as j3

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def oc_update(prob, rho, dc, dv, move=0.2):
    l1, l2 = 0.0, 1e9
    rho = np.asarray(rho); dc = np.asarray(dc); dv = np.asarray(dv)
    while (l2 - l1) / max(l1 + l2, 1e-12) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        rho_new = np.clip(rho * np.sqrt(np.maximum(-dc / dv / lmid, 0.0)),
                          np.maximum(0.0, rho - move), np.minimum(1.0, rho + move))
        if float(prob.volume(rho_new)) > prob.volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


def run(prob, cfg):
    rho = np.full(prob.nelem, prob.volfrac)
    history = []
    move = cfg["optim"]["move"]
    for it in range(cfg["optim"]["max_iter"]):
        c, dc = prob.value_and_grad_compliance(rho)
        _, dv = jax.value_and_grad(prob.volume)(rho)
        c = float(c)
        rho_new = oc_update(prob, rho, dc, dv, move=move)
        change = float(np.max(np.abs(rho_new - rho)))
        rho = rho_new
        vol = float(prob.volume(rho))
        history.append({"iter": it, "compliance": c, "volume": vol, "change": change})
        print(f"it {it:2d}  C={c:.5f}  vol={vol:.3f}  change={change:.4f}")
        if change < cfg["optim"]["tol_change"]:
            break
    return {"rho": np.asarray(prob.filt(rho)), "history": history,
            "final_compliance": history[-1]["compliance"],
            "final_volume": history[-1]["volume"]}


def main():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    p = cfg["problem"]
    prob = j3.TopOpt3D(p["nelx"], p["nely"], p["nelz"], p["volfrac"],
                       penal=p["penal"], rmin=p["rmin"])
    print(f"3D problem: {p['nelx']}x{p['nely']}x{p['nelz']} = {prob.nelem} elems, "
          f"ndof={prob.ndof}, volfrac={p['volfrac']}, device={jax.devices()[0]}")
    res = run(prob, cfg)
    RESULTS.mkdir(exist_ok=True)
    rho3d = np.asarray(res["rho"]).reshape(p["nelx"], p["nely"], p["nelz"])
    np.save(RESULTS / "reference3d_density.npy", rho3d)
    out = {"final_compliance": res["final_compliance"],
           "final_volume": res["final_volume"],
           "history": res["history"], "problem": p}
    (RESULTS / "reference3d.json").write_text(json.dumps(out, indent=2))
    (HERE / "metrics.json").write_text(json.dumps({
        "reference_compliance": res["final_compliance"],
        "reference_volume": res["final_volume"],
        "n_iterations": len(res["history"]),
        "grid": [p["nelx"], p["nely"], p["nelz"]],
        "ndof": int(prob.ndof),
    }, indent=2))
    print(f"\nReference 3D compliance: {res['final_compliance']:.5f}  "
          f"volume: {res['final_volume']:.3f}")
    print("Wrote results/reference3d.json, reference3d_density.npy, metrics.json")


if __name__ == "__main__":
    main()
