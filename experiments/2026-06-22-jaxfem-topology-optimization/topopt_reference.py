"""Reference topology optimization (no LLM): Optimality-Criteria minimisation of
compliance at a fixed volume fraction, using the JAX autodiff sensitivities from
jaxfem.py. Establishes the reference compliance the agent is graded against, and
saves the optimal density field (npy + pgm + ascii).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

import jaxfem

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def oc_update(prob, rho, dc, dv, move=0.2):
    """Optimality-criteria update with bisection on the volume multiplier."""
    l1, l2 = 0.0, 1e9
    rho = np.asarray(rho)
    dc, dv = np.asarray(dc), np.asarray(dv)
    while (l2 - l1) / max(l1 + l2, 1e-12) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        rho_new = np.clip(
            rho * np.sqrt(np.maximum(-dc / dv / lmid, 0.0)),
            np.maximum(0.0, rho - move), np.minimum(1.0, rho + move))
        if float(prob.volume(rho_new)) > prob.volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


def run(prob, cfg) -> dict:
    rho = np.full(prob.nelem, prob.volfrac)
    history = []
    move = cfg["optim"]["move"]
    for it in range(cfg["optim"]["max_iter"]):
        c, dc = prob.value_and_grad_compliance(rho)
        _, dv = __import__("jax").value_and_grad(prob.volume)(rho)
        c = float(c)
        rho_new = oc_update(prob, rho, dc, dv, move=move)
        change = float(np.max(np.abs(rho_new - rho)))
        rho = rho_new
        history.append({"iter": it, "compliance": c, "volume": float(prob.volume(rho)),
                        "change": change})
        print(f"it {it:2d}  C={c:.5f}  vol={float(prob.volume(rho)):.3f}  change={change:.4f}")
        if change < cfg["optim"]["tol_change"]:
            break
    return {"rho": np.asarray(prob.filt(rho)), "history": history,
            "final_compliance": history[-1]["compliance"],
            "final_volume": history[-1]["volume"]}


def save_density(rho_phys, nelx, nely, stem: Path):
    img = np.asarray(rho_phys).reshape(nelx, nely).T   # (nely, nelx), row 0 = top
    np.save(stem.with_suffix(".npy"), img)
    g = (np.clip(1 - img, 0, 1) * 255).astype(np.uint8)   # solid = dark
    with open(stem.with_suffix(".pgm"), "wb") as fh:
        fh.write(f"P5\n{nelx} {nely}\n255\n".encode())
        fh.write(g.tobytes())
    chars = " .:-=+*#%@"
    ascii_rows = ["".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row)
                  for row in img]
    return "\n".join(ascii_rows)


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    p = cfg["problem"]
    prob = jaxfem.TopOptProblem(p["nelx"], p["nely"], p["volfrac"],
                                penal=p["penal"], rmin=p["rmin"])
    print(f"Problem: {p['nelx']}x{p['nely']} elements, ndof={prob.ndof}, "
          f"volfrac={p['volfrac']}, device={__import__('jax').devices()[0]}")
    res = run(prob, cfg)
    RESULTS.mkdir(exist_ok=True)
    art = save_density(res["rho"], p["nelx"], p["nely"], RESULTS / "reference_density")
    print("\nOptimised topology (solid = dense):\n" + art)
    print(f"\nReference compliance: {res['final_compliance']:.5f}  "
          f"volume: {res['final_volume']:.3f}")

    out = {"final_compliance": res["final_compliance"], "final_volume": res["final_volume"],
           "history": res["history"], "problem": p, "ascii": art}
    (RESULTS / "reference.json").write_text(json.dumps(out, indent=2))
    (HERE / "metrics.json").write_text(json.dumps({
        "reference_compliance": res["final_compliance"],
        "reference_volume": res["final_volume"],
        "n_iterations": len(res["history"]),
    }, indent=2))
    print("\nWrote results/reference.json, reference_density.{npy,pgm}, metrics.json")


if __name__ == "__main__":
    main()
