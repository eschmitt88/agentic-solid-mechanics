"""Loop-2 agentic harness for 3D topology optimization (trial 4).

Drives the Claude Code CLI headless (`claude -p`, subscription auth, no API key —
see trial 1/2). Each trial: the agent is handed ONLY the differentiable 3D solver
(jaxfem3d_mf.py, matrix-free) and must write its OWN gradient-based optimizer to
minimise compliance subject to a volume constraint, run it on the GPU, and save
its final design. We then INDEPENDENTLY recompute the compliance from the agent's
saved density (anti-fabrication) and grade it against the reference optimum.

This is the 3D analogue of trial 3's 2D loop-2 demo, on the now-scalable
matrix-free substrate.

Usage:
  .venv/bin/python agent_topopt3d.py --trials 5 --model claude-opus-4-8
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import jaxfem3d_mf as mf

HERE = Path(__file__).parent
PROJ = HERE.parents[1]
PY = str(PROJ / ".venv" / "bin" / "python")

# fixed problem (same as the reference optimum we grade against)
NX, NY, NZ = 24, 8, 8
VOLFRAC, PENAL, RMIN = 0.30, 3.0, 1.5

SYSTEM = (
    "You are a numerical-optimization agent. You are given a DIFFERENTIABLE 3D "
    "finite-element solver and must write your OWN gradient-based optimizer to "
    "solve a topology-optimization problem (decide where to place material to "
    "make the stiffest structure for a fixed material budget). You write Python, "
    "run it, inspect the result, and iterate until it converges. Do not look for "
    "any reference or pre-computed answer — derive the optimizer yourself."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "final_compliance": {"type": "number", "description": "compliance of your final design"},
        "final_volume": {"type": "number", "description": "mean filtered density of your final design"},
        "n_iterations": {"type": "integer"},
        "optimizer": {"type": "string", "description": "one line: what optimizer you wrote"},
    },
    "required": ["final_compliance", "final_volume"],
}


def prompt() -> str:
    return (
        "Work ENTIRELY in the current directory. Do NOT read any file outside it and "
        "do NOT look for any reference/optimal solution — write the optimizer yourself.\n\n"
        "You have `jaxfem3d_mf.py` (and its dependency `jaxfem3d.py`) in this directory. "
        "It exposes a differentiable 3D finite-element solver:\n\n"
        "    import jaxfem3d_mf as mf\n"
        f"    prob = mf.TopOpt3DMatrixFree({NX}, {NY}, {NZ}, {VOLFRAC}, penal={PENAL}, rmin={RMIN})\n"
        "    prob.nelem                      # number of design cells (length of the density vector)\n"
        "    c       = prob.compliance(rho)  # differentiable scalar; LOWER = stiffer = better\n"
        "    c, dc   = prob.value_and_grad_compliance(rho)   # value + gradient wrt rho (jax array)\n"
        "    v       = prob.volume(rho)      # mean filtered density (the volume measure)\n"
        "    vfrac   = prob.volfrac          # the volume-fraction budget (constraint upper bound)\n\n"
        "`rho` is a 1-D array of length prob.nelem with entries in [0,1] (cell material amounts).\n\n"
        "TASK — minimise prob.compliance(rho) subject to prob.volume(rho) <= prob.volfrac, with "
        "0 <= rho <= 1. Write your own gradient-based optimizer (the gradient is available above). "
        "A standard choice is the Optimality-Criteria update or projected gradient; you decide. "
        "Iterate until the design converges.\n\n"
        f"Run your code with this GPU Python interpreter:\n    JAX_PLATFORMS=cuda {PY} your_script.py\n\n"
        "When done: save your final design variable to `agent_density.npy` with\n"
        "    import numpy as np; np.save('agent_density.npy', np.asarray(rho))\n"
        "(save the SAME rho you pass to prob.compliance — the raw design variable, not a filtered "
        "copy). Then report final_compliance, final_volume, n_iterations, and a one-line optimizer "
        "description via the structured schema."
    )


def grade(keep_dir: Path, ref_compliance: float, tol_rel=0.05, vol_tol=0.02) -> dict:
    dpath = keep_dir / "agent_density.npy"
    if not dpath.exists():
        return {"density_saved": False}
    rho = np.load(dpath).reshape(-1).astype(float)
    prob = mf.TopOpt3DMatrixFree(NX, NY, NZ, VOLFRAC, penal=PENAL, rmin=RMIN)
    if rho.size != prob.nelem:
        return {"density_saved": True, "error": f"wrong size {rho.size} != {prob.nelem}"}
    recomputed = float(prob.compliance(rho))
    vol = float(prob.volume(rho))
    pct = (recomputed - ref_compliance) / ref_compliance * 100
    return {
        "density_saved": True,
        "recomputed_compliance": recomputed,
        "final_volume": vol,
        "reference_compliance": ref_compliance,
        "pct_above_reference": pct,
        "compliance_pass": recomputed <= ref_compliance * (1 + tol_rel),
        "volume_pass": vol <= VOLFRAC + vol_tol,
        "feasible": bool(recomputed <= ref_compliance * (1 + tol_rel) and vol <= VOLFRAC + vol_tol),
    }


def leakage(keep_dir: Path) -> dict:
    forbidden = ("reference3d", "metrics.json", "topopt3d_reference", "49.9", "matrixfree_topopt")
    hits = []
    for f in keep_dir.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".md", ".txt", ".json", ".sh"} and f.name != "agent_density.npy":
            try:
                t = f.read_text(errors="ignore")
            except Exception:
                continue
            for tok in forbidden:
                if tok in t:
                    hits.append({"file": f.name, "token": tok})
    return {"clean": not hits, "hits": hits[:10]}


def run_trial(model, keep_dir, ref_compliance, timeout=1800, max_turns=70) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="asm_topopt3d_"))
    for src in ("jaxfem3d_mf.py", "jaxfem3d.py"):
        shutil.copy2(HERE / src, tmp / src)
    cmd = [
        "claude", "-p", prompt(),
        "--append-system-prompt", SYSTEM,
        "--model", model, "--max-turns", str(max_turns),
        "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
        "--dangerously-skip-permissions",
    ]
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        env = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except subprocess.TimeoutExpired:
        timed_out, env = True, {}
    except json.JSONDecodeError:
        env = {"is_error": True, "result": "unparseable CLI output"}

    keep_dir.mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        if f.is_file():
            shutil.copy2(f, keep_dir / f.name)
    shutil.rmtree(tmp, ignore_errors=True)

    ans = env.get("structured_output") or {}
    g = grade(keep_dir, ref_compliance)
    lk = leakage(keep_dir)
    return {
        "submitted": bool(ans) and not env.get("is_error") and not timed_out,
        "answer": ans, "grade": g, "leakage": lk,
        "feasible": g.get("feasible", False),
        "pct_above_reference": g.get("pct_above_reference"),
        "num_turns": env.get("num_turns"),
        "cost_usd": env.get("total_cost_usd"),
        "error": ("timeout" if timed_out else env.get("result")) if not ans else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-turns", type=int, default=70)
    args = ap.parse_args()

    ref = json.loads((HERE / "metrics.json").read_text())["reference_compliance"]
    print(f"Reference compliance (grid {NX}x{NY}x{NZ}, our OC): {ref:.4f}\n")

    trials = []
    for i in range(args.trials):
        wd = HERE / "results" / "agent_runs" / f"{args.model}_trial_{i:02d}"
        try:
            r = run_trial(args.model, wd, ref, timeout=args.timeout, max_turns=args.max_turns)
        except Exception as e:  # noqa: BLE001 — never let one trial kill the batch
            r = {"submitted": False, "grade": {}, "leakage": {"clean": True}, "error": f"harness: {e}"}
        g = r.get("grade", {})
        c = g.get("recomputed_compliance")
        if r.get("feasible") and c is not None:
            print(f"trial {i}: feasible=True C={c:.3f} ({g.get('pct_above_reference'):+.2f}% vs ref) "
                  f"vol={g.get('final_volume'):.3f} leak_clean={r['leakage']['clean']} turns={r.get('num_turns')}",
                  flush=True)
        elif c is not None:
            print(f"trial {i}: INFEASIBLE C={c:.3f} ({g.get('pct_above_reference'):+.2f}% vs ref) "
                  f"vol={g.get('final_volume'):.3f}", flush=True)
        else:
            print(f"trial {i}: FAILED — {r.get('error') or g.get('error') or 'no valid density saved'}", flush=True)
        trials.append(r)

    feas = [t for t in trials if t.get("feasible")]
    pcts = [t["pct_above_reference"] for t in feas if t.get("pct_above_reference") is not None]
    summary = {
        "model": args.model, "n_trials": args.trials,
        "grid": [NX, NY, NZ], "reference_compliance": ref,
        "feasible_rate": len(feas) / args.trials if args.trials else 0,
        "all_leakage_clean": all(t["leakage"]["clean"] for t in trials),
        "best_pct_above_reference": min(pcts, default=None),
        "mean_pct_above_reference": (sum(pcts) / len(pcts)) if pcts else None,
        "trials": trials,
    }
    (HERE / "results" / f"agent_topopt3d_trials_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfeasible {len(feas)}/{args.trials}; best {summary['best_pct_above_reference']}% vs ref; "
          f"leakage clean across all = {summary['all_leakage_clean']}")


if __name__ == "__main__":
    main()
