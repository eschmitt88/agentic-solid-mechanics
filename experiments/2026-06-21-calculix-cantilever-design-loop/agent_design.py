"""Automated agent-as-designer harness for the cantilever design loop (trial 2).

Drives the Claude Code CLI headless (`claude -p`, subscription auth, no API key —
see trial 1's agent_operator.py) once per trial: the agent iterates on the
cross-section (b, h) — author deck, run ccx, measure deflection, revise — to
minimize mass subject to the deflection + nominal-stress constraints, then
reports its final design. Each design is re-graded on the fixed grading mesh.

Usage:
  python agent_design.py --trials 5 --model claude-opus-4-8
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import design as D

HERE = Path(__file__).parent

SYSTEM = (
    "You are an engineering design agent operating CalculiX (ccx 2.23), a finite "
    "element solver reading an Abaqus-style .inp text deck. You iterate on a "
    "design: author a deck, run ccx, read the result, and revise — to MINIMISE "
    "mass while satisfying the constraints. Author decks yourself (a 3D solid "
    "mesh suited to bending, e.g. C3D20R; avoid fully-integrated linear hex). "
    "Verify your final design by FE before reporting. SI units."
)

CCX = ('PATH="$HOME/.local/bin:$PATH" MAMBA_ROOT_PREFIX="$HOME/projects/.micromamba" '
       'micromamba run -n solidmech ccx')

SCHEMA = {
    "type": "object",
    "properties": {
        "b_m": {"type": "number", "description": "chosen width in metres"},
        "h_m": {"type": "number", "description": "chosen height in metres"},
        "achieved_tip_deflection_m": {"type": "number"},
        "achieved_mass_kg": {"type": "number"},
        "n_designs_evaluated": {"type": "integer"},
        "reasoning": {"type": "string", "description": "one or two sentences on the design choice"},
    },
    "required": ["b_m", "h_m"],
}


def prompt_for(prob: D.DesignProblem) -> str:
    p = prob.p
    return (
        "Work ENTIRELY in the current directory. Do NOT read any file outside it "
        "and do NOT look for any reference/optimal solution — design it yourself.\n\n"
        f"Run a job `beam` (reads beam.inp → beam.dat) with:\n    {CCX} beam\n\n"
        "DESIGN TASK — choose the cross-section of a cantilever beam to MINIMISE its "
        "mass while meeting the constraints.\n"
        f"- Length L = {p['length_m']} m, encastre (all DOF fixed) at x=0.\n"
        f"- Rectangular section: width b (along y) and height h (along z) are YOUR design variables.\n"
        f"- Steel: E = {p['youngs_modulus_pa']:g} Pa, nu = {p['poisson_ratio']}, "
        f"density = {p['density_kg_m3']} kg/m^3.\n"
        f"- Tip load: total P = {p['tip_load_n']} N downward (-z) at the free end x=L.\n\n"
        "CONSTRAINTS:\n"
        f"- Tip vertical deflection must be <= {prob.delta_allow*1e3:.1f} mm.\n"
        f"- Nominal bending stress 6*P*L/(b*h^2) must be <= {prob.sigma_allow/1e6:.0f} MPa.\n"
        f"- Bounds: {prob.b_bounds[0]*1e3:.0f} mm <= b <= {prob.b_bounds[1]*1e3:.0f} mm; "
        f"{prob.h_bounds[0]*1e3:.0f} mm <= h <= {prob.h_bounds[1]*1e3:.0f} mm.\n\n"
        "OBJECTIVE: minimise mass = density * L * b * h. Iterate (try a design, solve, "
        "check the deflection, revise) until you have the lightest feasible section, then "
        "report b_m, h_m, your FE-measured deflection, the mass, and how many designs you "
        "evaluated — via the structured schema."
    )


def run_trial(model: str, prob: D.DesignProblem, keep_dir: Path,
              timeout: int = 1800, max_turns: int = 80) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="asm_design_"))
    cmd = [
        "claude", "-p", prompt_for(prob),
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

    ans = env.get("structured_output")
    if not ans or "b_m" not in ans or env.get("is_error"):
        return {"submitted": False,
                "error": "timeout" if timed_out else env.get("result", "no structured_output")}
    # Re-grade the agent's reported design on the fixed grading mesh.
    g = prob.grade(ans["b_m"], ans["h_m"], prob._mass_opt, keep_dir / "_grade")
    return {"submitted": True, "answer": ans, "grade": g,
            "feasible": g["feasible"], "pct_above_optimum": g["pct_above_optimum"],
            "cost_usd": env.get("total_cost_usd"), "num_turns": env.get("num_turns")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-turns", type=int, default=80)
    args = ap.parse_args()

    prob = D.load(HERE / "config.yaml")
    ref = json.loads((HERE / "results" / "reference.json").read_text())
    prob._mass_opt = ref["fe_optimum"]["mass_kg"]
    print(f"Reference optimum: {prob._mass_opt:.4f} kg (b={ref['fe_optimum']['b']*1e3:.1f} mm, "
          f"h={ref['fe_optimum']['h']*1e3:.2f} mm)\n")

    trials = []
    for i in range(args.trials):
        wd = HERE / "results" / "agent_runs" / f"{args.model}_trial_{i:02d}"
        r = run_trial(args.model, prob, wd, timeout=args.timeout, max_turns=args.max_turns)
        if r["submitted"]:
            g = r["grade"]
            print(f"trial {i}: feasible={r['feasible']} mass={g['mass_kg']:.3f} kg "
                  f"(+{r['pct_above_optimum']:.1f}% vs opt) b={g['b']*1e3:.1f}mm h={g['h']*1e3:.1f}mm "
                  f"defl={g['check_fe_deflection_m']*1e3:.3f}mm turns={r.get('num_turns')}")
        else:
            print(f"trial {i}: FAILED — {r['error']}")
        trials.append(r)

    feas = [t for t in trials if t.get("feasible")]
    summary = {
        "model": args.model, "n_trials": args.trials,
        "reference_optimum_mass_kg": prob._mass_opt,
        "feasible_rate": len(feas) / args.trials if args.trials else 0,
        "best_pct_above_optimum": min((t["pct_above_optimum"] for t in feas), default=None),
        "mean_pct_above_optimum": (sum(t["pct_above_optimum"] for t in feas) / len(feas)) if feas else None,
        "trials": trials,
    }
    (HERE / "results" / f"agent_trials_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfeasible {len(feas)}/{args.trials}; "
          f"best +{summary['best_pct_above_optimum']}% above optimum")


if __name__ == "__main__":
    main()
