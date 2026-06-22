"""Automated agent-as-designer harness for the TAPERED design loop (trial 2b).

Same `claude -p` subscription-auth approach as trials 1/2 (no API key). The
agent designs a tapered cantilever (b, h_root, h_tip) to minimise mass under
the FE deflection + nominal-stress constraints; each design is re-graded on the
fixed grading mesh.

Usage:  python agent_design.py --trials 5 --model claude-opus-4-8
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
    "You are an engineering design agent operating CalculiX (ccx 2.23). You iterate "
    "on a TAPERED beam design — author a deck, run ccx, read the result, revise — to "
    "MINIMISE mass under the constraints. A tapered cantilever has no closed-form "
    "deflection, so you MUST use FE. Author decks yourself (3D solid C3D20R mesh whose "
    "z-coordinates scale with the local height h(x); avoid linear hex which shear-locks). "
    "Verify the final design by FE before reporting. SI units."
)
CCX = ('PATH="$HOME/.local/bin:$PATH" MAMBA_ROOT_PREFIX="$HOME/projects/.micromamba" '
       'micromamba run -n solidmech ccx')
SCHEMA = {
    "type": "object",
    "properties": {
        "b_m": {"type": "number"}, "h_root_m": {"type": "number"}, "h_tip_m": {"type": "number"},
        "achieved_tip_deflection_m": {"type": "number"},
        "achieved_mass_kg": {"type": "number"},
        "n_designs_evaluated": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["b_m", "h_root_m", "h_tip_m"],
}


def prompt_for(prob: D.DesignProblem) -> str:
    p = prob.p
    return (
        "Work ENTIRELY in the current directory. Do NOT read any file outside it or any "
        "reference solution — design it yourself.\n\n"
        f"Run a job `beam` (reads beam.inp -> beam.dat) with:\n    {CCX} beam\n\n"
        "DESIGN a TAPERED cantilever to MINIMISE mass.\n"
        f"- Length L = {p['length_m']} m, encastre at x=0.\n"
        "- Constant width b; height varies LINEARLY from h_root at the clamp to h_tip at the "
        "free end. Design variables: b, h_root, h_tip.\n"
        f"- Steel: E = {p['youngs_modulus_pa']:g} Pa, nu = {p['poisson_ratio']}, "
        f"density = {p['density_kg_m3']} kg/m^3.\n"
        f"- Tip load P = {p['tip_load_n']} N downward (-z) at x=L.\n\n"
        "CONSTRAINTS:\n"
        f"- FE tip deflection <= {prob.delta_allow*1e3:.1f} mm (no beam formula for a taper — simulate).\n"
        f"- Max nominal bending stress over the length, 6*P*(L-x)/(b*h(x)^2), <= {prob.sigma_allow/1e6:.0f} MPa.\n"
        f"- Bounds: b in [{prob.b_bounds[0]*1e3:.0f},{prob.b_bounds[1]*1e3:.0f}] mm; "
        f"h_root, h_tip in [{prob.hr_bounds[0]*1e3:.0f},{prob.hr_bounds[1]*1e3:.0f}] mm.\n\n"
        "OBJECTIVE: minimise mass = density * b * L * (h_root + h_tip)/2. Concentrate height "
        "where the bending moment is large (near the root). Iterate to the lightest feasible "
        "taper, then report via the schema."
    )


def run_trial(model, prob, keep_dir: Path, timeout=1800, max_turns=100) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="asm_taper_"))
    cmd = ["claude", "-p", prompt_for(prob), "--append-system-prompt", SYSTEM,
           "--model", model, "--max-turns", str(max_turns),
           "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
           "--dangerously-skip-permissions"]
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
        return {"submitted": False, "error": "timeout" if timed_out else env.get("result", "no output")}
    g = prob.grade(ans["b_m"], ans["h_root_m"], ans["h_tip_m"], prob._mass_opt, keep_dir / "_grade")
    return {"submitted": True, "answer": ans, "grade": g, "feasible": g["feasible"],
            "pct_above_optimum": g["pct_above_optimum"],
            "cost_usd": env.get("total_cost_usd"), "num_turns": env.get("num_turns")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-turns", type=int, default=100)
    args = ap.parse_args()

    prob = D.load(HERE / "config.yaml")
    ref = json.loads((HERE / "results" / "reference.json").read_text())
    prob._mass_opt = ref["tapered_optimum"]["mass_kg"]
    print(f"Reference tapered optimum: {prob._mass_opt:.4f} kg\n")

    trials = []
    for i in range(args.trials):
        wd = HERE / "results" / "agent_runs" / f"{args.model}_trial_{i:02d}"
        r = run_trial(args.model, prob, wd, timeout=args.timeout, max_turns=args.max_turns)
        if r["submitted"]:
            g = r["grade"]
            print(f"trial {i}: feasible={r['feasible']} mass={g['mass_kg']:.3f}kg "
                  f"({r['pct_above_optimum']:+.1f}% vs opt) b={g['b']*1e3:.0f} "
                  f"h_root={g['h_root']*1e3:.0f} h_tip={g['h_tip']*1e3:.0f}mm "
                  f"defl={g['check_fe_deflection_m']*1e3:.3f}mm turns={r.get('num_turns')}")
        else:
            print(f"trial {i}: FAILED — {r['error']}")
        trials.append(r)

    feas = [t for t in trials if t.get("feasible")]
    summary = {"model": args.model, "n_trials": args.trials,
               "reference_optimum_mass_kg": prob._mass_opt,
               "feasible_rate": len(feas) / args.trials if args.trials else 0,
               "best_pct_above_optimum": min((t["pct_above_optimum"] for t in feas), default=None),
               "trials": trials}
    (HERE / "results" / f"agent_trials_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfeasible {len(feas)}/{args.trials}")


if __name__ == "__main__":
    main()
