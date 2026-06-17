"""Automated agent-as-operator harness for the cantilever trial.

Drives the **Claude Code CLI in headless mode** (`claude -p`) once per trial:
the agent authors its own CalculiX deck, runs `ccx` via Bash, parses the
output, and returns two numbers under an enforced JSON schema. Each trial is
then graded against the Euler-Bernoulli reference. pass@k over --trials.

Why the CLI and not the Anthropic SDK: `claude -p` authenticates with the
logged-in Claude Code subscription (`~/.claude/.credentials.json`) — NO
ANTHROPIC_API_KEY, billed against the Max subscription's token pool. (The
Agent SDK / raw API would need a separate API key + API credits.) Do NOT pass
--bare: it skips credential loading and the run comes back "Not logged in".

Usage (no API key needed):
  python agent_operator.py --trials 10 --model claude-sonnet-4-6

Each trial runs in results/agent_runs/trial_NN/ (isolated; the agent is told
not to read any reference solution). Solver byproducts there are gitignored.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

import cantilever as C

HERE = Path(__file__).parent

SYSTEM = (
    "You are an engineering-simulation agent operating CalculiX (ccx 2.23), a "
    "finite-element solver that reads an Abaqus-style .inp text deck. Author the "
    "deck yourself, run it, read any solver errors and fix them, parse the "
    "result, and report it. Use a 3D solid mesh suited to bending (e.g. C3D20R "
    "quadratic hex; avoid fully-integrated linear hex, which shear-locks). "
    "Refine and re-run to confirm mesh convergence before reporting. SI units."
)

CCX = ('PATH="$HOME/.local/bin:$PATH" MAMBA_ROOT_PREFIX="$HOME/projects/.micromamba" '
       'micromamba run -n solidmech ccx')

SCHEMA = {
    "type": "object",
    "properties": {
        "tip_deflection_m": {"type": "number"},
        "max_von_mises_pa": {"type": ["number", "null"]},
        "element_type": {"type": "string"},
        "ccx_runs": {"type": "integer"},
    },
    "required": ["tip_deflection_m"],
}


def prompt_for(spec: C.Spec) -> str:
    return (
        "Work ENTIRELY in the current directory. Do NOT read any file outside it, "
        "and do NOT look for or read any reference/analytical solution — author the "
        "CalculiX deck from your own knowledge (this tests whether you can operate "
        "the solver unaided).\n\n"
        f"Run a job named e.g. `beam` (reads beam.inp, writes beam.dat/beam.frd) with:\n"
        f"    {CCX} beam\n"
        "(job name WITHOUT the .inp extension).\n\n"
        "Solve for the vertical tip deflection of a cantilever beam:\n"
        f"- Length L = {spec.length_m} m (x), encastre (all DOF fixed) at x=0.\n"
        f"- Rectangular section: width b = {spec.width_m} m (y), height h = {spec.height_m} m (z).\n"
        f"- Linear elastic steel: E = {spec.youngs_modulus_pa:g} Pa, nu = {spec.poisson_ratio}.\n"
        f"- Total downward load P = {spec.tip_load_n} N (-z) at the free end x=L.\n\n"
        "Report tip_deflection_m (magnitude of z-displacement at the loaded end, in "
        "metres), and max_von_mises_pa if you can extract it, via the structured schema. "
        "Report only once mesh-converged."
    )


def _salvage(spec: C.Spec, tmp: Path) -> dict | None:
    """If the agent solved but didn't self-report, parse any *.dat it left."""
    dats = sorted(tmp.glob("*.dat"), key=lambda p: p.stat().st_mtime, reverse=True)
    for dat in dats:
        try:
            d = C.parse_displacement(dat)
            return {"tip_deflection_m": d,
                    "deflection_rel_err": C.grade(spec, d, None)["deflection_rel_err"],
                    "from_file": dat.name}
        except Exception:
            continue
    return None


def run_trial(model: str, spec: C.Spec, keep_dir: Path,
              timeout: int = 1500, max_turns: int = 60) -> dict:
    # Isolate in /tmp so the nested `claude -p` does NOT auto-load the project's
    # CLAUDE.md / hooks / skills (big per-invocation overhead) and cannot reach
    # the reference solution. Artifacts are copied to keep_dir afterward.
    tmp = Path(tempfile.mkdtemp(prefix="asm_trial_"))
    cmd = [
        "claude", "-p", prompt_for(spec),
        "--append-system-prompt", SYSTEM,
        "--model", model,
        "--max-turns", str(max_turns),
        "--output-format", "json",
        "--json-schema", json.dumps(SCHEMA),
        "--dangerously-skip-permissions",   # unattended Bash in an isolated dir
    ]
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True,
                              text=True, timeout=timeout)
        env = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except subprocess.TimeoutExpired:
        timed_out, env = True, {}
    except json.JSONDecodeError:
        env = {"is_error": True, "result": "unparseable CLI output"}

    keep_dir.mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        if f.is_file():
            shutil.copy2(f, keep_dir / f.name)

    ans = env.get("structured_output")
    if ans and "tip_deflection_m" in ans and not env.get("is_error"):
        g = C.grade(spec, ans["tip_deflection_m"], ans.get("max_von_mises_pa"))
        shutil.rmtree(tmp, ignore_errors=True)
        return {"submitted": True, "answer": ans, "grade": g,
                "deflection_pass": g["deflection_pass"],
                "cost_usd": env.get("total_cost_usd"),
                "num_turns": env.get("num_turns")}

    salvage = _salvage(spec, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"submitted": False,
            "error": "timeout" if timed_out else env.get("result", "no structured_output"),
            "salvaged_from_dat": salvage}   # solved-but-didn't-self-report, if any


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="model under test (e.g. claude-opus-4-8)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1500, help="per-trial seconds")
    ap.add_argument("--max-turns", type=int, default=60, help="agent turn budget")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    p = cfg["problem"]
    spec = C.Spec(p["length_m"], p["width_m"], p["height_m"],
                  p["youngs_modulus_pa"], p["poisson_ratio"], p["tip_load_n"])

    trials = []
    for i in range(args.trials):
        wd = HERE / "results" / "agent_runs" / f"{args.model}_trial_{i:02d}"
        r = run_trial(args.model, spec, wd, timeout=args.timeout, max_turns=args.max_turns)
        err = r.get("grade", {}).get("deflection_rel_err")
        salv = (r.get("salvaged_from_dat") or {}).get("deflection_rel_err")
        print(f"trial {i}: submitted={r['submitted']} "
              f"pass={r.get('deflection_pass')} err={err} "
              f"turns={r.get('num_turns')} cost=${r.get('cost_usd')}"
              + (f" | salvaged_err={salv}" if salv is not None else "")
              + (f" | {r['error']}" if not r['submitted'] else ""))
        trials.append(r)

    done = [t for t in trials if t["submitted"]]
    n_pass = sum(1 for t in done if t.get("deflection_pass"))
    summary = {
        "model": args.model, "n_trials": args.trials,
        "n_submitted": len(done),
        "deflection_pass_at_k": n_pass / args.trials if args.trials else 0,
        "submission_rate": len(done) / args.trials if args.trials else 0,
        "trials": trials,
    }
    out = HERE / "results" / f"agent_trials_{args.model}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\npass@{args.trials} (deflection within tol): {n_pass}/{args.trials}; "
          f"submitted {len(done)}/{args.trials}. Wrote {out.name}")


if __name__ == "__main__":
    main()
