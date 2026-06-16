"""Automated agent-as-operator harness for the cantilever trial.

Drives an LLM through the operator loop with two tools — `run_ccx` (write a
.inp deck, run CalculiX, get stdout + .dat back) and `submit_answer` — then
grades the submitted answer against the Euler-Bernoulli reference.

This is the *repeatable / batch* version of the live subagent demo recorded in
log.md. It needs:
  * `ANTHROPIC_API_KEY` in the environment,
  * the `anthropic` package (run via: `uv run --with anthropic python agent_operator.py`),
  * CalculiX in the micromamba `solidmech` env (same as the deterministic sweep).

Default model is the latest Opus (claude-opus-4-8); pass --model claude-sonnet-4-6
for a cheaper run. Runs are nondeterministic — use --trials N for pass@k.

Usage:
  ANTHROPIC_API_KEY=... uv run --with anthropic python agent_operator.py --trials 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

import cantilever as C

HERE = Path(__file__).parent

SYSTEM = """You are an engineering-simulation agent operating CalculiX (ccx 2.23), \
a finite-element solver that reads an Abaqus-style .inp text deck. You author the \
deck yourself, run it with the run_ccx tool, fix any errors from the solver \
output, parse the result, and submit the answer. Use a 3D solid mesh suited to \
bending (e.g. C3D20R quadratic hex; avoid fully-integrated linear hex, which \
shear-locks). Refine and re-run to confirm mesh convergence before submitting. \
Work in SI units."""

TOOLS = [
    {
        "name": "run_ccx",
        "description": "Write a CalculiX .inp deck and run it. Returns the ccx "
                       "stdout and the first part of the .dat results file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_name": {"type": "string", "description": "job name, no extension"},
                "inp_contents": {"type": "string", "description": "full text of the .inp deck"},
            },
            "required": ["job_name", "inp_contents"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final result once you are confident it is mesh-converged.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tip_deflection_m": {"type": "number"},
                "max_von_mises_pa": {"type": ["number", "null"]},
                "notes": {"type": "string"},
            },
            "required": ["tip_deflection_m"],
        },
    },
]


def prompt_for(spec: C.Spec) -> str:
    return (
        "Solve for the vertical tip deflection of a cantilever beam.\n"
        f"- Length L = {spec.length_m} m (x), encastre (all DOF fixed) at x=0.\n"
        f"- Rectangular section: width b = {spec.width_m} m (y), height h = {spec.height_m} m (z).\n"
        f"- Linear elastic steel: E = {spec.youngs_modulus_pa:g} Pa, nu = {spec.poisson_ratio}.\n"
        f"- Total downward load P = {spec.tip_load_n} N (-z) at the free end x=L.\n"
        "Report the tip vertical deflection magnitude in metres (and max von Mises in Pa "
        "if you can). Submit only once mesh-converged."
    )


def make_run_ccx(workdir: Path, env: str, ccx_bin: str):
    workdir.mkdir(parents=True, exist_ok=True)

    def run_ccx(job_name: str, inp_contents: str) -> str:
        job = workdir / job_name
        job.with_suffix(".inp").write_text(inp_contents)
        try:
            proc = subprocess.run(
                ["micromamba", "run", "-n", env, ccx_bin, job_name],
                cwd=workdir, capture_output=True, text=True, timeout=1800,
            )
        except subprocess.TimeoutExpired:
            return "ccx TIMED OUT after 1800 s — your mesh is likely too large."
        out = [f"ccx return code: {proc.returncode}", "--- stdout (tail) ---",
               proc.stdout[-3000:]]
        if proc.returncode != 0:
            out += ["--- stderr (tail) ---", proc.stderr[-1500:]]
        dat = job.with_suffix(".dat")
        if dat.exists():
            out += ["--- beginning of .dat ---", dat.read_text()[:4000]]
        return "\n".join(out)

    return run_ccx


def run_trial(client, model: str, spec: C.Spec, workdir: Path,
              env: str, ccx_bin: str, max_steps: int = 12) -> dict:
    run_ccx = make_run_ccx(workdir, env, ccx_bin)
    messages = [{"role": "user", "content": prompt_for(spec)}]
    submitted, n_runs = None, 0

    for _ in range(max_steps):
        resp = client.messages.create(model=model, max_tokens=8000,
                                       system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name == "run_ccx":
                n_runs += 1
                result = run_ccx(**block.input)
            elif block.name == "submit_answer":
                submitted = block.input
                result = "Answer recorded."
            else:
                result = f"unknown tool {block.name}"
            tool_results.append({"type": "tool_result",
                                 "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})
        if submitted is not None:
            break

    if submitted is None:
        return {"submitted": False, "ccx_runs": n_runs}
    g = C.grade(spec, submitted["tip_deflection_m"],
                submitted.get("max_von_mises_pa"))
    return {"submitted": True, "ccx_runs": n_runs,
            "answer": submitted, "grade": g,
            "deflection_pass": g["deflection_pass"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the agentic harness. "
                         "(The deterministic ground truth is reference_sweep.py.)")
    import anthropic

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    p = cfg["problem"]
    spec = C.Spec(p["length_m"], p["width_m"], p["height_m"],
                  p["youngs_modulus_pa"], p["poisson_ratio"], p["tip_load_n"])
    env, ccx_bin = cfg["solver"]["micromamba_env"], cfg["solver"]["ccx_bin"]
    client = anthropic.Anthropic()

    trials = []
    for i in range(args.trials):
        wd = HERE / "results" / "agent_runs" / f"trial_{i:02d}"
        r = run_trial(client, args.model, spec, wd, env, ccx_bin)
        passed = r.get("deflection_pass")
        err = r.get("grade", {}).get("deflection_rel_err")
        print(f"trial {i}: submitted={r['submitted']} pass={passed} "
              f"err={err} ccx_runs={r['ccx_runs']}")
        trials.append(r)

    n_pass = sum(1 for t in trials if t.get("deflection_pass"))
    summary = {"model": args.model, "n_trials": args.trials,
               "deflection_pass_at_k": n_pass / args.trials if args.trials else 0,
               "trials": trials}
    (HERE / "results" / "agent_trials.json").write_text(json.dumps(summary, indent=2))
    print(f"\npass@{args.trials} (deflection): {n_pass}/{args.trials}")


if __name__ == "__main__":
    main()
