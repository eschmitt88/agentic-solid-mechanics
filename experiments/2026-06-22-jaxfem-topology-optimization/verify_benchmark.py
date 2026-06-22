"""verify_benchmark.py — external-gold verification of the trial-3 machinery.

The trial-3 grading compares the agent to a reference optimum *we* compute. This
script checks that reference machinery against targets defined OUTSIDE this repo,
answering "how do we know our reference is right":

  (1) FORWARD-MODEL gold (closed form): our differentiable FEM, on a uniform
      cantilever, must reproduce analytical beam theory (Euler-Bernoulli +
      Timoshenko shear). compliance(ones) == tip deflection under the unit load.

  (2) OPTIMIZER gold (published problem): reproduce the canonical MBB beam
      [Sigmund 2001, "A 99 line topology optimization code"; Andreassen et al.
      2011, "Efficient topology optimization in MATLAB using 88 lines of code",
      Struct Multidisc Optim 43:1-16] — the textbook benchmark. The problem
      definition, BCs, and the expected optimal topology come from the
      literature, not from us; reproducing them validates our solver+optimizer.

Run (GPU auto):  ~/projects/research/agentic-solid-mechanics/.venv/bin/python verify_benchmark.py
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import jaxfem
import topopt_reference as tr

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).parent
RESULTS = HERE / "results"
NU = 0.3


# --------------------------------------------------------------------------- #
# (1) forward-model gold — analytical beam theory
# --------------------------------------------------------------------------- #
def beam_theory_check() -> dict:
    """Uniform-density cantilever: FEM tip deflection vs EB and Timoshenko.

    Unit-square elements => length L=nelx, height h=nely, thickness t=1, E=1,
    nu=0.3, point load P=1 at mid-right edge, left edge clamped. Compliance
    f.u with a unit point load equals the deflection at the loaded node."""
    rows = []
    for nelx, nely in [(100, 10), (160, 8), (48, 16)]:
        prob = jaxfem.TopOptProblem(nelx, nely, 0.5, penal=1.0, rmin=1.5)
        delta = float(prob.compliance(jnp.ones(prob.nelem)))  # = tip deflection (P=1)
        L, h, E = float(nelx), float(nely), 1.0
        I = h ** 3 / 12.0
        d_eb = L ** 3 / (3 * E * I)
        G, kappa, A = E / (2 * (1 + NU)), 5.0 / 6.0, h * 1.0
        d_timo = d_eb + L / (kappa * G * A)
        rows.append({
            "L_over_h": L / h,
            "fem_tip_deflection": delta,
            "euler_bernoulli": d_eb,
            "timoshenko": d_timo,
            "err_vs_eb_pct": 100 * (delta - d_eb) / d_eb,
            "err_vs_timoshenko_pct": 100 * (delta - d_timo) / d_timo,
        })
    worst_timo = max(abs(r["err_vs_timoshenko_pct"]) for r in rows)
    return {"rows": rows, "max_abs_err_vs_timoshenko_pct": worst_timo,
            "pass": worst_timo < 2.0}


# --------------------------------------------------------------------------- #
# (2) optimizer gold — canonical MBB beam
# --------------------------------------------------------------------------- #
def make_mbb(nelx, nely, volfrac, penal, rmin) -> jaxfem.TopOptProblem:
    """Build the MBB beam by reusing the trial-3 solver and swapping only the
    boundary conditions + load to the published MBB definition (88-line code):
      * left edge: symmetry -> fix x-DOF of every left-column node;
      * bottom-right node: vertical roller -> fix its y-DOF;
      * load: unit downward force at the top-left node.
    Everything else (element stiffness, connectivity, density filter, assembly,
    differentiable solve) is the unmodified trial-3 machinery."""
    prob = jaxfem.TopOptProblem(nelx, nely, volfrac, penal=penal, rmin=rmin)
    ndof = prob.ndof
    fixed = {2 * n for n in range(nely + 1)}          # x-dofs of left column (n=ely)
    br = nelx * (nely + 1) + nely                      # bottom-right node
    fixed.add(2 * br + 1)                              # its y-dof (roller)
    prob.free = jnp.array([d for d in range(ndof) if d not in fixed])
    f = np.zeros(ndof)
    f[2 * 0 + 1] = -1.0                                # downward load at top-left node
    prob.f = jnp.asarray(f)
    return prob


def mbb_check() -> dict:
    # standard textbook settings
    nelx, nely, volfrac, penal, rmin = 60, 20, 0.5, 3.0, 1.5
    prob = make_mbb(nelx, nely, volfrac, penal, rmin)
    cfg = {"optim": {"max_iter": 250, "move": 0.2, "tol_change": 0.005}}
    res = tr.run(prob, cfg)
    art = tr.save_density(res["rho"], nelx, nely, RESULTS / "mbb_density")
    return {
        "problem": {"nelx": nelx, "nely": nely, "volfrac": volfrac,
                    "penal": penal, "rmin": rmin, "filter": "density (88-line ft=2)"},
        "our_compliance": res["final_compliance"],
        "our_volume": res["final_volume"],
        "n_iterations": len(res["history"]),
        "literature_reference": {
            "source": "Sigmund 2001 (99-line) / Andreassen et al. 2011 (88-line)",
            "rigorous_check": "topology",
            "note": "The unambiguous external gold is the reproduced canonical MBB "
                    "topology (top + bottom chords with a triangulated diagonal web). "
                    "The published compliance SCALAR is setup-sensitive (filter type, "
                    "rmin, penalty continuation) and typically falls in the ~200-220 "
                    "range for 60x20/vf0.5/p3; we report ours rather than assert a "
                    "single literature number we cannot re-derive offline.",
        },
        "ascii": art,
    }


def main():
    print("device:", jax.devices()[0])
    print("\n=== (1) forward-model gold: FEM vs beam theory ===")
    beam = beam_theory_check()
    for r in beam["rows"]:
        print(f"  L/h={r['L_over_h']:4.1f}  FEM={r['fem_tip_deflection']:.3f}  "
              f"EB {r['err_vs_eb_pct']:+.1f}%  Timoshenko {r['err_vs_timoshenko_pct']:+.2f}%")
    print(f"  max |err vs Timoshenko| = {beam['max_abs_err_vs_timoshenko_pct']:.2f}%  "
          f"-> {'PASS' if beam['pass'] else 'FAIL'}")

    print("\n=== (2) optimizer gold: canonical MBB beam ===")
    mbb = mbb_check()
    print(f"  our compliance = {mbb['our_compliance']:.3f} (vol {mbb['our_volume']:.3f}, "
          f"{mbb['n_iterations']} iters); literature: canonical MBB topology, scalar ~200-220")
    print("  reproduced topology:\n" + mbb["ascii"])

    out = {"forward_model_gold": beam, "optimizer_gold_mbb": mbb}
    (RESULTS / "verification.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results/verification.json + results/mbb_density.{npy,pgm}")


if __name__ == "__main__":
    main()
