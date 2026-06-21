"""Design-loop problem definition, FE-true reference optimum, and grader.

Objective: minimize mass of a cantilever by choosing (b, h), subject to
  * tip deflection <= delta_allow   (graded by FE — the binding constraint)
  * nominal bending stress <= sigma_allow  (closed-form 6PL/(b h^2))
  * b, h within bounds.

The reference optimum is found numerically with FE in the loop (so it accounts
for shear, unlike the Euler-Bernoulli formula). The grader re-checks a candidate
design on a fixed, converged grading mesh (mesh_n_grade) — the same mesh used to
find the reference optimum, and independent of whatever mesh the agent chose.
"""
from __future__ import annotations

from pathlib import Path

import yaml

import fea


class DesignProblem:
    def __init__(self, cfg: dict):
        p = cfg["problem"]
        self.p = p
        self.b_bounds = (cfg["design_vars"]["width_m"]["min"], cfg["design_vars"]["width_m"]["max"])
        self.h_bounds = (cfg["design_vars"]["height_m"]["min"], cfg["design_vars"]["height_m"]["max"])
        self.delta_allow = cfg["constraints"]["max_tip_deflection_m"]
        self.sigma_allow = cfg["constraints"]["max_nominal_bending_stress_pa"]
        self.n_grade = cfg["mesh_n_grade"]
        self.tol = cfg["feasibility_tol_rel"]
        self.env = cfg["solver"]["micromamba_env"]
        self.ccx = cfg["solver"]["ccx_bin"]

    def beam(self, b: float, h: float) -> fea.Beam:
        return fea.Beam(self.p["length_m"], b, h, self.p["youngs_modulus_pa"],
                        self.p["poisson_ratio"], self.p["density_kg_m3"], self.p["tip_load_n"])

    def evaluate(self, b: float, h: float, workdir: Path, n: int | None = None,
                 tag: str = "eval") -> dict:
        beam = self.beam(b, h)
        defl = fea.fe_deflection(beam, n or self.n_grade, workdir, tag=tag,
                                 env=self.env, ccx_bin=self.ccx)
        return {"b": b, "h": h, "mass_kg": beam.mass_kg(),
                "fe_deflection_m": defl,
                "nominal_stress_pa": beam.nominal_bending_stress_pa()}

    def _min_h_for_deflection(self, b: float, workdir: Path) -> float | None:
        """Bisection: smallest h in bounds whose FE deflection == delta_allow."""
        lo, hi = self.h_bounds
        # need the tall end to satisfy the constraint, else infeasible at this b
        if fea.fe_deflection(self.beam(b, hi), self.n_grade, workdir, tag="b", env=self.env, ccx_bin=self.ccx) > self.delta_allow:
            return None
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            d = fea.fe_deflection(self.beam(b, mid), self.n_grade, workdir, tag="b", env=self.env, ccx_bin=self.ccx)
            if d > self.delta_allow:
                lo = mid
            else:
                hi = mid
        return hi

    def reference_optimum(self, workdir: Path, n_b: int = 6) -> dict:
        """FE-true min-mass feasible design, searching b over a grid and
        bisecting h for the binding deflection constraint at each b."""
        bmin, bmax = self.b_bounds
        best = None
        scan = []
        for i in range(n_b):
            b = bmin + (bmax - bmin) * i / (n_b - 1)
            h = self._min_h_for_deflection(b, workdir)
            if h is None:
                continue
            beam = self.beam(b, h)
            row = {"b": b, "h": h, "mass_kg": beam.mass_kg(),
                   "nominal_stress_pa": beam.nominal_bending_stress_pa(),
                   "stress_ok": beam.nominal_bending_stress_pa() <= self.sigma_allow}
            scan.append(row)
            if row["stress_ok"] and (best is None or row["mass_kg"] < best["mass_kg"]):
                best = row
        return {"optimum": best, "scan": scan}

    def grade(self, b: float, h: float, mass_opt: float, workdir: Path) -> dict:
        within_bounds = (self.b_bounds[0] <= b <= self.b_bounds[1] and
                         self.h_bounds[0] <= h <= self.h_bounds[1])
        beam = self.beam(b, h)
        # independent feasibility check on the fixed converged grading mesh
        defl = fea.fe_deflection(beam, self.n_grade, workdir, tag="check",
                                 env=self.env, ccx_bin=self.ccx)
        sigma = beam.nominal_bending_stress_pa()
        feas_defl = defl <= self.delta_allow * (1 + self.tol)
        feas_stress = sigma <= self.sigma_allow
        mass = beam.mass_kg()
        return {
            "b": b, "h": h, "mass_kg": mass,
            "within_bounds": within_bounds,
            "check_fe_deflection_m": defl, "delta_allow_m": self.delta_allow,
            "deflection_feasible": feas_defl,
            "nominal_stress_pa": sigma, "sigma_allow_pa": self.sigma_allow,
            "stress_feasible": feas_stress,
            "mass_ratio_to_opt": mass / mass_opt if mass_opt else None,
            "pct_above_optimum": (mass / mass_opt - 1) * 100 if mass_opt else None,
            "feasible": within_bounds and feas_defl and feas_stress,
        }


def load(cfg_path: Path) -> DesignProblem:
    return DesignProblem(yaml.safe_load(cfg_path.read_text()))
