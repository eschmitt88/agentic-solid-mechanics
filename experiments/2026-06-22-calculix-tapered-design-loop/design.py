"""Tapered-cantilever design problem, FE-true reference optimum, and grader.

Design vars: width b, root height h_root, tip height h_tip (linear taper).
Objective: minimise mass = rho * b * L * (h_root + h_tip)/2.
Constraints: FE tip deflection <= delta_allow (no closed form for a taper);
max nominal bending stress over the length <= sigma_allow (closed form);
all vars within bounds.

Reference optimum: width is held at its lower bound (stiffness-per-mass favours
height over width — same corner as the prismatic trial, spot-checked here), and
the taper ratio r = h_tip/h_root is searched; for each r the root height is
bisected to make the FE deflection exactly meet the cap.
"""
from __future__ import annotations

from pathlib import Path

import yaml

import fea


class DesignProblem:
    def __init__(self, cfg: dict):
        self.p = cfg["problem"]
        dv = cfg["design_vars"]
        self.b_bounds = (dv["width_m"]["min"], dv["width_m"]["max"])
        self.hr_bounds = (dv["h_root_m"]["min"], dv["h_root_m"]["max"])
        self.ht_bounds = (dv["h_tip_m"]["min"], dv["h_tip_m"]["max"])
        self.delta_allow = cfg["constraints"]["max_tip_deflection_m"]
        self.sigma_allow = cfg["constraints"]["max_nominal_bending_stress_pa"]
        self.n_grade = cfg["mesh_n_grade"]
        self.tol = cfg["feasibility_tol_rel"]
        self.env = cfg["solver"]["micromamba_env"]
        self.ccx = cfg["solver"]["ccx_bin"]

    def beam(self, b, hr, ht) -> fea.TaperedBeam:
        p = self.p
        return fea.TaperedBeam(p["length_m"], b, hr, ht, p["youngs_modulus_pa"],
                               p["poisson_ratio"], p["density_kg_m3"], p["tip_load_n"])

    def _defl(self, b, hr, ht, wd):
        if not (self.ht_bounds[0] <= ht <= self.ht_bounds[1]):
            return None
        return fea.fe_deflection(self.beam(b, hr, ht), self.n_grade, wd, "b",
                                 env=self.env, ccx_bin=self.ccx)

    def _scale_to_deflection(self, b, r, wd):
        """Bisect root height so FE deflection == delta_allow, with h_tip = r*h_root."""
        lo, hi = self.hr_bounds
        d_hi = self._defl(b, hi, r * hi, wd)
        if d_hi is None or d_hi > self.delta_allow:
            return None                      # even the tallest section is too soft / out of bounds
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            d = self._defl(b, mid, r * mid, wd)
            if d is None or d > self.delta_allow:
                lo = mid
            else:
                hi = mid
        return hi, r * hi

    def reference_optimum(self, wd: Path) -> dict:
        bmin = self.b_bounds[0]
        ratios = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12]
        scan, best = [], None
        for r in ratios:
            res = self._scale_to_deflection(bmin, r, wd)
            if res is None:
                continue
            hr, ht = res
            beam = self.beam(bmin, hr, ht)
            sigma = beam.max_nominal_stress_pa()
            row = {"taper_ratio": r, "b": bmin, "h_root": hr, "h_tip": ht,
                   "mass_kg": beam.mass_kg(), "max_nominal_stress_pa": sigma,
                   "stress_ok": sigma <= self.sigma_allow}
            scan.append(row)
            if row["stress_ok"] and (best is None or row["mass_kg"] < best["mass_kg"]):
                best = row
        # spot-check that widening b only adds mass (confirms the b=b_min corner)
        wide = self._scale_to_deflection(self.b_bounds[1], 0.5, wd)
        wide_mass = self.beam(self.b_bounds[1], *wide).mass_kg() if wide else None
        return {"optimum": best, "scan": scan, "b_max_check_mass_kg": wide_mass}

    def grade(self, b, hr, ht, mass_opt, wd: Path) -> dict:
        within = (self.b_bounds[0] <= b <= self.b_bounds[1] and
                  self.hr_bounds[0] <= hr <= self.hr_bounds[1] and
                  self.ht_bounds[0] <= ht <= self.ht_bounds[1])
        beam = self.beam(b, hr, ht)
        defl = fea.fe_deflection(beam, self.n_grade, wd, "check", env=self.env, ccx_bin=self.ccx)
        sigma = beam.max_nominal_stress_pa()
        feas_d = defl <= self.delta_allow * (1 + self.tol)
        feas_s = sigma <= self.sigma_allow
        mass = beam.mass_kg()
        return {
            "b": b, "h_root": hr, "h_tip": ht, "mass_kg": mass,
            "within_bounds": within,
            "check_fe_deflection_m": defl, "delta_allow_m": self.delta_allow,
            "deflection_feasible": feas_d,
            "max_nominal_stress_pa": sigma, "sigma_allow_pa": self.sigma_allow,
            "stress_feasible": feas_s,
            "mass_ratio_to_opt": mass / mass_opt if mass_opt else None,
            "pct_above_optimum": (mass / mass_opt - 1) * 100 if mass_opt else None,
            "feasible": within and feas_d and feas_s,
        }


def load(cfg_path: Path) -> DesignProblem:
    return DesignProblem(yaml.safe_load(cfg_path.read_text()))
