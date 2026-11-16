# Trial 13 — module contracts (shared by every builder)

Read `DESIGN.md` for the method and `hf.py` for the frozen problem.
Decisions that override `DESIGN.md` (made after the Fable review and the
evaluator studies in `log.md`):

- Coarse (LF) grid 40 × 20 elements, hc = 2.5 mm. Fine (HF) grid 400 × 200
  pixels, h = 0.25 mm. Phasor period λ = 7.5 mm (= 3 hc).
- Geometry/BCs: `hf.GIRDER`. Coarse non-design masks: `hf.coarse_masks()`
  (keys skin, pads, solid_fixed, void_fixed, design; [iy, ix]).
  Supports are distributed springs (k_line per mm) under the pads: left pad x+y,
  right pad y. Load: total P = 1 downward, uniform on the top edge.
- Material: E = 1, ν = 0.3, plane stress, unit thickness.

## Array conventions (all modules)

- Every field is a numpy array indexed `[iy, ix]`, `iy = 0` at y = 0 (bottom),
  y up. Coarse fields have shape (20, 40); fine fields (200, 400).
- Element centres: coarse x = (ix + 0.5)·hc, fine x = (ix + 0.5)·h.
- Angles in radians, measured from +x, counter-clockwise.

## The design description X (what LF produces and dehom consumes)

Per coarse element:
- `w1` ∈ [0, 1]: realised width (fraction of the period λ) of bar 1, which runs
  ALONG direction θ;
- `w2` ∈ [0, 1]: realised width of bar 2, which runs along θ + π/2;
- `theta`: bar-1 direction.
Effective widths already include the void indicator (w_i = μ_i · ρ̄).
Element volume of the cross cell: w1 + w2 − w1·w2. In non-design solid
elements w1 = w2 = 1; in non-design void elements w1 = w2 = 0.
Bars thinner than `W_VOID = 0.05` are treated as absent; w ≥ 0.99 as solid.

## Module APIs

`homog.py`
- `CH(w1, w2) -> (..., 3, 3)` local-frame homogenised tensor (Voigt, engineering
  shear), from numerical homogenisation of the cross cell + polynomial fit,
  PD-projected, isotropic floor. Vectorised, JAX-differentiable.

`lf.py`
- `lf_optimise(V0, Lmin, seed=0, rmin=None, theta_jitter=0.0, ...) -> dict`
  with w1, w2, theta (20, 40), mu1, mu2, rho, compliance_lf, vol_lf, history.
- `lf_displacement(w1, w2, theta) -> (ux, uy)` element-centre displacement
  field (20, 40) of the homogenised design under the design load.
- `lf_compliance(w1, w2, theta) -> float`.

`dehom.py`
- `dehomogenise(w1, w2, theta, phase_seed=None) -> dict` with `phi` (200, 400)
  bool (before island removal; masks applied), `level` (200, 400) float
  continuous field whose 0 level set is the boundary (positive = solid), and
  timing. `phase_seed` randomises the global phase/alignment order (for the
  re-realisation noise test); `None` = deterministic default.

`hf.py` (exists)
- `Problem(GIRDER, H_FINE).evaluate(phi, buckling=bool, nev=1) -> dict`
  (vf, connected, compliance, umax, blf).

`verify_ccx.py`
- `verify(level_or_phi, ...) -> dict` compliance, umax, blf (+ mesh stats) from a
  body-fitted CalculiX model of the same design.

## Rules for builders

- Python via `../../.venv/bin/python` (uv-managed). Add a dependency only with
  `uv add` from the project root and say so in your report.
- Write only the files your brief names, plus `scratch/<your-module>/`.
  Do not edit `hf.py`, `DESIGN.md`, this file, or another builder's module.
  Do not commit; the orchestrator commits.
- No long multi-core runs (> 10 min or > 8 cores): this box runs scheduled jobs.
