# Log — dic-plasticity-identification

All times UTC, 2026-11-15. Phase 2, track E.

## 16:28 — orientation
- Read the shortlist (track E), the phase-2 summary page, trial 9's README / PREREGISTRATION / log for house style.
- Box: GPU 380 MiB used of 16 GB, load 1.3, 2.4 TB free on /mnt/projects.

## 16:30 — data hunt: the first-choice dataset is open
- Crossref record of doi:10.1038/s41597-024-03949-y lists the data DOI 10.25452/figshare.plus.25483534.
  Figshare API (`api.figshare.com/v2/articles/25483534`) returns 200 files, licence CC0, no login, direct
  `ndownloader.figshare.com` URLs with md5 sums. Total several hundred GB (raw images + IR + processed .mat).
- Saved the paper (PDF + text) and the API record under `raw/sandia-304l-dic/`; downloaded the small files
  (test matrix, drawings, mill cert, 8 nominal-geometry .mat, 27 dog-bone .mat), all md5-verified.
- Test matrix: bespoke plates O1/O5/O14/O15/O18 are all transverse, one slow test each; X-specimen has one slow
  transverse (XT10) and one slow rolling (XR5); the D-specimen was only tested fast. Fast tests self-heat
  (paper Fig. 10), so restricting to slow tests lets a rate-independent isothermal model stand.
- Started background downloads (curl, ~20 MB/s): O1-3, O14-3, O5-4 into `large/calibration/`, XT10, XR5, O15-3
  into `large/heldout/`. Held-out files are md5-checked only, not opened. O18-3 (15.3 GB) skipped as
  disproportionate. Planned total ≈ 26 GB.
- `uv pip install h5py` (MAT v7.3 files).

## 16:33–16:38 — solver written and unit-checked before any data
- `fem.py`: T6 plane-stress, log-strain finite plasticity, quadratic (Hill-48 family) yield, Voce(+Voce)+linear
  hardening, scalar return map with bisection-safeguarded Newton, forward (direct) sensitivities propagated
  through the plastic history with `jax.jvp`; sparse solves by pypardiso.
- Dead end 1: first Newton scheme (move the boundary nodes, start from the old interior) diverged on a 2.5 %
  first step — boundary elements see a huge strain. Fixed with the standard linearised predictor
  du_f = −K_ff⁻¹ K_fp Δu_p using the last converged tangent. The earlier secant extrapolation was removed.
- `test_unit.py`: uniaxial bar to 50 % stretch vs closed form: 9e-14 (von Mises) and 6e-10 (Hill, loading along y);
  sensitivities vs central FD on a clamped tension+shear block, 9 parameters: dF/dp ≤ 3e-7, du/dp ≤ 4e-8 relative.

## 16:37 — first look at one calibration file (O1-3)
- 618 frames × 106 226 DIC points; force peaks at 32.6 kN at 23.2 mm LVDT extension, then tears progressively.
  Mean IR temperature at peak force 22.8 °C — isothermal as hoped. 0.4 % of points unmatched at peak force.
- Five spurious force drops (0.4–1.0 kN) before the peak, as the paper warns → exclusion rule in the pre-registration.

## 16:39 — pre-registration written (split, metrics, tolerances, gates). No fit has been run.

## 16:40–16:42 — pre-processing of the calibration files; one amendment to the pre-registration
- `data.py`: stations, cut lines, cubic boundary fits, valid-point mask → `results/cache/<spec>.npz` (not in git).
- Bug: LVDT extension is NaN for the pre-test frames, `argmin` returned frame 0 for every station. Fixed.
- The pre-registered force-drop rule (rolling max over ±5 frames) flagged stations 0–7 (O1-3) and 0–10 (O14-3):
  on a rising curve the following frames are always higher. Changed to the preceding 5 frames and recorded as
  Amendment 1 in PREREGISTRATION.md (before any fit; held-out files still unopened). O14-3 has a band of
  load-cell chatter near its peak (stations 38 and 40 excluded from force scoring).
- Cubic boundary fit residual grows to 0.056 mm rms at station 40 (displacements ≈ 10 mm there; DIC noise 0.004 mm):
  the cubic is not a perfect description of the edge profile late in the test. Left as pre-registered.
- O1-3: x_F = 23.19 mm, F_max = 32.6 kN, 99.6 % of DIC points valid at the three field stations.
  O14-3: x_F = 17.10 mm, F_max = 28.3 kN.

## 16:43–16:52 — real geometry meshed, first forward solve, baseline fitted, warm-up launched
- `meshutil.py`: outline → gmsh `.geo` → T6 mesh. Two dead ends: (a) splitting the outline only at sharp corners
  gave one spline through straight edges *and* fillets and gmsh produced no surface; now straight runs are Lines and
  curved runs Splines. (b) the nominal outlines overlap themselves by 0.1 mm at the loop seam ("Impossible to recover
  edge"); zero-area spikes are now removed. A scratch file `/tmp/glob.py` of mine shadowed the stdlib `glob`
  and broke `import jax`; deleted, scratch scripts now live in this folder.
- `specimen.py`: mesh + DIC-driven boundary motion + levelling operators. O1-3 calibration mesh (2.5 mm far field, 1.0 mm
  at notches/holes): 8 880 nodes, 4 282 T6; 99 065 of 106 226 DIC points kept (rest: within 0.2 mm of the cut
  lines, unmatched, or outside the nominal outline).
- My levelled strains vs VIC-3D's own strains on O1-3: 2.8 % relative L2 at all three field stations (not graded).
- First forward solve with **guessed** parameters (s0 300, Q 900, b 3, H 300): 29 s, force NRMSE 2.7 %, strain
  rel-L2 9–10 %, peak Eyy (Green–Lagrange) 1.15 at station 40. Noted: a guess already scores well on the strain
  metric because the boundary motion is imposed — the null-model comparison in the pre-registration matters.
- `baseline.py` (uniaxial-only baseline, dog bones BT3/CT3/CT4): apparent modulus below 250 MPa is 187–194 GPa
  (so the fixed E = 193 GPa agrees with this sheet). BT3 contains a spurious unloading at ε_p ≈ 0.085 that made
  the three curves "disagree" by 47 %; after removing points more than 1 % of the maximum below the running
  maximum the spread is ≤ 3.3 %. Fits to the mean true curve up to ε_p = 0.473 (UTS 676–686 MPa):
  M1 s0 342, Q 1134, b 2.23 (rms 4.6 MPa); M2 s0 332, Q 370, b 5.31, H 880 (rms 2.8); M3 rms 2.8 (no gain over M2).
- `femu.py` (LM + one worker process per specimen + Gauss–Newton identifiability) and `warmup.py` written;
  warm-up launched 16:47. Processes pinned to cores 20–31 with taskset after noticing a worker at ~1000 % CPU.

## 16:47–17:20 — warm-up: two runs stopped, optimiser reworked (optimiser stall, recorded as data)
- Warm-up truth: Hill + Voce + linear, p = [s0 280, Q 700, b 2.5, H 600, a 1.15, h 0.42, c 2.7] on a 30 055-node mesh
  (truth solve 106 s), α_95 = 0.367, α_max = 0.794; noisy synthetic field at the real O1-3 point cloud; boundary
  fits redone from the noisy field. Identification on the 8 880-node mesh from start A = [400, 450, 4, 900, 0.8, 0.6, 3.6].
- **Run 1 (stopped by PID at 17:07 after 14 iterations, log kept as `results/warmup_run1_killed.log`).** Two LM
  iterations brought the cost from 193 to 0.043 and the Hill coefficients to 1.155 / 0.4214 / 2.699 (truth
  1.15 / 0.42 / 2.7). Then the optimiser **crawled**: Q, b, H drifted by 2–3 % per iteration toward the truth
  (Q 436 → 632, b 3.37 → 2.69, H 909 → 664) while the cost moved 0.0317 → 0.0303. The hardening *curve* was already
  right at iteration 3: max error of σ_y(α) on [0, α_95] 0.27 % (it. 3), 0.15 % (it. 14); on [0, α_max] 2.2 % → 0.3 %.
  Reading: the data pin the curve, not the individual Voce/linear constants (a curved, nearly flat valley).
- Each iteration took 70–100 s. Profile of one solve: 310 pardiso calls at 66 ms (symbolic analysis redone every
  time, plus scipy fancy-indexing of K). Reworked `fem.py`: one symbolic analysis per mesh, numeric refactorisation
  only, free/prescribed blocks assembled by precomputed index maps, the converged tangent's factorisation reused
  for the sensitivity solves and the next predictor. Bug on the way: after a step cut the predictor used a
  factorisation from a failed iterate (350 Newton its, 20 cuts instead of 142 / 7); fixed by restoring the last
  converged tangent. 44 s → 20 s for a 20-station solve with 4 sensitivities; unit checks unchanged.
- **Run 2 (same optimiser, faster solver, stop rule "cost drop < 0.2 % twice"; stopped by PID at 17:20 after 5
  iterations, `results/warmup_run2_killed.log`)**: same crawl, still ~2 min per iteration under CPU contention.
- Fix for the crawl: inside the optimiser the hardening parameters are replaced by **log σ_y at fixed plastic
  strains** (3/4/6 nodes for M1/M2/M3), Hill coefficients stay as logs; Gauss–Newton steps are taken in those
  coordinates and mapped back with a small root solve (`femu.CurveCoords`). Falls back to log-parameters when the
  map is singular (it is for the uniaxial M3 fit, whose b1 = 1362 is invisible at the nodes). Run 3 launched 17:20.
- `ccx_check.py` written; development run on calibration specimen O1-3 with the uniaxial M2 parameters started
  17:08 (CalculiX takes ~30 s per load step with spooles).

## 17:20–17:31 — warm-up run 3 (reparametrised optimiser): start A passes; CalculiX pipeline checked
- Start A converged in 3 iterations (cost 193 → 0.0302): p = [280.4, 738.7, 2.424, 559.1 | 1.1550, 0.4214, 2.6994]
  vs truth [280, 700, 2.5, 600 | 1.15, 0.42, 2.7]. σ_y error on [0, α_95] 0.15 % (tolerance 2 %), Hill errors
  +0.005 / +0.001 / −0.001 (tolerances 0.05 / 0.05 / 0.3): **pass**. Q, b, H individually are 5 %, 3 %, 7 % off
  while the curve is right — the valley again. Start B running.
- CalculiX development run (calibration specimen O1-3, uniaxial M2 parameters, 40 load steps, 18 min wall):
  JAX vs CalculiX force NRMSE 0.38 %, levelled strain fields 0.9 / 0.9 / 1.3 % at stations 13 / 27 / 40. The deck
  writer and parser work; the two solvers agree far inside the pre-registered 2 % / 3 %.
- 17:31 — real-data fits M1 and M2 launched (two worker processes each, cores 20–31).

## 17:31–17:52 — warm-up gate closed (pass); first real fits; where the misfit lives
- Warm-up start B = [180, 1000, 1.5, 300, 1.5, 0.3, 2.0] reached the *same* optimum as start A to 8 digits, 6 solver
  evaluations each. Both pass: σ_y error 0.15 % on [0, α_95], 0.36 % on [0, α_max]; Hill +0.005 / +0.0014 / −0.0006.
  Linearised width of the 1 %-cost valley: Q ±30 %, b ±20 %, H ±37 %, but σ_y at the nodes ±0.1–0.6 %, s0 ±0.3 %,
  a ±1.2 %, h ±0.6 %, c ±0.4 %. Gradient gate: direct sensitivities vs central FD ≤ 1.1e-6 (`test_unit.py`), limit 1e-5.
- Real data, M1 (von Mises + Voce), two starts → identical optimum J = 1.98674, p = [331.3, 1063, 2.442].
  M2 (+ linear term), first two starts → identical J = 1.90258, p = [322.6, 430.8, 5.166, 701.0]. Third start running.
  Each start takes 7–11 min (box load average 40–50; my processes are pinned to 12 cores).
- The uniaxial baseline is already close on the calibration plates before any full-field fit: force NRMSE 2.7 %
  (O1-3) and 0.6 % (O14-3), strain error 7–9 %. The full-field fit lowers J from 3.18 to 1.90, mostly through force.
- `dev_errmap.py` on calibration plate O1-3 (baseline M2 parameters): the null (elastic) model scores 38–50 % strain
  error and a hot spot 41 % low, so the strain metric is not satisfied by the imposed boundary motion alone.
  The plastic model's 7–9 % residual is not concentrated at edges or cut lines (same number 3 mm inside). The worst
  component is the sideways strain Exx (12–14 % vs 7 % for Eyy): the von Mises model contracts sideways about 10 %
  more than measured (mean Exx −0.0397 vs −0.0358 at station 40) — a signature of plastic anisotropy (r < 1), which
  M4 can absorb and M1–M3 cannot.
- 17:52 — M3 and M4 launched.
- 17:52 M4 run crashed at start: M3 and M4 were launched together and both re-ran gmsh into the same .msh file (meshio read a half-written file). Meshes are now generated once to a temp name and renamed; relaunched M4 at 18:04 (11 min lost). M2's third start reached the same optimum (J = 1.90258).

## 18:04–18:43 — all four forms fitted; development scoring; model selection
- M4 (Hill + Voce + linear), three starts (isotropic M2 optimum; a far start; a start with the anisotropy pushed the
  wrong way) → identical optimum J = 1.30955: p = [s0 330.6, Q 379.3, b 5.147, H 842.2 | a 1.0735, h 0.4460, c 2.928].
  J drops 31 % against M2, all of it in the strain term (calibration strain error 7.1–8.4 % → 5.5–6.5 %).
  The hardening part of M4 lands within 1–4 % of the uniaxial dog-bone fit (332, 370, 5.31, 880).
- M3 (double Voce): start from the uniaxial fit → J = 1.89367 (0.5 % below M2; the extra Voce term degenerates into
  an instant yield-point shift, b1 = 1766). **Optimiser stalls recorded:** start "far" stopped in a different
  minimum at J = 4.34 after 24 min (H → 8, Q2 → 3162, b2 → 0.39: second Voce playing the linear term badly);
  start "m2like" wandered up to J ≈ 4.9 with b2 → 330 and was still running at 18:43. The six-parameter form is
  ill-conditioned; nothing in the data asks for it.
- Development specimen O5-4 (`score.py dev`, results/dev_scores.json), force NRMSE / strain error at 30-60-90 % / hot spot:
  M1 0.67 % / 8.9-7.6-8.4 % / +1.9 %; M2 0.56 % / 8.2-7.4-7.6 % / +2.9 %; M3 0.53 % / 8.3-7.4-7.6 % / +2.7 %;
  **M4 0.49 % / 5.0-4.4-4.7 % / +2.7 %**; uniaxial-M2 baseline 0.96 % / 8.5-7.5-7.2 % / −1.4 %;
  null (elastic) model: strain error 41–49 %, hot spot −54 %.
- Selection score S = F/0.05 + mean strain/0.15: M1 0.687, M2 0.628, M3 0.623, M4 0.410. Only M4 is within 10 % of
  the best → **primary model = M4**.
- Pre-registration wording issue: M4 was to use "the hardening law of the better of M2/M3". I fitted M4 with the M2
  law before the development scores existed. On the scores M3 is nominally "better" than M2 (S 0.623 vs 0.628,
  0.8 %), so the literal reading asks for Hill + double Voce (9 parameters, "M4b"). Launched M4b at 18:43 so the
  selection rule can be applied over both; if M4b is not more than 10 % better on O5-4, M4 (7 parameters) stays primary.

## 18:43–19:06 — M4b fitted, models frozen
- M4b (Hill + double Voce, 9 parameters), two starts → J = 1.2990 (0.8 % below M4), same Hill coefficients
  (1.0723, 0.4456, 2.921). On O5-4: S = 0.407 vs 0.411 for M4. Both are "within 10 % of the best"; the rule picks the
  one with fewer parameters → **primary = M4**, baseline = uniaxial-M2 (same hardening law, von Mises),
  von Mises form for the CalculiX check = M2 (same rule applied among M1–M3).
- M3 third start ("m2like") finally stopped at J = 1.9096 after 35 min and 30+ iterations, short of the 1.8937 found
  from the uniaxial start: three starts, three different end points for the six-parameter form.
- `freeze.py` wrote `results/frozen_models.json` at 19:06:01 UTC. Up to this moment no held-out file has been
  opened by any script (only curl + md5sum at download; `dev_mesh_dryrun.py` used the held-out *outlines* from the
  nominal-geometry files, with guessed cut lines, to check that gmsh can mesh them).

## 19:07 — the held-out pass (`score.py heldout`)
- Opened 19:06:13 UTC (recorded with the hashes of the frozen-model file and the pre-registration in
  `results/heldout_access.json`), finished 19:15:45. Pre-processing of the three held-out files ran blind with the
  rules developed on the calibration plates; nothing needed fixing. Meshes: O15-3 6 902 nodes, XT10 9 396, XR5 9 579.
- Result (force NRMSE / force at last station / strain error at 30-60-90 % / hot spot), tolerances 5 % / 5 % / 15 % / 15 %:
  - O15-3: **M4 0.74 % / −0.96 % / 4.0-3.8-3.8 % / +0.8 %**; uniaxial-M2 0.53 % / +0.39 % / 6.7-6.5-6.6 % / −3.0 %; null 31–39 % strain, hot spot −47 %.
  - XT10: **M4 1.53 % / −1.53 % / 7.2-6.2-5.9 % / −1.0 %**; uniaxial-M2 0.32 % / −0.05 % / 9.6-8.1-7.6 % / −2.4 %; null 31–38 %, −31 %.
  - XR5 (rolling direction): M4 2.64 % / −3.26 % / 7.2-7.6-8.6 % / +3.8 %; uniaxial-M2 1.27 % / −1.81 % / 8.7-8.3-9.2 % / +2.4 %.
  Every plastic model form passes every pre-registered check on every held-out specimen, with wide margins.
- What the comparison says: the full-field model (with anisotropy) predicts the **strain fields** better than the
  uniaxial baseline (O15-3 6.6 → 3.8 %, XT10 8.1 → 6.2 %), but the **force** slightly worse (0.7–1.5 % vs 0.3–0.5 %).
  The baseline, fitted to three dog bones in seconds, would have passed the same exam.
- 19:16 — launched the convergence runs (calibration mesh with 80 load increments; half-size mesh) for M4 and
  uniaxial-M2 on the three held-out specimens, and the CalculiX cross-check (M2 on O15-3).
- Supporting numbers while the gates run: equivalent plastic strain in the calibration plates at station 40 (M4):
  median 0.10 / 0.04, 95th percentile 0.30 / 0.26, maximum 0.84 / 0.92 (O1-3 / O14-3) — most material sits below the
  end of the tension test (0.47), a few per cent of it far beyond.
- Elastic-constant sensitivity on O5-4 (M4): first attempt changed E and ν inside one process and "found" zero
  effect to all digits — the constants are baked into the jit-compiled kernels at first trace, so nothing had
  changed. Redone with one process per setting (`elastic_sensitivity.py`): E 193 → 203 GPa moves force NRMSE
  0.495 % → 0.441 % and strain errors by ≤ 0.02 points; ν 0.29 → 0.33 moves them by ≤ 0.01 points. Negligible, as assumed.

## 19:16–19:28 — verification gates on the held-out predictions; one post-hoc look
- **Load-step convergence** (80 instead of 40 increments, calibration mesh): every metric of M4 and uniaxial-M2 on the
  three held-out specimens moves by ≤ 0.09 percentage points (largest: O15-3 hot spot +0.80 → +0.72 %).
- **Mesh convergence** (far field 1.25 mm, 0.5 mm at notches; 26 434 / 35 819 / 36 508 nodes instead of
  6 902 / 9 396 / 9 579): every metric moves by ≤ 0.07 points (largest: O15-3 hot spot +0.80 → +0.87 %). No pass
  depends on the mesh or the step size.
- **CalculiX cross-check** (M2, held-out plate O15-3, CPS6, NLGEOM, 40 steps, 9.7 min): force curve JAX vs CalculiX
  0.24 % NRMSE (max 0.31 %), levelled strain fields 0.43 / 0.59 / 1.18 % at stations 13 / 27 / 40 — pass (limits 2 % / 3 %).
  CalculiX's own score against the experiment: force 0.63 %, strain 6.6 / 6.3 / 6.3 %, hot spot −0.9 % (JAX: 0.83 %,
  6.5 / 6.3 / 6.2 %, +0.0 %). The check covers the von Mises path only; CalculiX has no Hill yield surface.
- Because of that gap, `test_hill.py`: closed-form uniaxial checks of the anisotropic part with the identified M4
  constants — stress along y and along x to 1e-14, lateral strain (−ν T/E − h ε_p, and the x-direction analogue)
  to 3e-14 and 7e-10. (First version of the test had the sign of the lateral stretch wrong; fixed.)
- **Post hoc, not graded** (`posthoc_rolling.py`): XR5 (pulled along the rolling direction) is the one place where
  the full-field model is clearly worse than the baseline on force (−3.3 % vs −1.8 % at the last station). M4's
  Hill coefficient a = 1.0735 implies the rolling direction is 3.5 % *weaker* than the transverse one. The slow
  rolling dog bones AR3/BR3 (never used for fitting) say it is 1–4 % *stronger* (flow-stress ratio 1.010–1.038 at
  plastic strains 0.02–0.4). So the anisotropy identified from plates pulled in one direction improves the strain
  maps but gets the strength in the other direction wrong in sign; a is also the least constrained Hill
  coefficient (±5.5 % for a 1 % worse fit). M4's implied r-value in the transverse direction is h/(1−h) = 0.80.
- All computation finished 19:28. Writing up.

## 19:28–19:34 — write-up
- `make_figures.py`, `build_page.py` (metrics.json + `docs/qa/trial-11-identification/index.html`, 14 figures copied),
  README, config. Figures were first drawn 7 inches wide; at 420 px the text was unreadable, so they were redrawn
  4.6 inches wide with the four-panel strain maps as a 2 × 2 grid. Page checked with headless Chromium at 420 px
  and 1100 px (tables stack into cards on narrow screens).
- GPU: spot checks with nvidia-smi showed 300–850 MiB per worker process and 2.2–2.6 GB in total on the card
  (siblings included); at most six workers ran at once. Not monitored continuously.
- Not done: `_meta/index.md`, `NOTES.md`, git — left to the orchestrator as instructed.
