# Pre-registration — trial 14 (follow-ups to trial 13)

Written 2026-11-17 ~18:30 UTC. At this point: the decoder bug is fixed (θ/θ+π tie; `log.md` 17:40), the
evaluator is solid-only and factorises once (`hf.py`), gen-0 caches are rebuilt with it, `heldout.py`
exists, the problem-grain harness (`run_pg.py`, `pg_ops.py`) passed a 3-generation smoke test, and the
double-clamped-beam problem is being built in `experiments/2026-11-17-dehom-dcbeam/`. **No trial-14
arm has run.** Nothing below is edited after the first arm starts; deviations go in dated addenda.
Design rationale and the technical review: `DESIGN.md`, `log.md`.

## Fixed set-up

Girder problem, LF gen 0, de-homogenisation, PCA (inv5, 506 dims), NSGA-II, BLX-0.25, VAE
(506→256→128→latent, β = 1, 1000 epochs), image mutation as in trial 13, except: fixed decoder,
solid-only evaluator. 110 HF evaluations per generation (ROB3: 330). Buckling constraint reference =
gen-0 nominal envelope. Double-clamped beam (paper Sec. 4.1): objective (vf, umax), stiffness only.

## Primary metric (all hypotheses unless stated): held-out HV gain

`heldout.py`: the feasible non-dominated set of a run's archive and of gen 0 are re-evaluated under
10 held-out phasor seeds (5000–5009, never used in search); objectives = mean vf and geometric-mean
compliance (S) / BLF (B) / umax (beam); a design with any disconnected realisation is dropped; no
stiffness constraint in the held-out buckling HV. Gain = HV(gen-0 ∪ run) / HV(gen-0) − 1, shared
normaliser, r = (1.1, 1.1). Reported with the nominal archive HV gain (secondary) and their ratio
(winner's-curse index). "pp" = percentage points of gain.

## Runs (compute-limited: the box is memory-bandwidth bound at ~4.5 buckling or ~20 stiffness evals/s)

| block | arms × seeds × generations |
|---|---|
| S headline | FULL, PCA × s0–s2 × 300; FULL, PCA × s3–s4 × 100 (all five give a gen-100 snapshot) |
| S controls | LOTTERY × s0 × 100; VAE32, VAE64, RAWTH × s0–s1 × 100; ROB3 × s0–s1 × 100 (= 33 000 evals); PCANOV × s0–s1 × 300 |
| B | FULL, PCA × s0–s2 × 100 |
| Beam (S, umax) | FULL, PCA × s0–s2 × 100 |
| C pilot | run_pg RAND, LLM × s0 × 100 (S) |
| SIMP | multi-start: rmin ∈ {1.0, 1.5, 2.0} mm × penal ∈ {3, 4} at vf ∈ {0.25 … 0.50}; best per vf |

Cut order if time runs out: PCANOV, then VAE64, then ROB3 s1, then the C pilot.

## Hypotheses and pass rules

- **H1 replication (clean evaluator).** FULL held-out gain > 0 on every seed: S at gen 300, B at gen 100,
  beam at gen 100. Magnitudes reported against the paper's 15.1 % / 19.4 % (300 generations).
- **H2 VAE vs PCA crossover.** Per paired seed d = gain(PCA) − gain(FULL). "PCA > VAE" if d ≥ +5 pp on
  every seed; "VAE > PCA" if d ≤ −5 pp on every seed; otherwise "no difference". Graded separately for
  S gen 300 (3 seeds), S gen 100 (5 seeds), B gen 100 (3), beam gen 100 (3).
- **H3 is the VAE gap a hyperparameter artefact?** At S gen 100 (seeds 0–1): "artefact" if the best of
  {VAE32, VAE64, RAWTH} has mean gain ≥ mean(PCA) − 2 pp; otherwise "robust to latent size and encoding".
- **H4 noise mining.** "Search > lottery" if LOTTERY's gain (s0, 11 000 evals) is ≥ 5 pp below both
  FULL's and PCA's mean gen-100 gain (S).
- **H5 robust selection.** ROB3 (s0–s1, 33 000 evals) vs PCA gen 300 (s0–s1, 33 000 evals): "ROB3 better"
  if ≥ +5 pp on both seeds, "worse" if ≤ −5 pp on both, else "no difference". Secondary: winner's-curse
  index (held-out / nominal gain) of ROB3 vs PCA.
- **H6 novelty archive.** PCANOV vs PCA at gen 300 (s0–s1), same two-sided 5 pp rule.
- **H7 lattice vs SIMP.** At vf ∈ {0.30, 0.35, 0.40, 0.45}: the best held-out design within ±0.01 vf from
  any S arm vs the best SIMP multi-start design (same evaluator) in compliance; same for B in BLF.
  Expectation stated in advance: SIMP wins compliance at all four; BLF open.
- **H8 beam.** H1 and H2 on the paper's double-clamped beam (umax objective).
- **H9 problem-grain pilot.** Per-child elite-entry rate (child enters P in its generation) of the 10
  problem-grain children per generation: LLM vs RAND, two-proportion z-test, α = 0.05; HV descriptive.
  The LLM prompt's one strategy hint was removed before the pilot (log).
- **Select-then-verify** (descriptive): on PCA/FULL archives, re-rank the nominal front by held-out
  objectives and report how many nominal-front designs survive.

## Evaluator integrity

Hashes of hf.py, dehom.py, ea.py, run_arm.py, heldout.py, run_pg.py, pg_ops.py recorded in log.md at
freeze. CalculiX is not used in this trial's grading (the held-out phase seeds are the oracle here);
trial 13's CalculiX calibration (rank agreement 0.99/0.90) is the cited evidence for the pixel evaluator.
