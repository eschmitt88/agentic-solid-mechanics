# Trial 14 design draft — three follow-ups to trial 13, and the publication question

Status: DRAFT for Fable review. Not a pre-registration.

## What trial 13 established (experiments/2026-11-16-evolutionary-dehomogenization/README.md)

Reproduction of Xu et al. 2026 (evolutionary phasor de-homogenisation; Eng. Struct.; arXiv 2510.08830)
on a new problem (100 × 50 girder, spring bearings, deck load, lined duct). Pixel Q4 in-loop HF,
CalculiX body-fitted grading. Pre-registered H1–H7:

| | verdict | numbers |
|---|---|---|
| H1 EA gain replicates | PASS | HV(r=1.1) +19.5 % stiffness (100 gens), +14.9 % buckling (30 gens) |
| H2a EA vs matched extra LF designs | EA wins | S: more-LF +1.6 %; B: more-LF +10.1 % (+17 % from its raw baseline) |
| H2b selection matters | PASS | no selection +0–6 % |
| H3 mutation | MISS (null) | −1.7 pp; mutants ≤ 2 % of survivors; PCA keeps 62 % of a mutation |
| H4 VAE | MISS (hurts) | PCA-space BLX +10.2 pp better; VAE shrinks diversity 31 → 18–25, PCA-BLX grows it to 46–57 |
| H5 B vs SIMP | MISS | SIMP better in BLF at 3/4 volumes and 17–42 % stiffer (CalculiX agrees) |
| H6 phase re-draw | PASS | gain retained 86 % S / 68 % B; per design evolved fronts degrade 23–33 % vs 10–18 % gen 0 |
| H7 pixel vs CalculiX | MISS | Spearman 0.99/0.90; one selected design is a pixel-connectivity artefact (C ×2.6) |

## New finding today (before any trial-14 run): the evaluator is discontinuous at machine precision

`scratch/chaos.py`: perturb a gen-0 design's PCA coordinates by ε · N(0, I) and re-evaluate.

| ε | designs with any pixel change | \|ΔC\|/C > 1 % | > 10 % | median \|ΔC\|/C |
|---|---|---|---|---|
| 1e-12 | 0.98 | 0.88 | 0.43 | 8.7 % |
| 1e-9 | 1.00 | 0.90 | 0.45 | 7.9 % |
| 1e-6 | 1.00 | 0.92 | 0.47 | 9.7 % |
| 1e-3 | 1.00 | 0.88 | 0.42 | 7.2 % |

Root causes traced so far:
1. (Our bug, fixed in trial 14's `run_arm.py`) non-design elements decoded θ from zero channels →
   rounding-noise directions in the duct, fed to the orientation clean-up. Fixing it did NOT remove
   the chaos (same table within noise).
2. The orientation clean-up (port of deHomTop808 `filterVectorField`): greedy discrete choice among
   {n, −n, ±R90 n} from the sum of fixed neighbours; where neighbours have opposite sign the sum
   cancels and the 4 scores tie → rounding decides → cascades (with 0 alignment sweeps, ε = 1e-12
   flips coarse widths by up to 0.9 via the layer swap and orientation vectors by sign).
3. Phase alignment (Gauss–Seidel on unit phases, 20 sweeps): even when the clean-up output is
   identical (Δw ~ 5e-13), one sweep gives phase differences up to π somewhere.
So the map design → geometry is effectively stochastic with ~9 % compliance noise, the same size as
re-drawing the global phase seed. Very likely inherent to the 808-style algorithm (never exercised by
one-shot use), not only our port — but we did not run 808 itself (blocked) so this is unverified.

Implication: the EA in the paper (and trial 13) selects partly on realisation noise — consistent with
H6's per-design winner's curse.

Speed fix today: pypardiso refactorised on every eigsh iteration; factorise-once makes a buckling
evaluation ~1.4 s (was 6–13 s), identical numbers. 300-generation buckling runs are now affordable
(~110 × 1.5 s / 10 workers ≈ 17 s per generation ≈ 1.5 h per run).

## The three follow-ups the user asked to explore

**A. Diversity-preserving recombination, 300 generations (paper's horizon), S and B.**
Arms: FULL (VAE, paper), PCA (BLX in whitened PCA space = trial 13 NOVAE), PCA+NOV (PCA-BLX with a
novelty archive: survival selection adds a third objective = mean distance to k = 15 nearest in the
archive in PCA space; HV still measured on the 2 real objectives). 3 seeds each. Question: does the gap
to SIMP close at 300 generations; does diversity keep improving the front.

**B. Adversarial / robust selection against realisation noise and evaluator artefacts.**
Arms (built on the best base from A, likely PCA-BLX):
- NOM: nominal evaluation (one phase draw) — = A's arm.
- ROB-K: each child evaluated under K = 3 phase seeds (fixed per child = hash of z), objective = mean
  (or worst) over the K draws. 3× cost per child.
- ADV (zhu2026bad style): archive of adversarial phase seeds; every 10 generations draw 8 fresh seeds,
  test the current front; seeds that degrade any front member by > 10 % enter the archive (cap 4);
  children are scored as worst over nominal + archive seeds.
- (optional) CCX-AUDIT: every 25 generations CalculiX-audit the front; designs whose ccx/pixel
  compliance ratio > 1.3 are struck from the population.
Held-out evaluation (HCE-like, never in loop): fronts re-scored with 5 fresh phase seeds (1000–1004)
and CalculiX on 8 front designs per run. Matched comparison by HF evaluations (ROB-K at 1/3 the
generations) and by generations.

**C. Problem-grain mutation: LLM vs random operator (AlphaEvolve-style; evolutionary-search-grain).**
Replace image mutation (inert) with "problem-grain" mutants: a parent's LF problem parameters
(V0, Lmin, rmin, theta_jitter, init_jitter, + new knobs: local volume caps in 4 × 2 regions, a
secondary load case weight (to stiffen against buckling), passive solid/void boxes) are edited and the
LF optimiser re-run (6 s) → de-homogenised → HF. 10 mutants per generation.
- RAND: random edits of the same parameters.
- LLM: `claude -p` (pinned Sonnet) sees the parent's picture, HF numbers (C, BLF, buckling-mode
  location, fragment count), its parameters and the schema; returns JSON edits.
- LLM-BLIND: same prompt without the picture/numbers (tests whether feedback is used).
Metrics: HV gain at matched evaluations; survival rate of operator children (vs ~1–2 % for image
mutation); structural diversity of proposals (attractor check, gurkan2026mutation). 100 generations,
S and B, 3 seeds. ~3 000 LLM calls per run.

## Questions for Fable

1. Given the chaos finding, should we (a) try to make de-homogenisation continuous (e.g. MST-based
   frame clean-up from a fixed root, continuous phase alignment), (b) treat it as noise and fix the
   evaluation instead (B), or (c) both? What is the cheapest decisive diagnostic that it is inherent to
   808 and not our port (we cannot run MATLAB/Octave here)?
2. Critique A/B/C designs: arms, controls, matched budgets, seeds, metrics, pass rules. What would a
   reviewer at SMO / CMAME / Eng. Struct. demand that is missing? (We have one problem; the paper had
   three; their mutation claim is on max-stress, our mutation null is on stiffness/buckling.)
3. Publication direction. Candidates:
   P1 critical replication + ablation ("what actually drives evolutionary de-homogenisation"),
   P2 noise/discontinuity of phasor de-homogenisation as the central finding, with robust selection as remedy,
   P3 problem-grain LLM operators for multifidelity topology design (novel, risky),
   P4 an agentic-science framing (an AI agent replicating and stress-testing a paper, with pre-registration).
   Which is the strongest single paper, which venue, and what is the minimum additional evidence
   (problems, objectives, seeds, comparisons) to make it publishable? Should the paper's own
   double-clamped beam and a max-stress L-bracket be added?
4. Compute: box has 32 cores, ~14 h fully free now then constrained; quota is plentiful. Prioritise.
