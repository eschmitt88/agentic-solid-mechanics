# Design draft — trial 13: evolutionary de-homogenization, new geometry

Status: DRAFT for technical review (Fable). Not the pre-registration; that is
written after review, before any optimisation run.

## Source

Xu, Guo, Kawabe, Yaji, "Data-driven multifidelity and multiscale topology
optimization based on phasor-based evolutionary de-homogenization",
Eng. Struct. 371 (2027) 124076, doi 10.1016/j.engstruct.2026.124076;
preprint arXiv 2510.08830 (text: `scratch/papers/xu.txt`). Phasor
de-homogenisation per Woldseth, Sigmund & Jensen, deHomTop808,
SMO 67:205 (2024), arXiv 2405.14321 (text: `scratch/papers/woldseth808.txt`),
and Woldseth, Bærentzen & Sigmund, CMAME 418:116551 (2024)
(`scratch/papers/woldseth_phasor.txt`).

Paper pipeline: z (PCA coefficients) → X = X̄ + Φz (per macro element:
μ1, μ2 bar widths, θ angle; ρ scales widths) → phasor de-homogenisation →
binary φ → marching squares → equal-arc-length resampling → DXF → COMSOL
free-triangle mesh → HF FE (max displacement, BLF, max von Mises) →
NSGA-II on (volume, objective) → VAE (FC, 2 hidden layers, retrained every
generation on elites) for latent crossover → image-deformation mutation
(LF displacement field, radial attenuation, bicubic Keys a = −0.5).
Initial population from gradient homogenisation-based TO (LF), varied over
V0 ∈ [0.25, 0.50] × Lmin ∈ {0.10, 0.15, 0.20} = 480 designs.
Reported: HV +15.1 % (stiffness, 300 gens), +19.4 % (buckling w/ stiffness
constraint), L-bracket stress +107 % (w/ mutation) vs +55 % (w/o).

## New geometry and boundary conditions (paper: double-clamped beam, L-bracket, multi-load part)

"Duct-pierced simply-supported girder":
- Domain 100 × 50 mm (2:1), plane stress, unit thickness, E = 1, ν = 0.3
  (normalised, as the paper's LF).
- Pin support: bottom-left 5 mm solid pad, ux = uy = 0. Roller: bottom-right
  5 mm pad, uy = 0.
- Uniformly distributed downward load on a 2 mm non-design solid top skin
  (deck). Total load P (normalised).
- Non-design circular void (service duct), r = 7.5 mm, centred at
  (25, 22) mm: quarter span, where shear is high, so the design must route
  around it.
- Design region: everything else.

## Fidelities

- **LF (initial designs)**: homogenisation-based TO on a 50 × 25 coarse
  grid (2 mm). Variables per element μ1, μ2 ∈ [Lmin, 1], θ, ρ ∈ [ρmin, 1];
  C̃ = T(θ)ᵀ C^H(μ1ρ, μ2ρ) T(θ), paper Eq. 31. C^H(μ1, μ2) of the cross
  cell from numerical homogenisation (Andreassen & Andreasen 2014 style,
  periodic, 100 × 100 px cell, E_void = 1e-9) on a 21 × 21 (μ1, μ2) grid,
  bivariate polynomial fit (Eq. 4). Optimiser: widths/ρ by MMA or OC with
  JAX autodiff sensitivities; θ by principal-stress alignment each
  iteration (as deHomTop808), not as a gradient variable.
- **Phasor de-homogenisation**: port of deHomTop808 (period, kernel
  bandwidth, phase alignment, sawtooth, threshold by width) to
  Python/numpy. Period ~8 mm. Fine grid 0.2 mm → 500 × 250 px, so a
  Lmin = 0.10 bar is 4 px wide.
- **HF (in loop)**: binary φ on the 500 × 250 Q4 pixel grid (one element
  per pixel), ersatz void E = 1e-6, islands not connected to the load path
  removed by connected-component labelling first. Static: compliance and
  max |u|, pypardiso. Buckling: linear eigenproblem (K + λ K_σ) ψ = 0, void
  K_σ zeroed, shift-invert eigsh, first positive λ.
- **Independent oracle (grading only, never in loop)**: marching-squares
  contour (skimage) → equal-arc-length resampling (paper Eqs. 15–20) →
  gmsh body-fitted CPS6 triangles → CalculiX 2.23 static + *BUCKLE. Used
  on the final fronts of every arm and on gen 0.

## Evolutionary loop

- Channels: (μ1ρ, μ2ρ, cos 2θ, sin 2θ). θ is encoded as a 2θ unit vector
  because a laminate direction is π-periodic and raw θ has branch jumps
  that PCA/VAE would blend into garbage. (Paper uses raw θ in ±4π.)
- PCA per channel over the spatial dimension, fit on gen 0; Nc chosen for
  99 % explained variance (cap 64).
- Archive = every HF-evaluated design. Each generation: NSGA-II rank +
  crowding → Ns = 100 elites → train FC VAE (in → 256 → 128 → 8 latent;
  β-weighted KL) → 100 children by latent crossover (encode two elites,
  convex blend + small noise, decode) → mutation on 10 children × 10
  random centres (paper settings r = 20, p0 = 2 in coarse units?) with the
  displacement field of a cheap LF solve of that design → HF evaluate.
- Objectives: stiffness arm (Gvf, compliance); buckling arm (Gvf, −BLF)
  subject to compliance ≤ gen-0-front compliance at the same volume
  (constrained domination).
- HV on normalised objectives with a reference point fixed from gen 0.
- 100–150 generations (the paper's own cost study uses 100).

## Arms and controls

1. FULL-S: full method, stiffness, 3 seeds.
2. FULL-B: full method, buckling + stiffness constraint, 1–3 seeds.
3. NOMUT-S: no mutation (paper's Case 2 analogue).
4. NOVAE-S: VAE crossover replaced by Gaussian perturbation in PCA space,
   matched child count.
5. NULL-S (selection-only null, from hub concept `zhou2026does`): spend
   the same number of HF evaluations on more LF designs (denser V0 × Lmin
   sweep, varied filter radius, varied phasor period and global phase)
   and take the non-dominated set. Tests whether "HV +15 %" is search or
   just best-of-N among more evaluated designs.
6. SIMP baseline at 5 matched volumes, evaluated in the same HF (paper
   Dsg.4).

## Draft hypotheses (to be fixed in PREREGISTRATION.md)

- H1 replication: FULL-S HV gain over gen 0 ≥ 10 %; FULL-B ≥ 10 %.
- H2 null: FULL-S final HV > NULL-S HV at matched HF budget, on every seed.
- H3 mutation: FULL-S > NOMUT-S final HV on mean over seeds.
- H4 buckling: FULL-B beats SIMP in BLF at matched volume, compliance
  within 5 %.
- H5 oracle: pixel HF vs CalculiX body-fitted within 5 % compliance,
  10 % BLF; arm ordering by HV unchanged under CalculiX.

## Compute

32 cores, 91 GB, RTX 5080. HF static ≈ 1 s, buckling ≈ 5 s at 250 k DOF
(to measure). 100 children × 100 gens = 1e4 evals per arm. Shared box:
chemrage-drift 08:10–08:26 UTC takes 25 cores, so use ≤ 16 workers and
launch via `claude-coordinator-jobs run`.
