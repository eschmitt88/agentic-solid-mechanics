# NOTES

Running log of work sessions. `/wrap` appends a new dated section at the
end of each session with **Did / Findings / Next** subsections. The
SessionEnd hook backstops this if you forget.

<!-- entries go below this line, newest at bottom -->

## 2026-06-16 — Project founding: scope, literature base, solver decision

### Did
- Scaffolded the project (repo + GitHub Pages, `agency: max`, RAM=91GB). Set the
  two-loop framing in README/CLAUDE: (1) agent-as-operator, (2) differentiable
  inverse design; single-physics solid mechanics first.
- Ran `/discover` (3 parallel sweeps → 15-entry triage) on agentic simulation,
  differentiable FEM, and the open-source solver landscape.
- **ADR 0001**: committed **CalculiX** (Abaqus-style `.inp` text decks) as the
  operator solver — cleanest text-in/text-out surface for an agent. JAX-FEM is
  the working assumption for loop 2 (its own ADR when that loop starts).
- Ingested all 15 ranked candidates (10 papers + 5 solver repos), declined 6
  honorable mentions with reasons, archived the triage to `_done/`.
- Built the concept graph: 6 concepts across 3 agentic design axes
  (drive / differentiate / write-from-scratch a solver) + cross-cutting
  (design-optimization, self-correction, physics-grounded-evaluation).
  All wikilinks resolve.

### Findings
- The field already ships the evaluation harnesses we'd want — **FEM-Bench**
  (objective pass@5) and **EngDesign** (simulation-in-the-loop scoring) — which
  map cleanly onto our HCE rule. **ALL-FEM** (2026, FEniCS, 39-task benchmark)
  is the closest published analog; **Foam-Agent** is the best "agent drives a
  CLI solver" architecture (RAG + dependency-aware scheduling, MCP tools).
- Known agent failure modes are specific: geometric nonlinearity,
  buckling/eigenvalue problems, writing discriminative tests → a ready-made
  difficulty ladder for trials.
- Install caveats logged: FEniCSx needs conda (this box is uv-only → reason we
  picked CalculiX); JAX-FEM needs recent jaxlib (CUDA 12.6+) for the RTX 5080's
  Blackwell sm_120 before loop-2 trials.

### Next
- Trial roadmap (in README, not yet authorized to build):
  1. CalculiX cantilever operator baseline (agent writes deck → runs → parses →
     validates vs. analytical beam + mesh convergence).
  2. Operator design loop (min mass s.t. stress constraint, iterative).
  3. Differentiable inverse design on JAX-FEM (topology / param ID, GPU).
- Borrow FEM-Bench's pass@5 + EngDesign's sim-in-the-loop scoring for HCE grading.

### Did (cont.) — Trial 1 built and run
- Installed CalculiX 2.23 + Gmsh via micromamba (`solidmech` env, no sudo, on SN850X).
- `experiments/2026-06-16-calculix-cantilever-baseline/`: full harness —
  `cantilever.py` (analytical + C3D20R deck gen + ccx runner + .dat parser +
  grader), `reference_sweep.py` (deterministic ground truth), `agent_operator.py`
  (automated LLM operator loop w/ run_ccx + submit_answer tools, pass@k).
- Ran it: deterministic reference mesh-converged (0.30% deflection err); a live
  subagent operated ccx **unaided** and hit **0.275% deflection error (PASS)**,
  authoring its own mesh generator + deck, 2 runs, 0 errors.

### Findings (cont.)
- Agent-as-operator loop works end to end on the cantilever; the agent reasoned
  about physics (chose C3D20R to avoid shear locking, explained FEM>EB shear gap).
- Stress grading exposed a real subtlety: agent reported surface-peak vM (incl.
  clamp stress concentration, 10% over nominal) vs the grader's nominal bending
  stress — a metric-definition gap to fix, not an agent error.
- Two harness bugs found+fixed: PyYAML needs signed exponents (`2.1e+11`);
  mesh convergence = successive-solution-change→0, not error-vs-EB→0.

### Next (updated)
- `agent_operator.py --trials 10` for pass@k + deck-error rate (needs ANTHROPIC_API_KEY).
- Trial 2 (operator design loop: size section to a stress constraint).
- Add a problem *distribution* + HCE held-out test split; disambiguate the
  stress metric and add a .frd von-Mises parser.

## 2026-06-21 — Trial 2: operator design loop

### Did
- Built `experiments/2026-06-21-calculix-cantilever-design-loop/`: size a
  cantilever (b,h) to minimise mass s.t. tip deflection ≤ 3mm (FE-graded) +
  nominal bending stress ≤ 200MPa (closed-form — applies trial-1's lesson that
  the FE clamp-stress peak is a mesh-dependent singularity, ill-posed as a constraint).
- `fea.py` (reuses trial-1 C3D20R gen), `design.py` (FE-true reference via b-scan +
  h-bisection; grader on fixed n=8 mesh so reference passes its own grader),
  `reference_design.py`, `agent_design.py` (claude -p harness, pass@k, no API key).
- FE-true optimum: b=10mm, h=108.5mm, mass=8.519kg (deflection-binding, stress slack).

### Findings
- Live subagent designer: **+0.07% above the FE-true optimum**, feasible, 3 FE evals.
  Found the non-obvious corner solution (drive width b→min; reasoned mass∝b^(2/3) at
  the deflection limit) AND closed the FE loop (noticed FE deflection > Euler-Bernoulli
  seed due to shear, nudged h up). Genuine constrained optimisation, not a blind sweep.
- This beam is slender (L/h≈9) so the EB-vs-FE gap is only ~0.2% — feasibility is
  easy; the discriminator is optimality. To make FE *essential*, next variants need a
  stubbier/tapered section or both constraints active.

### Next
- `agent_design.py --trials 10` → feasible-rate + mass-gap distribution (pass@k).
- Harder variants: both constraints active (raise P), or tapered/stepped section (no
  closed form) so FE is genuinely in the loop.
- Trial 3: differentiable inverse design on JAX-FEM (loop 2, GPU).

## 2026-06-22 — Trial 2b (tapered) + Trial 3 (differentiable, loop 2)

### Did
- Trial 2b: tapered cantilever design loop (no closed-form deflection → FE
  genuinely required). Agent ran a 29-design FE search, matched the FE-true
  optimum (b=10/h_root=148/h_tip=37mm, 7.26kg, 14.8% lighter than prismatic),
  and explicitly flagged "beam theory unsafe" — discovered the FE-essential point.
- Trial 3 (first loop-2): differentiable inverse design. Built jaxfem.py (compact
  differentiable plane-stress FEM, SIMP, autodiff compliance; stands in for
  JAX-FEM). Agent wrote its OWN OC optimizer from jax.value_and_grad and solved
  min-compliance topology optimization → +0.31% vs reference, volume-feasible, PASS.
- Installed jax[cuda12] (GPU-ready; CPU now).

### Findings
- All three agentic axes now demonstrated: drive a solver (trial 1), design with a
  solver (trials 2/2b), differentiate a solver (trial 3). Agent reached/within
  ~0.3% of optimum in every case, authoring its own decks/optimizers, graded
  against physics — no fabrication (independently recomputed).
- **GPU is down**: NVIDIA driver package-updated (NVML 580.167) but old kernel
  module (NVRM 580.159.03) still loaded → nvidia-smi + CUDA fail
  (CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE). Needs a reboot / sudo module reload.
  Trial 3 ran on CPU (6s); same code uses GPU once fixed.

### Next
- After reboot: re-run trial 3 on GPU, scale grid, migrate to JAX-FEM proper (3D).
- pass@10 batch for trial 2 (prismatic) running in background — record when done.
- Headless agent_topopt.py harness for loop-2 pass@k.

### Did (cont.) — QA review surface + GPU back
- Built a static, interactive **QA review site** at `docs/qa/` (Pages, linked from
  the landing page → "QA review"): one page per trial grading two axes —
  **physics** (interactive 3D deformed/σ_Mises scenes via pyvista `export_html`,
  PNG fallback; convergence/design-space/topology plots) and **agentic
  legitimacy** (leakage scan, independent metric recompute, FE-eval/turn counts,
  pass@k). Tooling: `_meta/qa/qa_lib.py` + `build_qa.py`, headless via VTK EGL
  (no X/Xvfb). Design-trial decks only `*NODE PRINT`→.dat, so showcased designs
  are re-solved with `*NODE FILE`/`*EL FILE` to colour by stress.
- Chose static-baked over the agentic-CAD project's live FastAPI viewer: FEA
  experiment results are frozen once run, so no service is warranted.
- **GPU back up**: reboot reconciled the driver/kernel mismatch. `nvidia-smi` OK
  (RTX 5080, 580.167.08, 16 GB free); project `.venv` JAX 0.10.2 reports
  `CudaDevice(id=0)`. Trial-3-on-GPU + JAX-FEM-3D follow-ups are unblocked.

### Findings (cont.)
- Caught a real QA-viz bug pre-publish: agent saved the **raw flat column-major**
  topology vector while the reference is saved image-oriented; naive
  `reshape(nely,nelx)` made a plausible-but-wrong picture. Fixed to
  `reshape(nelx,nely).T` — agent truss now visibly matches reference. Logged in
  `_meta/qa/README.md` gotchas. (Reminder: a wrong visualization is worse than none.)
- Ground-truth provenance is **mixed** and worth stating plainly: trial 1 is
  anchored to a genuine closed-form benchmark (Euler–Bernoulli) + mesh
  convergence; trials 2/2b/3 are graded against **self-computed reference
  optima** (brute scan + FE-bisection; our own OC optimizer) — independent in
  method and internally consistent, but not externally-published gold answers.
  Strengthening path: adopt a published benchmark (NAFEMS, the 88-line topopt
  MBB/cantilever values, or FEM-Bench/EngDesign/ALL-FEM) for an external gold.

### Did (cont.) — GPU back, external-gold verification, 3D FEM (trial 4)
- **GPU re-run (trial 3):** confirmed working post-reboot; GPU result is
  bit-identical to CPU (C=295.69531 → determinism) and ~4× faster at 48×16.
- **External-gold verification (trial 3, `verify_benchmark.py`):** answers "how do
  we know the reference is right." (1) Forward-model gold: the differentiable FEM,
  uniform density, reproduces analytical beam theory — max |err vs Timoshenko|
  0.86%, and L/h=3 shows EB off +8.9% (shear) while Timoshenko holds to +0.2%.
  (2) Optimizer gold: reproduces the canonical MBB beam topology [Sigmund 2001 /
  Andreassen 2011], C≈218.5 (published scalar is filter/rmin-dependent ~200–220 →
  topology is the rigorous match).
- **GPU grid scaling (trial 3):** 120×40 (ndof 9922) runs on GPU in ~70 s — dense
  direct solve is the wall (ndof ∝ resolution³).
- **Trial 4 — 3D differentiable FEM** (`2026-06-22-jaxfem3d-cantilever-topology/`):
  hand-rolled 8-node hex (B8) via Gauss quadrature, SIMP, 3D density filter,
  autodiff compliance, on GPU. Element exact (symmetric, 6 rigid-body zero-eigs,
  ~0 force under rigid translation). Forward model converges monotonically to
  Euler–Bernoulli under through-thickness refinement (−12.4%→−3.4%→−1.4%→−0.61%);
  residual = textbook shear locking of fully-integrated trilinear hex. 3D topology
  opt (24×8×8, ndof 6075) → C=49.98, vol 0.30, sensible 3D load path.
- **QA site extended:** trial-4 3D interactive scene + beam-convergence plot +
  element-gold panel; trial-3 page gained the external-gold verification (Timoshenko
  table + MBB image) and GPU-scaling sections. Rebuilt + pushed.

### Did (cont.) — QA pages rewritten for newcomers + matrix-free GPU solver
- **QA pages made self-contained** (user feedback: too much insider jargon): each
  page now opens with a plain-language objective, a boundary-condition diagram
  (`qa_lib.bc_beam`/`bc_domain_2d`/`bc_domain_3d`/`bc_mbb`), a setup table, and an
  explicit "what is measured"; acronyms are spelled out and a per-page glossary
  (`qa_lib.GLOSSARY`/`terms_block`) defines them. Removed "loop-2 substrate",
  "forward-model/optimizer gold", etc. Structure documented in `_meta/qa/README.md`
  so future pages inherit it.
- **Matrix-free (sparse) GPU solver** (`jaxfem3d_mf.py`): element-by-element
  Jacobi-preconditioned conjugate gradient instead of dense assemble+solve, AND a
  3D-convolution density filter instead of the dense O(nelem²) H matrix (the H
  matrix was the real memory wall — a 64³-ish grid's H alone is ~10 GB). Verified
  **bit-identical to dense** (compliance ~1e-13, gradient ~1e-9 → autodiff
  sensitivities preserved). ~600× faster at 40×12×12 (11 ms vs 6.6 s); solves
  **121,875 unknowns in 0.43 s** (dense matrix would be ~119 GB — impossible on
  16 GB). Full 48×16×16 optimization (42,483 unknowns) in 3.3 s. All pushed.

### Next (updated)
- Loop-2 agentic pass@k on the (now scalable) matrix-free 3D substrate.
- Reduced-integration/B-bar hex to cut B8 shear locking on coarse meshes.
- Warm-started CG across optimisation iterations for even larger 3D grids.
