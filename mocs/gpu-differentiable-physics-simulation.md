---
kind: moc
name: "GPU-Accelerated & Differentiable Physics Simulation"
added: "2026-06-25"
status: growing
related_concepts:
  - gpu-differentiable-physics-simulation
  - weak-form-fem-dsl
  - implicit-differentiation-through-solve
  - boussinesq-natural-convection
  - differentiable-lattice-boltzmann
  - multi-backend-differentiable-physics-framework
  - differentiable-inverse-design
  - code-from-scratch-numerical-solver-agent
related_experiments:
  - 2026-06-22-jaxfem3d-cantilever-topology
  - 2026-06-23-vhb-viscohyperelastic-calibration
tags: [gpu, differentiable-simulation, cfd, fem, lbm, moc]
---

# GPU-Accelerated & Differentiable Physics-Simulation Landscape

Synthesis of a verified research sweep, mapped to our two loops:
- **Loop 1** — an LLM agent *operates* a physics solver (drives, configures, runs, reads output).
- **Loop 2** — *differentiable / inverse design* — gradients flow through the solve for optimization, sensitivity, parameter ID.

Every tool below is scored on three axes the project actually cares about: **GPU** (does it run on our Blackwell sm_120 RTX 5080, no-sudo conda), **Differentiable** (real end-to-end autodiff vs. asserted), and **Maturity** (peer-reviewed / actively maintained / packaged). The dominant honest caveat across the whole CFD column is the same one: **nobody ships a turnkey Boussinesq natural-convection solver** — that physics is DIY everywhere.

---

## Category 1 — Kernel DSLs / compilers (write-your-own-GPU-PDE substrate)

The "author arbitrary GPU kernels in Python, get adjoints for free" layer. Maximum generality, maximum implementation burden.

- **NVIDIA Warp** — JIT-compiles `@wp.kernel` Python to CUDA PTX (forward-compatible to sm_120), auto-generates reverse-mode adjoints via `wp.Tape`. First-party NVIDIA, Apache-2.0, monthly releases (v1.14.0, Jun 2026), conda-forge `warp-lang`. **GPU: yes. Diff: yes (per-kernel adjoint, standard AD caveats). Maturity: high.** The most general substrate we have for both loops.
- **warp.fem** — the weak-form Galerkin FEM layer inside Warp: declare integrands/function spaces in Python, GPU sparse assembly + solve, differentiable through `wp.Tape`. Ships 2D Navier-Stokes + convection-diffusion examples (the exact pieces of a Boussinesq solve) but **no buoyancy-coupled example**. v1.13 added float64; v1.14 added batched multi-environment solves (→ parallel slat-angle sweeps). Diff is Warp-Tape, not JAX-native; crosses to JAX via dlpack arrays only.
- **Taichi / DiffTaichi** *(named in the project brief and survey references; treat as landscape context)* — the original "embed a differentiable physics DSL in Python with a megakernel compiler" effort. Establishes the category that Warp now leads with first-party vendor backing. Coordinated lineage (DiffTaichi → Taichi) but less GPU-vendor-aligned than Warp for Blackwell.

**Loop fit:** Loop 1 (agent writes/drives kernels) and Loop 2 (adjoints) both — this is the "code-from-scratch numerical solver agent" category.

## Category 2 — JAX physics ecosystem (compose inside `jax.grad`/`jit`/`vmap`)

The most coherent, *coordinated* corner of the field — a real stack, not scattered tools. This is where our hand-rolled JAX FEM already lives, so adoption friction is near zero and sm_120 risk is inherited from the already-working jaxlib.

- **JAX-FEM** (DeepModeling) — pure-Python differentiable FEM; AD produces the Newton tangent *and* design sensitivities. Ships Poisson/heat, elasticity, hyperelasticity, plasticity, **Stokes flow** (no Navier-Stokes/Boussinesq). Assembled-matrix (petsc4py dependency), **GPL-3.0**. Peer-reviewed (CPC 2023), active (Jun 2026). The closest published analog to our own JAX FEM. **GPU: yes. Diff: yes (core design). Maturity: high but 0.0.x release hygiene — pin commits.**
- **Lineax** (Kidger) — differentiable linear solves + least-squares behind one operator abstraction; matrix-free `JacobianLinearOperator`, stable implicit-diff through the solve. Apache-2.0, active. **Infrastructure**, not a PDE solver. Pairs with / could replace our ad-hoc CG plumbing.
- **Optimistix** (Kidger) — differentiable nonlinear root-find / minimize / least-squares / fixed-point, implicit-function-theorem adjoints through converged solutions. conda-forge noarch, Apache-2.0, active. The canonical JAX answer to "I have a differentiable residual, give me a robust differentiable nonlinear solver" — exactly the outer loop for a steady Boussinesq solve.
- **PhiFlow** (TUM) — backend-agnostic (NumPy/Torch/JAX/TF) grid incompressible Navier-Stokes + heat advection-diffusion, ICML 2024, MIT. With the JAX backend it *adds no kernels of its own* → zero sm_120 build risk. Ships Lid-Driven Cavity / Smoke Plume but **no Boussinesq**; slats need immersed-boundary masking. Stale 3.4.0 tag vs. live develop branch — pin a commit.

**Coordination note:** Lineax + Optimistix + (our FEM) compose cleanly — Optimistix wraps the nonlinear outer loop, Lineax/our CG the linear inner solve. This is the most *integrated* sub-stack in the whole landscape and the natural home for Loop 2.

## Category 3 — GPU Lattice-Boltzmann CFD

- **XLB** (Autodesk) — fully differentiable LBM on JAX (+ Warp + Neon backends), `jax.grad` through the time-stepping loop, shipped differentiable inverse-problem example (May 2026). Apache-2.0, peer-reviewed (CPC 2024), active. Installs via `pip install "xlb[cuda]"` → jax[cuda13], matching our stack. **GPU: yes. Diff: yes (JAX backend only; Warp/Neon are perf, not AD). Maturity: high.** Decisive caveat: **isothermal — thermal/Boussinesq is roadmap WIP**, so not turnkey for the window-shade problem. Best-architected differentiable-CFD reference we have.

## Category 4 — GPU FVM / spectral / high-order CFD

- **JAX-Fluids** *(via the survey)* — high-order compressible, two-phase, JAX. Differentiable, GPU. But compressible/two-phase ≠ incompressible buoyant natural convection — wrong regime for the window-shade problem.

This column is the *thinnest* for our specific need: the differentiable-CFD tools that exist target compressible (JAX-Fluids) or isothermal (XLB) regimes, not incompressible Boussinesq.

## Category 5 — Differentiable physics engines / surveys (orientation)

- **Newbury et al. (2024), "A Review of Differentiable Simulators"** (IEEE Access) — taxonomy by gradient method (autodiff vs. analytical/adjoint), versatility-vs-speed-vs-accuracy trade-off triangle. Robotics/contact-centric; **fluid coverage is shallow** (PhiFlow, JAX-Fluids, FluidLab — none thermal). Confirms: turnkey differentiable natural-convection does not exist; we will adapt or hand-roll. Good citation anchor, not a tool.

---

## What's coordinated vs. scattered

- **Coordinated:** (a) the **JAX scientific stack** — Lineax/Optimistix/Diffrax/Equinox + JAX-FEM + PhiFlow's JAX backend + XLB's JAX backend all interoperate under one autodiff/XLA umbrella; (b) **NVIDIA's Warp/warp.fem** as a single vendor-backed kernel+FEM substrate with a JAX FFI bridge.
- **Scattered / siloed:** Taichi/DiffTaichi (separate compiler lineage), the various robotics diff-engines in the Newbury survey (Brax/Nimble/Dojo/TDS), and LBM (XLB) each live in their own world. Crossing between them is dlpack-array-level, not autodiff-composable.

## Where the field is heading

1. **Vendor consolidation of the DSL layer** — Warp (NVIDIA, monthly releases, JAX FFI, batched solves) is absorbing the "differentiable GPU kernel DSL" niche DiffTaichi pioneered.
2. **JAX as the lingua franca of differentiable science** — the most composable inverse-design tooling (implicit-diff solvers, matrix-free operators) is JAX-native; backends increasingly *defer* to JAX rather than ship their own CUDA (PhiFlow's JAX path, XLB's JAX backend).
3. **Differentiability is now real, not marketing** — every "differentiable" claim in this sweep is backed by shipped examples (XLB inverse-problem example, warp.fem perturbation-optimization, Optimistix IFT adjoints). The gap has moved from "is it differentiable" to "is the *right physics* shipped" — and for thermal/buoyant CFD it is not.
4. **The persistent hole: incompressible thermal/Boussinesq CFD with autodiff.** Confirmed by the survey and by every CFD tool's limitation list. This is exactly our window-shade problem, and it means hand-rolling/assembling is the honest path.

## Loop mapping summary

| Tool | Loop 1 (operate) | Loop 2 (diff/inverse) | sm_120 | Maturity |
|---|---|---|---|---|
| Warp / warp.fem | strong | strong (Warp Tape) | yes | high |
| JAX-FEM | strong | strong (AD-Newton) | yes | high (GPL-3.0) |
| XLB | strong | strong (isothermal only) | yes | high |
| PhiFlow | strong | strong | inherited | mid (pin commit) |
| Optimistix | weak (plumbing) | strong (IFT adjoints) | inherited | mid |
| Lineax | weak (plumbing) | strong (stable linear-solve grads) | inherited | mid |
| Newbury survey | n/a | orientation only | n/a | published |

---

## Map — notes in the graph

**Concepts** (new this sweep): [[gpu-differentiable-physics-simulation]] · [[weak-form-fem-dsl]] · [[implicit-differentiation-through-solve]] · [[boussinesq-natural-convection]] · [[differentiable-lattice-boltzmann]] · [[multi-backend-differentiable-physics-framework]]

**Tools** (literature/repos): [[xlb]] · [[phiflow]] · [[lineax]] · [[optimistix]] · [[pyfr]] · [[mujoco-mjx]] · [[taichi]] · [[nvidia-warp]] · [[deepmodeling-jax-fem]] · [[fenics-dolfinx]] · [[sfepy]]

**Papers**: [[newbury2024diffsim]] · [[sapienza2024diffprog]] · [[difvm2026]] · [[zheng2022venetianblinds]] · [[xue2022jaxfem]]

## Window-shade recommendation

For the 2D venetian-blind natural-convection study (effective h vs. slat angle, Boussinesq buoyancy), no surveyed tool is turnkey — every differentiable CFD option is either isothermal (XLB) or ships only the uncoupled pieces. So the real choice is which substrate to assemble the Boussinesq coupling on. Concrete recommendation, primary + fallback:

PRIMARY — extend our existing hand-rolled JAX FEM (or JAX-FEM's get_universal_kernel custom-weak-form path) to a steady incompressible Boussinesq formulation, with Optimistix driving the nonlinear outer solve (Newton / fixed-point, implicit-function-theorem adjoints) and Lineax/our matrix-free CG handling the linear inner solves. Rationale: (1) it stays entirely inside the JAX/CUDA stack we already run on the RTX 5080, so sm_120 support is inherited from the working jaxlib with zero build risk; (2) Optimistix's IFT adjoints give correct, memory-cheap d(h)/d(angle) gradients through the converged steady flow — exactly Loop 2; (3) it composes with our SIMP/topology-opt machinery. Cost: must author the temperature-momentum buoyancy coupling and SUPG-style stabilization ourselves, and supply a preconditioner for the convection-dominated system. JAX-FEM is the reference template here; note its GPL-3.0 if any of this becomes a derivative work.

FAST-PROTOTYPE / VALIDATION PATH — PhiFlow on its JAX backend to stand up the 2D enclosure (grid incompressible NS + scalar heat advection-diffusion, immersed-boundary masking for the angled slats) quickly, with a hand-added buoyancy source term in the momentum equation. Use this to validate flow fields and ballpark h(angle) before committing to the FEM build; it inherits our jaxlib so no sm_120 risk, but pin a develop-branch commit (the 3.4.0 tag is stale) and don't expect FLUENT-grade accuracy.

ALTERNATIVE GPU SUBSTRATE — warp.fem if we want a CUDA-native (not JAX-native) assembly path: it ships the 2D Navier-Stokes and convection-diffusion examples to assemble Boussinesq from, with v1.14 batched multi-environment solves mapping cleanly onto a parallel slat-angle sweep, and conda-forge install. Trade-off: differentiation is Warp-Tape (crosses to JAX only via dlpack arrays, does not compose inside jax.grad), so prefer it for Loop-1 agent operation and forward sweeps rather than tightly-coupled JAX inverse design. Smoke-test on the actual 5080 first (sm_120 inferred from CUDA 12.6/13 coverage, not explicitly stated).

AVOID for this specific task — XLB (isothermal; thermal/Boussinesq is roadmap WIP) and JAX-Fluids (compressible/two-phase, wrong regime). Keep XLB as a differentiable-CFD architecture reference and an isothermal-flow agent target.
