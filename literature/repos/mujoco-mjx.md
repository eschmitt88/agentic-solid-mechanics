---
kind: repo
name: "MuJoCo MJX"
url: https://mujoco.readthedocs.io/en/stable/mjx.html
commit:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 2
status: skimmed
related_experiments: []
related_concepts:
  - gpu-differentiable-physics-simulation
  - differentiable-inverse-design
tags: [jax, gpu, physics-engine, robotics, differentiable]
---

# MuJoCo MJX

## Purpose

MuJoCo MJX (MuJoCo XLA) is Google DeepMind's JAX/XLA reimplementation of the MuJoCo physics engine, expressing rigid-body dynamics, collision, constraints, and integration as differentiable XLA primitives that run on NVIDIA/AMD GPUs, Apple Silicon, and TPU. Its purpose is massively-parallel, hardware-accelerated, end-to-end-differentiable rigid-body simulation, primarily for reinforcement learning and gradient-based control/system-identification. It is API-compatible with the MuJoCo C engine but implements a feature subset. It is a mature, actively maintained, Apache-2.0 reference example of how to express a full physics engine in JAX for GPU autodiff.

## Shape

- **GPU:** Yes. MJX is a JAX/XLA reimplementation that runs on all XLA targets: NVIDIA + AMD GPUs, Apple Silicon, and TPU. Because it is pure JAX it inherits whatever jaxlib/CUDA the host JAX provides — no separate native CUDA build of MJX itself.
- **Differentiable:** Yes, with a caveat. MJX-JAX supports automatic differentiation / gradients through the dynamics (this is its main appeal over the MuJoCo C engine). The MJX-Warp backend does NOT support autodiff. Skeptic note: rigid-body contact dynamics are inherently non-smooth, so contact-medi…
- **Install:** pip — `pip install mujoco-mjx` (optional `[warp]` extra). On PyPI, Apache-2.0, Python >=3.10. No conda-forge build needed; it is pure-Python JAX and rides on the project's existing JAX/jaxlib CUDA stack. Works fine no-su…
- **License:** Apache License 2.0
- **Maintained:** active — latest mujoco-mjx release v3.10.0 on 2026-06-22 (days before review); maintained by Google DeepMind, version-locked to the MuJoCo C engine and updated in lockstep.

Capabilities:
- JAX/XLA reimplementation of MuJoCo rigid-body dynamics, collision, constraints, integration as differentiable XLA primitives
- Runs and batches across NVIDIA/AMD GPU, Apple Silicon, TPU; designed for massively-parallel vmapped rollouts (RL / sim-to-real)
- End-to-end autodiff through contact dynamics for gradient-based control and system ID (MJX-JAX backend)
- API-compatible with MuJoCo C, drop-in for many MJCF models

## Useful bits

- Concrete, production-grade proof that an entire physics solver can be re-expressed as XLA primitives to get GPU execution + end-to-end autodiff for free — directly relevant as an architectural pattern for our hand-rolled JAX FEM and any future differentiable CFD solver, even though the physics domain differs.
- Pure-JAX, pip-install, Apache-2.0, version-locked to MuJoCo and updated in lockstep (v3.10.0, 2026-06-22) — trivially adoptable on our existing JAX/CUDA RTX-5080 stack with no native build.
- Honest scope boundary worth recording: rigid-body only, fluid = lumped inertia model, contact gradients non-smooth — so it is a methodology/pattern reference, not a tool for continuum or thermal-fluid problems.

## Reality check

Claims hold up — this is not vaporware. GPU execution, JAX-native autodiff, multi-backend support, and active DeepMind maintenance are all verified (release days before review). The honest caveats are: 'fully differentiable' glosses over non-smooth contact gradients; it implements only a subset of MuJoCo features; the Warp backend is not differentiable; and 'general domain' means general rigid-body dynamics, not general physics — no continuum mechanics, no fluids, no thermal. Solid, mature project; just narrower in scope than a casual reading of the marketing suggests, and orthogonal to the project's CFD need.

## Relevance here

- **Window-shade (natural convection):** Essentially zero direct relevance. MJX is a rigid-body multibody engine; it has no CFD, no natural convection, no Boussinesq buoyancy, and no heat transfer. Its only fluid capability is MuJoCo's crude lumped 'inertia model', which cannot produce a boundary-layer h(angle) for a venetian-blind cavity. Use a real differentiable-CFD tool for that problem.
- **Project:** Moderate as a methodological/landscape reference, low as a usable tool. It is the canonical example of a full physics engine rebuilt in JAX/XLA for GPU + autodiff, which is exactly the pattern the project pursues with its hand-rolled JAX FEM and differentiable-inverse-design loop. Relevant to the 'agent operates a simulator' loop (it is a real simulator an agent could drive) and to differentiable …

## Follow-up

- Trial the most relevant capability against our stack before relying on it.
