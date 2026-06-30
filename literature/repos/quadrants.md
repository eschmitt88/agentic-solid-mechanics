---
kind: repo
name: "Quadrants"
url: https://github.com/Genesis-Embodied-AI/quadrants
commit:
source: "(user-surfaced, assessed 2026-06-30)"
added: "2026-06-30"
relevance: 3
status: skimmed
related_experiments: []
related_concepts:
  - gpu-differentiable-physics-simulation
  - code-from-scratch-numerical-solver-agent
  - differentiable-inverse-design
tags: [gpu, compiler, dsl, autodiff, multi-platform, python]
---

# Quadrants

## Purpose

Quadrants (Genesis Embodied AI) is a multi-platform **GPU kernel compiler**, not a
physics solver: it compiles plain Python into optimized parallel kernels targeting
NVIDIA CUDA, Vulkan/SPIR-V, Apple Metal, AMD ROCm/HIP, and x86/ARM64 CPUs, with
built-in reverse- and forward-mode automatic differentiation. It sits in the same
"write-your-own differentiable GPU-PDE substrate" category as [[nvidia-warp]] and
[[taichi]] — maximum generality, maximum implementation burden — and is almost
certainly the in-house compiler backend of the **Genesis** physics engine (which
moved off a Taichi fork) spun out as a standalone package. Apache-2.0.

## Shape

- **GPU:** multi-backend — CUDA, Vulkan (SPIR-V), Metal, ROCm (HIP), and CPU
  (x86/ARM64). Broader hardware reach than Warp (CUDA-centric).
- **Differentiable:** yes — README claims it "computes the gradient of any kernel
  transparently using reverse-mode differentiation" with **dynamic loops and
  runtime-based memory allocation**; forward-mode + custom gradients supported.
- **Install:** `pip install quadrants`. **License:** Apache-2.0.
- **Maintained:** active — v1.1.0 (2026-06-26), 60 releases, ~11,400 commits; but
  only ~158 stars / 26 forks (newly public, small external community).

## Useful bits

- **The standout claim is AD through dynamic control flow + dynamic allocation.**
  Differentiating through dynamic loops / iterative solvers is exactly where Warp's
  tape-based adjoint has documented caveats; if Quadrants' version is robust it is a
  real edge for loop-2 (back-prop through an iterative Boussinesq/Newton solve).
- **Multi-backend portability** (Vulkan/Metal/ROCm, not just CUDA) — a portability
  layer like PyFR's codegen, but as a general *differentiable* kernel compiler.
- **It is a substrate, not a solver** — no shipped physics/FEM module (unlike
  `warp.fem`); you author the weak form / discretization / time-stepping yourself.

## Reality check

Healthy skepticism warranted. The parent Genesis engine launched with overstated
performance marketing (the "43M FPS" headline conflated parallel environments and
drew justified criticism), so the "gradient of *any* kernel" claim should be checked
against a real iterative-solver example before relying on it. 158 stars + newly
public = heavy internal development but unproven external adoption, vs. Warp's
NVIDIA-first-party backing and ~6.8k stars. The README is also coy about the Genesis
relationship.

## Relevance here

- **Window-shade (natural convection):** changes nothing — like Warp/Taichi you would
  hand-write the entire CFD solver in its kernels, pulling us *out* of the JAX
  ecosystem where Optimistix/Lineax/our FEM already compose. More friction than the
  standing recommendation (PhiFlow-JAX or extend-our-JAX-FEM).
- **Project:** a credible [[nvidia-warp]] alternative to track in Category 1 of
  [[gpu-differentiable-physics-simulation]]. Its advantages (non-NVIDIA portability,
  multi-backend) are not things we need on a single RTX 5080; Warp's Blackwell
  first-party support + JAX FFI bridge + `warp.fem` still win for our stack. Would
  only flip if its dynamic-loop AD proves materially more robust than Warp's tape.

## Follow-up

- If ever evaluated: benchmark its reverse-mode AD through an iterative linear solve
  (CG) against Warp's tape — the dynamic-loop-AD claim is the only thing that would
  change our Warp-default.
