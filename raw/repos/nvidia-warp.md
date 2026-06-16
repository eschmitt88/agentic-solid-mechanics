---
repo: NVIDIA/warp
url: https://github.com/NVIDIA/warp
fetched: 2026-06-16
---

# NVIDIA Warp

**[Documentation](https://nvidia.github.io/warp/stable/)** | [Changelog](https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md)

Warp is a Python framework for GPU-accelerated simulation, robotics, and machine learning. Warp takes
regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp comes with a rich set of primitives for physics simulation, robotics, geometry processing,
and more. Warp kernels are differentiable and can be used as part of machine-learning pipelines
with frameworks such as PyTorch, JAX and Paddle.

## Quick Start

Simulate one million particles under gravitational attraction, in 20 lines:

```python
import warp as wp
import numpy as np

num_particles = 1_000_000
dt = 0.01

@wp.kernel
def gravity_step(pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
    i = wp.tid()
    position = pos[i]
    dist_sq = wp.length_sq(position) + 0.01  # softened distance
    acc = -1000.0 / dist_sq * wp.normalize(position)  # gravitational pull toward origin
    vel[i] = vel[i] + acc * dt
    pos[i] = pos[i] + vel[i] * dt

rng = np.random.default_rng(42)
positions = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3)
velocities = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3)

for _ in range(100):
    wp.launch(gravity_step, dim=num_particles, inputs=[positions, velocities])

print(positions.numpy())
```

## Installing

Python version 3.10 or newer is required. Warp can run on x86-64 and ARMv8 CPUs on Windows and Linux, and on Apple Silicon (ARMv8) on macOS.
GPU support requires a CUDA-capable NVIDIA GPU and driver (minimum GeForce GTX 9xx).

The easiest way to install Warp is from [PyPI](https://pypi.org/project/warp-lang/):

```text
pip install warp-lang
```

You can also use `pip install warp-lang[examples]` to install additional dependencies for running examples and USD-related features.

For nightly builds, conda, CUDA 13 builds, building from source, and CUDA driver requirements, see the
[Installation Guide](https://nvidia.github.io/warp/stable/user_guide/installation.html).
(Pre-built PyPI wheels bundle CUDA 12 runtime; minimum CUDA driver >= 525 for the CUDA 12 builds.)

## Running Examples

The `warp/examples` directory contains examples covering physics simulation, geometry processing,
optimization, and tile-based GPU programming. Install optional example deps with
`pip install warp-lang[examples]`. Run via `python -m warp.examples.<example_subdir>.<example>`.

### warp/examples/fem

The `warp.fem` module provides FEM building blocks. Bundled examples include:
diffusion 3d, mixed elasticity, apic fluid, streamlines, distortion energy, taylor green,
kelvin helmholtz, magnetostatics, adaptive grid, nonconforming contact, darcy level-set
optimization, elastic shape optimization.

### warp/examples/optim

diffray, fluid checkpoint, particle repulsion, navier-stokes perturbation.

### warp/examples/core

dem, fluid, graph capture, marching cubes, mesh, nvdb, raycast, raymarch, sample mesh, sph,
torch, wave, 2-D incompressible turbulence.

### warp/examples/tile

mlp, nbody, mcgp.

## Learn More

* [Product Page](https://developer.nvidia.com/warp-python)
* [SIGGRAPH 2024 Course Slides](https://dl.acm.org/doi/10.1145/3664475.3664543)
* [SIGGRAPH Asia 2021 Differentiable Simulation Course](https://dl.acm.org/doi/abs/10.1145/3476117.3483433)

## License

Warp is provided under the Apache License, Version 2.0.

This project will download and install additional third-party open source software projects.
When building from source, the build downloads NVIDIA libmathdx (governed by the NVIDIA Software
License Agreement); pre-built PyPI packages already include libmathdx statically linked.

## Publications & Citation

[PUBLICATIONS.md](https://github.com/NVIDIA/warp/blob/main/PUBLICATIONS.md) lists academic and
research publications leveraging Warp.
