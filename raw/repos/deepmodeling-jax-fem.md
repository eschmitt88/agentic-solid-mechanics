---
repo: deepmodeling/jax-fem
url: https://github.com/deepmodeling/jax-fem
fetched: 2026-06-16
source: gh api repos/deepmodeling/jax-fem/readme
---

# JAX-FEM (README snapshot)

JAX-FEM is a differentiable finite element package based on [JAX](https://github.com/google/jax).

## Documentation

For installation and user guide, please visit our [documentation](https://deepmodeling.github.io/jax-fem/) for details.

## Key features

JAX-FEM is Automatic Differentiation (AD) + Finite Element Method (FEM), and we support the following features:

- 2D quadrilateral/triangle elements
- 3D hexahedron/tetrahedron elements
- First and second order elements
- Dirichlet/Neumann/Robin boundary conditions
- Linear and nonlinear analysis including
  - Heat equation
  - Linear elasticity
  - Hyperelasticity
  - Plasticity (macro and crystal plasticity)
- Multi-physics problems
- Integration with PETSc for solver options
- Differentiable programming for solving inverse/design problems __without__ deriving sensitivities by hand, e.g.,
  - Topology optimization
  - Optimal thermal control

## Examples

- Thermal profile in direct energy deposition (DED).
- Linear static analysis of a bracket (von Mises stress).
- Crystal plasticity: grain structure and stress-xx fields.
- Stokes flow: velocity and pressure.
- Topology optimization with differentiable simulation.

## JAX-FEM Express

We have built an LLM Agent for JAX-FEM: Try JAX-FEM Express! https://www.bohrium.com/apps/jax-fem-express
(Video: https://www.youtube.com/watch?v=LsBPWGrDhiY)

## License

This project is licensed under the GNU General Public License v3 - see the LICENSE
(https://www.gnu.org/licenses/) for details. For commercial use, contact Tianju Xue
(https://ce.hkust.edu.hk/people/tian-ju-xue-xuetianju).

## Citations

```bibtex
@article{xue2023jax,
  title={JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science},
  author={Xue, Tianju and Liao, Shuheng and Gan, Zhengtao and Park, Chanwook and Xie, Xiaoyu and Liu, Wing Kam and Cao, Jian},
  journal={Computer Physics Communications},
  pages={108802},
  year={2023},
  publisher={Elsevier}
}
```
