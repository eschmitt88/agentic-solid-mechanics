---
repo: https://github.com/FEniCS/dolfinx
fetched: 2026-06-16
---

# DOLFINx (README snapshot)

[![DOLFINx CI](https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml)
[![Actions Spack build](https://github.com/FEniCS/dolfinx/actions/workflows/spack.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/spack.yml)
[![Actions Conda install](https://github.com/FEniCS/dolfinx/actions/workflows/conda.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/conda.yml)
[![Actions macOS/Homebrew install](https://github.com/FEniCS/dolfinx/actions/workflows/macos.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/macos.yml)
[![Actions Docker images](https://github.com/FEniCS/dolfinx/actions/workflows/docker-end-user.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/docker-end-user.yml)
[![Actions Windows/vcpkg install](https://github.com/FEniCS/dolfinx/actions/workflows/windows.yml/badge.svg)](https://github.com/FEniCS/dolfinx/actions/workflows/windows.yml)

DOLFINx is the computational environment of
[FEniCSx](https://fenicsproject.org) and implements the FEniCS Problem
Solving Environment in C++ and Python. DOLFINx is a new version of
DOLFIN and is actively developed.

For questions about using DOLFINx, visit the [FEniCS
Discourse](https://fenicsproject.discourse.group/) page or use the
[FEniCS Slack channel](https://fenicsproject.slack.com/) (use
[this](https://join.slack.com/t/fenicsproject/shared_invite/zt-1lraknsp1-6_3Js5kueDIyWgF192d3nA)
link to sign up to the Slack channel).

## Documentation

Documentation can be viewed [here](https://docs.fenicsproject.org).

## Installation

### From source

For detailed instructions and a list of dependencies, see
[here](https://docs.fenicsproject.org/dolfinx/main/python/installation).

#### C++ core

To build and install the C++ core, in the `cpp/` directory, run:

```shell
mkdir build
cd build
cmake ..
make install
```

#### Python interface

To install the Python interface, first install the C++ core, and then in
the `python/` directory run:

```shell
pip install scikit-build-core
python -m scikit_build_core.build requires | python -c "import sys, json; print(' '.join(json.load(sys.stdin)))" | xargs pip install
pip install --check-build-dependencies --no-build-isolation .
```

### Spack

To build the most recent release using
[Spack](https://spack.readthedocs.io/) (assuming a bash-compatible
shell):

```shell
git clone https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
spack install --add py-fenics-dolfinx+petsc4py+slepc4py
```

Spack is the recommended approach for HPC systems.

### Binary packages

**Recommendations**

- macOS: conda.
- Linux: apt (Ubuntu/Debian), docker or conda. See also Spack.
- Windows: docker, or install WSL2 and use Ubuntu. conda packages in beta testing.
- High performance computers: Spack or from source, both using system-provided MPI.

#### conda

To install the latest release of the Python interface, including pyvista for
visualisation, using [conda](https://conda.io):

```shell
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista # Linux and macOS
conda install -c conda-forge fenics-dolfinx pyvista pyamg # Windows
```

*Windows only*: Windows conda packages are currently in beta testing. PETSc
and petsc4py are not available on Windows.

The recipe is hosted on
[conda-forge](https://github.com/conda-forge/fenics-dolfinx-feedstock).

#### Ubuntu packages

The [Ubuntu PPA](https://launchpad.net/~fenics-packages/+archive/ubuntu/fenics)
provides FEniCSx packages:

```shell
add-apt-repository ppa:fenics-packages/fenics
apt update
apt install fenicsx
```

#### Debian packages

DOLFINx is included with various versions of Debian. Install with
`apt-get install fenicsx`.

#### Docker images

To run a Docker image with the latest release of DOLFINx:

```shell
docker run -ti dolfinx/dolfinx:stable
```

A Jupyter Lab environment with the latest release of DOLFINx:

```shell
docker run --init -ti -p 8888:8888 dolfinx/lab:stable  # Access at http://localhost:8888
```

The Docker images support arm64 and amd64 architectures. For a full list
of tags see <https://hub.docker.com/u/dolfinx>

## License

DOLFINx is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License (LGPL), either
version 3 of the License, or (at your option) any later version.

DOLFINx is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY. See the GNU Lesser General Public License for more details.
<https://www.gnu.org/licenses/>
</content>
</invoke>
