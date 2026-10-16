# PyTorch-based Libcint Interface

<table>
  <tr>
    <td>Compatibility:</td>
    <td>
      <img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue.svg" alt="Python Versions"/>
      <img src="https://img.shields.io/badge/PyTorch-%3E=2.6.0-blue.svg" alt="PyTorch Versions"/>
      <img src="https://img.shields.io/badge/OS-Linux-blue.svg" alt="Operating Systems">
    </td>
  </tr>
  <tr>
    <td>Availability:</td>
    <td>
      <a href="https://github.com/tad-mctc/tad-libcint/releases/latest">
        <img src="https://img.shields.io/github/v/release/tad-mctc/tad-libcint?color=orange" alt="Release"/>
      </a>
      <a href="https://pypi.org/project/tad-libcint/">
        <img src="https://img.shields.io/pypi/v/tad-libcint?color=orange" alt="PyPI"/>
      </a>
      <a href="http://www.apache.org/licenses/LICENSE-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="Apache-2.0"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Status:</td>
    <td>
      <a href="https://github.com/tad-mctc/tad-libcint/actions/workflows/ubuntu.yaml">
        <img src="https://github.com/tad-mctc/tad-libcint/actions/workflows/ubuntu.yaml/badge.svg" alt="Tests Ubuntu"/>
      </a>
      <a href="https://github.com/tad-mctc/tad-libcint/actions/workflows/release.yaml">
        <img src="https://github.com/tad-mctc/tad-libcint/actions/workflows/release.yaml/badge.svg" alt="Build Status"/>
      </a>
      <a href="https://tad-libcint.readthedocs.io">
        <img src="https://readthedocs.org/projects/tad-libcint/badge/?version=latest" alt="Documentation Status"/>
      </a>
      <a href="https://results.pre-commit.ci/latest/github/tad-mctc/tad-libcint/main">
        <img src="https://results.pre-commit.ci/badge/github/tad-mctc/tad-libcint/main.svg" alt="pre-commit.ci Status"/>
      </a>
      <a href="https://codecov.io/gh/tad-mctc/tad-libcint">
        <img src="https://codecov.io/gh/tad-mctc/tad-libcint/branch/main/graph/badge.svg?token=9faLOhisRx" alt="Coverage"/>
      </a>
    </td>
  </tr>
</table>


## Content

This package contains a PyTorch-based interface to the *[libcint](https://github.com/sunqm/libcint)* high-performance integral library (check version [here](https://github.com/tad-mctc/libcint/blob/dxtb/README.rst)).
The interface facilitates automatic differentiation with custom backward functions that call *libcint*'s C backend for derivative calculation.

It is based on [PySCF's lib directory](https://github.com/pyscf/pyscf/tree/master/pyscf/lib). Modifications include

- only the driver for two-centre integral matrices (`GTOint2c`) and its helpers are kept
- a new CMake build that compiles the libcint submodule directly and suffixes all libraries with `_tad`

The project was heavily inspired by *[diffqc/dqc](https://github.com/diffqc/dqc)* and *[diffqc/dqclibs](https://github.com/diffqc/dqclibs)*.

This interface was mainly written for the calculation of integrals within *[grimme-lab/dxtb](https://github.com/grimme-lab/dxtb)*.


## Usage

A `Basis` holds the shells of every element: their angular momenta, exponents and contraction coefficients.
The atoms are passed to every integral call as a [tad-mctc](https://github.com/tad-mctc/tad-mctc) `Structure` (atomic numbers and positions in Bohr), so one basis serves all conformers of a molecule as well as different molecules.
Contraction coefficients are used as given, i.e., they must include the normalization of the primitives.

```python
import torch
from tad_mctc.io.structure import Structure
from tad_libcint import Basis, Shell, int1e, overlap

dd = {"dtype": torch.double}

# STO-3G shells of each element: angular momentum, exponents, coefficients
# (coefficients include the normalization in libcint's convention)
shells = {
    1: [
        Shell(0, torch.tensor([3.42525, 0.623914, 0.168855], **dd), torch.tensor([0.981707, 0.949464, 0.295906], **dd)),
    ],
    8: [
        Shell(0, torch.tensor([130.709, 23.8089, 6.44361], **dd), torch.tensor([15.0727, 14.5777, 4.54323], **dd)),
        Shell(0, torch.tensor([5.03315, 1.1696, 0.380389], **dd), torch.tensor([-0.848697, 1.1352, 0.856753], **dd)),
        Shell(1, torch.tensor([5.03315, 1.1696, 0.380389], **dd), torch.tensor([3.42907, 2.15629, 0.341592], **dd)),
    ],
}

water = Structure(
    numbers=torch.tensor([8, 1, 1]),
    positions=torch.tensor([[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]], **dd),
)

basis = Basis(shells)
s = overlap(basis, water)  # (nao, nao)
dipole = int1e("j", basis, water)  # (3, nao, nao), about each ket's atom
```

Gradients with respect to positions, exponents (`basis.alphas`) and coefficients (`basis.coeffs`) are available through autograd and are computed with *libcint*'s derivative integrals.
The exponents and coefficients are shared by all atoms of an element.

Periodic structures (with a `lattice`) are not supported yet and raise an error.

### Batches and `torch.func`

The integrals are pure functions of the structure, exponents and coefficients and compose with `torch.vmap`, `torch.func.grad` and `torch.func.jacrev`, to any derivative order (forward-mode AD is not supported).
A batched `Structure` (one leading batch dimension) is evaluated structure by structure, and `torch.vmap` maps over a `Structure` directly:

- **Conformers** repeat the atomic numbers, e.g. a multi-frame XYZ file read with `tad_mctc.io.read`.
- **Different structures** are zero-padded to a common number of atoms (atomic number 0 marks padding), e.g. with `tad_mctc.io.structure.pack_structures`.
  The integrals of each structure are zero-padded to the largest number of AOs in the batch, with the structure's AOs first.
  Padding atoms get zero gradients.

The `torch.func` transforms differentiate with respect to tensors, not structures (the atomic numbers are integers).
Replace the positions inside the differentiated function instead, `structure.replace(positions=positions)`.

```python
from torch.func import jacrev, vmap
from tad_mctc.io import read
from tad_mctc.io.structure import pack_structures

conformers = read.read("conformers.xyz", dtype=torch.double)  # positions: (nconf, nat, 3)
s = overlap(basis, conformers)  # (nconf, nao, nao)

molecules = [read.read(f, dtype=torch.double) for f in ("water.xyz", "methane.xyz")]
batch = pack_structures(molecules)
s = overlap(basis, batch)  # (2, nao_max, nao_max)

def dipole(structure, positions):
    return int1e("j", basis, structure.replace(positions=positions))

# Jacobians w.r.t. the positions, (2, 3, nao_max, nao_max, nat_max, 3)
jac = vmap(jacrev(dipole, argnums=1))(batch, batch.positions)
```

Since *libcint* is C code, a batch is evaluated one structure after another.
Combining very different structures in one batch wastes memory on padding.


## Installation

### pip <a href="https://pypi.org/project/tad-libcint/"><img src="https://img.shields.io/pypi/v/tad-libcint" alt="PyPI Version"></a> <a href="https://pypi.org/project/tad-libcint/"><img src="https://img.shields.io/pypi/dm/tad-libcint?color=orange" alt="PyPI Downloads"></a>

The `tad-libcint` package can be obtained from pip.

```console
pip install tad-libcint
```

### Build from source

You can also build the Python wheels from source.

Get the repository (including libcint submodule).

```console
git clone --recursive git@github.com:tad-mctc/tad-libcint.git
cd tad-libcint
```

The [libcint fork](https://github.com/tad-mctc/libcint/tree/dxtb) contains additional integrals required for `dxtb` and removes unneccessary ones.
Therefore, downloading from our fork is recommended.

If you already have the repository, you can update the submodule with

```console
git submodule update --remote
```

The build requires CMake (≥ 3.18) and a C compiler (with OpenMP for parallel integral evaluation).
Extra CMake options are passed through the `CMAKE_ARGS` environment variable, e.g., `CMAKE_ARGS="-DBUILD_MARCH_NATIVE=ON" pip install .`.

The compiler is taken from `CC`. Besides GCC, the Intel oneAPI compiler (`icx`) is supported, e.g., after `module load intel`.
The libraries then find Intel's runtime libraries in the compiler installation, so the compiler environment is not needed at runtime.

All shared libraries carry the suffix `_tad` (`libcint_tad.so`, `libcgto_tad.so`), so that they do not clash with PySCF's libraries of the same name when both packages are used in one process.

#### CIBuildWheel

Building with `cibuildwheel` is as simple as running

```console
cibuildwheel --platform linux
```

For a specific version, use

```console
CIBW_BUILD="cp312-manylinux_x86_64" CIBW_MANYLINUX_X86_64_IMAGE="manylinux2014" cibuildwheel --platform linux
```

#### Conda

1. Create an environment (`conda`) for building the wheels. The example here uses Python 3.10, but we support 3.10-3.14.

```console
mamba create -n wheel-310 --yes python=3.10 auditwheel c-compiler cxx-compiler fortran-compiler cffi cmake git make meson ninja numpy patchelf pkgconfig pip python-build unzip wheel
mamba activate wheel-310
```

2. Build the wheel on your system.

```console
python -m build --wheel
```

3. Repair the wheels for cross-distribution packaging.

```console
auditwheel repair -w wheels --plat manylinux_2_17_x86_64 dist/*-cp310-cp310-linux_x86_64.whl
```

4. Or only install this project with `pip` in the environment.

```console
pip install .
```

## Compatibility <img src="https://img.shields.io/badge/OS-Linux_only-black.svg" alt="Linux only">

| PyTorch \ Python | 3.10               | 3.11               | 3.12               | 3.13               | 3.14               |
| ---------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| 2.6.0            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :white_check_mark: | :x:                |
| 2.7.1            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :white_check_mark: | :x:                |
| 2.8.0            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :white_check_mark: | :x:                |
| 2.9.1            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :white_check_mark: |
| 2.10.0           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :white_check_mark: |
| 2.11.0           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :white_check_mark: |
| 2.12.1           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :white_check_mark: |
| 2.13.0           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :white_check_mark: |
| 2.14.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

:white_check_mark: tested in CI &nbsp;&nbsp; :heavy_check_mark: supported, but not tested in CI (should still work) &nbsp;&nbsp; :x: not supported

Note that only the latest bug fix version is listed, but all preceding bug fix minor versions are supported.
For example, although only version 2.7.1 is listed, version 2.7.0 is also supported.


## Development

For development, additionally install the following tools in your environment.

```console
mamba install black covdefaults coverage mypy pre-commit pylint tox
```

With pip, add the option `-e` for installing in development mode, and add `[dev]` for the development dependencies

```console
pip install -e .[dev]
```

The pre-commit hooks are initialized by running the following command in the root of the repository.

```console
pre-commit install
```

For testing all Python environments, simply run `tox`.

```console
tox
```

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional *posargs*.

```console
tox -- test
```
