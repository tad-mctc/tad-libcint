# PyTorch-based Libcint Interface

<table>
  <tr>
    <td>Compatibility:</td>
    <td>
      <img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg" alt="Python Versions"/>
      <img src="https://img.shields.io/badge/PyTorch-%3E=1.11.0-blue.svg" alt="PyTorch Versions"/>
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
      <a href="https://github.com/tad-mctc/tad-libcint/actions/workflows/python.yaml">
        <img src="https://github.com/tad-mctc/tad-libcint/actions/workflows/python.yaml/badge.svg" alt="Test Status"/>
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
        <img src="https://codecov.io/gh/tad-mctc/tad-libcint/branch/main/graph/badge.svg?token=OGJJnZ6t4G" alt="Coverage"/>
      </a>
    </td>
  </tr>
</table>

<br>

## Content

This package contains a PyTorch-based interface to the *[libcint](https://github.com/sunqm/libcint)* high-performance integral library.
The interface facilitates automatic differentiation with custom backward functions that call *libcint*'s C backend for derivative calculation.

It is based on [PySCF's lib directory](https://github.com/pyscf/pyscf/tree/master/pyscf/lib). Modifications include

- removal of unnecessary code
- minor adaptation of build instructions (CMakeLists.txt)

The project was heavily inspired by *[diffqc/dqc](https://github.com/diffqc/dqc)* and *[diffqc/dqclibs](https://github.com/diffqc/dqclibs)*.

This interface was mainly written for the calculation of integrals within *[grimme-lab/dxtb](https://github.com/grimme-lab/dxtb)*.

## Installation

### pip

The `tad-libcint` package can be obtained from pip.

```console
pip install tad-libcint
```

### Build from source

You can also build the Python wheels from source.

1. Get the repository (including libcint submodule).

```console
git clone --recursive git@github.com:grimme-lab/tad-libcint.git
cd tad-libcint
```

The [libcint fork](https://github.com/tad-mctc/libcint/tree/dxtb) contains some additional integrals required for `dxtb`. Therefore, downloading from our fork is recommended.

If you already have the repository, you can update the submodule with

```console
git submodule update --remote
```

2. Create an environment (`conda`) for building the wheels. The example here uses Python 3.10, but we support 3.8-3.11.

```console
mamba create -n wheel-310 --yes python=3.10 auditwheel c-compiler cxx-compiler fortran-compiler cffi cmake git make meson ninja numpy patchelf pkgconfig pip python-build unzip wheel
mamba activate wheel-310
```

3. Build the wheel on your system.

```console
python -m build --wheel
```

4. Repair the wheels for cross-distribution packaging.

```console
auditwheel repair -w wheels --plat manylinux_2_12_x86_64 dist/*-cp310-cp310-linux_x86_64.whl
```

5. Or only install this project with `pip` in the environment.

```console
pip install .
```


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

```
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
