# dxtb's libcint interface

## Content

This package contains an interface to and extended version of `libcint` for the calculation of integrals within `dxtb`.

It is based on [PySCF's lib directory](https://github.com/pyscf/pyscf/tree/master/pyscf/lib). Modifications include

- removal of unnecessary code
- minor adaptation of build instructions (CMakeLists.txt)

## Installation

### pip

The `tad-libcint` package can be obtained from pip.

```console
pip install tad-libcint
```

### Build from source

You can also build the Python wheels from source.

1. Get the repository (including libcint submodule)

```console
git clone --recursive git@github.com:grimme-lab/tad-libcint.git
cd tad-libcint
```

The [libcint fork](https://github.com/marvinfriede/libcint/tree/dxtb) contains some additional integrals required for `dxtb`. Therefore, downloading from our fork is recommended.

If you already have the repository, you can update the submodule with

```console
git submodule update --remote
```

2. Create an environment (`conda`) for building the wheels. The example here uses Python 3.10, but we support 3.8-3.11.

```console
mamba create -n wheel-310 --yes python=3.10 auditwheel c-compiler cmake fortran-compiler numpy python pip python-build pkgconfig patchelf unzip wheel
```

3. Build the wheel on your system

```console
python -m build --wheel
```

4. Repair the wheels for cross-distribution packaging.

```console
auditwheel repair -w wheel --plat manylinux_2_12_x86_64 dist/tad-libcint-*-cp310-cp310-linux_x86_64.whl
```
