# C-Interfaces for dxtb

## Content

This package contains an interface to `libcint` for the calculation of integrals within `dxtb`.

It is based on [PySCF's lib directory](https://github.com/pyscf/pyscf/tree/master/pyscf/lib), where unnecessary code was removed.

## Building the library

1. Get the repository

```console
git clone git@github.com:grimme-lab/dxtblibs.git
cd dxtblibs
```

2. Obtain `libcint`. For `dxtb`, some additional integrals are required. Therefore, downloading from our fork is recommended.

```console
pushd dxtblibs/libs
git clone git@github.com:marvinfriede/libcint.git
popd
```

3. Create an environment (`conda`) for building the wheels. The example here uses Python 3.10, but we support 3.8-3.11.

```console
mamba create -n wheel-310 --yes python=3.10 auditwheel c-compiler cmake fortran-compiler numpy python pip python-build pkgconfig patchelf unzip wheel
```

4. Build the wheel on your system

```console
python -m build --wheel
```

5. Repair the wheels for cross-distribution packaging.

```console
auditwheel repair -w wheel --plat manylinux_2_12_x86_64 dist/dxtblibs-*-cp310-cp310-linux_x86_64.whl
```
