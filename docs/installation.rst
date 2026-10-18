Installation
------------

pip
~~~

.. image:: https://img.shields.io/pypi/v/tad-libcint
    :target: https://pypi.org/project/tad-libcint/
    :alt: PyPI

The `tad-libcint` package can be obtained from pip.

.. code::

    pip install tad-libcint

Wheels are available for Linux (x86_64), macOS (Apple silicon) and Windows (x86_64), the platforms of PyTorch's wheels.
One wheel serves all supported Python versions.


Build from source
~~~~~~~~~~~~~~~~~

This project is hosted on GitHub at `tad-mctc/tad-libcint <https://github.com/tad-mctc/tad-libcint>`__.

.. code::

    git clone --recursive https://github.com/tad-mctc/tad-libcint
    cd tad-libcint


1. Get the repository (including libcint submodule).

.. code::

    git clone --recursive https://github.com/tad-mctc/tad-libcint
    cd tad-libcint

The `libcint fork <https://github.com/tad-mctc/libcint/tree/dxtb>`__ contains some additional integrals required for `dxtb <https://github.com/grimme-lab/dxtb>`__. Therefore, downloading from our fork is recommended.

If you already have the repository, you can update the submodule with

.. code::

    git submodule update --remote

The build requires CMake (≥ 3.18) and a C compiler (with OpenMP for parallel integral evaluation).
On macOS, Apple's Clang (Xcode Command Line Tools) builds with OpenMP if Homebrew's libomp is installed (``brew install libomp``). The libraries then use PyTorch's OpenMP runtime, since a second one in the same process aborts, so ``torch.set_num_threads`` also sets their number of threads. Editable installs find it only with PyTorch in the build environment (``pip install --no-build-isolation -e .``) and otherwise build without OpenMP.
On Windows, *libcint* needs the GCC of `MinGW-w64 <https://www.msys2.org/>`__, since MSVC lacks C99 complex numbers, e.g., the MSYS2 packages ``mingw-w64-ucrt-x86_64-gcc``, ``mingw-w64-ucrt-x86_64-libgomp`` (OpenMP) and ``mingw-w64-ucrt-x86_64-make`` on the ``PATH``.
The steps below use conda and apply to Linux.

2. Create an environment for building the wheels.

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.
The example here uses Python 3.10, but we support 3.10-3.14.

.. code::

    mamba create -n wheel-310 --yes python=3.10 auditwheel c-compiler cxx-compiler fortran-compiler cffi cmake git make meson ninja numpy patchelf pkgconfig pip python-build unzip wheel
    mamba activate wheel-310

3. Build the wheel.

.. code::

    python -m build --wheel

4. Repair the wheels for cross-distribution packaging.

.. code::

    auditwheel repair -w wheels --plat manylinux_2_17_x86_64 dist/*-py3-none-linux_x86_64.whl

5. Or only install this project with ``pip`` in the environment.

.. code::

    pip install .


Development
-----------

For development, additionally install the following tools in your environment.

.. code::

    mamba install black covdefaults coverage mypy pre-commit pylint tox

With pip, add the option ``-e`` for installing in development mode, and add ``[dev]`` for the development dependencies

.. code::

    pip install -e .[dev]

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code::

    pre-commit install

For testing all Python environments, simply run `tox`.

.. code::

    tox

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional *posargs*.

.. code::

    tox -- test
