Installation
------------

pip
~~~

The `tad-libcint` package can be obtained from pip.

.. code::

    pip install tad-libcint


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

2. Create an environment for building the wheels.

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.
The example here uses Python 3.10, but we support 3.8-3.11.

.. code::

    mamba env create -f env-310.yaml
    mamba activate wheel-310

3. Build the wheel.

.. code::

    python -m build --wheel

4. Repair the wheels for cross-distribution packaging.

.. code::

    auditwheel repair -w wheels --plat manylinux_2_12_x86_64 dist/*-cp310-cp310-linux_x86_64.whl

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
