PyTorch-based Libcint Interface
===============================

.. image:: https://img.shields.io/badge/python-%3E=3.8-blue.svg
    :target: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg
    :alt: Python Versions

.. image:: https://img.shields.io/github/v/release/tad-mctc/tad-libcint
    :target: https://github.com/tad-mctc/tad-libcint/releases/latest
    :alt: Release

.. image:: https://img.shields.io/pypi/v/tad-libcint
    :target: https://pypi.org/project/tad-libcint/
    :alt: PyPI

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: http://www.apache.org/licenses/LICENSE-2.0
    :alt: Apache-2.0

.. image:: https://github.com/tad-mctc/tad-libcint/actions/workflows/python.yaml/badge.svg
    :target: https://github.com/tad-mctc/tad-libcint/actions/workflows/python.yaml
    :alt: CI

.. image:: https://readthedocs.org/projects/tad-libcint/badge/?version=latest
    :target: https://tad-mctc.readthedocs.io
    :alt: Documentation Status

.. image:: https://codecov.io/gh/tad-mctc/tad-libcint/branch/main/graph/badge.svg?token=9faLOhisRx
    :target: https://codecov.io/gh/tad-mctc/tad-libcint
    :alt: Coverage

.. image:: https://results.pre-commit.ci/badge/github/tad-mctc/tad-libcint/main.svg
    :target: https://results.pre-commit.ci/latest/github/tad-mctc/tad-libcint/main
    :alt: pre-commit.ci status


This package contains a PyTorch-based interface to the `libcint <https://github.com/sunqm/libcint>`__ high-performance integral library.
The interface facilitates automatic differentiation with custom backward functions that call *libcint*'s C backend for derivative calculation.

It is based on `PySCF's lib directory <https://github.com/pyscf/pyscf/tree/master/pyscf/lib>`__. Modifications include

- removal of unnecessary code
- minor adaptation of build instructions (CMakeLists.txt)

The project was heavily inspired by `diffqc/dqc <https://github.com/diffqc/dqc>`__ and `diffqc/dqclibs <https://github.com/diffqc/dqclibs>`__.

This interface was mainly written for the calculation of integrals within `grimme-lab/dxtb <https://github.com/grimme-lab/dxtb>`__.


.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   modules/index
