# This file is part of tad-libcint, modified from diffqc/dqc.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the Apache License, Version 2.0 by diffqc/dqc.
# Modifications made by Grimme Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Integrals
=========

Differentiable two-centre one-electron integrals.

The integrals are pure functions of the structure (atomic numbers and
positions) and the basis parameters (exponents and coefficients). They support
first and higher derivatives with respect to positions and parameters through
libcint's derivative integrals, and compose with :func:`torch.vmap` and the
other :mod:`torch.func` transforms. Since libcint is C code, a batch of
structures is evaluated once per structure. A batch of cotangents of one
structure, as in :func:`torch.func.jacrev`, shares one evaluation per chunk
of cotangents. In large structures, only shell pairs of nearby atoms are
evaluated (screened with :func:`tad_mctc.neighbor.pairs_within_radii`): the
others fall below libcint's own threshold for negligible products of
Gaussians.

Example
-------
>>> import torch
>>> from tad_mctc.io.structure import Structure
>>> from tad_libcint import Basis, Shell, overlap
>>> dd = {"dtype": torch.double}
>>> one = torch.tensor([1.0], **dd)
>>> s = Shell(angmom=0, alphas=one, coeffs=one).normalized()
>>> basis = Basis(shells={1: [s], 8: [s]})
>>> numbers = torch.tensor([1, 1])
>>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd)
>>> h2 = Structure(numbers=numbers, positions=positions)
>>> overlap(basis, h2)
tensor([[1.0000, 0.3753],
        [0.3753, 1.0000]], dtype=torch.float64)

Batches are evaluated element by element, either through leading batch
dimensions of the structure or explicitly with :func:`torch.vmap`. A conformer
set repeats the atomic numbers:

>>> conformers = Structure(
...     numbers=numbers.expand(2, -1),
...     positions=torch.stack([positions, 1.1 * positions]),
... )
>>> overlap(basis, conformers).shape
torch.Size([2, 2, 2])
>>> torch.vmap(lambda s: overlap(basis, s))(conformers).shape
torch.Size([2, 2, 2])

Different structures are zero-padded to a common number of atoms (atomic
number 0 marks padding), e.g. with :func:`tad_mctc.io.structure.pack_structures`.
The integrals of each structure are then zero-padded to the largest number of
AOs in the batch, with the structure's AOs first:

>>> from tad_mctc.io.structure import pack_structures
>>> water = Structure(
...     numbers=torch.tensor([1, 8, 1]),
...     positions=torch.tensor(
...         [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 0.0, -1.4]], **dd
...     ),
... )
>>> s = overlap(basis, pack_structures([h2, water]))
>>> s.shape
torch.Size([2, 3, 3])
>>> s[0]
tensor([[1.0000, 0.3753, 0.0000],
        [0.3753, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]], dtype=torch.float64)

The :mod:`torch.func` transforms differentiate w.r.t. tensors, not w.r.t. a
structure (its atomic numbers are integers). Replace the positions inside the
differentiated function instead:

>>> from torch.func import jacfwd, jacrev
>>> def func(positions):
...     return overlap(basis, h2.replace(positions=positions))
>>> jacrev(func)(h2.positions).shape
torch.Size([2, 2, 2, 3])

For the Jacobian of an integral matrix, prefer :func:`torch.func.jacfwd`. It
batches one tangent per input (e.g. ``3 * nat`` for the positions), whereas
:func:`torch.func.jacrev` batches one cotangent per matrix element, which
costs time and, through its one-hot cotangents, memory of order ``nao**4``:

>>> torch.allclose(jacfwd(func)(h2.positions), jacrev(func)(h2.positions))
True

Forward mode is evaluated through reverse mode (there are no native
Jacobian-vector products). For the Hessian of a scalar function,
``jacrev(grad(f))`` is therefore somewhat faster than
:func:`torch.func.hessian` (``jacfwd(jacrev(f))``), which works as well.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor import pairs_within_radii
from torch import Tensor

from . import library
from .basis import Basis, Layout
from .names import (
    DERIVATIVE_OPERATORS,
    KET_CENTRED_OPERATORS,
    QUADRUPOLE_OPERATORS,
    IntegralName,
    Side,
)

__all__ = ["int1e", "overlap"]


class Inputs(NamedTuple):
    """Tensor inputs of an integral of one structure"""

    numbers: Tensor
    """Atomic numbers, shape ``(nat,)`` (not differentiable)."""

    charges: Tensor
    """Nuclear charges, shape ``(nat,)`` (not differentiable)."""

    positions: Tensor
    """Cartesian coordinates, shape ``(nat, 3)``."""

    alphas: Tensor
    """Exponents of the basis, shape ``(nparam,)``."""

    coeffs: Tensor
    """Coefficients of the basis, shape ``(nparam,)``."""


def int1e(
    name: str,
    basis: Basis,
    structure: Structure,
    *,
    nuclear_charges: Tensor | None = None,
    alphas: Tensor | None = None,
    coeffs: Tensor | None = None,
    packed: bool = True,
) -> Tensor:
    """
    Two-centre one-electron integral over all AOs of a structure.

    Parameters
    ----------
    name : str
        libcint's short name of the integral, e.g. ``"ovlp"`` (overlap),
        ``"r0"``/``"r0r0"`` (multipoles about the origin), ``"j"``/``"jj"``
        (multipoles about the ket's centre) or ``"nuc"`` (nuclear attraction).
    basis : Basis
        The basis set.
    structure : Structure
        Atomic numbers, shape ``([batch,] nat)``, where zeros mark padding,
        and Cartesian coordinates in Bohr, shape ``([batch,] nat, 3)``. The
        total charge, unpaired electrons and bonds do not enter the
        integrals.
    nuclear_charges : Tensor | None, optional
        Nuclear charges for ``"nuc"`` integrals, shape ``([batch,] nat)``.
        Non-integer charges are allowed, but not differentiable. Defaults to
        the atomic numbers.
    alphas : Tensor | None, optional
        Exponents of the basis, shape ``(nparam,)``. Defaults to
        ``basis.alphas``.
    coeffs : Tensor | None, optional
        Coefficients of the basis, shape ``(nparam,)``. Defaults to
        ``basis.coeffs``.
    packed : bool, optional
        Only return the 6 unique components of the quadrupoles ``"r0r0"``
        and ``"jj"`` instead of all 9: the lower triangle xx, yx, yy, zx,
        zy, zz as in tblite (but not traceless). The other components are
        never stored. Defaults to ``True``.

    Returns
    -------
    Tensor
        Integral of shape ``([batch,] *components, nao, nao)``, with
        libcint's components (e.g. 3 for ``"j"``, 9 or 6 when packed for
        ``"jj"``). In a batch of different structures, ``nao`` is the
        largest number of AOs and the integrals of smaller structures are
        zero-padded.

    Raises
    ------
    ValueError
        If the name is not understood, an input has the wrong shape or dtype,
        or the basis lacks an element.
    NotImplementedError
        If the structure is periodic (has a lattice, even one without
        periodic directions), or if the libcint build lacks the integral or
        one of the derivatives required for a gradient.
    """
    if structure.lattice is not None:
        raise NotImplementedError(
            "Integrals of periodic structures (with a lattice) are not "
            "supported yet."
        )

    iname = IntegralName.parse(name)
    if packed and iname.op in QUADRUPOLE_OPERATORS:
        iname = replace(iname, packed=True)
    numbers, positions = structure.numbers, structure.positions
    charges = numbers if nuclear_charges is None else nuclear_charges
    alphas = basis.alphas if alphas is None else alphas
    coeffs = basis.coeffs if coeffs is None else coeffs
    _check_inputs(basis, Inputs(numbers, charges, positions, alphas, coeffs))

    func = _StructureIntegral(iname, basis)

    def single(n: Tensor, c: Tensor, p: Tensor) -> Tensor:
        out = _PerStructure.apply(func, n, c, p, alphas, coeffs)
        assert isinstance(out, tuple)
        return out[0]

    if numbers.ndim == 1:
        return single(numbers, charges, positions)

    # a batch of structures, which may share the nuclear charges
    charges = charges.expand(numbers.shape)
    return torch.vmap(single)(numbers, charges, positions)


def overlap(
    basis: Basis,
    structure: Structure,
    *,
    alphas: Tensor | None = None,
    coeffs: Tensor | None = None,
) -> Tensor:
    """Overlap integral, see :func:`int1e`"""
    return int1e("ovlp", basis, structure, alphas=alphas, coeffs=coeffs)


def _check_inputs(basis: Basis, inputs: Inputs) -> None:
    # Only shapes and dtypes: under vmap, the values are not accessible here.
    # The atomic numbers and positions are checked by the `Structure`, but
    # stacking batched structures bypasses its check of the batch dimensions.
    numbers, charges, positions, alphas, coeffs = inputs
    if numbers.ndim > 2:
        raise ValueError(
            "A structure has at most one batch dimension, got atomic numbers "
            f"of shape {tuple(numbers.shape)}."
        )
    if charges.requires_grad:
        raise NotImplementedError(
            "Gradients w.r.t. nuclear charges are not supported."
        )
    if charges.shape not in (numbers.shape, numbers.shape[-1:]):
        raise ValueError(
            f"Expected nuclear charges of shape {tuple(numbers.shape)}, got "
            f"{tuple(charges.shape)}."
        )
    for label, x in (("alphas", alphas), ("coeffs", coeffs)):
        if x.shape != (basis.nparam,):
            raise ValueError(
                f"Expected {label} of shape ({basis.nparam},), got "
                f"{tuple(x.shape)}."
            )
    if not positions.dtype == alphas.dtype == coeffs.dtype:
        raise ValueError(
            "Positions, exponents and coefficients must share one dtype, got "
            f"{positions.dtype}, {alphas.dtype} and {coeffs.dtype}."
        )


class _Block(NamedTuple):
    """
    Block of an integral matrix: its bra (rows) and ket (columns) shells.

    With `pairs` (bra and ket shell, sorted by bra shell), only these shell
    pairs of the block are evaluated and all others are zero (screening, see
    `_screened_pairs`).
    """

    bra: range
    ket: range
    pairs: np.ndarray | None

    @classmethod
    def full(cls, layout: Layout, pairs: np.ndarray | None) -> _Block:
        """The whole matrix"""
        return cls(range(layout.nsh), range(layout.nsh), pairs)

    @property
    def shls_slice(self) -> tuple[int, int, int, int]:
        """The block in libcint's format"""
        return (self.bra.start, self.bra.stop, self.ket.start, self.ket.stop)

    @property
    def is_diagonal(self) -> bool:
        """Whether bra and ket have the same shells"""
        return self.bra == self.ket

    def transposed(self) -> _Block:
        """The block with bra and ket exchanged"""
        pairs = None if self.pairs is None else _sorted(self.pairs[:, ::-1])
        return _Block(self.ket, self.bra, pairs)

    def rows(self, bra: range) -> _Block | None:
        """
        The rows of the bra shells `bra`. With `pairs`, only the columns of
        their shell pairs, and ``None`` without any.
        """
        if self.pairs is None:
            return _Block(bra, self.ket, None)

        first, last = np.searchsorted(self.pairs[:, 0], [bra.start, bra.stop])
        pairs = self.pairs[first:last]
        if len(pairs) == 0:
            return None
        ket = range(int(pairs[:, 1].min()), int(pairs[:, 1].max()) + 1)
        return _Block(bra, ket, pairs)

    def uncontracted(self, layout: Layout) -> _Block:
        """The block of :attr:`Basis.uncontracted` (layout of this basis)"""
        bra = layout.uncontracted_shells(self.bra)
        ket = layout.uncontracted_shells(self.ket)
        if self.pairs is None:
            return _Block(bra, ket, None)
        return _Block(bra, ket, _sorted(layout.uncontracted_pairs(self.pairs)))


def _sorted(pairs: np.ndarray) -> np.ndarray:
    """Shell pairs sorted by bra, then by ket shell"""
    # one key per pair sorts much faster than two keys
    nket = int(pairs[:, 1].max(initial=0)) + 1
    keys = pairs[:, 0].astype(np.int64) * nket + pairs[:, 1]
    return np.stack(np.divmod(np.sort(keys), nket), axis=-1)


_SCREENING = 60.0
"""
Negative logarithm of the smallest Gaussian product factor of an evaluated
shell pair (libcint's own threshold for pairs of primitives).
"""

_SCREENING_MIN_PAIRS = 2**15
"""
Number of shell pairs from which on screening pays off: the neighbour search
costs about a millisecond, libcint about 0.1 µs per (even negligible) pair.
"""


def _screened_pairs(layout: Layout, inputs: Inputs) -> np.ndarray:
    """
    Shell pairs of atoms within their screening distance, sorted.

    The product of two Gaussians with exponents a and b at distance R carries
    the factor exp(-ab/(a+b) R²), which decays slowest for the smallest
    exponents of the two atoms. A pair of atoms is screened beyond the
    distance where this factor falls below exp(-_SCREENING), i.e. beyond
    their radii sqrt(_SCREENING / a) added in quadrature.
    """
    # smallest exponent of each atom (infinite for padding)
    alphas = inputs.alphas.detach().cpu().numpy()[layout.prim_param]
    prim_atom = np.repeat(layout.shell_atom, layout.shell_nprim)
    smallest = np.full(layout.nat, np.inf)
    np.minimum.at(smallest, prim_atom, alphas)

    # on the CPU, like libcint
    structure = Structure(
        numbers=inputs.numbers.cpu(), positions=inputs.positions.detach().cpu()
    )
    radii = torch.from_numpy(np.sqrt(_SCREENING / smallest))
    i, j = pairs_within_radii(structure, radii)
    i, j = i.numpy(), j.numpy()

    # each neighbour pair once, in both directions, and every atom with itself
    atoms = np.arange(layout.nat)
    bra, ket = np.concatenate([i, j, atoms]), np.concatenate([j, i, atoms])
    return _sorted(layout.shell_pairs(np.stack([bra, ket], axis=-1)))


def _integral(
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
    values: bool,
) -> Tensor:
    out = _Int2c.apply(*inputs, basis, name, block, values)
    assert isinstance(out, Tensor)
    return out


def _evaluate(
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
    values: bool = True,
) -> Tensor:
    """
    Block of the integral `name`, or the transpose of ``name.transposed()``.

    The integral of a bra/ket symmetric operator equals the transpose of the
    integral with bra and ket operators exchanged, so libcint only needs to
    provide one of the two (e.g. ``"ipovlp"`` for ``"ovlpip"``). Without
    `values`, only the autograd graph is built (see `_Int2c`).
    """
    symbol = name.symbol(basis.spherical)
    if library.has_symbol(symbol):
        return _integral(name, basis, inputs, block, values)

    source = name.transposed()
    missing = f"'{symbol}'"
    if name.is_symmetric:
        if library.has_symbol(source.symbol(basis.spherical)):
            integral = _integral(
                source, basis, inputs, block.transposed(), values
            )
            return _transpose(integral, source)
        missing += f" and its transpose '{source.symbol(basis.spherical)}'"

    raise NotImplementedError(
        f"The integral '{name}' is not available in this libcint build "
        f"(missing {missing})."
    )


# Structures
#
# Inside `torch.vmap`, the atomic numbers may differ between batch elements
# (padded batches of different structures). Everything that depends on their
# values (the layout) must therefore happen where the batch is resolved into
# single structures: in the vmap rule of `_PerStructure` below. Its backward
# pass rebuilds the autograd graph of the structure's integral (without
# evaluating it) and differentiates it, again through `_PerStructure`, so
# that derivatives of any order resolve the batch as well. If only the
# vectors of such a vector-Jacobian product are batched (e.g. the cotangents
# of a Jacobian), the structure is evaluated once per chunk of them.
# Forward mode (Jacobian-vector products) reuses the vector-Jacobian
# products, see `_PerStructure.jvp`. Within one structure, `_Int2c` provides
# (blocks of) the integrals and their derivatives.


_CHUNK_ELEMENTS = 2**22
"""Number of vector elements in one chunk of a batched product."""


class _StructureIntegral:
    """The integral `name` of one structure as a function of its inputs"""

    def __init__(self, name: IntegralName, basis: Basis) -> None:
        self.name = name
        self.basis = basis
        self._screening: tuple[list[Tensor], np.ndarray] | None = None

    def __call__(
        self, *tensors: Tensor, values: bool = True
    ) -> tuple[Tensor, ...]:
        inputs = Inputs(*tensors)
        block = self.block(inputs)
        return (_evaluate(self.name, self.basis, inputs, block, values),)

    def block(self, inputs: Inputs) -> _Block:
        """The whole matrix, screened for large structures"""
        layout = self.basis.layout(inputs.numbers)
        return _Block.full(layout, self._shell_pairs(layout, inputs))

    def _shell_pairs(self, layout: Layout, inputs: Inputs) -> np.ndarray | None:
        """
        The screened shell pairs (`_screened_pairs`), or ``None`` (all pairs)
        for small structures. They are reused for the structure of the last
        call, as the backward pass evaluates the integral (its graph) of the
        forward pass again.
        """
        if layout.nsh**2 < _SCREENING_MIN_PAIRS:
            return None

        structure = [inputs.numbers, inputs.positions, inputs.alphas]
        if self._screening is not None:
            last, pairs = self._screening
            if all(
                a.shape == b.shape and torch.equal(a, b)
                for a, b in zip(last, structure)
            ):
                return pairs

        pairs = _screened_pairs(layout, inputs)
        self._screening = ([t.detach().clone() for t in structure], pairs)
        return pairs


class _SelectedOutputs:
    """The outputs flagged in `selected` of a structure function `func`"""

    def __init__(
        self,
        func: Callable[..., tuple[Tensor, ...]],
        selected: tuple[bool, ...],
    ) -> None:
        self.func = func
        self.selected = selected

    def __call__(
        self, numbers: Tensor, *tensors: Tensor, values: bool = True
    ) -> tuple[Tensor, ...]:
        outputs = self.func(numbers, *tensors, values=values)
        return tuple(out for out, s in zip(outputs, self.selected) if s)


class _VectorJacobianProduct:
    """
    Vector-Jacobian product of a structure function `func` w.r.t. the inputs
    flagged in `needs`, as a structure function of the inputs and the vectors.

    With `vector_dims`, the vectors carry a batch dimension at these positions
    (``None``: shared by the batch), and the products of the whole batch are
    stacked along a leading dimension. As the product is linear in the
    vectors, `func` and its derivatives are only evaluated once per chunk of
    the batch (the contractions with the vectors dominate, not libcint).
    """

    def __init__(
        self,
        func: Callable[..., tuple[Tensor, ...]],
        needs: tuple[bool, ...],
        vector_dims: tuple[int | None, ...] | None = None,
    ) -> None:
        self.func = func
        self.needs = needs
        self.vector_dims = vector_dims

    def __call__(
        self, numbers: Tensor, *args: Tensor, values: bool = True
    ) -> tuple[Tensor, ...]:
        del values  # the products are the values of their own graph
        tensors, vectors = args[: len(self.needs)], args[len(self.needs) :]

        # Called from `_PerStructure.forward` (gradients disabled), differentiate
        # w.r.t. fresh leaves: the inputs, in particular the vectors, may
        # depend on each other in the caller's graph, which is accounted for
        # outside. Nested inside another product, keep its leaves connected
        # and the graph of the product, which the outer one differentiates.
        nested = torch.is_grad_enabled()
        with torch.enable_grad():
            leaves = [
                (
                    t
                    if nested and (t.requires_grad or not need)
                    else t.detach().requires_grad_(need)
                )
                for t, need in zip(tensors, self.needs)
            ]
            # the product only needs the graph of `func`, not its values
            outputs = self.func(numbers, *leaves, values=False)
            inputs = [t for t, need in zip(leaves, self.needs) if need]

            # skip outputs that do not depend on the inputs
            keep = [o.requires_grad for o in outputs]

            def product(*vectors: Tensor) -> tuple[Tensor, ...]:
                # Vectors of a padded batch: drop the padding of this
                # structure.
                pairs = [
                    (o, v[tuple(slice(0, n) for n in o.shape)])
                    for o, v, k in zip(outputs, vectors, keep)
                    if k
                ]
                if not pairs:
                    return tuple(torch.zeros_like(t) for t in tensors)

                grads = iter(
                    torch.autograd.grad(
                        [o for o, _ in pairs],
                        inputs,
                        [v for _, v in pairs],
                        retain_graph=True,  # chunks of a batch share it
                        create_graph=nested,
                        allow_unused=True,
                        materialize_grads=True,
                    )
                )
                return tuple(
                    next(grads) if need else torch.zeros_like(t)
                    for t, need in zip(tensors, self.needs)
                )

            if self.vector_dims is None:
                return product(*vectors)

            # One backward pass through the graph for a chunk of vectors. The
            # chunks bound the memory of the intermediates, which hold one
            # integral per vector.
            batched = [
                (v, dim)
                for v, dim in zip(vectors, self.vector_dims)
                if dim is not None
            ]
            elements = sum(
                math.prod(v.shape[:dim] + v.shape[dim + 1 :])
                for v, dim in batched
            )
            chunk_size = max(1, _CHUNK_ELEMENTS // max(1, elements))

            # torch's chunked vmap fails for an empty batch
            v, dim = batched[0]
            return torch.vmap(
                product,
                in_dims=self.vector_dims,
                chunk_size=chunk_size if v.shape[dim] > 0 else None,
            )(*vectors)

    def batched(
        self, vector_dims: tuple[int | None, ...]
    ) -> _VectorJacobianProduct:
        """The product for vectors with batch dimensions `vector_dims`"""
        return _VectorJacobianProduct(self.func, self.needs, vector_dims)


class _JacobianVectorProduct:
    """
    Jacobian-vector product of the integral of a structure w.r.t. the inputs
    flagged in `needs`, as a structure function of the inputs and their
    tangents (the vectors).

    Unlike forward mode through two vector-Jacobian products, which holds
    all derivative integrals in the graph of the inner one, the product is
    accumulated from row blocks of the derivative integrals (see
    `_tangent`). With `vector_dims`, as for `_VectorJacobianProduct`, the
    products of a batch of vectors share the derivative integrals.
    """

    def __init__(
        self,
        func: _StructureIntegral,
        needs: tuple[bool, ...],
        vector_dims: tuple[int | None, ...] | None = None,
    ) -> None:
        self.func = func
        self.needs = needs
        self.vector_dims = vector_dims

    def __call__(
        self, numbers: Tensor, *args: Tensor, values: bool = True
    ) -> tuple[Tensor, ...]:
        del values  # the products need the values of the derivatives
        tensors, vectors = args[: len(self.needs)], args[len(self.needs) :]
        inputs = Inputs(numbers, *tensors)

        # vectors with a leading batch dimension, and only the needed ones
        dims = self.vector_dims or (None,) * len(vectors)
        batch = [
            v.unsqueeze(0) if dim is None else v.movedim(dim, 0)
            for v, dim in zip(vectors, dims)
        ]
        _, positions, alphas, coeffs = (  # charges are not differentiable
            v if need else None for v, need in zip(batch, self.needs)
        )

        out = _tangent(
            self.func.name,
            self.func.basis,
            inputs,
            self.func.block(inputs),
            _Tangents(positions, alphas, coeffs),
            nbatch=max(len(v) for v in batch),
        )
        return (out if self.vector_dims else out[0],)

    def batched(
        self, vector_dims: tuple[int | None, ...]
    ) -> _JacobianVectorProduct:
        """The product for vectors with batch dimensions `vector_dims`"""
        return _JacobianVectorProduct(self.func, self.needs, vector_dims)


class _PerStructure(torch.autograd.Function):
    """
    Differentiable evaluation of `func(numbers, *tensors)` for one structure,
    batched over structures by looping and zero-padding the results.
    """

    # signatures are prescribed by torch.autograd.Function
    # pylint: disable=abstract-method,arguments-differ

    @staticmethod
    def forward(
        func: Callable[..., tuple[Tensor, ...]],
        numbers: Tensor,
        *tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        return tuple(out.detach() for out in func(numbers, *tensors))

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        func, *tensors = inputs
        ctx.func = func
        ctx.save_for_backward(*tensors)
        ctx.save_for_forward(*tensors)
        ctx.outputs = [(out.shape, out.dtype) for out in output]
        # Unused outputs and inputs without tangents get None instead of
        # zeros, which tells which derivatives are needed.
        ctx.set_materialize_grads(False)

    @staticmethod
    def jvp(ctx: Any, *tangents: Tensor | None) -> tuple[Tensor | None, ...]:
        numbers, *tensors = ctx.saved_tensors
        tangents = tangents[2:]
        if isinstance(ctx.func, _StructureIntegral):
            jvp_integral = _JacobianVectorProduct(
                ctx.func, tuple(dt is not None for dt in tangents)
            )
            out = _PerStructure.apply(
                jvp_integral,
                numbers,
                *tensors,
                *(
                    torch.zeros_like(t) if dt is None else dt
                    for t, dt in zip(tensors, tangents)
                ),
            )
            assert isinstance(out, tuple)
            return out

        # Other functions (products), forward mode through reverse mode: the
        # vector-Jacobian product J^T u is linear in u, and its
        # vector-Jacobian product w.r.t. u with the tangents t as vectors is
        # the Jacobian-vector product J t. Any u will do, e.g. zeros.
        us = [
            torch.zeros((), dtype=dtype, device=numbers.device).expand(shape)
            for shape, dtype in ctx.outputs
        ]
        vjp = _VectorJacobianProduct(
            ctx.func, tuple(dt is not None for dt in tangents)
        )
        differentiable = tuple(u.is_floating_point() for u in us)
        jvp = _VectorJacobianProduct(
            vjp, (False,) * len(tensors) + differentiable
        )

        out = _PerStructure.apply(
            jvp,
            numbers,
            *tensors,
            *us,
            *(
                torch.zeros_like(t) if dt is None else dt
                for t, dt in zip(tensors, tangents)
            ),
        )
        assert isinstance(out, tuple)
        return tuple(
            dout if diff else None
            for dout, diff in zip(out[len(tensors) :], differentiable)
        )

    @staticmethod
    def backward(
        ctx: Any, *vectors: Tensor | None
    ) -> tuple[Tensor | None, ...]:
        numbers, *tensors = ctx.saved_tensors
        needs = tuple(ctx.needs_input_grad[2:])
        if not any(needs):
            return (None,) * (2 + len(tensors))

        # only the outputs with vectors (None: unused) enter the product
        used = tuple(v is not None for v in vectors)
        func = ctx.func if all(used) else _SelectedOutputs(ctx.func, used)
        vjp = _VectorJacobianProduct(func, needs)
        given = (v for v in vectors if v is not None)
        grads = _PerStructure.apply(vjp, numbers, *tensors, *given)
        assert isinstance(grads, tuple)
        return (
            None,
            None,
            *(g if need else None for g, need in zip(grads, needs)),
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        func: Callable[..., tuple[Tensor, ...]],
        *tensors: Tensor,
    ) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
        # Only the vectors of a product are batched (e.g. the cotangents or
        # tangents of a Jacobian): one structure, evaluated once (per chunk)
        # for all vectors
        if (
            isinstance(func, (_VectorJacobianProduct, _JacobianVectorProduct))
            and func.vector_dims is None
        ):
            # dimensions of the numbers and tensors, then of the vectors
            nstructure = 1 + len(func.needs)
            if all(dim is None for dim in in_dims[1 : 1 + nstructure]):
                product = func.batched(in_dims[1 + nstructure :])
                out = _PerStructure.apply(product, *tensors)
                assert isinstance(out, tuple)
                return out, (0,) * len(out)

        # libcint has no batch dimension: evaluate each structure
        def element(i: int) -> list[Tensor]:
            return [
                t if dim is None else t.select(dim, i)
                for t, dim in zip(tensors, in_dims[1:])
            ]

        def structure(i: int) -> tuple[Tensor, ...]:
            out = _PerStructure.apply(func, *element(i))
            assert isinstance(out, tuple)
            return out

        # different structures: pad to the largest number of AOs
        outputs = [structure(i) for i in range(info.batch_size)]
        columns = [list(outs) for outs in zip(*outputs)]
        del outputs  # the columns hold the only references
        batched = tuple(_stack_padded(column) for column in columns)
        return batched, (0,) * len(batched)


def _stack_padded(tensors: list[Tensor]) -> Tensor:
    """
    Stack tensors zero-padded at the end of every dimension to a common
    shape. Each padded tensor replaces its original in `tensors`, which is
    emptied, so that at most one padded copy is held besides the stack.
    """
    shape = [max(sizes) for sizes in zip(*(t.shape for t in tensors))]
    for k, t in enumerate(tensors):
        pad = [
            p
            for n, size in zip(reversed(shape), reversed(t.shape))
            for p in (0, n - size)
        ]
        if any(pad):
            tensors[k] = F.pad(t, pad)
        del t
    stacked = torch.stack(tensors)
    tensors.clear()
    return stacked


class _Int2c(torch.autograd.Function):
    """
    Block of a two-centre integral with gradients through libcint's
    derivatives.

    Without `values`, the result is an unmaterialised placeholder with the
    integral's autograd graph: a vector-Jacobian product only differentiates
    the integral, which needs the derivative integrals but not its values.
    """

    # signatures are prescribed by torch.autograd.Function
    # pylint: disable=abstract-method,arguments-differ
    # pylint: disable=too-many-arguments,too-many-positional-arguments

    @staticmethod
    def forward(
        numbers: Tensor,
        charges: Tensor,
        positions: Tensor,
        alphas: Tensor,
        coeffs: Tensor,
        basis: Basis,
        name: IntegralName,
        block: _Block,
        values: bool,
    ) -> Tensor:
        layout = basis.layout(numbers)
        rows, cols = layout.aos(block.bra), layout.aos(block.ket)
        shape = (
            *name.comp_shape,
            rows.stop - rows.start,
            cols.stop - cols.start,
        )
        if not values:
            return positions.new_zeros(()).expand(shape)
        if 0 in shape:
            return positions.new_zeros(shape)

        out = library.int2c(
            name.symbol(basis.spherical),
            name.unpacked.comp_shape,
            name.is_hermitian and block.is_diagonal,
            shls_slice=block.shls_slice,
            pairs=block.pairs,
            components=name.components,
            atm=layout.atm(charges),
            bas=layout.bas,
            env=layout.env(charges, positions, alphas, coeffs),
            ao_loc=layout.ao_loc,
        ).reshape(shape)
        return torch.from_numpy(out).to(positions.device, positions.dtype)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        *tensors, basis, name, block, _ = inputs
        ctx.save_for_backward(*tensors)
        ctx.basis = basis
        ctx.name = name
        ctx.block = block

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        inputs = Inputs(*ctx.saved_tensors)
        need_pos, need_alphas, need_coeffs = ctx.needs_input_grad[2:5]
        args = (grad_out, ctx.name, ctx.basis, inputs, ctx.block)

        return (
            None,  # numbers
            None,  # charges
            _grad_positions(*args) if need_pos else None,
            _grad_alphas(*args) if need_alphas else None,
            _grad_coeffs(*args) if need_coeffs else None,
            None,  # basis
            None,  # name
            None,  # block
            None,  # values
        )

    @staticmethod
    def vmap(info: Any, in_dims: tuple[int | None, ...], *args: Any) -> Any:
        # Under vmap, torch skips this rule if no input is batched, as for the
        # derivative integrals in a backward pass with batched cotangents.
        # Batches of structures are resolved by `_PerStructure` beforehand.
        raise NotImplementedError(
            "The integrals of one structure cannot be batched."
        )


# Gradients
#
# Notation: `grad_out` is the incoming gradient of a block of the integral
# with shape (*components, nao_bra, nao_ket) and `g` the same with flattened
# components. The derivative integrals are the largest tensors of a backward
# pass. They are evaluated for row blocks of the block (see `_row_blocks`)
# with the shape (directions, *g[:, rows].shape), a single direction for
# derivatives without one, and contracted right away.


_BLOCK_ELEMENTS = 2**21
"""Number of elements of a derivative integral evaluated at once (memory)."""

_CONTRACT_ELEMENTS = 2**17
"""Number of elements in one chunk of a contraction's products (cache)."""


def _grad_positions(
    grad_out: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
) -> Tensor:
    if name.op == "nuc":
        raise NotImplementedError(
            "Gradients of 'nuc' integrals w.r.t. the positions are not "
            "supported: the operator depends on the nuclear positions."
        )

    layout = basis.layout(inputs.numbers)
    g = _flat_components(grad_out, name)

    def nabla(side: Side, rows: _Block) -> Tensor:
        d = _derivative("ip", side, name, basis, inputs, block=rows)
        return d.reshape(3, len(g), *d.shape[-2:])

    # In a diagonal block of a hermitian integral, the ket derivative is the
    # transpose of the bra derivative.
    symmetric = name.is_hermitian and block.is_diagonal

    # libcint's nabla acts on the electron coordinate, which is minus the
    # derivative with respect to the function's centre
    grad = torch.zeros_like(inputs.positions)
    for rows, cols, sub in _row_blocks(layout, block, 3 * len(g)):
        weights = _weights(g, rows, cols, symmetric=symmetric)
        bra = _contract(weights, nabla("bra", sub), "bra")
        bra_atoms = layout.ao_to_atom[layout.aos(sub.bra)]
        grad = grad.index_add(0, bra_atoms, -bra.mT)
        if symmetric:
            continue

        ket = _contract(weights, nabla("ket", sub), "ket")
        if not name.is_symmetric:
            ket = ket - _ket_operator_term(
                grad_out[..., rows, cols], name, basis, inputs, sub
            )
        ket_atoms = layout.ao_to_atom[layout.aos(sub.ket)]
        grad = grad.index_add(0, ket_atoms, -ket.mT)
    return grad


def _ket_operator_term(
    grad_out: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
) -> Tensor:
    r"""
    Gradient from the operator's own dependence on the ket centre.

    For :math:`(\mathbf{r} - \mathbf{R}_\nu)^{\otimes n}`, the derivative
    w.r.t. :math:`R_{\nu,d}` replaces each of the ``n`` factors in turn by
    :math:`-\delta_{d,\cdot}`, which leaves the operator of rank ``n - 1``.
    """
    nao = grad_out.shape[-2:]
    if name.packed:
        # the components left out do not enter the packed integral
        g = _flat_components(grad_out, name)
        index = torch.tensor(name.components, device=g.device)
        name = name.unpacked
        full = g.new_zeros(math.prod(name.comp_shape), *nao)
        grad_out = full.index_copy(0, index, g)

    rank = len(name.op)  # "j", "jj", "jjj"
    lower_name = replace(name, op=KET_CENTRED_OPERATORS[name.op])
    lower = _flat_components(
        _evaluate(lower_name, basis, inputs, block), lower_name
    )

    g = grad_out.reshape(*name.bra_shape, *(3,) * rank, *name.ket_shape, *nao)
    first = len(name.bra_shape)
    terms = []
    for factor in range(rank):
        gk = g.movedim(first + factor, 0).reshape(3, *lower.shape)
        terms.append(_contract(lower, gk, "ket"))
    return -torch.stack(terms).sum(0)


def _grad_alphas(
    grad_out: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
) -> Tensor:
    # d/dα exp(-α r²) = -r² exp(-α r²), with r relative to the centre ("rr")
    g = _flat_components(grad_out, name)
    ubasis, ulayout, ublock, bra_ao, ket_ao = _uncontracted(
        basis, inputs, block
    )

    def rr(side: Side, rows: _Block) -> Tensor:
        d = _derivative("rr", side, name, ubasis, inputs, block=rows)
        return d.reshape(1, len(g), *d.shape[-2:])

    # the ket derivative of a symmetric block, see `_grad_positions`
    symmetric = name.is_hermitian and block.is_diagonal

    grad = torch.zeros_like(inputs.alphas)
    for rows, cols, sub in _row_blocks(ulayout, ublock, len(g)):
        weights = _weights(g, bra_ao[rows], ket_ao[cols], symmetric=symmetric)
        bra = _contract(weights, rr("bra", sub), "bra")
        bra_params = ulayout.ao_to_param[ulayout.aos(sub.bra)]
        grad = grad.index_add(0, bra_params, -bra[0])
        if not symmetric:
            ket = _contract(weights, rr("ket", sub), "ket")
            ket_params = ulayout.ao_to_param[ulayout.aos(sub.ket)]
            grad = grad.index_add(0, ket_params, -ket[0])
    return grad


def _grad_coeffs(
    grad_out: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
) -> Tensor:
    # An uncontracted integral is the product of its bra and ket coefficients
    # and the integral with unit coefficients.
    g = _flat_components(grad_out, name)
    ubasis, ulayout, ublock, bra_ao, ket_ao = _uncontracted(
        basis, inputs, block
    )
    unit_inputs = inputs._replace(coeffs=torch.ones_like(inputs.coeffs))

    grad = torch.zeros_like(inputs.coeffs)
    for rows, cols, sub in _row_blocks(ulayout, ublock, len(g)):
        unit = _evaluate(name, ubasis, unit_inputs, sub)
        unit = unit.reshape(len(g), *unit.shape[-2:])
        bra_params = ulayout.ao_to_param[ulayout.aos(sub.bra)]
        ket_params = ulayout.ao_to_param[ulayout.aos(sub.ket)]

        # summed over the components, then over the other side's AOs with
        # their coefficients
        weights = _weights(g, bra_ao[rows], ket_ao[cols])
        products = (weights * unit).sum(-3)
        bra = products @ inputs.coeffs[ket_params]
        ket = inputs.coeffs[bra_params] @ products
        grad = grad.index_add(0, bra_params, bra)
        grad = grad.index_add(0, ket_params, ket)
    return grad


# Tangents
#
# Forward mode, the transpose of the gradients above: each row block of a
# derivative integral is contracted with the tangents of its AOs' atoms or
# parameters and added to the product `out` right away, which has the shape
# (batch, ncomp, nao_bra, nao_ket) with flattened components. The tangents
# carry a leading batch dimension (of one if shared by the batch).


class _Tangents(NamedTuple):
    """Tangents of the differentiable inputs (``None``: zero)"""

    positions: Tensor | None
    alphas: Tensor | None
    coeffs: Tensor | None


def _tangent(
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
    tangents: _Tangents,
    *,
    nbatch: int,
) -> Tensor:
    """
    Jacobian-vector products of a block of the integral `name` with a batch
    of `nbatch` tangents, shape ``(nbatch, *components, nao_bra, nao_ket)``
    """
    layout = basis.layout(inputs.numbers)
    rows, cols = layout.aos(block.bra), layout.aos(block.ket)
    # in the (column-major) memory layout of libcint's integrals, as forward
    # mode copies a tangent with another layout than its primal
    ncomp, nrows, ncols = (
        math.prod(name.comp_shape),
        rows.stop - rows.start,
        cols.stop - cols.start,
    )
    out = inputs.positions.new_empty_strided(
        (nbatch, ncomp, nrows, ncols),
        (ncomp * nrows * ncols, nrows * ncols, 1, nrows),
    ).zero_()
    terms = (
        (_tangent_positions, tangents.positions),
        (_tangent_alphas, tangents.alphas),
        (_tangent_coeffs, tangents.coeffs),
    )
    for add, tangent in terms:
        if tangent is not None:
            add(out, tangent, name, basis, inputs, block=block)
    return out.reshape(nbatch, *name.comp_shape, *out.shape[-2:])


def _tangent_positions(
    out: Tensor,
    tangent: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    *,
    block: _Block,
) -> None:
    if name.op == "nuc":
        raise NotImplementedError(
            "Derivatives of 'nuc' integrals w.r.t. the positions are not "
            "supported: the operator depends on the nuclear positions."
        )

    layout = basis.layout(inputs.numbers)
    nbatch, ncomp = out.shape[:2]

    def nabla(side: Side, rows: _Block) -> Tensor:
        d = _derivative("ip", side, name, basis, inputs, block=rows)
        return d.reshape(3, ncomp, *d.shape[-2:])

    def atoms(shells: range) -> Tensor:
        # tangents of the atoms of the AOs, shape (batch, nao, 3)
        return tangent[:, layout.ao_to_atom[layout.aos(shells)]]

    # the ket derivative of a symmetric block, see `_grad_positions`
    symmetric = name.is_hermitian and block.is_diagonal

    # minus libcint's nabla, see `_grad_positions`
    for rows, cols, sub in _row_blocks(layout, block, ncomp * max(3, nbatch)):
        bra = -_directional(nabla("bra", sub), atoms(sub.bra), "bra")
        out[..., rows, cols] += bra
        if symmetric:
            out[..., cols, rows] += bra.mT
            continue

        ket = -_directional(nabla("ket", sub), atoms(sub.ket), "ket")
        if not name.is_symmetric:
            ket = ket - _ket_operator_tangent(
                atoms(sub.ket), name, basis, inputs, sub
            )
        out[..., rows, cols] += ket


def _ket_operator_tangent(
    tangent: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
) -> Tensor:
    r"""
    Tangent from the operator's own dependence on the ket centre, for the
    tangents of the ket AOs' atoms, shape ``(batch, nao_ket, 3)``.

    The transpose of `_ket_operator_term`: each factor
    :math:`(\mathbf{r} - \mathbf{R}_\nu)_{a_i}` in turn is replaced by the
    tangent :math:`t_{a_i}`, which leaves the operator of rank ``n - 1``.
    """
    full = name.unpacked
    rank = len(name.op)  # "j", "jj", "jjj"
    lower_name = replace(full, op=KET_CENTRED_OPERATORS[name.op])
    lower = _evaluate(lower_name, basis, inputs, block)
    nbra, nket = len(full.bra_shape), len(full.ket_shape)
    nao = lower.shape[-2:]
    lower = lower.reshape(
        *full.bra_shape, *(3,) * (rank - 1), *full.ket_shape, *nao
    )

    nbatch = len(tangent)
    terms = []
    for factor in range(rank):
        # the tangent's direction at the factor's axis, the ket AOs last
        t = tangent.mT.reshape(
            nbatch,
            *(1,) * (nbra + factor),
            3,
            *(1,) * (rank - 1 - factor + nket),
            1,
            nao[1],
        )
        terms.append(lower.unsqueeze(nbra + factor) * t)
    total = torch.stack(terms).sum(0).reshape(nbatch, -1, *nao)
    return total if name.components is None else total[:, name.components]


def _tangent_alphas(
    out: Tensor,
    tangent: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    *,
    block: _Block,
) -> None:
    # d/dα exp(-α r²) = -r² exp(-α r²), see `_grad_alphas`
    nbatch, ncomp = out.shape[:2]
    ubasis, ulayout, ublock, bra_ao, ket_ao = _uncontracted(
        basis, inputs, block
    )

    def rr(side: Side, rows: _Block) -> Tensor:
        d = _derivative("rr", side, name, ubasis, inputs, block=rows)
        return d.reshape(1, ncomp, *d.shape[-2:])

    def params(shells: range) -> Tensor:
        # tangents of the exponents of the AOs, shape (batch, nao, 1)
        return tangent[:, ulayout.ao_to_param[ulayout.aos(shells)], None]

    # the ket derivative of a symmetric block, see `_grad_positions`
    symmetric = name.is_hermitian and block.is_diagonal

    for rows, cols, sub in _row_blocks(ulayout, ublock, ncomp * nbatch):
        bra = -_directional(rr("bra", sub), params(sub.bra), "bra")
        if symmetric:
            _add_contracted(out, bra, bra_ao[rows], ket_ao[cols])
            _add_contracted(out, bra.mT, ket_ao[cols], bra_ao[rows])
            continue

        ket = -_directional(rr("ket", sub), params(sub.ket), "ket")
        _add_contracted(out, bra + ket, bra_ao[rows], ket_ao[cols])


def _tangent_coeffs(
    out: Tensor,
    tangent: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    *,
    block: _Block,
) -> None:
    # bra and ket coefficients times the integral with unit coefficients,
    # see `_grad_coeffs`
    nbatch, ncomp = out.shape[:2]
    ubasis, ulayout, ublock, bra_ao, ket_ao = _uncontracted(
        basis, inputs, block
    )
    unit_inputs = inputs._replace(coeffs=torch.ones_like(inputs.coeffs))

    for rows, cols, sub in _row_blocks(ulayout, ublock, ncomp * nbatch):
        unit = _evaluate(name, ubasis, unit_inputs, sub)
        unit = unit.reshape(ncomp, *unit.shape[-2:])
        bra_params = ulayout.ao_to_param[ulayout.aos(sub.bra)]
        ket_params = ulayout.ao_to_param[ulayout.aos(sub.ket)]
        coeffs = (
            tangent[:, None, bra_params, None] * inputs.coeffs[ket_params]
            + inputs.coeffs[bra_params, None]
            * tangent[:, None, None, ket_params]
        )
        _add_contracted(out, coeffs * unit, bra_ao[rows], ket_ao[cols])


def _directional(derivative: Tensor, tangent: Tensor, side: Side) -> Tensor:
    """
    Sum of ``derivative[k] * tangent[..., k]`` over the directions ``k`` of
    the derivative, with the tangent per AO of the bra or ket, shape
    ``(batch, nao, ndir)``. Shape ``(batch, ncomp, nao_bra, nao_ket)``.
    """
    if side == "bra":
        tangent = tangent[:, None, :, None, :]
    else:
        tangent = tangent[:, None, None, :, :]
    out = derivative[0] * tangent[..., 0]
    for k in range(1, len(derivative)):
        out = out + derivative[k] * tangent[..., k]
    return out


def _add_contracted(
    out: Tensor, x: Tensor, bra_ao: Tensor, ket_ao: Tensor
) -> None:
    """
    Add `x`, of uncontracted AOs, to `out` at the contracted AOs `bra_ao` and
    `ket_ao` (within the block) of the uncontracted AOs (see `_uncontracted`)
    """
    first, last = int(ket_ao.min()), int(ket_ao.max()) + 1
    x = x.expand(len(out), *x.shape[1:])
    columns = x.new_zeros(*x.shape[:-1], last - first)
    columns.index_add_(-1, ket_ao - first, x)
    out[..., first:last].index_add_(-2, bra_ao, columns)


def _row_blocks(
    layout: Layout, block: _Block, ncomp: int
) -> Iterator[tuple[slice, slice, _Block]]:
    """
    Split the bra shells of `block` into row blocks with at most about
    `_BLOCK_ELEMENTS` elements of an integral with `ncomp` components.

    Yields the rows and columns of each row block within `block` and the row
    block. With screening, a row block only spans the columns of its shell
    pairs, and row blocks without any are skipped (see `_Block.rows`).
    """
    bra, ket = layout.aos(block.bra), layout.aos(block.ket)

    def fits(start: int, stop: int) -> bool:
        sub = block.rows(range(start, stop))
        if sub is None:
            return True
        rows, cols = layout.aos(sub.bra), layout.aos(sub.ket)
        size = ncomp * (rows.stop - rows.start) * (cols.stop - cols.start)
        return size <= _BLOCK_ELEMENTS

    def within(aos: slice, outer: slice) -> slice:
        return slice(aos.start - outer.start, aos.stop - outer.start)

    start = block.bra.start
    while start < block.bra.stop:
        # As many shells as fit, but at least one: double their number while
        # they fit, then bisect (with screening, the columns of a row block
        # grow with its rows).
        stop, step = start + 1, 1
        while stop + step <= block.bra.stop and fits(start, stop + step):
            stop, step = stop + step, 2 * step
        while step > 1:
            step //= 2
            if stop + step <= block.bra.stop and fits(start, stop + step):
                stop += step

        sub = block.rows(range(start, stop))
        start = stop
        if sub is not None:
            rows = within(layout.aos(sub.bra), bra)
            yield rows, within(layout.aos(sub.ket), ket), sub


def _contract(weights: Tensor, derivative: Tensor, side: Side) -> Tensor:
    """
    Sum of ``weights[x] * derivative[:, x]`` over the components ``x`` and
    the AOs of the other side, shape ``(len(derivative), nao)``.

    The products are summed in chunks of rows in memory order (libcint's
    integrals are column-major), which keeps the temporaries in the cache.
    einsum instead makes permuted copies of the whole derivative.
    """
    # In memory order (contiguous last dimension), the remaining AOs run
    # along the rows or along the columns.
    along_rows = side == "bra"
    if derivative.stride(-2) < derivative.stride(-1):
        weights, derivative = weights.mT, derivative.mT
        along_rows = not along_rows

    step = max(1, _CONTRACT_ELEMENTS // derivative[..., 0, :].numel())
    products: Iterable[Tensor]
    if step >= derivative.shape[-2]:
        products = [weights * derivative]
    else:
        # chunks by `split`, whose backward joins their gradients once (that
        # of slicing writes each one into zeros of the full size)
        chunks = zip(weights.split(step, -2), derivative.split(step, -2))
        products = (w * d for w, d in chunks)
    if along_rows:
        return torch.cat([p.sum((1, 3)) for p in products], dim=-1)
    return torch.stack([p.sum((1, 2)) for p in products]).sum(0)


def _weights(
    g: Tensor,
    rows: slice | Tensor,
    cols: slice | Tensor,
    symmetric: bool = False,
) -> Tensor:
    """
    The rows and columns (slices, or indices of the uncontracted AOs) of the
    gradient `g` of a block, or of ``g + g.mT`` if `symmetric`, which is
    never formed for the whole block.
    """

    def take(bra: slice | Tensor, ket: slice | Tensor) -> Tensor:
        if isinstance(bra, Tensor):
            return g[:, bra[:, None], ket]
        return g[:, bra, ket]

    if symmetric:
        return take(rows, cols) + take(cols, rows).mT
    return take(rows, cols)


def _flat_components(integral: Tensor, name: IntegralName) -> Tensor:
    """(Block of an) integral `name` with flattened components"""
    return integral.reshape(math.prod(name.comp_shape), *integral.shape[-2:])


def _uncontracted(
    basis: Basis, inputs: Inputs, block: _Block
) -> tuple[Basis, Layout, _Block, Tensor, Tensor]:
    """
    Uncontracted basis, layout and block, and the contracted AO within
    `block` of each uncontracted AO of its bra and of its ket, with which a
    gradient of the block spreads onto the uncontracted AOs (`_weights`).
    """
    layout = basis.layout(inputs.numbers)
    ubasis = basis.uncontracted
    ulayout = ubasis.layout(inputs.numbers)
    ublock = block.uncontracted(layout)

    def index(shells: range, ushells: range) -> Tensor:
        return layout.uao_to_ao[ulayout.aos(ushells)] - layout.aos(shells).start

    bra_ao = index(block.bra, ublock.bra)
    ket_ao = index(block.ket, ublock.ket)
    return ubasis, ulayout, ublock, bra_ao, ket_ao


def _derivative(
    deriv: str,
    side: Side,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    *,
    block: _Block,
) -> Tensor:
    """
    Block of the integral with an additional derivative operator on the bra
    or ket.

    An axis added by the operator is moved to the front.
    """
    result = _evaluate(name.differentiate(deriv, side), basis, inputs, block)
    if not DERIVATIVE_OPERATORS[deriv]:
        return result
    axis = 0 if side == "bra" else len(name.bra_shape) + len(name.op_shape)
    return result.movedim(axis, 0)


def _transpose(integral: Tensor, name: IntegralName) -> Tensor:
    """Turn the integral `name` into the integral ``name.transposed()``"""
    nbra, nop = len(name.bra_shape), len(name.op_shape)
    ncomp = len(name.comp_shape)
    bra = list(range(nbra))
    op = list(range(nbra, nbra + nop))
    ket = list(range(nbra + nop, ncomp))
    return integral.permute(*ket, *op, *bra, ncomp + 1, ncomp)
