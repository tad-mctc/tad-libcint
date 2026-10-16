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
other :mod:`torch.func` transforms. Since libcint is C code, a vmapped call
evaluates the integrals once per batch element.

Example
-------
>>> import torch
>>> from tad_mctc.io.structure import Structure
>>> from tad_libcint import Basis, Shell, overlap
>>> dd = {"dtype": torch.double}
>>> # normalized s function (libcint's s functions include 1/sqrt(4 pi))
>>> alpha, coeff = torch.tensor([1.0], **dd), torch.tensor([2.52647511], **dd)
>>> basis = Basis({1: [Shell(0, alpha, coeff)], 8: [Shell(0, alpha, coeff)]})
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

>>> from torch.func import jacrev
>>> def func(positions):
...     return overlap(basis, h2.replace(positions=positions))
>>> jacrev(func)(h2.positions).shape
torch.Size([2, 2, 2, 3])
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from tad_mctc.io.structure import Structure
from tad_mctc.math import einsum
from torch import Tensor

from . import library
from .basis import Basis, Layout
from .names import (
    DERIVATIVE_OPERATORS,
    KET_CENTRED_OPERATORS,
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

    Returns
    -------
    Tensor
        Integral of shape ``([batch,] *components, nao, nao)``. In a batch of
        different structures, ``nao`` is the largest number of AOs and the
        integrals of smaller structures are zero-padded.

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
    # The atomic numbers and positions are checked by the `Structure`.
    numbers, charges, positions, alphas, coeffs = inputs
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


def _integral(name: IntegralName, basis: Basis, inputs: Inputs) -> Tensor:
    out = _Int2c.apply(*inputs, basis, name)
    assert isinstance(out, Tensor)
    return out


def _evaluate(name: IntegralName, basis: Basis, inputs: Inputs) -> Tensor:
    """
    Integral `name`, or the transpose of ``name.transposed()``.

    The integral of a bra/ket symmetric operator equals the transpose of the
    integral with bra and ket operators exchanged, so libcint only needs to
    provide one of the two (e.g. ``"ipovlp"`` for ``"ovlpip"``).
    """
    symbol = name.symbol(basis.spherical)
    if library.has_symbol(symbol):
        return _integral(name, basis, inputs)

    source = name.transposed()
    missing = f"'{symbol}'"
    if name.is_symmetric:
        if library.has_symbol(source.symbol(basis.spherical)):
            return _transpose(_integral(source, basis, inputs), source)
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
# pass recomputes the structure's integral and differentiates it, again
# through `_PerStructure`, so that derivatives of any order resolve the batch
# as well. Within one structure, `_Int2c` provides the integrals and their
# derivatives.


class _StructureIntegral:
    """The integral `name` of one structure as a function of its inputs"""

    def __init__(self, name: IntegralName, basis: Basis) -> None:
        self.name = name
        self.basis = basis

    def __call__(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        return (_evaluate(self.name, self.basis, Inputs(*tensors)),)


class _VectorJacobianProduct:
    """
    Vector-Jacobian product of a structure function `func` w.r.t. the inputs
    flagged in `needs`, as a structure function of the inputs and the vectors
    """

    def __init__(
        self,
        func: Callable[..., tuple[Tensor, ...]],
        needs: tuple[bool, ...],
    ) -> None:
        self.func = func
        self.needs = needs

    def __call__(self, numbers: Tensor, *args: Tensor) -> tuple[Tensor, ...]:
        tensors, vectors = args[: len(self.needs)], args[len(self.needs) :]

        # Called from `_PerStructure.forward` (gradients disabled), differentiate
        # w.r.t. fresh leaves: the inputs, in particular the vectors, may
        # depend on each other in the caller's graph, which is accounted for
        # outside. Nested inside another product, keep its leaves connected.
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
            outputs = self.func(numbers, *leaves)

            # Vectors of a padded batch: drop the padding of this structure.
            # Skip outputs that do not depend on the inputs, and outputs with
            # a zero vector, which also avoids requesting derivative
            # integrals that are not needed. A zero vector of a nested
            # product must stay if it is differentiated (the product's
            # derivative w.r.t. the vector does not vanish at zero).
            pairs = [
                (o, v[tuple(slice(0, n) for n in o.shape)])
                for o, v in zip(outputs, vectors)
                if o.requires_grad
                and (bool(v.any()) or (nested and v.requires_grad))
            ]
            if not pairs:
                return tuple(torch.zeros_like(t) for t in tensors)

            grads = iter(
                torch.autograd.grad(
                    [o for o, _ in pairs],
                    [t for t, need in zip(leaves, self.needs) if need],
                    [v for _, v in pairs],
                    create_graph=True,
                    allow_unused=True,
                    materialize_grads=True,
                )
            )
        return tuple(
            next(grads) if need else torch.zeros_like(t)
            for t, need in zip(tensors, self.needs)
        )


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

    @staticmethod
    def backward(ctx: Any, *vectors: Tensor) -> tuple[Tensor | None, ...]:
        numbers, *tensors = ctx.saved_tensors
        needs = tuple(ctx.needs_input_grad[2:])
        if not any(needs):
            return (None,) * (2 + len(tensors))

        vjp = _VectorJacobianProduct(ctx.func, needs)
        grads = _PerStructure.apply(vjp, numbers, *tensors, *vectors)
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
        # libcint has no batch dimension: evaluate each structure
        def element(i: int) -> list[Tensor]:
            return [
                t if dim is None else t.select(dim, i)
                for t, dim in zip(tensors, in_dims[1:])
            ]

        outputs = []
        for i in range(info.batch_size):
            out = _PerStructure.apply(func, *element(i))
            assert isinstance(out, tuple)
            outputs.append(out)

        # different structures: pad to the largest number of AOs
        batched = tuple(torch.stack(_pad(list(outs))) for outs in zip(*outputs))
        return batched, (0,) * len(batched)


def _pad(tensors: list[Tensor]) -> list[Tensor]:
    """Zero-pad tensors at the end of every dimension to a common shape"""
    shape = [max(sizes) for sizes in zip(*(t.shape for t in tensors))]
    padded = []
    for t in tensors:
        pad = [
            p
            for n, size in zip(reversed(shape), reversed(t.shape))
            for p in (0, n - size)
        ]
        padded.append(F.pad(t, pad))
    return padded


class _Int2c(torch.autograd.Function):
    """Two-centre integral with gradients through libcint's derivatives."""

    # signatures are prescribed by torch.autograd.Function
    # pylint: disable=abstract-method,arguments-differ,too-many-positional-arguments

    @staticmethod
    def forward(
        numbers: Tensor,
        charges: Tensor,
        positions: Tensor,
        alphas: Tensor,
        coeffs: Tensor,
        basis: Basis,
        name: IntegralName,
    ) -> Tensor:
        layout = basis.layout(numbers)
        if layout.nsh == 0:
            return positions.new_zeros((*name.comp_shape, 0, 0))

        out = library.int2c(
            name.symbol(basis.spherical),
            name.comp_shape,
            name.is_hermitian,
            atm=layout.atm(charges),
            bas=layout.bas,
            env=layout.env(charges, positions, alphas, coeffs),
            ao_loc=layout.ao_loc,
        )
        return torch.from_numpy(out).to(positions.device, positions.dtype)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        *tensors, basis, name = inputs
        ctx.save_for_backward(*tensors)
        ctx.basis = basis
        ctx.name = name

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        inputs = Inputs(*ctx.saved_tensors)
        need_pos, need_alphas, need_coeffs = ctx.needs_input_grad[2:5]
        args = (grad_out, ctx.name, ctx.basis, inputs)

        return (
            None,  # numbers
            None,  # charges
            _grad_positions(*args) if need_pos else None,
            _grad_alphas(*args) if need_alphas else None,
            _grad_coeffs(*args) if need_coeffs else None,
            None,  # basis
            None,  # name
        )


# Gradients
#
# Notation: `grad_out` is the incoming gradient of the integral with shape
# (*components, nao, nao) and `g` the same with flattened components. In the
# einsum strings, `m`/`n` are bra/ket AOs, `x` runs over the flattened
# components and `d` over cartesian directions.


def _grad_positions(
    grad_out: Tensor, name: IntegralName, basis: Basis, inputs: Inputs
) -> Tensor:
    if name.op == "nuc":
        raise NotImplementedError(
            "Gradients of 'nuc' integrals w.r.t. the positions are not "
            "supported: the operator depends on the nuclear positions."
        )

    layout = basis.layout(inputs.numbers)
    g = grad_out.reshape(-1, layout.nao, layout.nao)
    dbra = _derivative("ip", "bra", name, basis, inputs).reshape(3, *g.shape)

    # libcint's nabla acts on the electron coordinate, which is minus the
    # derivative with respect to the function's centre
    if name.is_hermitian:
        # the ket derivative is the transpose of the bra derivative
        grad_ao = -einsum("xmn,dxmn->dm", g + g.mT, dbra)
    else:
        dket = _derivative("ip", "ket", name, basis, inputs)
        dket = dket.reshape(3, *g.shape)
        grad_ao = -einsum("xmn,dxmn->dm", g, dbra)
        grad_ao = grad_ao - einsum("xmn,dxmn->dn", g, dket)
        if not name.is_symmetric:
            grad_ao = grad_ao + _ket_operator_term(
                grad_out, name, basis, inputs
            )

    zeros = torch.zeros_like(inputs.positions)
    return zeros.index_add(0, layout.ao_to_atom, grad_ao.mT)


def _ket_operator_term(
    grad_out: Tensor, name: IntegralName, basis: Basis, inputs: Inputs
) -> Tensor:
    r"""
    Gradient from the operator's own dependence on the ket centre.

    For :math:`(\mathbf{r} - \mathbf{R}_\nu)^{\otimes n}`, the derivative
    w.r.t. :math:`R_{\nu,d}` replaces each of the ``n`` factors in turn by
    :math:`-\delta_{d,\cdot}`, which leaves the operator of rank ``n - 1``.
    """
    nao = grad_out.shape[-1]
    rank = len(name.op)  # "j", "jj", "jjj"
    lower_name = replace(name, op=KET_CENTRED_OPERATORS[name.op])
    lower = _evaluate(lower_name, basis, inputs).reshape(-1, nao, nao)

    g = grad_out.reshape(
        *name.bra_shape, *(3,) * rank, *name.ket_shape, nao, nao
    )
    first = len(name.bra_shape)
    term = torch.zeros((3, nao), dtype=grad_out.dtype, device=grad_out.device)
    for factor in range(rank):
        gk = g.movedim(first + factor, 0).reshape(3, *lower.shape)
        term = term - einsum("dxmn,xmn->dn", gk, lower)
    return term


def _grad_alphas(
    grad_out: Tensor, name: IntegralName, basis: Basis, inputs: Inputs
) -> Tensor:
    # d/dα exp(-α r²) = -r² exp(-α r²), with r relative to the centre ("rr")
    ubasis, ulayout, g = _uncontracted(grad_out, basis, inputs)
    dbra = _derivative("rr", "bra", name, ubasis, inputs).reshape(g.shape)
    if name.is_hermitian:
        # the ket derivative is the transpose of the bra derivative
        grad_uao = -einsum("xmn,xmn->m", g + g.mT, dbra)
    else:
        dket = _derivative("rr", "ket", name, ubasis, inputs).reshape(g.shape)
        grad_uao = -einsum("xmn,xmn->m", g, dbra)
        grad_uao = grad_uao - einsum("xmn,xmn->n", g, dket)

    zeros = torch.zeros_like(inputs.alphas)
    return zeros.index_add(0, ulayout.ao_to_param, grad_uao)


def _grad_coeffs(
    grad_out: Tensor, name: IntegralName, basis: Basis, inputs: Inputs
) -> Tensor:
    # An uncontracted integral is the product of its bra and ket coefficients
    # and the integral with unit coefficients.
    ubasis, ulayout, g = _uncontracted(grad_out, basis, inputs)
    ones = torch.ones_like(inputs.coeffs)
    unit = _evaluate(name, ubasis, inputs._replace(coeffs=ones))
    unit = unit.reshape(g.shape)

    c = inputs.coeffs[ulayout.ao_to_param]
    grad_uao = einsum("xmn,xmn,n->m", g, unit, c)
    grad_uao = grad_uao + einsum("xmn,xmn,m->n", g, unit, c)

    zeros = torch.zeros_like(inputs.coeffs)
    return zeros.index_add(0, ulayout.ao_to_param, grad_uao)


def _uncontracted(
    grad_out: Tensor, basis: Basis, inputs: Inputs
) -> tuple[Basis, Layout, Tensor]:
    """
    Uncontracted basis and layout, and the gradient spread from the
    contracted AOs onto the uncontracted AOs, shape ``(ncomp, nuao, nuao)``
    """
    ubasis = basis.uncontracted
    ulayout = ubasis.layout(inputs.numbers)
    index = basis.layout(inputs.numbers).uao_to_ao
    g = grad_out.index_select(-2, index).index_select(-1, index)
    return ubasis, ulayout, g.reshape(-1, ulayout.nao, ulayout.nao)


def _derivative(
    deriv: str, side: Side, name: IntegralName, basis: Basis, inputs: Inputs
) -> Tensor:
    """
    Integral with an additional derivative operator on the bra or ket.

    An axis added by the operator is moved to the front.
    """
    result = _evaluate(name.differentiate(deriv, side), basis, inputs)
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
