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

A periodic structure (with a lattice) gives lattice sums over the images of
the ket, at the Γ point or at k-points (see :func:`int1e`). They are
differentiable w.r.t. the lattice as well:

>>> cell = h2.replace(lattice=3.0 * torch.eye(3, **dd))
>>> s = overlap(basis, cell)
>>> bool(s[0, 0] > 1.0)  # an atom overlaps with its own images
True
>>> jacrev(lambda lat: overlap(basis, cell.replace(lattice=lat)))(
...     cell.lattice
... ).shape
torch.Size([2, 2, 3, 3])
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
from tad_mctc.kpoints import KPoints, to_fractional
from tad_mctc.neighbor import pairs_within_radii, periodic_pairs_within_radii
from torch import Tensor

from . import library
from .basis import Basis, Layout, _to_numpy
from .names import (
    DERIVATIVE_OPERATORS,
    QUADRUPOLE_OPERATORS,
    IntegralName,
    Side,
)

__all__ = ["PERIODIC_OPERATORS", "int1e", "overlap"]


PERIODIC_OPERATORS = ("ovlp", "j", "jj", "jjj")
"""
Central operators of periodic integrals. Moments about a fixed origin are not
translation invariant, and the nuclear attraction of all images diverges.
"""


class Inputs(NamedTuple):
    """
    Tensor inputs of an integral of one structure. The fields of a periodic
    structure are absent (``None``) for a molecule.
    """

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

    lattice: Tensor | None = None
    """Lattice vectors as rows, shape ``(3, 3)``."""

    periodic: Tensor | None = None
    """Periodic axes, shape ``(3,)`` (not differentiable)."""

    kpoints: Tensor | None = None
    """
    Fractional k-points, shape ``(nk, 3)``, none (``(0, 3)``) for the real
    integral at the Γ point (not differentiable).
    """

    kmask: Tensor | None = None
    """``False`` for padding k-points, shape ``(nk,)``."""


def int1e(
    name: str,
    basis: Basis,
    structure: Structure,
    *,
    nuclear_charges: Tensor | None = None,
    alphas: Tensor | None = None,
    coeffs: Tensor | None = None,
    packed: bool = True,
    kpoints: KPoints | Tensor | None = None,
    precision: float = 1e-10,
) -> Tensor:
    r"""
    Two-centre one-electron integral over all AOs of a structure.

    For a periodic structure (with a ``lattice``), the integral is the
    lattice sum over the images of the ket within reach,

    .. math::

        A_{\mu\nu}(\mathbf{k}) = \sum_{\mathbf{T}}
        e^{i \mathbf{k} \cdot \mathbf{T}}
        \langle \mu | \hat{O} | \nu(\mathbf{T}) \rangle,

    at the Γ point (real) or at the given k-points (complex), with the
    translations :math:`\mathbf{T} = \mathbf{n} \mathbf{A}` along the
    periodic axes. A ket-centred operator (``"j"``, ``"jj"``) is centred on
    the ket's image. Derivatives w.r.t. the lattice are taken at fixed
    fractional k-points (the k-points are not differentiable).

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
        and Cartesian coordinates in Bohr, shape ``([batch,] nat, 3)``. A
        periodic structure also has lattice vectors as rows, shape
        ``([batch,] 3, 3)``, and its periodic axes. The total charge,
        unpaired electrons and bonds do not enter the integrals.
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
    kpoints : KPoints | Tensor | None, optional
        k-points of a periodic structure, Cartesian in 1/Bohr as in PySCF,
        shape ``([batch,] nk, 3)``, e.g. from
        :func:`tad_mctc.kpoints.make_kpoints`. A batch with sets of
        different size is packed with :func:`tad_mctc.kpoints.pack_kpoints`,
        whose padding k-points give zeros. They are converted to fractional
        coordinates (from their values) and are not differentiable. Defaults
        to the Γ point only, with a real result.
    precision : float, optional
        Largest neglected contribution to an element of a periodic integral:
        the images and shell pairs beyond their cutoff, estimated per pair
        of elements for the operator's rank. Defaults to ``1e-10``.
        Molecular integrals are only screened where libcint's own threshold
        would neglect a pair anyway.

    Returns
    -------
    Tensor
        Integral of shape ``([batch,] *components, nao, nao)``, with
        libcint's components (e.g. 3 for ``"j"``, 9 or 6 when packed for
        ``"jj"``), and with k-points complex of shape
        ``([batch,] nk, *components, nao, nao)``. In a batch of different
        structures, ``nao`` is the largest number of AOs and the integrals
        of smaller structures are zero-padded.

    Raises
    ------
    ValueError
        If the name is not understood, an input has the wrong shape or dtype,
        the basis lacks an element, a periodic integral has an origin- or
        nucleus-dependent operator (see :data:`PERIODIC_OPERATORS`), or
        k-points are given for a molecule or along a non-periodic axis.
    NotImplementedError
        If the libcint build lacks the integral or one of the derivatives
        required for a gradient.
    """
    iname = IntegralName.parse(name)
    if packed and iname.op in QUADRUPOLE_OPERATORS:
        iname = replace(iname, packed=True)
    numbers, positions = structure.numbers, structure.positions
    charges = numbers if nuclear_charges is None else nuclear_charges
    alphas = basis.alphas if alphas is None else alphas
    coeffs = basis.coeffs if coeffs is None else coeffs
    _check_inputs(
        basis,
        numbers=numbers,
        charges=charges,
        positions=positions,
        alphas=alphas,
        coeffs=coeffs,
    )

    if structure.lattice is None:
        if kpoints is not None:
            raise ValueError("k-points require a periodic structure.")
        return _molecular(
            iname,
            basis,
            numbers=numbers,
            charges=charges,
            positions=positions,
            alphas=alphas,
            coeffs=coeffs,
        )

    if iname.op not in PERIODIC_OPERATORS:
        raise ValueError(
            f"The operator '{iname.op}' has no periodic integral. Periodic "
            f"structures support {list(PERIODIC_OPERATORS)}."
        )
    if math.isnan(precision) or precision <= 0.0:
        raise ValueError(f"Expected a positive precision, got {precision}.")
    return _periodic(
        iname,
        basis,
        structure,
        charges=charges,
        alphas=alphas,
        coeffs=coeffs,
        kpoints=kpoints,
        precision=precision,
    )


def _molecular(
    name: IntegralName,
    basis: Basis,
    *,
    numbers: Tensor,
    charges: Tensor,
    positions: Tensor,
    alphas: Tensor,
    coeffs: Tensor,
) -> Tensor:
    func = _StructureIntegral(name, basis)

    def single(n: Tensor, c: Tensor, p: Tensor) -> Tensor:
        out = _PerStructure.apply(func, n, c, p, alphas, coeffs)
        assert isinstance(out, tuple)
        return out[0]

    if numbers.ndim == 1:
        return single(numbers, charges, positions)

    # a batch of structures, which may share the nuclear charges
    charges = charges.expand(numbers.shape)
    return torch.vmap(single)(numbers, charges, positions)


def _periodic(
    name: IntegralName,
    basis: Basis,
    structure: Structure,
    *,
    charges: Tensor,
    alphas: Tensor,
    coeffs: Tensor,
    kpoints: KPoints | Tensor | None,
    precision: float,
) -> Tensor:
    numbers, positions = structure.numbers, structure.positions
    lattice, periodic = structure.lattice, structure.periodic
    assert lattice is not None and periodic is not None
    batch = numbers.shape[:-1]

    # The k-points are construction data: fractional coordinates from
    # their values (fixed under strain), none at the Gamma point.
    gamma = kpoints is None
    if isinstance(kpoints, KPoints):
        points, mask = kpoints.points, kpoints.mask
    else:
        if kpoints is None:
            kpoints = positions.new_zeros((*batch, 0, 3))
        points = kpoints
        mask = torch.ones(
            points.shape[:-1], dtype=torch.bool, device=points.device
        )
    if points.ndim not in (2, 3) or points.shape[-1] != 3:
        raise ValueError(
            f"Expected k-points of shape ([batch,] nk, 3), got "
            f"{tuple(points.shape)}."
        )
    if points.ndim == 3 and points.shape[:-2] != batch:
        raise ValueError(
            f"Expected one k-point set per structure, {tuple(batch)}, got "
            f"{tuple(points.shape[:-2])}."
        )
    fractional = to_fractional(
        points.detach().to(positions.dtype), lattice.detach()
    )

    func = _StructureIntegral(name, basis, precision=precision)

    # positional, as `vmap` maps them
    # pylint: disable-next=too-many-positional-arguments,too-many-arguments
    def single(
        n: Tensor,
        c: Tensor,
        p: Tensor,
        lat: Tensor,
        per: Tensor,
        k: Tensor,
        km: Tensor,
    ) -> Tensor:
        out = _PerStructure.apply(
            func, n, c, p, alphas, coeffs, lat, per, k, km
        )
        assert isinstance(out, tuple)
        # rows of the lattice sums: 1, or cos and sin per k-point
        sums = out[0]
        if gamma:
            return sums[0]
        return torch.complex(sums[0::2], sums[1::2])

    if numbers.ndim == 1:
        return single(
            numbers, charges, positions, lattice, periodic, fractional, mask
        )

    # a batch of structures, which may share the nuclear charges, the
    # lattice, the periodic axes and the k-points
    charges = charges.expand(numbers.shape)
    lattice = lattice.expand(*batch, 3, 3)
    periodic = periodic.expand(*batch, 3)
    fractional = fractional.expand(*batch, *fractional.shape[-2:])
    mask = mask.expand(*batch, mask.shape[-1])
    return torch.vmap(single)(
        numbers, charges, positions, lattice, periodic, fractional, mask
    )


def overlap(
    basis: Basis,
    structure: Structure,
    *,
    alphas: Tensor | None = None,
    coeffs: Tensor | None = None,
) -> Tensor:
    """Overlap integral, see :func:`int1e`"""
    return int1e("ovlp", basis, structure, alphas=alphas, coeffs=coeffs)


def _check_inputs(
    basis: Basis,
    *,
    numbers: Tensor,
    charges: Tensor,
    positions: Tensor,
    alphas: Tensor,
    coeffs: Tensor,
) -> None:
    # Only shapes and dtypes: under vmap, the values are not accessible here.
    # The atomic numbers and positions are checked by the `Structure`, but
    # stacking batched structures bypasses its check of the batch dimensions.
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


class _Images(NamedTuple):
    """
    Lattice images of the kets of a block (see `_Block`), each entering
    weighted sums of the block (the integral gets a leading axis of sums).
    """

    shifts: np.ndarray
    """Integer image shifts, shape ``(nimg, 3)``."""

    weights: np.ndarray
    """Weight of each image in each sum, shape ``(nw, nimg)``."""

    def translations(self, lattice: Tensor) -> np.ndarray:
        """Cartesian translations of the images, shape ``(nimg, 3)``"""
        return self.shifts @ _to_numpy(lattice)



class _Block(NamedTuple):
    """
    Block of an integral matrix: its bra (rows) and ket (columns) shells.

    With `pairs` (bra and ket shell, sorted by bra shell), only these shell
    pairs of the block are evaluated and all others are zero (screening, see
    `_screened_pairs`).

    With `images`, the block holds weighted lattice sums over images of the
    kets: `pairs` has a third column, the image of the ket, and is sorted by
    bra shell, ket shell and image. The ket shells must belong to atoms of
    their own (see `_with_ket_copies`). The integral of the block has a
    leading axis of the `nw` sums.
    """

    bra: range
    ket: range
    pairs: np.ndarray | None
    images: _Images | None = None

    @classmethod
    def full(cls, layout: Layout, pairs: np.ndarray | None) -> _Block:
        """The whole matrix"""
        return cls(range(layout.nsh), range(layout.nsh), pairs)

    @property
    def lead(self) -> tuple[int, ...]:
        """Leading axes of the block's integral: the sums of images"""
        return () if self.images is None else (len(self.images.weights),)

    @property
    def shls_slice(self) -> tuple[int, int, int, int]:
        """The block in libcint's format"""
        return (self.bra.start, self.bra.stop, self.ket.start, self.ket.stop)

    @property
    def is_diagonal(self) -> bool:
        """Whether bra and ket have the same shells"""
        return self.bra == self.ket

    def transposed(self) -> _Block:
        """
        The block with bra and ket exchanged. A lattice sum of
        :math:`\\langle \\mu | \\nu(\\mathbf{T}) \\rangle` is one of
        :math:`\\langle \\nu | \\mu(-\\mathbf{T}) \\rangle` (translation
        invariance), so the images are negated.
        """
        images = self.images
        if images is not None:
            images = images._replace(shifts=-images.shifts)
        pairs = self.pairs
        if pairs is not None:
            order = [1, 0, *range(2, pairs.shape[1])]
            pairs = _sorted(pairs[:, order])
        return self._replace(
            bra=self.ket, ket=self.bra, pairs=pairs, images=images
        )

    def rows(self, bra: range) -> _Block:
        """
        The rows of the bra shells `bra`. With `pairs`, only the columns of
        their shell pairs (every shell has some: those with the shells of its
        own atom).
        """
        if self.pairs is None:
            return self._replace(bra=bra)

        first, last = np.searchsorted(self.pairs[:, 0], [bra.start, bra.stop])
        pairs = self.pairs[first:last]
        ket = range(int(pairs[:, 1].min()), int(pairs[:, 1].max()) + 1)
        return self._replace(bra=bra, ket=ket, pairs=pairs)

    def uncontracted(self, layout: Layout) -> _Block:
        """The block of :attr:`Basis.uncontracted` (layout of this basis)"""
        bra = layout.uncontracted_shells(self.bra)
        ket = layout.uncontracted_shells(self.ket)
        pairs = self.pairs
        if pairs is not None:
            pairs = _sorted(layout.uncontracted_pairs(pairs))
        return self._replace(bra=bra, ket=ket, pairs=pairs)


def _sorted(pairs: np.ndarray) -> np.ndarray:
    """
    Shell pairs sorted by bra, then by ket shell (and then by image, for a
    third column)
    """
    # one key per pair sorts much faster than several keys
    sizes = [int(column.max(initial=0)) + 1 for column in pairs.T]
    keys = np.ravel_multi_index(tuple(pairs.T.astype(np.int64)), sizes)
    return np.stack(np.unravel_index(np.sort(keys), sizes), axis=-1)


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


# Periodic screening
#
# The images of a lattice sum are screened per pair of elements: beyond a
# cutoff, every block and the images within one decay length of it add less
# than the precision (see `_pair_cutoff`). The bound and its margin were
# benchmarked on GFN2-xTB cells in 3D, 2D and 1D (`notes/periodic.md`).


_PRECISION_MARGIN = 10.0
"""
Factor between the precision and the bound of a block at the cutoff: the
error jumps as discrete images cross the cutoff.
"""

_COMMON_FACTOR = {0: 0.282094791773878143, 1: 0.488602511902919921}
"""libcint's normalization of s and p functions (1 for higher ones)."""

_CUTOFF_STEP = 0.05
"""Resolution of the cutoffs in Bohr."""


def _operator_rank(name: IntegralName) -> int:
    """
    Power of the distance with which a block can grow: one per factor of a
    ket-centred moment and per derivative
    """
    return name.op.count("j") + len(name.bra) + len(name.ket)


def _pair_cutoff(
    bra: tuple[np.ndarray, np.ndarray, int],
    ket: tuple[np.ndarray, np.ndarray, int],
    rank: int,
    threshold: float,
    cell: tuple[float, int],
) -> float:
    """
    Distance beyond which the blocks of two shells are below `threshold`.

    The block of the primitives with exponents a, b and coefficients c_a,
    c_b (angular momenta l_a, l_b) of an operator of rank n about the ket is
    bounded by

        |c_a c_b| f_a f_b (pi/(a+b))^1.5 (bR/(a+b) + w)^l_a
            (aR/(a+b) + w)^(l_b + n) exp(-ab/(a+b) R^2),

    with the product's width w = 1/sqrt(2(a+b)), times the number of
    images within one decay length 1/(2 mu R) at R (from the cell's measure:
    volume, area or length of its `dim` periodic axes).
    """
    (a, ca, la), (b, cb, lb) = bra, ket
    measure, dim = cell
    a, b = a[:, None, None], b[None, :, None]
    p, mu = a + b, a * b / (a + b)
    pref = (
        np.abs(ca[:, None, None] * cb[None, :, None])
        * _COMMON_FACTOR.get(la, 1.0)
        * _COMMON_FACTOR.get(lb, 1.0)
        * (np.pi / p) ** 1.5
    )

    # beyond the Gaussian's decay to the threshold (plus the polynomial)
    rmax = np.sqrt((np.log(pref.max() / threshold) + 100.0) / mu.min())
    r = np.arange(0.0, rmax + _CUTOFF_STEP, _CUTOFF_STEP)
    width = 1.0 / np.sqrt(2.0 * p)
    bound = (
        pref
        * (b * r / p + width) ** la
        * (a * r / p + width) ** (lb + rank)
        * np.exp(-mu * r**2)
    )
    decay = 1.0 / (2.0 * mu * np.maximum(r, 1e-3))
    shell = {3: 4.0 * np.pi * r**2, 2: 2.0 * np.pi * r, 1: 2.0}.get(dim, 0.0)
    count = np.maximum(1.0, shell * decay / measure)
    worst = (bound * count).max(axis=(0, 1))

    above = np.flatnonzero(worst > threshold)
    return 0.0 if len(above) == 0 else float(r[min(above[-1] + 1, len(r) - 1)])


def _cell_measure(
    lattice: np.ndarray, periodic: np.ndarray
) -> tuple[float, int]:
    """
    Volume, area or length of the periodic axes (1 for none), and their
    number: the square root of the Gram determinant
    """
    axes = lattice[periodic]
    return float(np.sqrt(np.linalg.det(axes @ axes.T))), len(axes)


def _element_cutoffs(
    basis: Basis,
    inputs: Inputs,
    elements: list[int],
    *,
    rank: int,
    precision: float,
    cell: tuple[float, int],
) -> np.ndarray:
    """
    Cutoff of the blocks of a bra atom and a ket atom of each pair of
    `elements`, shape ``(nelem, nelem)``
    """
    shells = dict(basis.element_shells)
    alphas, coeffs = _to_numpy(inputs.alphas), _to_numpy(inputs.coeffs)
    threshold = precision / _PRECISION_MARGIN

    def params(
        shell: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        l, n, first = shell
        return alphas[first : first + n], coeffs[first : first + n], l

    cut = np.zeros((len(elements), len(elements)))
    for i, zi in enumerate(elements):
        for j, zj in enumerate(elements):
            cut[i, j] = max(
                _pair_cutoff(params(si), params(sj), rank, threshold, cell)
                for si in shells[zi]
                for sj in shells[zj]
            )
    return cut


def _periodic_block(
    name: IntegralName, basis: Basis, inputs: Inputs, precision: float
) -> _Block:
    """
    The lattice sums of a periodic structure: the shell pairs of each bra
    atom and the ket copy (see `_with_ket_copies`) of each atom image within
    the cutoff of their elements, with the weights of the requested sums.
    """
    assert inputs.lattice is not None and inputs.periodic is not None
    assert inputs.kpoints is not None and inputs.kmask is not None
    numbers = inputs.numbers.cpu()
    positions = inputs.positions.detach().cpu().to(torch.float64)
    lattice = inputs.lattice.detach().cpu().to(torch.float64)
    periodic = inputs.periodic.cpu()
    layout = basis.layout(inputs.numbers)

    # cutoffs of each pair of elements, as bra and as ket
    elements = sorted(set(numbers.tolist()) - {0})
    cell = _cell_measure(lattice.numpy(), periodic.numpy())
    rank = _operator_rank(name)
    cut = _element_cutoffs(
        basis, inputs, elements, rank=rank, precision=precision, cell=cell
    )
    index = np.zeros(int(numbers.numpy().max(initial=0)) + 1, dtype=np.int64)
    index[elements] = np.arange(len(elements))
    element = index[numbers.numpy()]

    # all pairs within the larger cutoff of either direction, then each
    # direction within its own
    real = numbers.numpy() != 0
    radius = np.maximum(cut, cut.T).max(axis=1, initial=0.0)[element] * real
    i, j, n = (
        x.numpy()
        for x in periodic_pairs_within_radii(
            Structure(
                numbers=numbers,
                positions=positions,
                lattice=lattice,
                periodic=periodic,
            ),
            torch.from_numpy(radius),
        )
    )
    distance = np.linalg.norm(
        positions.numpy()[j] + n @ lattice.numpy() - positions.numpy()[i],
        axis=-1,
    )
    forward = distance <= cut[element[i], element[j]]
    backward = distance <= cut[element[j], element[i]]
    onsite = np.flatnonzero(real)
    bra = np.concatenate([i[forward], j[backward], onsite])
    ket = np.concatenate([j[forward], i[backward], onsite])
    shift = np.concatenate(
        [n[forward], -n[backward], np.zeros((len(onsite), 3), dtype=np.int64)]
    )

    # images, and the shell pairs of each atom pair with its image
    shifts, image = np.unique(shift.reshape(-1, 3), axis=0, return_inverse=True)
    image = image.reshape(-1)
    pairs = layout.shell_pairs(np.stack([bra, ket, image], axis=-1))
    pairs[:, 1] += layout.nsh  # the ket copies
    pairs = _sorted(pairs)

    images = _Images(shifts, _image_weights(inputs, shifts))
    return _Block(
        range(layout.nsh), range(layout.nsh, 2 * layout.nsh), pairs, images
    )


def _image_weights(inputs: Inputs, shifts: np.ndarray) -> np.ndarray:
    """
    Weights of the images in the sums: a single sum at the Γ point, or the
    real and imaginary part of the Bloch phases of each k-point (zero for
    padding), shape ``(nw, nimg)``
    """
    assert inputs.kpoints is not None and inputs.kmask is not None
    assert inputs.periodic is not None
    kpoints = _to_numpy(inputs.kpoints)
    kmask = inputs.kmask.cpu().numpy()
    if len(kpoints) == 0:
        return np.ones((1, len(shifts)))

    periodic = inputs.periodic.cpu().numpy()
    offaxis = np.abs(kpoints[kmask][:, ~periodic])
    if offaxis.size > 0 and offaxis.max() > 1e-8:
        raise ValueError(
            "The k-points have components along a non-periodic axis "
            f"(fractional, up to {offaxis.max():.3g})."
        )

    phase = 2.0 * np.pi * kpoints @ shifts.T.astype(np.float64)
    weights = np.stack([np.cos(phase), np.sin(phase)], axis=1)
    weights *= kmask[:, None, None]
    return weights.reshape(2 * len(kpoints), len(shifts))


def _with_ket_copies(inputs: Inputs) -> Inputs:
    """
    The inputs with a copy of every atom for the kets of a lattice sum,
    which the C driver moves to the images (atoms ``nat`` to ``2 nat``).
    Gradients of both copies add up through the concatenation.
    """
    return inputs._replace(
        numbers=torch.cat([inputs.numbers, inputs.numbers]),
        charges=torch.cat([inputs.charges, inputs.charges]),
        positions=torch.cat([inputs.positions, inputs.positions]),
    )


def _integral(
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
    values: bool,
) -> Tensor:
    # the fields one by one: Dynamo fails on slices of namedtuples with defaults
    out = _Int2c.apply(
        inputs.numbers,
        inputs.charges,
        inputs.positions,
        inputs.alphas,
        inputs.coeffs,
        inputs.lattice,
        basis,
        name,
        block,
        values,
    )
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
            return _transpose(integral, source, len(block.lead))
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
    """
    The integral `name` of one structure as a function of its inputs. A
    periodic structure (with a lattice among the inputs) gives the weighted
    lattice sums of its block (see `_periodic_block`), with a leading axis
    of the sums.
    """

    def __init__(
        self, name: IntegralName, basis: Basis, precision: float = 0.0
    ) -> None:
        self.name = name
        self.basis = basis
        self.precision = precision
        self._screening: tuple[list[Tensor], _Block] | None = None

    def __call__(
        self, *tensors: Tensor, values: bool = True
    ) -> tuple[Tensor, ...]:
        inputs = Inputs(*tensors)
        block = self.block(inputs)
        return (
            _evaluate(
                self.name, self.basis, self.internal(inputs), block, values
            ),
        )

    @staticmethod
    def internal(inputs: Inputs) -> Inputs:
        """The inputs of the block: with ket copies for a lattice sum"""
        return inputs if inputs.lattice is None else _with_ket_copies(inputs)

    def block(self, inputs: Inputs) -> _Block:
        """
        The whole matrix, screened for large structures, or the lattice
        sums of a periodic structure. Either is reused for the structure of
        the last call, as the backward pass evaluates the integral (its
        graph) of the forward pass again.
        """
        if inputs.lattice is None:
            layout = self.basis.layout(inputs.numbers)
            if layout.nsh**2 < _SCREENING_MIN_PAIRS:
                return _Block.full(layout, None)
            structure = [inputs.numbers, inputs.positions, inputs.alphas]
        else:
            structure = [t for t in inputs if t is not None]

        if self._screening is not None:
            last, block = self._screening
            if all(
                a.shape == b.shape and torch.equal(a, b)
                for a, b in zip(last, structure)
            ):
                return block

        if inputs.lattice is None:
            block = _Block.full(layout, _screened_pairs(layout, inputs))
        else:
            block = _periodic_block(
                self.name, self.basis, inputs, self.precision
            )
        self._screening = ([t.detach().clone() for t in structure], block)
        return block


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
        # charges, periodic axes and k-points are not differentiable
        needed = [v if need else None for v, need in zip(batch, self.needs)]
        # in the order of `Inputs` (no lattice for a molecule)
        _, positions, alphas, coeffs, lattice, *_ = [*needed, None]
        if inputs.lattice is not None and positions is not None:
            positions = torch.cat([positions, positions], dim=1)  # ket copies

        out = _tangent(
            self.func.name,
            self.func.basis,
            self.func.internal(inputs),
            self.func.block(inputs),
            _Tangents(positions, alphas, coeffs, lattice),
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
        lattice: Tensor | None,
        basis: Basis,
        name: IntegralName,
        block: _Block,
        values: bool,
    ) -> Tensor:
        layout = basis.layout(numbers)
        rows, cols = layout.aos(block.bra), layout.aos(block.ket)
        shape = (
            *block.lead,
            *name.comp_shape,
            rows.stop - rows.start,
            cols.stop - cols.start,
        )
        if not values:
            return positions.new_zeros(()).expand(shape)
        if 0 in shape:
            return positions.new_zeros(shape)

        if block.images is not None:
            assert lattice is not None and block.pairs is not None
            out = library.int2c_lattice(
                name.symbol(basis.spherical),
                name.unpacked.comp_shape,
                shls_slice=block.shls_slice,
                pairs=block.pairs,
                translations=block.images.translations(lattice),
                weights=block.images.weights,
                components=name.components,
                atm=layout.atm(charges),
                bas=layout.bas,
                env=layout.env(charges, positions, alphas, coeffs),
                ao_loc=layout.ao_loc,
            ).reshape(shape)
            return torch.from_numpy(out).to(positions.device, positions.dtype)

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
        numbers, charges, positions, alphas, coeffs, lattice = ctx.saved_tensors
        inputs = Inputs(
            numbers, charges, positions, alphas, coeffs, lattice=lattice
        )
        needs = ctx.needs_input_grad
        need_pos, need_alphas, need_coeffs, need_lattice = needs[2:6]
        args = (grad_out, ctx.name, ctx.basis, inputs, ctx.block)

        return (
            None,  # numbers
            None,  # charges
            _grad_positions(*args) if need_pos else None,
            _grad_alphas(*args) if need_alphas else None,
            _grad_coeffs(*args) if need_coeffs else None,
            _grad_lattice(*args) if need_lattice else None,
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
    g = _flat_components(grad_out)

    # In a diagonal block of a hermitian integral, the ket derivative is the
    # transpose of the bra derivative.
    symmetric = name.is_hermitian and block.is_diagonal
    invariant = _translation_invariant(name)

    # libcint's nabla acts on the electron coordinate, which is minus the
    # derivative with respect to the function's centre
    grad = torch.zeros_like(inputs.positions)
    for rows, cols, sub in _row_blocks(layout, block, 3 * len(g)):
        weights = _weights(g, rows, cols, symmetric=symmetric)
        nabla = _derivative("ip", "bra", name, basis, inputs, block=sub)
        nabla = nabla.reshape(3, *weights.shape)
        bra = _contract(weights, nabla, "bra")
        bra_atoms = layout.ao_to_atom[layout.aos(sub.bra)]
        grad = grad.index_add(0, bra_atoms, -bra.mT)
        if symmetric:
            continue

        if invariant:
            ket = -_contract(weights, nabla, "ket")
        else:
            ket = _ket_term(weights, name, basis, inputs, block=sub)
        ket_atoms = layout.ao_to_atom[layout.aos(sub.ket)]
        grad = grad.index_add(0, ket_atoms, -ket.mT)
    return grad


def _grad_lattice(
    grad_out: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
) -> Tensor:
    r"""
    Gradient w.r.t. the lattice vectors (as rows). The ket of image
    :math:`\mathbf{n}` sits at :math:`\mathbf{R} + \mathbf{n} \mathbf{A}`,
    so row ``a`` is the ket term of the position gradient with the images
    weighted by :math:`n_a`, summed over all kets. By translation invariance
    (see `_translation_invariant`), the ket term is minus the bra term.
    """
    assert inputs.lattice is not None
    layout = basis.layout(inputs.numbers)
    g = _flat_components(grad_out)
    axes, stacked = _lattice_axes(block)

    grad = inputs.lattice.new_zeros(3, 3)
    if not axes:
        return grad
    index = torch.tensor(axes, device=grad.device)
    for rows, cols, sub in _row_blocks(layout, stacked, 3 * len(axes) * len(g)):
        weights = _weights(g, rows, cols)
        nabla = _derivative("ip", "bra", name, basis, inputs, block=sub)
        nabla = nabla.reshape(3 * len(axes), *weights.shape)
        # minus the ket term, see `_grad_positions`
        ket = _contract(weights, nabla, "ket").sum(-1)
        grad = grad.index_add(0, index, ket.reshape(3, len(axes)).mT)
    return grad


def _translation_invariant(name: IntegralName) -> bool:
    """
    Whether the integral is invariant under a common translation of its bra
    and ket (and the operator, if centred on the ket). Its derivative w.r.t.
    the ket centre is then minus that w.r.t. the bra centre.
    """
    return name.op in PERIODIC_OPERATORS


def _lattice_axes(block: _Block) -> tuple[list[int], _Block]:
    """
    The lattice axes along which the block has images, and the block with
    the sums of its images weighted by the shift along each of them, axis
    by axis (leading axis of the integral ``(naxes * nw,)``).
    """
    assert block.images is not None
    shifts, weights = block.images
    axes = [a for a in range(3) if shifts[:, a].any()]
    # (naxes, nw, nimg), also for no axes
    stacked = shifts.T[axes, None, :] * weights
    images = block.images._replace(weights=stacked.reshape(-1, len(shifts)))
    return axes, block._replace(images=images)


def _ket_term(
    weights: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    *,
    block: _Block,
) -> Tensor:
    """
    The block's derivatives w.r.t. the electron coordinate in the ket
    function, contracted with `weights` (the gradient of the block,
    flattened components), shape ``(3, nao_ket)``, for an operator that does
    not depend on the ket centre
    """
    assert name.is_symmetric
    nabla = _derivative("ip", "ket", name, basis, inputs, block=block)
    return _contract(weights, nabla.reshape(3, *weights.shape), "ket")


def _grad_alphas(
    grad_out: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    block: _Block,
) -> Tensor:
    # d/dα exp(-α r²) = -r² exp(-α r²), with r relative to the centre ("rr")
    g = _flat_components(grad_out)
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
    g = _flat_components(grad_out)
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
    lattice: Tensor | None = None


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
    of `nbatch` tangents, shape ``(nbatch, *lead, *components, nao_bra,
    nao_ket)`` (with the leading sums of a block's images)
    """
    layout = basis.layout(inputs.numbers)
    rows, cols = layout.aos(block.bra), layout.aos(block.ket)
    # in the (column-major) memory layout of libcint's integrals, as forward
    # mode copies a tangent with another layout than its primal
    ncomp, nrows, ncols = (
        math.prod(block.lead) * math.prod(name.comp_shape),
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
        (_tangent_lattice, tangents.lattice),
    )
    for add, tangent in terms:
        if tangent is not None:
            add(out, tangent, name, basis, inputs, block=block)
    return out.reshape(nbatch, *block.lead, *name.comp_shape, *out.shape[-2:])


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

    def atoms(shells: range) -> Tensor:
        # tangents of the atoms of the AOs, shape (batch, nao, 3)
        return tangent[:, layout.ao_to_atom[layout.aos(shells)]]

    # the ket derivative of a symmetric block, see `_grad_positions`
    symmetric = name.is_hermitian and block.is_diagonal
    invariant = _translation_invariant(name)

    # minus libcint's nabla, see `_grad_positions`
    for rows, cols, sub in _row_blocks(layout, block, ncomp * max(3, nbatch)):
        nabla = _derivative("ip", "bra", name, basis, inputs, block=sub)
        nabla = nabla.reshape(3, ncomp, *nabla.shape[-2:])
        bra = -_directional(nabla, atoms(sub.bra), "bra")
        out[..., rows, cols] += bra
        if symmetric:
            out[..., cols, rows] += bra.mT
            continue

        if invariant:
            out[..., rows, cols] += _directional(nabla, atoms(sub.ket), "ket")
        else:
            out[..., rows, cols] += _ket_tangent(
                atoms(sub.ket), ncomp, name, basis, inputs, block=sub
            )


def _tangent_lattice(
    out: Tensor,
    tangent: Tensor,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    *,
    block: _Block,
) -> None:
    """
    The transpose of `_grad_lattice`: the kets of image n move by
    ``n @ tangent``, the tangent of the lattice vectors, shape
    ``(batch, 3, 3)``
    """
    layout = basis.layout(inputs.numbers)
    nbatch, ncomp = out.shape[:2]
    axes, stacked = _lattice_axes(block)
    if not axes:
        return
    move = tangent[:, axes]
    size = ncomp * len(axes) * max(3, nbatch)
    for rows, cols, sub in _row_blocks(layout, stacked, size):
        nabla = _derivative("ip", "bra", name, basis, inputs, block=sub)
        nabla = nabla.reshape(3, len(axes), ncomp, *nabla.shape[-2:])
        # minus the ket tangent, see `_grad_lattice`
        out[..., rows, cols] += torch.einsum("dacij,bad->bcij", nabla, move)


def _ket_tangent(
    tangent: Tensor,
    ncomp: int,
    name: IntegralName,
    basis: Basis,
    inputs: Inputs,
    *,
    block: _Block,
) -> Tensor:
    """
    The transpose of `_ket_term`: the tangent of the block for the tangents
    of its ket AOs' centres, shape ``(batch, nao_ket, 3)``, shape
    ``(batch, ncomp, nao_bra, nao_ket)``
    """
    assert name.is_symmetric
    nabla = _derivative("ip", "ket", name, basis, inputs, block=block)
    nabla = nabla.reshape(3, ncomp, *nabla.shape[-2:])
    return -_directional(nabla, tangent, "ket")


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
    pairs (see `_Block.rows`).
    """
    bra, ket = layout.aos(block.bra), layout.aos(block.ket)

    def fits(start: int, stop: int) -> bool:
        sub = block.rows(range(start, stop))
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


def _flat_components(integral: Tensor) -> Tensor:
    """
    (Block of an) integral with flattened components, including the leading
    sums of a block's images
    """
    ncomp = math.prod(integral.shape[:-2])
    return integral.reshape(ncomp, *integral.shape[-2:])


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
    return result.movedim(len(block.lead) + axis, 0)


def _transpose(integral: Tensor, name: IntegralName, nlead: int) -> Tensor:
    """
    Turn the integral `name` into the integral ``name.transposed()``, with
    `nlead` leading axes (the sums of a block's images) kept in front
    """
    nbra, nop = len(name.bra_shape), len(name.op_shape)
    ncomp = nlead + len(name.comp_shape)
    lead = list(range(nlead))
    bra = list(range(nlead, nlead + nbra))
    op = list(range(nlead + nbra, nlead + nbra + nop))
    ket = list(range(nlead + nbra + nop, ncomp))
    return integral.permute(*lead, *ket, *op, *bra, ncomp + 1, ncomp)
