"""
Prototype: Gamma-point lattice sums of two-centre integrals with the existing
tad-libcint (no library changes), by folding ghost images.

    A_{mu nu} = sum_T < mu(R_i) | op | nu(R_j + T) >

Checks
1. overlap vs PySCF's ``cell.pbc_intor("int1e_ovlp")``
2. ket-centred "j"/"jj" folded over ket images == tblite's convention
   (bra translated, moment about the untranslated column atom)
3. overlap is symmetric, "j" is not
4. autograd gradients w.r.t. positions and lattice vs finite differences
"""

import math
import time

import numpy as np
import torch
from pyscf.pbc import gto as pbcgto
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor.images import build_periodic_shifts

from tad_libcint import Basis, Shell, int1e

torch.set_printoptions(precision=3, linewidth=140)
dd = {"dtype": torch.float64}

# small triclinic cell, two atoms, s + p shells with a diffuse exponent
LATTICE = torch.tensor(
    [[5.5, 0.0, 0.0], [1.0, 6.0, 0.0], [0.5, -0.7, 6.5]], **dd
)
NUMBERS = torch.tensor([3, 1])
POSITIONS = torch.tensor([[0.2, 0.1, 0.0], [2.3, 2.9, 3.1]], **dd)
SHELLS = [[0, [3.4, 0.3], [0.6, 0.7]], [1, [0.8, 1.0]]]

SCREENING = 60.0  # libcint's own threshold, -log of the Gaussian product


def pyscf_cell(lattice, positions, dimension=3):
    return pbcgto.M(
        dimension=dimension,
        # images along periodic axes only (PySCF default 2D sums all 3)
        low_dim_ft_type="inf_vacuum" if dimension < 3 else None,
        atom=[(int(z), p.tolist()) for z, p in zip(NUMBERS, positions)],
        a=lattice.numpy(),
        unit="Bohr",
        basis={"Li": SHELLS, "H": SHELLS},
        cart=True,  # (x, y, z) in both codes, no AO permutation needed
        precision=1e-16,
        verbose=0,
    )


def tad_basis(cell):
    elements = {}
    for ish in range(cell.nbas):
        z = int(NUMBERS[cell.bas_atom(ish)])
        elements.setdefault(z, {}).setdefault(cell.bas_atom(ish), []).append(
            Shell(
                angmom=int(cell.bas_angular(ish)),
                alphas=torch.tensor(cell.bas_exp(ish), **dd),
                coeffs=torch.tensor(cell._libcint_ctr_coeff(ish)[:, 0], **dd),
            )
        )
    shells = {
        z: next(iter(per_atom.values())) for z, per_atom in elements.items()
    }
    return Basis(shells=shells, spherical=False)


def images(lattice, cutoff, periodic=(True, True, True)):
    """Integer shifts along the periodic axes, the zero shift first"""
    mask = torch.tensor(periodic)
    shifts = build_periodic_shifts(lattice.detach(), mask, cutoff).shifts
    # build_periodic_shifts leaves dropping non-periodic axes to the consumer
    shifts = shifts[(shifts[:, ~mask] == 0).all(-1)]
    zero = (shifts == 0).all(-1)
    return torch.cat([shifts[zero], shifts[~zero]])


def bloch_int1e(name, basis, positions, lattice, shifts, kfrac, kcart=None):
    """
    A(k) = sum_T exp(i k.T) <mu|op|nu(T)> for k-points in fractional
    (reduced) coordinates, k.T = 2 pi kfrac.n: the phases are constants, only
    the image positions depend on the lattice. With `kcart` (Cartesian, 1/bohr)
    instead, exp(i k.T) with T = n.h. Shape (nk, *comp, nao, nao).
    """
    nimg = len(shifts)
    trans = shifts.to(positions.dtype) @ lattice
    ext = Structure(
        numbers=NUMBERS.repeat(nimg),
        positions=(positions[None] + trans[:, None]).reshape(-1, 3),
    )
    full = int1e(name, basis, ext, packed=False)
    nao = full.shape[-1] // nimg
    blocks = full[..., :nao, :].unflatten(-1, (nimg, nao))
    if kcart is None:
        phase = torch.exp(2j * math.pi * (kfrac @ shifts.T.to(kfrac.dtype)))
    else:  # Cartesian k (PySCF's convention), phases depend on the lattice
        phase = torch.exp(1j * (kcart @ trans.T))
    return torch.einsum("...inj,kn->k...ij", blocks.to(phase.dtype), phase)


def periodic_int1e(name, basis, positions, lattice, shifts, fold="ket"):
    """
    Lattice sum through a ghost supercell: AOs of image k are the k-th block
    of nao AOs. fold="ket": rows of the central cell, image columns folded
    (operator about the ket image). fold="bra": the transpose arrangement
    (tblite's _lat routines: bra translated, operator about the central ket).
    """
    nimg = len(shifts)
    trans = shifts.to(positions.dtype) @ lattice  # differentiable in lattice
    ext = Structure(
        numbers=NUMBERS.repeat(nimg),
        positions=(positions[None] + trans[:, None]).reshape(-1, 3),
    )
    full = int1e(name, basis, ext, packed=False)
    nao = full.shape[-1] // nimg
    if fold == "ket":
        return full[..., :nao, :].unflatten(-1, (nimg, nao)).sum(-2)
    return full[..., :, :nao].unflatten(-2, (nimg, nao)).sum(-3)


def main():
    cell = pyscf_cell(LATTICE, POSITIONS)
    basis = tad_basis(cell)
    amin = float(basis.alphas.min())
    # Gaussian product exp(-a b/(a+b) R^2) of the two most diffuse primitives
    cutoff = math.sqrt(2 * SCREENING / amin)
    shifts = images(LATTICE, cutoff)
    print(f"min alpha {amin}, cutoff {cutoff:.1f} bohr, {len(shifts)} images")

    # 1. overlap vs PySCF
    t = time.perf_counter()
    s = periodic_int1e("ovlp", basis, POSITIONS, LATTICE, shifts)
    t_tad = time.perf_counter() - t
    t = time.perf_counter()
    ref = torch.from_numpy(cell.pbc_intor("int1e_ovlp", hermi=0))
    t_pyscf = time.perf_counter() - t
    print(
        f"1. |S - S_pyscf|max = {(s - ref).abs().max():.2e}"
        f"  (tad {t_tad:.2f} s, pyscf {t_pyscf:.3f} s)"
    )
    print(
        f"   molecular S would differ by {(s - int1e('ovlp', basis, Structure(numbers=NUMBERS, positions=POSITIONS))).abs().max():.2e}"
    )

    # 2./3. multipole conventions and symmetry
    for name in ("j", "jj"):
        ket = periodic_int1e(name, basis, POSITIONS, LATTICE, shifts, "ket")
        bra = periodic_int1e(name, basis, POSITIONS, LATTICE, shifts, "bra")
        print(
            f"2. {name}: |ket-fold - bra-fold (tblite)|max = "
            f"{(ket - bra).abs().max():.2e},  asymmetry "
            f"{(ket - ket.mT).abs().max():.2e}"
        )
    print(f"3. overlap asymmetry {(s - s.mT).abs().max():.2e}")

    # wrapping an atom into a neighbouring cell leaves the Gamma sum unchanged
    wrapped = POSITIONS.clone()
    wrapped[1] -= LATTICE[2]
    sw = periodic_int1e("ovlp", basis, wrapped, LATTICE, shifts)
    jw = periodic_int1e("j", basis, wrapped, LATTICE, shifts)
    j0 = periodic_int1e("j", basis, POSITIONS, LATTICE, shifts)
    print(
        f"   wrapped atom: dS {(sw - s).abs().max():.2e}, dj {(jw - j0).abs().max():.2e}"
    )

    # 4. gradients w.r.t. positions and lattice (fixed Cartesian positions)
    for name in ("ovlp", "j"):
        w = torch.randn_like(
            periodic_int1e(name, basis, POSITIONS, LATTICE, shifts)
        )

        def energy(pos, lat):
            return (w * periodic_int1e(name, basis, pos, lat, shifts)).sum()

        pos = POSITIONS.clone().requires_grad_(True)
        lat = LATTICE.clone().requires_grad_(True)
        gpos, glat = torch.autograd.grad(energy(pos, lat), (pos, lat))

        h = 1e-5

        def fd(x, f):
            g = torch.zeros_like(x)
            for idx in torch.cartesian_prod(*map(torch.arange, x.shape)):
                idx = tuple(idx.tolist())
                xp, xm = x.clone(), x.clone()
                xp[idx] += h
                xm[idx] -= h
                g[idx] = (f(xp) - f(xm)) / (2 * h)
            return g

        fpos = fd(POSITIONS, lambda p: energy(p, LATTICE))
        flat = fd(LATTICE, lambda l: energy(POSITIONS, l))
        print(
            f"4. {name}: |dE/dR - FD| {(gpos - fpos).abs().max():.2e}, "
            f"|dE/dh - FD| {(glat - flat).abs().max():.2e}"
        )

        # strain derivative (tblite's sigma) from the lattice and position
        # gradients: sigma = (dE/dh)^T h + (dE/dR)^T R
        # (not symmetric here: random weights on Cartesian p AOs are not
        # rotationally invariant), checked against strained geometries
        sigma = glat.T @ LATTICE + gpos.T @ POSITIONS

        def strained(eps):
            e = torch.eye(3, **dd) + eps
            return energy(POSITIONS @ e.T, LATTICE @ e.T)

        fsigma = fd(torch.zeros(3, 3, **dd), strained)
        print(f"   |sigma - FD strain| {(sigma - fsigma).abs().max():.2e}")

    # 5. k-points and 1D/2D periodicity vs PySCF (dimension = number of
    # leading periodic axes there)
    kfrac = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.5], [0.5, -0.25, 0.125]], **dd
    )
    for dim in (3, 2, 1):
        periodic = tuple(a < dim for a in range(3))
        k = kfrac * torch.tensor(periodic, **dd)  # no k along vacuum axes
        # PySCF wants vacuum axes orthogonal to the periodic ones; they are
        # short here, so their images would lie within the cutoff
        lat_d = LATTICE.clone()
        lat_d[dim:] = torch.diag(LATTICE).diag()[dim:]
        cell_d = pyscf_cell(lat_d, POSITIONS, dimension=dim)
        shifts_d = images(lat_d, cutoff, periodic)
        out = bloch_int1e("ovlp", basis, POSITIONS, lat_d, shifts_d, k)
        ref = torch.from_numpy(
            np.asarray(
                cell_d.pbc_intor(
                    "int1e_ovlp", kpts=cell_d.get_abs_kpts(k.numpy())
                )
            )
        )
        herm = (out - out.mT.conj()).abs().max()
        print(
            f"5. {dim}D, {len(shifts_d)} images: |S(k) - S_pyscf(k)| "
            f"{(out - ref).abs().max():.2e}, |S(k) - S(k)^H| {herm:.2e}"
        )

    # 6. lattice gradient at k-points fixed in fractional coordinates
    w = torch.randn(len(kfrac), 3, *s.shape, dtype=torch.complex128)

    def energy_k(pos, lat):
        a = bloch_int1e("j", basis, pos, lat, shifts, kfrac)
        return (w.conj() * a).sum().real

    lat = LATTICE.clone().requires_grad_(True)
    (glat,) = torch.autograd.grad(energy_k(POSITIONS, lat), (lat,))
    flat = fd(LATTICE, lambda l: energy_k(POSITIONS, l))
    print(f"6. j(k): |dE/dh - FD| {(glat - flat).abs().max():.2e}")

    # 7. Cartesian k-points (PySCF's convention). Held fixed, the phases
    # depend on the lattice; built as 2 pi kfrac h^-T from the lattice, they
    # reproduce the gradient at fixed fractional k (6.)
    kcart = 2 * math.pi * kfrac @ torch.linalg.inv(LATTICE).T

    def energy_cart(pos, lat, kc):
        a = bloch_int1e("j", basis, pos, lat, shifts, None, kcart=kc(lat))
        return (w.conj() * a).sum().real

    lat = LATTICE.clone().requires_grad_(True)
    fixed = lambda l: kcart  # noqa: E731
    (gfix,) = torch.autograd.grad(energy_cart(POSITIONS, lat, fixed), (lat,))
    ffix = fd(LATTICE, lambda l: energy_cart(POSITIONS, l, fixed))
    print(
        f"7. j(k Cartesian, fixed): |dE/dh - FD| {(gfix - ffix).abs().max():.2e},"
        f" differs from fixed fractional k by {(gfix - glat).abs().max():.1e}"
    )
    built = lambda l: 2 * math.pi * kfrac @ torch.linalg.inv(l).T  # noqa: E731
    lat = LATTICE.clone().requires_grad_(True)
    (gbuilt,) = torch.autograd.grad(energy_cart(POSITIONS, lat, built), (lat,))
    print(
        f"   j(k = 2 pi kfrac h^-T): |dE/dh - (6.)| {(gbuilt - glat).abs().max():.2e}"
    )

    # 8. the chosen semantics: Cartesian input converted to fractional from
    # values (construction data, no gradient), phases exp(2 pi i kfrac.n).
    # The lattice gradient must not depend on how the k-points were built.
    def energy_conv(pos, lat, kc):
        k = kc(lat).detach()
        kf = k @ lat.detach().T / (2 * math.pi)
        a = bloch_int1e("j", basis, pos, lat, shifts, kf)
        return (w.conj() * a).sum().real

    for label, kc in (("constant", fixed), ("built from h", built)):
        lat = LATTICE.clone().requires_grad_(True)
        (gconv,) = torch.autograd.grad(energy_conv(POSITIONS, lat, kc), (lat,))
        print(
            f"8. Cartesian k {label}, converted: |dE/dh - (6.)| "
            f"{(gconv - glat).abs().max():.2e}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
