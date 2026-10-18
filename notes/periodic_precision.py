"""
Benchmark: accuracy setting -> real-space cutoffs of lattice sums.

For a few cells in the GFN2-xTB basis, every block <mu(R_i)|op|nu(R_j + T)>
is evaluated once with a near-exact reference cutoff (libcint's own
exp(-60)). Screening at a precision p by an estimator drops whole (atom pair,
image) blocks. For each estimator and p, this reports

- the error bound max_{mu nu} sum_T |dropped A_T|, valid at every k-point,
- the share of (atom pair, image) blocks kept (the cost).

Estimators (K = -ln p):

gauss   per atom r_a = sqrt(K / alpha_min(a)), pairs in quadrature: drops a
        block when the Gaussian product factor exp(-ab/(a+b) R^2) of the two
        most diffuse primitives is below p (tad-libcint's molecular
        screening, there with K = 60)
bound<n> per pair of elements, a bound of the primitive pairs' blocks for an
        operator of rank n, times the images within one decay length, with
        a margin SAFETY (`pair_bound`)
pyscf   PySCF's _estimate_rcut(alpha, l, c, p) per primitive, maximum over
        the cell, one global cutoff (cell.rcut)
tblite  sqrt(2 K / alpha_min) of the cell, capped at 40 bohr (get_cutoff,
        with intcut = K)

Run from the repository root (needs dxtb's sources for the STO-NG tables and
GFN2 parameters):

    DXTB_SRC=../dxtb/src PYTHONPATH=src:../tad-mctc/src \
        python notes/periodic_precision.py
"""

import math
import os
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from tad_mctc.neighbor.images import build_periodic_shifts

from tad_libcint import Basis, Shell, library
from tad_libcint.names import IntegralName

DXTB = Path(os.environ.get("DXTB_SRC", Path.home() / "Dokumente/dev/dxtb/src")) / "dxtb/_src"
dd = {"dtype": torch.float64}

PRECISIONS = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
REFERENCE = 60.0  # -ln of libcint's own threshold
INTEGRALS = ["ovlp", "j", "jj"]


# GFN2-xTB basis (STO-NG expansion of Slater functions, as dxtb)

STO_NG = [np.load(DXTB / f"basis/sto-ng/sto-{n}g.npy") for n in range(1, 7)]
DFACTORIAL = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0]


def slater_to_gauss(ng, n, l, zeta):
    itype = n + [0, 4, 7, 9, 10][l] - 1
    if n == 6 and ng == 6:
        itype = 15 + l
    coeff, alpha = STO_NG[ng - 1][:, itype, :]
    alpha = alpha * zeta**2
    coeff = (
        coeff * (2 / math.pi * alpha) ** 0.75 * np.sqrt(4 * alpha) ** l
        / math.sqrt(DFACTORIAL[l])
    )
    return alpha, coeff


def gfn2_basis(numbers):
    with open(DXTB / "param/gfn2/gfn2-xtb.toml", "rb") as f:
        param = tomllib.load(f)
    symbols = {
        1: "H", 3: "Li", 6: "C", 8: "O", 9: "F", 11: "Na", 12: "Mg",
        14: "Si", 17: "Cl",
    }
    shells = {}
    for z in sorted(set(numbers)):
        el = param["element"][symbols[z]]
        shells[z] = []
        for label, zeta, ng in zip(el["shells"], el["slater"], el["ngauss"]):
            n, l = int(label[0]), "spdf".index(label[1])
            alpha, coeff = slater_to_gauss(ng, n, l, zeta)
            shell = Shell(
                angmom=l,
                alphas=torch.tensor(alpha, **dd),
                coeffs=torch.tensor(coeff, **dd),
            )
            shells[z].append(shell.normalized())
    return Basis(shells=shells)


# Cells (bohr), lattice vectors as rows

def fcc(a):
    return 0.5 * a * torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], **dd)


def systems():
    a = 6.74  # diamond
    yield "diamond (3D)", [6, 6], fcc(a), [[0, 0, 0], [a / 4] * 3], (1, 1, 1)
    a = 10.26  # silicon
    yield "silicon (3D)", [14, 14], fcc(a), [[0, 0, 0], [a / 4] * 3], (1, 1, 1)
    a = 7.61  # LiF rocksalt
    yield "LiF (3D)", [3, 9], fcc(a), [[0, 0, 0], [a / 2] * 3], (1, 1, 1)
    a = 4.65  # graphene, 20 bohr vacuum vector
    lat = torch.tensor(
        [[a, 0, 0], [a / 2, a * math.sqrt(3) / 2, 0], [0, 0, 20]], **dd
    )
    pos = [[0, 0, 0], [a / 2, a / (2 * math.sqrt(3)), 0]]
    yield "graphene (2D)", [6, 6], lat, pos, (1, 1, 0)
    c = 4.83  # polyethylene, zigzag CH2 chain along x
    lat = torch.tensor([[c, 0, 0], [0, 20, 0], [0, 0, 20]], **dd)
    pos = [
        [0, 0, 0], [c / 2, 1.62, 0],
        [0, -1.2, 1.67], [0, -1.2, -1.67],
        [c / 2, 2.82, 1.67], [c / 2, 2.82, -1.67],
    ]
    yield "polyethylene (1D)", [6, 6, 1, 1, 1, 1], lat, pos, (1, 0, 0)

    # Not used to choose SAFETY (out of sample)
    a = 10.66  # NaCl rocksalt
    yield "NaCl (3D)*", [11, 17], fcc(a), [[0, 0, 0], [a / 2] * 3], (1, 1, 1)
    a = 7.96  # MgO rocksalt
    yield "MgO (3D)*", [12, 8], fcc(a), [[0, 0, 0], [a / 2] * 3], (1, 1, 1)
    # water at the density of ice, one molecule per cubic cell
    lat = 6.04 * torch.eye(3, **dd)
    pos = [[0, 0, 0.13], [0, 1.43, -0.98], [0, -1.43, -0.98]]
    yield "water (3D)*", [8, 1, 1], lat, pos, (1, 1, 1)


# Cutoff estimators: keep (i, j, T) if dist <= cutoff(i, j)

def pyscf_rcut(alpha, l, c, precision):
    theta = alpha * 0.5
    a1 = (alpha * 2) ** -0.5
    norm_ang = (2 * l + 1) / (4 * np.pi)
    fac = 2 * np.pi * c**2 * norm_ang / theta / precision
    fac *= 4 * alpha**2
    r0 = 20
    r0 = (np.log(fac * r0 * (r0 * 0.5 + a1) ** (2 * l + 2) + 1.0) / theta) ** 0.5
    r0 = (np.log(fac * r0 * (r0 * 0.5 + a1) ** (2 * l + 2) + 1.0) / theta) ** 0.5
    return r0


def estimators(basis, numbers, lattice_measure):
    amin = {z: min(float(s.alphas.min()) for s in basis.shells[z]) for z in numbers}
    a = np.array([amin[z] for z in numbers])

    def gauss(p):
        r2 = -math.log(p) / a
        return np.sqrt(r2[:, None] + r2[None, :])

    def pyscf(p):
        r = max(
            pyscf_rcut(s.alphas.numpy(), s.angmom, np.abs(s.coeffs.numpy()), p).max()
            for z in numbers for s in basis.shells[z]
        )
        return np.full((len(a), len(a)), r)

    def tblite(p):
        return np.full((len(a), len(a)), min(math.sqrt(2 * -math.log(p) / a.min()), 40.0))

    out = {"gauss": gauss, "pyscf": pyscf, "tblite": tblite}
    for rank in (0, 2):
        out[f"bound{rank}"] = lambda p, rank=rank: pair_bound(
            basis, numbers, lattice_measure, p, rank
        )
    return out


# libcint's angular prefactor of s and p functions (CINTcommon_fac_sp)
COMMON_FAC = {0: 0.282094791773878143, 1: 0.488602511902919921}
GRID = np.arange(0.0, 200.0, 0.05)

SAFETY = 10.0
"""Margin of the pair bound: the error jumps as discrete images cross the cutoff."""


def pair_bound(basis, numbers, measure, p, rank):
    """
    Per pair of elements, the smallest R beyond which every block is below
    p / SAFETY divided by the number of images within one decay length.

    A primitive pair (a, b) with |coefficients| c_a c_b, angular momenta
    l_a, l_b and a rank-n operator about the ket's centre is bounded by

        c_a c_b f_a f_b (pi/(a+b))^1.5 (bR/(a+b) + w)^l_a
            (aR/(a+b) + w)^(l_b + n) exp(-ab/(a+b) R^2),

    with w = 1/sqrt(2(a+b)) the product's width (as PySCF's `a1`), and the
    images in the shell of one decay length 1/(2 mu R) at R counted from the
    cell's measure (volume, area or length).
    """
    measure_, dim = measure
    elements = sorted(set(numbers))
    cut = {}
    for zi in elements:
        for zj in elements:
            worst = np.zeros_like(GRID)
            for si in basis.shells[zi]:
                for sj in basis.shells[zj]:
                    la, lb = si.angmom, sj.angmom
                    fac = COMMON_FAC.get(la, 1.0) * COMMON_FAC.get(lb, 1.0)
                    for a, ca in zip(si.alphas.numpy(), si.coeffs.numpy()):
                        for b, cb in zip(sj.alphas.numpy(), sj.coeffs.numpy()):
                            mu = a * b / (a + b)
                            w = 1 / math.sqrt(2 * (a + b))
                            bound = (
                                abs(ca * cb) * fac * (math.pi / (a + b)) ** 1.5
                                * (b * GRID / (a + b) + w) ** la
                                * (a * GRID / (a + b) + w) ** (lb + rank)
                                * np.exp(-mu * GRID**2)
                            )
                            # images within one decay length of the shell at R
                            dr = 1 / (2 * mu * np.maximum(GRID, 1e-3))
                            shell = {3: 4 * math.pi * GRID**2, 2: 2 * math.pi * GRID, 1: 2.0}[dim]
                            count = np.maximum(1.0, shell * dr / measure_)
                            worst = np.maximum(worst, bound * count)
            above = np.nonzero(worst > p / SAFETY)[0]
            cut[zi, zj] = GRID[above[-1] + 1] if len(above) else 0.0
    return np.array([[cut[zi, zj] for zj in numbers] for zi in numbers])


# Reference blocks

def image_blocks(name, basis, numbers, positions, shifts, lattice):
    """
    <mu(R_i)| op |nu(R_j + T)>, shape (ncomp, nao, nimg, nao), from the
    central cell's rows of a ghost supercell (libcint driver directly)
    """
    nimg, nat = len(shifts), len(numbers)
    trans = shifts.to(positions.dtype) @ lattice
    ext_numbers = torch.tensor(numbers).repeat(nimg)
    ext_positions = (positions[None] + trans[:, None]).reshape(-1, 3)
    layout = basis.layout(ext_numbers)
    nsh = layout.nsh // nimg
    iname = IntegralName.parse(name)
    charges = ext_numbers.to(positions.dtype)
    out = library.int2c(
        iname.symbol(basis.spherical),
        iname.comp_shape,
        False,
        shls_slice=(0, nsh, 0, layout.nsh),
        atm=layout.atm(charges),
        bas=layout.bas,
        env=layout.env(charges, ext_positions, basis.alphas, basis.coeffs),
        ao_loc=layout.ao_loc,
    )
    nao = out.shape[-2]
    return out.reshape(-1, nao, nimg, nao), layout.ao_to_atom[:nao].numpy()


def main():
    print(f"{'system':18s} {'p':>7s} {'estimator':>9s} {'kept':>7s} "
          + " ".join(f"{'err ' + n:>10s}" for n in INTEGRALS))
    for label, numbers, lattice, pos, periodic in systems():
        positions = torch.tensor(pos, **dd)
        mask = torch.tensor([bool(x) for x in periodic])
        basis = gfn2_basis(numbers)
        dim = int(mask.sum())
        lat_p = lattice[mask]
        if dim == 3:
            measure = abs(torch.linalg.det(lattice)).item()
        elif dim == 2:
            measure = torch.linalg.cross(lat_p[0], lat_p[1]).norm().item()
        else:
            measure = lat_p[0].norm().item()
        est = estimators(basis, numbers, (measure, dim))

        # reference: every image within the exp(-60) pair cutoff, plus the
        # largest intra-cell distance
        ref_cut = est["gauss"](math.exp(-REFERENCE)).max()
        span = torch.cdist(positions, positions).max().item()
        shifts = build_periodic_shifts(lattice, mask, ref_cut + span).shifts
        shifts = shifts[(shifts[:, ~mask] == 0).all(-1)]
        # the rows of image_blocks are the first image: the central cell
        zero = (shifts == 0).all(-1)
        shifts = torch.cat([shifts[zero], shifts[~zero]])

        t = time.perf_counter()
        blocks = {}
        for name in INTEGRALS:
            blocks[name], ao_atom = image_blocks(
                name, basis, numbers, positions, shifts, lattice
            )
        t = time.perf_counter() - t

        # distance of every (i, j, T)
        trans = shifts.to(torch.float64) @ lattice
        vec = positions[None, None] + trans[:, None, None] - positions[None, :, None]
        dist = vec.norm(dim=-1).permute(1, 2, 0).numpy()  # (i, j, T)
        ref_keep = dist <= est["gauss"](math.exp(-REFERENCE))[..., None]
        print(f"{label}: {len(shifts)} images, {ref_keep.sum()} reference "
              f"blocks, nao {len(ao_atom)}, {t:.1f} s")

        for p in PRECISIONS:
            for ename, cutoff in est.items():
                keep = dist <= cutoff(p)[..., None]
                dropped = ref_keep & ~keep  # (i, j, T)
                # per AO pair and image
                drop_ao = dropped[ao_atom[:, None], ao_atom[None, :], :]
                errs = []
                for name in INTEGRALS:
                    b = np.abs(blocks[name]).transpose(0, 1, 3, 2)
                    errs.append((b * drop_ao[None]).sum(-1).max())
                share = (keep & ref_keep).sum() / ref_keep.sum()
                flag = " " if max(errs) <= p else "!"
                print(f"{'':18s} {p:7.0e} {ename:>9s} {share:7.1%} "
                      + " ".join(f"{e:10.1e}" for e in errs) + f" {flag}")


if __name__ == "__main__":
    main()
