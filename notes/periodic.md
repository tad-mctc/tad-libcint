# Periodic integrals: investigation

Status: investigation (branch `feature/periodic`), no library changes yet.
Prototype: [`periodic_prototype.py`](periodic_prototype.py).

## What is needed

The consumer is dxtb, which follows tblite. Both work at the Γ point only.
tblite's periodic H0 (`xtb/h0.f90: get_hamiltonian`) and its `*_lat` integral
routines (`integral/native/integrals.f90`) build **real-space lattice sums**
of the molecular integrals:

$$
A_{\mu\nu} = \sum_{\mathbf T} \langle \mu(\mathbf R_i) \,|\, \hat O \,|\, \nu(\mathbf R_j + \mathbf T) \rangle,
\qquad \mathbf T = \mathbf n \cdot \mathbf h,\ \mathbf n \in \mathbb Z^3
$$

with `h` holding the lattice vectors as rows (tad-mctc convention). Required:

| Integral | tad-libcint name | Note |
| --- | --- | --- |
| overlap | `"ovlp"` | symmetric |
| dipole | `"j"` | about the ket's image, not symmetric |
| quadrupole | `"jj"` (packed) | as now: lower triangle, not traceless (dxtb removes the trace) |
| gradients | positions, **lattice/strain** | tblite: `gradient` and `sigma` in `get_hamiltonian_gradient` |
| gradients | exponents, coefficients | for dxtb's parameter fits, as in the molecular case |

Not needed, and these should raise for periodic structures:

- `"r0"`, `"r0r0"`: a moment about a fixed origin, summed over images, is not translation invariant (the position operator is ill-defined in a crystal).
- `"nuc"`: the sum over the nuclei's images diverges and needs Ewald summation.
- k-points: tblite and dxtb use Γ only. PySCF's Bloch-phase weights are the route if they are ever needed (see below).

### No new libcint integrals are needed

Periodicity is entirely a matter of the **driver**, not of libcint. libcint
evaluates `<i|op|j>` for whatever coordinates sit in `env`. A lattice sum
calls it with the ket's coordinates shifted by each `T`.

- PySCF does this in its own `libpbc` (`pyscf/lib/pbc/fill_ints.c`, `PBCnr2c_drv`), which receives a plain libcint function pointer such as `int1e_ovlp_sph`.
- The integrals of the `dxtb` fork are therefore enough: `ovlp`, `j`, `jj`, plus the `ip*` and `rr` derivatives already used for gradients.
- The pieces the fork removed are only needed for periodic Coulomb-type terms: 2c2e, 3c2e and 3c1e, range-separated Coulomb (`WITH_RANGE_COULOMB`), and grids. Those terms are periodic nuclear attraction, Gaussian density fitting and ERIs, which xTB does not use.

The ket-centred operators recentre automatically. libcint's `int1e_j*` use
`envs->rj` (`G1E_R_J`), the ket shell's coordinate, so shifting the ket moves
the operator centre onto the image.

tblite's convention is the same, though stored the other way round:
`multipole_cgto(cgto_j, cgto_i, ...)` takes moments about the untranslated
column atom `i` (`rpi = P - R_i`) and translates the row function. After the
sum over an inversion-symmetric image set, this equals the ket-image fold
(verified numerically, see below).

## How tblite does it

Steps:

1. A global cutoff, `sqrt(2*intcut/min_alpha)` with `intcut` from the accuracy, **capped at 40 bohr** (`basis/type.f90: get_cutoff`).
2. Image rings per axis from the interplanar spacing (`cutoff.f90: get_translations`; tad-mctc's `count_image_rings_mctclib` is a port).
3. A half adjacency list of `(iat, jat <= iat, T)` within the cutoff, dropping only `r = 0` (`adjlist.f90`).
4. `get_hamiltonian`: for each list entry, one `multipole_cgto` call. The block goes into `(j, i)` and, if `iat != jat`, the transpose built with `shift_operator` into `(i, j)`. Self-images `iat == jat, T != 0` are added once, but both `T` and `-T` are in the list.
5. Strain: `sigma += 0.5 (vec ⊗ dG + dG ⊗ vec)` per pair.

Improvements worth making (tblite's periodic code is not final):

- **Screening per pair.** Radii are added in quadrature per pair of atoms, as tad-libcint already does molecularly (`_screened_pairs`), instead of one radius from the global smallest exponent. A diffuse function on one element then no longer inflates the cutoff for every pair.
- **No silent cap.** The 40 bohr cap truncates the sum without warning for very diffuse bases.
- **Symmetry.** The `*_lat` routines loop over all `iat`, `jat` and `T`. For a symmetric operator, `A(-T) = A(T)ᵀ` halves the work, as the H0 builder already does.
- **Analytic strain.** Lattice gradients come from the same derivative integrals (below). PySCF computes the overlap/kinetic strain derivatives by **finite differences** (`pbc/grad/rks_stress.py: get_ovlp`).

## How PySCF does it

`cell.pbc_intor` → `intor_cross` (or `_intor_cross_screened`) → `libpbc.PBCnr2c_drv`:

- `conc_env(cell1, cell2)` duplicates atoms and shells, so that the ket shells refer to their own atom copies. Shifting those coordinates must not move a bra on the same atom.
- `Ls = cell.get_lattice_Ls(rcut)`, with `rcut` estimated from `cell.precision`.
- OpenMP over ket shells, each thread with its own copy of `env`. For each image, the ket atom's coordinates are shifted, libcint is called, and the result is added with the weight `expkL[k, L] = exp(i k·L)`.
- At Γ all weights are 1 and the result is taken real. For `kderiv`, the weights become `i L exp(i k·L)`: the same driver with other weights.
- The screened variant takes a per-shell-pair image list (`NeighborListOpt`).

## Prototype and results

`periodic_prototype.py` uses the **unchanged** library. It replicates the
cell's atoms at all integer shifts (ghost supercell), calls `int1e`, keeps
the rows of the central cell and folds the image columns onto their central
AOs. The test case is a triclinic two-atom cell with s and p shells,
343 images, and a 14 bohr cutoff from libcint's own `exp(-60)` threshold.

| Check | Result |
| --- | --- |
| overlap vs PySCF `cell.pbc_intor("int1e_ovlp")` (precision 1e-16) | 2e-16 |
| `"j"`, `"jj"`: ket-image fold vs tblite's convention (bra translated) | 2e-16 |
| overlap symmetric / `"j"` not | 1e-17 / 2e-2 |
| atom wrapped into the neighbouring cell | 1e-17 |
| dE/dR and dE/dh (autograd) vs finite differences, ovlp and `"j"` | ≤ 2e-10 |
| strain derivative `σ = (dE/dh)ᵀ h + (dE/dR)ᵀ R` vs strained geometries | 1e-10 |

The approach is correct, and the lattice gradient comes for free. With image
positions `R_j + n·h` formed in torch, autograd drives the existing
`_Int2c` backward (including `_ket_operator_term` for the moving operator
centre) with no new derivative code.

It does not scale, though. The block is dense, `nao × nimg·nao` (the
prototype even computes `(nimg·nao)²`). The cutoff fixes the number of
images (tad-mctc `build_periodic_shifts`):

| Cubic cell | Cutoff | Images |
| --- | --- | --- |
| 20 bohr | 40 bohr | 125 |
| 10 bohr | 40 bohr | 729 |
| 5 bohr | 40 bohr | 4913 |

## Proposed design

### C: a lattice-sum driver

Add a lattice-sum driver next to `GTOint2c` in `libs/gto/fill_int2c.c`: a
real, Γ-first port of `PBCnr2c_drv`.

- Inputs: `GTOint2c`'s plus the translations `Ls` (`nimg × 3`) and **weights** (`nw × nimg`), giving `nw` summed matrices of size `nao × nao`.
  - Weights of 1 give the integral.
  - Weights `n_T[a]` give the lattice derivative.
  - Complex `exp(i k·T)` would give k-points later.
- Duplicated ket atoms in `atm/bas/env` (PySCF's `conc_env`). `Layout` produces them.
- Screening: a CSR list of images per shell pair, from a periodic pair search (radii in quadrature).
- The hermitian shortcut stays valid for symmetric operators: an inversion-symmetric image set gives `Σ_T A(T) = (Σ_T A(T))ᵀ`.

Memory stays `nw × ncomp × nao²`, independent of the number of images.

### Python

- `library.int2c`: optional `translations` and `weights`.
- `Inputs`: add the lattice (differentiable). The integer shifts and the `periodic` mask belong to the structure function and its `_Block`. They are construction data, like the screened pairs, and carry no gradient.
- `int1e`: drop the `NotImplementedError`, and reject `r0`/`r0r0`/`nuc` for periodic structures.
- `_Block.pairs`: `(bra shell, ket shell)` becomes `(bra shell, ket shell, image)`, or a CSR per pair.
- Gradients:
  - Positions: as now, with lattice-summed derivative integrals. The bra term goes to atom `i` and the ket term (including the operator term) to atom `j`.
  - Lattice: `dE/dh_a = Σ_T n_T[a] · (ket term of image T)`, the same derivative integrals with weights `n_T` (3 weights × 3 directions).
  - Exponents and coefficients: unchanged, with summed integrals.
  - Forward mode (`_tangent_*`) needs the matching lattice tangent.
- `_PerStructure`/vmap: the lattice is one more per-structure tensor. tad-mctc already rejects batches that mix molecules and cells.

The ghost-supercell prototype becomes the reference implementation in the tests.

### tad-mctc

`neighbor.pairs_within_radii` raises for periodic structures. A periodic
variant returning `(i, j, shift)` is needed, built on
`build_periodic_shifts` or `NeighborList`. It must respect the half-list
rule: one entry stands for `(i, j, +n)` and `(j, i, -n)`, including
self-images, and shifts along non-periodic axes must be dropped (see
`build_periodic_shifts`' docstring).

## Open decisions

1. Γ only, or design the weights complex from the start for k-points?
2. Does the periodic pair search live in tad-mctc (preferred, reusable by dxtb) or here?
3. Cutoff: libcint's fixed `exp(-60)`, as molecular screening uses now, or an accuracy parameter like tblite's `acc`? And whether to keep a cap at all.
4. Should 1D/2D `periodic` masks be supported from the start? (They only change which shifts are generated.)
