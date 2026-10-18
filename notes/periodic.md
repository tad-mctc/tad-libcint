# Periodic integrals: investigation

Status: investigation (branch `feature/periodic`), no library changes yet.
Prototype: [`periodic_prototype.py`](periodic_prototype.py). It needs tad-mctc ≥ 0.10
(`neighbor.images`), PySCF with `pbc`, and the in-tree build on the path, e.g.

```console
PYTHONPATH=src:../tad-mctc/src python notes/periodic_prototype.py
```

## Decisions

1. **Γ point and k-points.** Complex Bloch weights from the start, with a real fast path at Γ.
2. **The periodic pair search lives in tad-mctc**, where dxtb can reuse it.
3. **An accuracy setting, as in other quantum chemistry programs** (PySCF `precision`, tblite `acc`, CP2K `EPS_PGF_ORB`). It replaces the fixed `exp(-60)` and the silent 40 bohr cap. The default is conservative, chosen by benchmark (see [Precision benchmark](#precision-benchmark)).
4. **1D and 2D periodicity are supported** through the structure's `periodic` mask.
5. **k-points as in PySCF:** Cartesian, in 1/bohr.
6. **Batches: each structure has its own k-point set.**

## What is needed

The consumer is dxtb, which follows tblite. tblite's periodic H0
(`xtb/h0.f90: get_hamiltonian`) and its `*_lat` integral routines
(`integral/native/integrals.f90`) build real-space lattice sums of the
molecular integrals at Γ. With k-points, these generalise to Bloch sums:

$$
A_{\mu\nu}(\mathbf k) = \sum_{\mathbf T} e^{i \mathbf k \cdot \mathbf T}
\langle \mu(\mathbf R_i) \,|\, \hat O \,|\, \nu(\mathbf R_j + \mathbf T) \rangle,
\qquad \mathbf T = \mathbf n \cdot \mathbf h,\ \mathbf n \in \mathbb Z^3
$$

Here `h` holds the lattice vectors as rows (tad-mctc convention), `n` runs
over the periodic axes only, and the phase convention is PySCF's (verified).
At Γ, `A` is real.

| Integral | tad-libcint name | Note |
| --- | --- | --- |
| overlap | `"ovlp"` | Hermitian at every k |
| dipole | `"j"` | about the ket's image, not Hermitian |
| quadrupole | `"jj"` (packed) | as now: lower triangle, not traceless (dxtb removes the trace) |
| gradients | positions, **lattice/strain** | tblite: `gradient` and `sigma` in `get_hamiltonian_gradient` |
| gradients | exponents, coefficients | for dxtb's parameter fits, as in the molecular case |

Not needed, and these should raise for periodic structures:

- `"r0"`, `"r0r0"`: a moment about a fixed origin, summed over images, is not translation invariant (the position operator is ill-defined in a crystal).
- `"nuc"`: the sum over the nuclei's images diverges and needs Ewald summation.

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
- OpenMP over ket shells, each thread with its own copy of `env`. For each image, the ket atom's coordinates are shifted, libcint is called, and the result is added with the weight `expkL[k, L] = exp(i k·L)`. The integrals of all images of a shell pair are buffered and contracted with the phases in one matrix product, so the libcint cost is shared by all k-points.
- At Γ all weights are 1 and the result is taken real. For `kderiv`, the weights become `i L exp(i k·L)`: the same driver with other weights.
- The screened variant takes a per-shell-pair image list (`NeighborListOpt`).
- **Accuracy:** `cell.precision` (default 1e-8) → `_estimate_rcut(alpha, l, c, precision)` per shell. This solves `c² r^(2l+2) e^(-α r²/2) ≈ precision` from the overlap of a function with its image, with a penalty factor for kinetic integrals. `rcut_by_shells` gives per-shell radii, and `estimate_rcut` takes the maximum.
- **Low dimensions:** with `cell.dimension = 2`, the default `low_dim_ft_type` still sums the 1e integrals over all **three** lattice vectors (`tools/pbc.py: get_lattice_Ls`, "necessary for atoms near the boundary") and relies on a large vacuum. Only `low_dim_ft_type="inf_vacuum"` restricts the sum to the periodic axes. With a short vacuum vector the two differ: 1.5e-3 in the prototype's overlap. We follow the `inf_vacuum` semantics, in which the mask decides.

## Prototype and results

`periodic_prototype.py` uses the **unchanged** library. It replicates the
cell's atoms at all integer shifts (ghost supercell), calls `int1e`, keeps
the rows of the central cell and folds the image columns onto their central
AOs, with Bloch phases for k-points. The test case is a triclinic two-atom
cell with s and p shells, 343 images in 3D, and a 14 bohr cutoff from
libcint's own `exp(-60)` threshold.

| Check | Result |
| --- | --- |
| Γ overlap vs PySCF `cell.pbc_intor("int1e_ovlp")` (precision 1e-16) | 2e-16 |
| `"j"`, `"jj"`: ket-image fold vs tblite's convention (bra translated) | 2e-16 |
| Γ overlap symmetric / `"j"` not | 1e-17 / 2e-2 |
| atom wrapped into the neighbouring cell | 1e-17 |
| dE/dR and dE/dh (autograd) vs finite differences, ovlp and `"j"` | ≤ 2e-10 |
| strain derivative `σ = (dE/dh)ᵀ h + (dE/dR)ᵀ R` vs strained geometries | 1e-10 |
| overlap at 3 k-points vs PySCF, 3D / 2D / 1D (343 / 49 / 7 images) | 2e-16 each |
| overlap Hermitian at every k | 1e-17 |
| `"j"` at k-points: dE/dh vs finite differences (k fixed in fractional coordinates) | 3e-11 |
| `"j"` at Cartesian k held fixed (phases depend on h): dE/dh vs finite differences | 5e-11 |
| `"j"` at k = 2π κ h⁻ᵀ built from the lattice: dE/dh vs fixed fractional k | 3e-17 |
| n-weighted Γ overlap `Σ_T n_T S(T)` antisymmetric | 1e-17 (\|W\| ≈ 7e-3) |

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

### API

```python
int1e(name, basis, structure, *, kpoints=None, precision=1e-10, ...)
```

- **Lattice and mask.** `structure.lattice` and `structure.periodic` come from tad-mctc's `Structure`, which already fills in the mask.
- **`kpoints`.** As in PySCF: **Cartesian**, in 1/bohr, shape `([batch,] nk, 3)`, with the phase `exp(i k·T)`. `None` means Γ: a real result of shape `([batch,] *comp, nao, nao)`. Otherwise the result is complex, `([batch,] nk, *comp, nao, nao)`.
  - **Phases are formed in torch** from `kpoints` and `T = n·h`, so they are differentiable in both. Holding Cartesian k fixed, a strain changes the phase too: `∂/∂h_ab e^{ik·T} = i n_a k_b e^{ik·T}`.
  - **The phase term is cheap.** It needs the n-weighted Bloch sums of the integral *values*, `Σ_T n_a e^{ik·T} A(T)`: the same libcint calls with three more weight rows, no derivative integrals. The same sums give `∂A/∂k = i Σ_T T e^{ik·T} A(T)`, which is PySCF's `kderiv=1`. The prototype checks this term: with Cartesian k fixed, the lattice gradient differs from the one at fixed fractional k by 5e-3, and both match finite differences.
  - **Monkhorst–Pack grids:** a user builds them from the lattice, `k = 2π κ h⁻ᵀ` (e.g. with a helper). Autograd then sees fractional k held fixed under strain, the physically meaningful stress (verified in the prototype with fractional k).
  - **Validation:** k-point components along non-periodic axes must be zero.
  - **Batches:** each structure has its own k-point set. Different numbers of k-points are zero-padded in the output, as AOs are padded now. The input is either a list of `(nk_b, 3)` tensors or a padded tensor with a mask (to decide when implementing).
- **`precision`.** An absolute threshold on the neglected integrals; see [Precision benchmark](#precision-benchmark) for the estimator and the default.
  - It also sets libcint's own primitive screening through `env[PTR_EXPCUTOFF] = -ln(precision)`. `g1e.c` uses it instead of the default 60, clamped to at least `MIN_EXPCUTOFF` (20, set in `libs/CMakeLists.txt`).
  - Molecular screening (`_SCREENING`) uses the same parameter, so molecules and cells behave alike.

### C: a lattice-sum driver

Add a lattice-sum driver next to `GTOint2c` in `libs/gto/fill_int2c.c`: a
port of `PBCnr2c_drv`.

- Inputs: `GTOint2c`'s plus the translations `Ls` (`nimg × 3`) and **weights** (`nw × nimg`), giving `nw` summed matrices of size `nao × nao`.
  - Real weights of 1 give the Γ integral.
  - Real weights `n_T[a]` give the Γ lattice derivative.
  - Complex `exp(i k·T)` give the k-points, and `n_T[a] exp(i k·T)` both their ket-derivative lattice terms (on derivative integrals) and the phase terms (on values).
  - As in PySCF, buffer the images of a shell pair and contract them with the weights in one product, so the libcint calls are shared by all k-points and derivative weights.
  - Keep a real path for Γ, since complex arithmetic doubles memory and work.
- Duplicated ket atoms in `atm/bas/env` (PySCF's `conc_env`). `Layout` produces them.
- Screening: a CSR list of images per shell pair, from tad-mctc's periodic pair search (see below).
- **Symmetric shortcut.** For a symmetric operator, `A(-T) = A(T)ᵀ`, so over an inversion-symmetric image set:
  - `Σ_T e^{ik·T} A(T)` is **Hermitian**, and real symmetric at Γ.
  - `Σ_T n_T e^{ik·T} A(T)` is **anti-Hermitian**, and real antisymmetric at Γ (checked above).

  The lower-triangle mirror (and `_grad_positions`' `symmetric` shortcut) must conjugate the mirrored triangle and, for the n-weighted derivatives, also flip its sign.
- **No libcint optimizer, or none built for the central cell.** A `CINTOpt` caches geometry-dependent pair data (see the comment in `library.int2c`). Once the ket's coordinates are shifted inside `env`, an optimizer built beforehand is stale and gives silently wrong numbers. PySCF's `intor_cross` passes a NULL optimizer for this reason. Measure the cost of going without, or build the pair data per image.

  The ghost-supercell prototype cannot catch this: every image there is a real atom with its own coordinates.

Memory stays `nw × ncomp × nao²`, independent of the number of images.

### Python

- `library.int2c`: optional `translations` and `weights`, real or complex.
- `Inputs`: add the lattice and the k-points, both differentiable. The integer shifts, the `periodic` mask and the precision belong to the structure function and its `_Block`. They are construction data, like the screened pairs, and carry no gradient.
- `int1e`: drop the `NotImplementedError`, and reject `r0`/`r0r0`/`nuc` for periodic structures.
- `_Block.pairs`: `(bra shell, ket shell)` becomes `(bra shell, ket shell, image)`, or a CSR per pair.
- Gradients of a real loss of a complex output: the incoming gradient is complex, and the contraction takes the real part (`Re Σ conj(g) · dA`).
  - Positions: as now, with Bloch-summed derivative integrals. The bra term goes to atom `i` and the ket term (including the operator term) to atom `j`.
  - Lattice, two terms:
    - **Ket term:** `Re Σ_T n_T[a] e^{ik·T} · (ket term of image T)`, the same derivative integrals with weights `n_T e^{ik·T}` (3 weights × 3 directions × nk).
    - **Phase term,** for Cartesian k held fixed: `Re Σ_T i n_T[a] k_b e^{ik·T} A(T)`, the n-weighted sums of the values contracted with the incoming gradient.

    Along non-periodic axes `n = 0`, so a placeholder vacuum vector gets a zero gradient by construction.
  - k-points: `Re Σ_T i T_b e^{ik·T} A(T)`, from the same n-weighted value sums.
  - Exponents and coefficients: unchanged, with summed integrals.
  - Forward mode (`_tangent_*`) needs the matching lattice tangent.
- `_PerStructure`/vmap: the lattice is one more per-structure tensor. tad-mctc already rejects batches that mix molecules and cells. A batch shares one k-point set, or pads it per structure (to decide when implementing).

The ghost-supercell prototype becomes the reference implementation in the tests.

### tad-mctc: periodic pair search

`neighbor.pairs_within_radii` raises for periodic structures. Extend it, or
add a sibling, to return `(i, j, shift)` for pairs within their radii added
in quadrature, `|R_j + n·h - R_i|² ≤ r_i² + r_j²`. Requirements:

- Built on `build_periodic_shifts` or `NeighborList` (ghost pool), honouring each structure's `periodic` mask. Shifts along non-periodic axes must be dropped by the consumer, because the cutoff check alone does not remove them for short placeholder vectors (see `build_periodic_shifts`' docstring and the prototype's `images`).
- Half list as in `NeighborList`: one entry stands for `(i, j, +n)` and `(j, i, -n)`. The self-images `i == j, n ≠ 0` appear once per `±n` pair, and the on-site `n = 0` pairs are added by the consumer, as now.
- Batched structures and `vmap` behave as in the molecular version (construction side, no gradient).

## Precision benchmark

[`periodic_precision.py`](periodic_precision.py) uses the GFN2-xTB basis,
STO-NG expanded from dxtb's `gfn2-xtb.toml` and `sto-ng` tables (`DXTB_SRC`
points to dxtb's `src`).

Cells:

- **In sample:** diamond, silicon and LiF (3D), graphene (2D, 20 bohr vacuum vector) and polyethylene (1D).
- **Out of sample**, added after the margin below was chosen: NaCl, MgO, and water at the density of ice (3D).

Method:

- Every block `<μ(R_i)|op|ν(R_j + T)>` is evaluated once, within the exp(-60) reference cutoff (up to 59319 images for LiF).
- For each estimator and precision p, the blocks it would skip are dropped.
- Error: `max_μν Σ_T |dropped A_T|`, a bound valid at every k-point, for `ovlp`, `j` and `jj`.
- Cost: the share of the reference's (atom pair, image) blocks that are kept.

Estimators (K = -ln p):

| Estimator | Cutoff |
| --- | --- |
| gauss | per atom `sqrt(K/α_min)`, pairs in quadrature: the Gaussian product factor below p (tad-libcint's molecular screening, there K = 60) |
| tblite | global `sqrt(2K/α_min)`, capped at 40 bohr (`get_cutoff` with `intcut = K`) |
| pyscf | global maximum of PySCF's `_estimate_rcut(α, l, c, p)` (`cell.rcut`) |
| bound⟨n⟩ | per element pair: largest bound over the primitive pairs (sketched below), times the number of images within one decay length `1/(2μR)` at R (from the cell's volume, area or length), compared with p/10 |

The primitive-pair bound for an operator of rank n about the ket is

    |c_a c_b| f_a f_b (π/(a+b))^1.5 (bR/(a+b) + w)^l_a (aR/(a+b) + w)^(l_b+n) exp(-ab/(a+b) R²)

with libcint's angular factors `f` and the product width `w = 1/sqrt(2(a+b))`.

Results over all eight cells. The error columns give the worst error/p
over the cells (≤ 1 is safe), first for the five in-sample cells, then for
the three out-of-sample ones.

| p | Estimator | Kept | Worst err/p, all three integrals (in / out) | Worst err/p, overlap | Worst absolute error |
| --- | --- | --- | --- | --- | --- |
| 1e-8 | gauss | 17–56% | 7e4 / 4e4 | 2e2 | 7e-4 |
| 1e-8 | tblite | 14–61% | 3e5 / 2e7 | 5e4 | 0.16 |
| 1e-8 | pyscf | 30–61% | 6e2 / 8e2 | 2 | 8e-6 |
| 1e-8 | bound0 | 22–60% | 20 / 30 | 0.1 | 3e-7 |
| 1e-8 | **bound2** | 29–61% | **0.08 / 0.04** | 0.003 | 8e-10 |
| 1e-10 | gauss | 23–61% | 1e5 / 8e4 | 6e2 | 1e-5 |
| 1e-10 | tblite | 14–61% | 3e7 / 2e9 | 5e6 | 0.16 |
| 1e-10 | pyscf | 40–71% | 6e2 / 9e2 | 1 | 9e-8 |
| 1e-10 | bound0 | 29–61% | 20 / 50 | 0.2 | 6e-9 |
| 1e-10 | **bound2** | 36–71% | **0.04 / 0.08** | 0.003 | 8e-12 |
| 1e-12 | gauss | 31–70% | 1e5 / 1e5 | 4e2 | 1e-7 |
| 1e-12 | tblite | 14–71% | 3e9 / 2e11 | 5e8 | 0.16 |
| 1e-12 | pyscf | 50–80% | 8e2 / 9e2 | 2 | 9e-10 |
| 1e-12 | bound0 | 36–71% | 20 / 20 | 0.3 | 3e-11 |
| 1e-12 | **bound2** | 47–75% | **0.1 / 0.03** | 0.002 | 1e-13 |

Findings:

- **The Gaussian product factor alone is not a precision.** It ignores the coefficients, the polynomial prefactors, the moment operators (which grow with R) and the sum over images. Its errors exceed p by up to 1e5×.
- **tblite's 40 bohr cap is a real accuracy limit for diffuse s and p shells.** The error stays fixed whatever the accuracy:
  - LiF (Li α_min = 0.020): 1e-5 in the overlap and 3e-3 in the quadrupole.
  - NaCl (Na 3p α_min = 0.014): 5e-4 in the overlap and **0.16** in the quadrupole.

  These are errors in integral matrix elements; their effect on tblite's energies was not measured. tblite's rule is otherwise the Gaussian product rule with a global exponent.
- **PySCF's estimate is safe for the overlap** (within 2×), but **not for the moments.** It is built for overlap and kinetic integrals, and fails for `j`/`jj` in the ionic cells (LiF, NaCl, MgO) by up to 900×.
- **The per-pair bound with the operator's rank is safe in all eight cells**, usually by orders of magnitude, at the cost of PySCF's estimate or less (LiF at 1e-10: 51% kept against PySCF's 57%, which fails).
  - Its margin of 10 was chosen on the five in-sample cells, after the bound without it overshot by up to 1.5× in graphene and polyethylene (the error jumps in steps as discrete images cross the cutoff). The three out-of-sample cells stay below 0.08 p.
  - `bound0` suffices for the overlap alone.

Recommendation:

- **Estimator:** use `bound⟨n⟩` with n the highest operator rank requested in one call. dxtb computes `ovlp`, `j` and `jj` together, so n = 2.
- **Default:** `precision = 1e-10`. With the margin of the bound, the actual errors stay below 1e-11 in all eight cells, which is conservative in effect. It is tighter than PySCF's 1e-8 and comparable to tblite's `acc = 1` (`intcut = 25`, about 1e-11). `1e-12` costs 10–25% more blocks.

Not covered yet:

- Errors of energies and gradients, rather than of integral matrices.
- The derivative integrals' own truncation error.
- Other bases (GFN1's second hydrogen s shell, the q-vSZP basis of g-xTB).

The gradients are exact for the truncated sum, because value and derivatives use the same pair list. Revisit the default once dxtb runs periodic SCF.

## Remaining questions

1. **Shape of the batched k-point input:** a list of `(nk_b, 3)` tensors, or a padded tensor with a mask?
2. **A Monkhorst–Pack helper:** provide one here (so autograd sees fractional k fixed under strain), or leave it to tad-mctc or dxtb?
