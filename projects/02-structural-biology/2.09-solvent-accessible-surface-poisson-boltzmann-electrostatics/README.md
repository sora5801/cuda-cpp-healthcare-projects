# 2.9 — Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **Educational only — not for clinical use.** Continuum electrostatics in
> reduced (illustrative) units on a *synthetic* molecule. Teaches the CUDA
> red-black stencil pattern; it is not a calibrated pKa/binding calculator.

## Summary

This project computes the **electrostatic potential of a molecule in salty
water** by solving the **linearized Poisson–Boltzmann equation (LPBE)** on a 3-D
grid, and reports the molecule's **solvent-accessible surface area (SASA)**. The
potential is found by **red-black Gauss–Seidel relaxation** of a 7-point finite-
difference stencil — the canonical way GPU PB solvers (APBS, DelPhi-GPU)
parallelize the solve. A serial CPU reference solves the same problem and the two
fields are checked for agreement.

## What this computes & why the GPU helps

A protein's charges live in a low-dielectric interior (ε≈2) surrounded by
high-dielectric, ion-screened water (ε≈80). The LPBE,

```
  ∇·(ε∇φ) − ε_w κ² φ = −ρ/ε0 ,
```

gives the mean-field potential φ from which pKa shifts, binding electrostatics,
and surface (zeta) potentials follow. Discretized on an `n³` grid it becomes a
huge sparse linear system solved by **iterative relaxation** — the **bottleneck**
is the 3-D finite-difference sweep, repeated hundreds of times. Each sweep
updates every grid cell from its six neighbours; with **red-black colouring**,
all cells of one colour are independent and update **in parallel**, one GPU
thread per cell. On the sample the GPU runs the 600-sweep solve about **9×**
faster than the serial CPU sweep (a teaching artifact — the edge grows with grid
size). See [THEORY.md](THEORY.md).

## The algorithm in brief

- **Build the grids** from the atoms: a sharp dielectric map (low inside any
  atom's radius, high in solvent), a screening map κ²(r) (0 inside the protein),
  and a charge source ρ(r) (each atom's charge on its nearest grid cell).
- **7-point finite-difference stencil** for `∇·(ε∇φ) − ε_w κ² φ`.
- **Red-black Gauss–Seidel**: colour cells by `(x+y+z) mod 2`; update all red
  cells, then all black — each colour is race-free and parallel.
- **SASA** by deterministic Shrake–Rupley sphere sampling (the geometric surface).
- Full derivation, complexity, and GPU mapping in [THEORY.md](THEORY.md).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3** (the
repo's ratified standard — see [`docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md)).

1. Open `build/solvent-accessible-surface-poisson-boltzmann-electrostatics.sln`.
2. Select **`Release|x64`**.
3. **Build → Build Solution** (`Ctrl+Shift+B`).

No path edits are needed: the project uses the CUDA `.props/.targets` integration
and `$(CUDA_PATH)`. A `Debug|x64` configuration (with `-G` device debug) is also
provided. Linux/CI users can build with the optional `CMakeLists.txt`.

## Run the demo

From this project folder:

```powershell
./demo/run_demo.ps1            # Windows: builds if needed, runs, verifies
```
```bash
./demo/run_demo.sh             # Linux/macOS via CMake
```

It runs the solver on the committed sample, prints the electrostatics summary +
SASA (stdout), shows timing (stderr), and diffs stdout against
[`demo/expected_output.txt`](demo/expected_output.txt). See
[`demo/README.md`](demo/README.md).

## Data

The committed sample [`data/sample/molecule.pqr`](data/sample/molecule.pqr) is a
**tiny synthetic "molecule"** — 11 atoms forming a deliberate **dipole** (net +1 e
on one lobe, −1 e on the other) plus a neutral scaffold — followed by the grid
and physics parameters. It is a `.pqr`-style format (atoms with partial **q** and
**radius**), the same information PDB2PQR produces from a real PDB. Regenerate or
resize it with `scripts/make_synthetic.py`; provenance, the file format, and the
real-data path (RCSB → PDB2PQR) are in [`data/README.md`](data/README.md) and
`scripts/download_data.*`. The data is **synthetic** and labelled as such.

## Expected output

```
2.9 -- Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
grid: 48x48x48 cells, h=0.60 A, eps_in=2.0 eps_out=80.0 kappa^2=0.1000, sweeps=600
atoms: 11  | SASA (probe=1.4 A) = 408.84 A^2
potential (kT/e): min=-0.367998  max=0.367998  center=-0.005102  sum|phi|=163.6273
phi along center x-line (8 samples): 0.0000 0.0035 0.0340 0.0486 -0.0295 -0.0516 -0.0047 0.0000
RESULT: PASS (GPU field matches CPU within tol=1.0e-09)
```

`RESULT: PASS` means the GPU field matches the CPU reference (worst cell
difference ≈ **5.6e-17**, machine precision — both run the identical red-black
arithmetic). The field is **antisymmetric** (`min = −max`) and ≈ 0 at the centre,
exactly as a symmetric dipole demands — a physical sanity check, not just
CPU==GPU. Timing is printed on **stderr** so stdout stays byte-identical for the
diff.

## Code tour

Read in this order:

1. [`src/pbe.h`](src/pbe.h) — the shared `__host__ __device__` per-cell
   relaxation (`pbe_relax_cell`); the one place the PDE is discretized.
2. [`src/main.cu`](src/main.cu) — load → build grids → CPU solve → GPU solve →
   verify → report.
3. [`src/kernels.cu`](src/kernels.cu) — `relax_color_kernel` (one thread/cell) and
   the host sweep loop (two launches per sweep).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — loader, grid builder, serial
   red-black reference, and SASA.

## Prior art & further reading

- **APBS** (<https://github.com/Electrostatics/apbs>) — the reference open-source
  PB solver; learn its multigrid backends and validation suite.
- **DelPhi** (<http://compbio.clemson.edu/delphi>) — classic finite-difference PB;
  its GPU branch parallelizes exactly this relaxation.
- **OpenMM GB force** (<https://github.com/openmm/openmm>) — the *analytic*
  Generalized-Born alternative (no grid) — a useful contrast.
- **PDB2PQR** (<https://github.com/Electrostatics/pdb2pqr>) — prepares the
  charged/radii input that PB solvers consume.

## Exercises

1. **Shared-memory tiling.** Stage each block's cells + a halo into shared memory
   so neighbour reads hit on-chip memory; measure the speed-up (see THEORY §4).
2. **Debye–Hückel boundary.** Replace the grounded `φ=0` shell with the analytic
   screened-Coulomb sum of the charges; check it on a tighter grid.
3. **Ionic strength sweep.** Run `make_synthetic.py --kappa2 0` (no salt) vs the
   default; observe how screening shortens the potential's range.
4. **Nonlinear PBE.** Add the `sinh` term with an outer Newton loop and compare to
   the linearized result for large charges.
5. **Harmonic-mean dielectric.** Use face-averaged ε between neighbouring cells
   instead of the centre cell's ε; note the change at the dielectric boundary.

## Limitations & honesty

- **Reduced-scope teaching version.** Linearized PBE, sharp van-der-Waals
  dielectric boundary, grounded-box boundary, nearest-grid-point charges, flat
  Gauss–Seidel (not multigrid), and **illustrative units** (not calibrated
  kT/e). THEORY §7 tabulates every simplification vs. production APBS/DelPhi.
- **Synthetic data.** The molecule is an abstract dipole, not a real protein; the
  numbers demonstrate the method, **not** any pKa, binding free energy, or zeta
  potential. No clinical or quantitative claim is made.
- **Timing is a teaching artifact**, never a benchmark — tiny grids are
  launch-bound; the GPU's edge grows with size (CLAUDE.md §12).
