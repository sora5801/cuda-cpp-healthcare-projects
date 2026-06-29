# 1.8 — Semi-Empirical & Tight-Binding Quantum Methods

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.8`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project is a hands-on introduction to **semi-empirical / tight-binding quantum chemistry on the
GPU**. It builds the simplest member of the tight-binding family — **Hückel Molecular Orbital (HMO)
theory** for the delocalised π-electrons of planar conjugated hydrocarbons — and runs it as a **batch of
many molecules at once**. For each molecule it constructs a small model Hamiltonian from the molecule's
bond graph, **diagonalises it**, fills the molecular orbitals with electrons, and reports physical
observables a chemist cares about: the total π-electron energy, the **HOMO–LUMO gap** (a reactivity /
stability proxy), and which molecule in the batch is most reactive. The GPU does what GPUs are good at
here: **build thousands of tiny Hamiltonians in parallel and diagonalise the whole batch in one library
call**, exactly the pattern production tools (xTB, DFTB+) use to screen huge molecule libraries.

## What this computes & why the GPU helps

Semi-empirical methods (PM7, GFN2-xTB) approximate quantum mechanics at **100–10000× lower cost than DFT**
by replacing expensive integrals with empirically parameterised expressions. They bridge force fields and
full DFT, enabling **geometry optimisation and reactivity screening of drug-like molecules at scale** —
conformer ranking, tautomer enumeration, QM-based ADMET. The expensive, parallelisable steps are
**constructing each molecule's (sparse) Hamiltonian** and **diagonalising it**. Because the molecules in a
screening library are independent, a GPU can optimise **thousands of small molecules simultaneously**: one
batched eigensolve replaces thousands of serial ones.

This teaching version keeps the *pipeline* identical to the real methods but swaps the elaborate
parameter functions for the one-line Hückel rules, so the linear algebra and the GPU mapping stand out
clearly. The bottleneck we parallelise is the **per-molecule Hamiltonian build + symmetric eigensolve**.

## The algorithm in brief

- **Model Hamiltonian (Hückel/tight-binding).** One π atomic orbital per sp² carbon. `H[i][i] = α`
  (on-site energy), `H[i][j] = β` if atoms `i,j` are bonded, else `0`. We use the textbook convention
  `α = 0, β = −1`, so energies come out in units of `|β|`.
- **Diagonalise.** The eigenvalues of `H` are the molecular-orbital (MO) energies; the eigenvectors are the
  MO coefficients. CPU reference: a **cyclic Jacobi** eigensolver. GPU: cuSOLVER's **batched** Jacobi
  eigensolver `cusolverDnDsyevjBatched`, which diagonalises every molecule in the batch in one launch.
- **Fill electrons (Aufbau).** Put 2 electrons per MO from the bottom up; sum occupied energies → total
  π-energy. Read the **HOMO** (highest occupied) and **LUMO** (lowest unoccupied) for the **gap**.
- **Batched build kernel.** A custom CUDA kernel builds all the padded Hamiltonians at once (one matrix
  element per thread), reusing the *same* `tb_hamiltonian_entry()` the CPU uses so the matrices are
  bit-identical.

See [`THEORY.md`](THEORY.md) for the science, the math, complexity, the GPU mapping, and the numerics.

## Build

Requires **Visual Studio 2026** (v145 toolset, *Desktop development with C++*) and **CUDA Toolkit 13.3**
(see [`docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md)). This project links **cuSOLVER** (and cuBLAS,
its dependency) — both ship with the toolkit.

1. Open [`build/semi-empirical-tight-binding-quantum-methods.sln`](build/semi-empirical-tight-binding-quantum-methods.sln) in Visual Studio 2026.
2. Select **`Release|x64`** (or `Debug|x64`).
3. **Build** (Ctrl+Shift+B). The runnable `.exe` lands in `build/x64/Release/`.

Command-line equivalent (Developer PowerShell):

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" `
  build\semi-empirical-tight-binding-quantum-methods.sln /p:Configuration=Release /p:Platform=x64
```

A cross-platform **CMake** build is also provided (`CMakeLists.txt`) for Linux/CI; the VS solution is the
required deliverable.

## Run the demo

```powershell
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1   # Windows
```
```bash
./demo/run_demo.sh                                           # Linux (CMake build)
```

The demo builds if needed, runs on the committed sample, prints the deterministic results, shows the
GPU-vs-CPU agreement and timing, and diffs stdout against [`demo/expected_output.txt`](demo/expected_output.txt).

## Data

The committed sample [`data/sample/molecules_sample.txt`](data/sample/molecules_sample.txt) is **synthetic**:
a tiny batch of eight **textbook conjugated hydrocarbons** (ethylene, allyl, butadiene, benzene,
cyclobutadiene, hexatriene, cyclopentadienyl, naphthalene), each described purely by its π-system
**connectivity graph** (no 3-D coordinates). These molecules were chosen because their Hückel spectra are
known in **closed form**, so the demo can check itself against analytic chemistry. Regenerate it with
`python scripts/make_synthetic.py`.

Real datasets the method is used with (large, require downloading) are documented in
[`data/README.md`](data/README.md) with pointers in `scripts/download_data.ps1` / `.sh`: **ANI-1**, **QM9**,
**GMTKN55**. We do not redistribute them; the scripts print fetch instructions and never bypass
registration/licensing.

## Expected output

The GPU path's eigenvalues are checked against the CPU Jacobi reference per molecule; the demo prints
`RESULT: PASS` when they agree within tolerance `1.0e-09` (we observe `~3e-15`, machine precision —
both sides diagonalise the same double-precision matrix). A second, stronger check is **analytic**: the
printed energies match textbook Hückel values exactly — benzene `E_π = 8.000 |β|`, butadiene `4.472`,
naphthalene `13.683`, and cyclobutadiene's **zero HOMO–LUMO gap** (antiaromatic, the most reactive in the
batch). See `THEORY.md` "How we verify correctness".

```
molecule         atoms           E_pi       HOMO       LUMO        gap
ethylene             2      -2.000000  -1.000000   1.000000   2.000000
benzene              6      -8.000000  -1.000000   1.000000   2.000000
cyclobutadiene       4      -4.000000   0.000000   0.000000   0.000000
naphthalene         10     -13.683239  -0.618034   0.618034   1.236068
...
RESULT: PASS (GPU batched eigensolve matches CPU Jacobi within tol=1.0e-09)
```

## Code tour

Read in this order:

1. [`src/tight_binding.h`](src/tight_binding.h) — the **shared `__host__ __device__` core**: the one true
   `tb_hamiltonian_entry()` both CPU and GPU call, and the padding trick.
2. [`src/main.cu`](src/main.cu) — the 5-step driver: load → CPU reference → GPU batch → verify → report.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted CPU pipeline: parse the batch, build each
   Hamiltonian, **Jacobi** diagonalise, fill electrons.
4. [`src/kernels.cuh`](src/kernels.cuh) / [`src/kernels.cu`](src/kernels.cu) — the GPU path: the batched
   matrix-build kernel and the **cuSOLVER batched eigensolver** wrapper.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timing, and host I/O helpers.

## Prior art & further reading

- **[xtb](https://github.com/grimme-lab/xtb)** — the GFN2-xTB reference implementation (CPU). Study it to
  see how the toy `α/β` of Hückel become element- and distance-dependent parameter *functions* in a real
  extended-tight-binding method, and how a self-consistent charge loop is added.
- **[DFTB+](https://github.com/dftbplus/dftbplus)** — density-functional tight binding with GPU
  acceleration (eigensolves via ELPA). Shows how the same "build H, diagonalise, fill" loop scales to
  periodic systems and large biomolecules.
- **[TBLite](https://github.com/tblite/tblite)** — a lightweight, well-documented tight-binding library; a
  good next read for the integral expressions.
- **[GFN-FF / xTB-IFF](https://github.com/grimme-lab)** — a force field *derived from* tight binding; the
  bridge back to the molecular-mechanics world.

## Exercises

1. **Charges and heteroatoms.** Hückel handles N/O by shifting `α` and scaling `β` (the "Hückel
   parameters"). Add per-atom `α` and per-bond `β` to `tight_binding.h` and model pyridine or pyrrole.
2. **Bond orders & charges.** Use the MO *coefficients* (already computed by the eigensolver) to compute
   π-bond orders and atomic π-charges. Add them to the report.
3. **Bigger batches.** Generate 10⁴ random conjugated graphs in `make_synthetic.py` and watch the batched
   GPU solve pull ahead of a serial CPU loop — plot time vs. batch size.
4. **Eigenvector check.** Extend the GPU wrapper to also return eigenvectors and verify them against the
   Jacobi reference (mind the arbitrary sign/phase of each eigenvector).
5. **Self-consistency.** Add a one-step Mulliken-charge feedback into `α` (a baby version of the SCC loop
   in DFTB/GFN) and observe how the energies shift.

## Limitations & honesty

- **This is the simplest tight-binding model**, not a production method. Hückel ignores electron–electron
  repulsion, σ electrons, 3-D geometry, and self-consistency. It is the right place to *start* because the
  GPU pipeline (build → batched diagonalise → fill) is **identical** to the grown-up methods; only the
  parameter functions get more elaborate (`THEORY.md` "Where this sits in the real world").
- **The data is synthetic** — idealised connectivity graphs of textbook molecules, labelled synthetic
  everywhere. No real coordinates, no experimental data, **no clinical meaning**.
- **Timings are a teaching artifact, not a benchmark.** On this tiny batch the GPU is dominated by launch
  and library-setup overhead; the batched solver's advantage appears only at thousands of molecules.
- Energies are in **units of `|β|`** (a relative scale), not kcal/mol; converting requires the empirical
  `β` value for the system, which is exactly the kind of parameter a real method supplies.
