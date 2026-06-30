# 2.23 — Protein-Ligand Interaction Energy Decomposition

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.23`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

When a drug binds a protein, only a few **residues** do most of the binding — and
they help in different ways: a charged residue forms a **salt bridge**
(electrostatics), a bulky hydrophobic residue packs against the ligand (van der
Waals). This project computes a **per-residue MM-GBSA energy decomposition**: for
each protein residue it reports its trajectory-averaged interaction energy with the
ligand, split into **electrostatic + van der Waals + Generalized-Born solvation**
components, and ranks the residues to surface the binding **hot spots**. That
ranking is what guides lead optimization and predicts where a tumour will mutate to
resist a kinase inhibitor. The evaluation over `N frames × M residues` is
embarrassingly parallel, so we map **one residue per GPU thread**. It runs on a
small **synthetic** system with a known answer, and verifies the GPU against a CPU
reference.

## What this computes & why the GPU helps

Per-residue energy decomposition (MM-GBSA per-residue) identifies which protein
residues contribute most to ligand binding, guiding lead optimization and
resistance-mutation analysis. MD trajectories provide snapshots; GPU-parallel
per-residue energy evaluation attributes a contribution to each residue. This
reveals hot-spot residues for mutational scanning and explains selectivity across
related proteins. Kinase resistance-mutation mapping in oncology is a prime
application.

**The parallel bottleneck:** the inner triple loop `M residues × F frames × L
ligand atoms` of pairwise energy evaluation — `M·F·L` independent O(1) pair
computations (each a Coulomb + Lennard-Jones + GB term with a `sqrt`/`exp`). It is
**compute-bound** and trivially data-parallel along the residue axis, so the GPU
gives every residue its own thread; the speed-up grows with real protein size
(hundreds of residues × thousands of frames).

## The algorithm in brief

- **Pairwise interaction energy** with three separated components:
  **electrostatics** (screened Coulomb), **van der Waals** (Lennard-Jones 12-6),
  and **Generalized-Born** pairwise desolvation (implicit solvent).
- **Lorentz–Berthelot** combining rules for LJ pair parameters; a distance
  **cutoff** to skip far pairs.
- **Per-residue accumulation** over all ligand atoms and all frames, then a
  trajectory average → `{elec, vdw, gb, total}` per residue.
- **Hot-spot ranking** by total energy (most favourable first).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-ligand-interaction-energy-decomposition.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-ligand-interaction-energy-decomposition.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-ligand-interaction-energy-decomposition.sln /p:Configuration=Release /p:Platform=x64
```

Both `Release|x64` and `Debug|x64` build with zero warnings. This project links
only the CUDA runtime (`cudart_static.lib`) — no extra CUDA libraries.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/complex_sample.txt`, prints the
per-residue decomposition table and hot-spot ranking, shows the GPU-vs-CPU
agreement check, and prints a timing line (on stderr).

## Data

- **Sample (committed):** `data/sample/complex_sample.txt` — a tiny, **synthetic**
  12-residue / 4-ligand-atom / 6-frame system so the demo runs offline with zero
  downloads. It embeds a known answer (an ARG salt-bridge hot spot + a LEU vdW hot
  spot).
- **Regenerate / resize:** `python scripts/make_synthetic.py [--residues N --frames F]`.
- **Real complexes:** `scripts/download_data.ps1` / `.sh` print where to obtain
  PDBbind / KLIFS / ChEMBL / ClinVar structures and how to prepare real MM-GBSA
  inputs (it never bypasses any registration).
- **Provenance, format & license:** see [data/README.md](data/README.md).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
per-residue table where **ARG41** ranks #1 (driven by a large favourable
electrostatic term, partly offset by the GB desolvation penalty) and **LEU88**
ranks #2 (pure van der Waals). The program computes the decomposition on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts they agree within **`1.0e-4 kcal/mol`** (typical `max_abs_err` ≈ `1e-15`,
since both run the same double-precision formula) — that agreement is the
correctness guarantee. Timings and the error go to stderr (not diffed).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the system, runs CPU + GPU, verifies, prints
   the table + hot-spot ranking.
2. [`src/mmgbsa.h`](src/mmgbsa.h) — the shared `__host__ __device__` physics core
   (Coulomb + LJ + GB) and the data model. **The heart of the project.**
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-residue idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **AMBER MMPBSA.py decomp** (https://ambermd.org/AmberTools.php) — the canonical
  per-residue MM-PBSA/GBSA decomposition; learn its component split and GB models.
- **gmx_MMPBSA** (https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) — MM-GBSA
  decomposition for GROMACS; great hot-spot output examples.
- **MDAnalysis** (https://github.com/MDAnalysis/mdanalysis) — trajectory I/O and
  pairwise residue–ligand contact analysis (how to make real per-frame coords).
- **ProLIF** (https://github.com/chemosim-lab/ProLIF) — interaction fingerprints
  for binding-mode decomposition; a complementary contact-based view.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs, one residue per thread** (PATTERNS.md §1, the `1.12 / 12.01`
family): the `N frames × M residues` work decomposes into M independent
accumulations, each a thread looping frames × ligand atoms and reducing in
registers — **no atomics, no shared-memory reduction**, so the result is exactly
deterministic. The per-pair physics lives in a single `__host__ __device__` header
(PATTERNS.md §2) shared by the CPU reference and the kernel. THEORY.md §4 explains
where the catalog's "cuBLAS for energy-matrix accumulation" would fit and why the
fused kernel is the right teaching choice.

## Exercises

1. **Shared-memory ligand.** The (small, read-only) ligand is re-read by every
   residue thread from global memory. Stage it into `__shared__` memory per block
   and measure the change. (THEORY §4.)
2. **All-atom residues.** Extend the data model so each residue has a variable atom
   count, and add an inner atom loop — the real decomposition. Does the hot-spot
   ranking change?
3. **The cuBLAS energy-matrix form.** Build the per-frame `M × L` pair-energy matrix
   and reduce per residue with `cublasDgemv`. Compare correctness and speed against
   the fused kernel; at what `M·L` does the matrix form win?
4. **Effective Born radii.** Replace the fixed Born radii with a per-frame GB-OBC
   calculation (an extra O(N²) pass) and observe how the `gb` component shifts.
5. **A resistance mutation.** Flip ARG41's charge to neutral (mutate it) in
   `make_synthetic.py`, rerun, and watch the hot-spot ranking collapse — the
   computational analogue of a resistance mutation.

## Limitations & honesty

- **Synthetic data.** The committed system is generated, not a real protein-ligand
  complex; every energy here is illustrative, **not** a real binding energy, and
  carries **no clinical meaning**. Labelled synthetic throughout.
- **Reduced scope.** One bead per residue (not all-atom); fixed Born radii (no
  per-frame GB-OBC); no non-polar surface-area term, no explicit water-bridge
  detection, no entropy/FEP terms. THEORY.md §7 details each simplification and the
  production approach.
- **Teaching timings only.** The tiny sample is dominated by launch/copy overhead;
  the timing line is a teaching artifact, never a benchmark claim (CLAUDE.md §12).
- **Not for any real decision.** Educational study material only — not for
  diagnosis, treatment, or lead-optimization use (CLAUDE.md §1, §8).
