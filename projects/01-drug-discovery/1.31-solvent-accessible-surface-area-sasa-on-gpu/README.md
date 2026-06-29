# 1.31 — Solvent-Accessible Surface Area (SASA) on GPU

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.31`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

The **solvent-accessible surface area (SASA)** of a molecule is the area of its
surface that a rolling water molecule can actually touch. It drives solvation
energy, buried-interface analysis between proteins, and the surface term in
implicit-solvent (GB/SA) models. This project computes SASA with the classic
**Shrake–Rupley** algorithm: sprinkle a fixed set of test points over each atom's
probe-inflated sphere and count how many are *not* buried inside a neighbouring
atom. Because every atom is an independent job, it maps cleanly onto the GPU —
**one thread per atom**. The GPU result is checked against a plain-C++ CPU
reference; the two share the exact same per-atom math, so they agree to the exact
integer count.

## What this computes & why the GPU helps

SASA measures the protein or ligand surface area accessible to solvent, used in
solvation energy estimation, buried surface analysis for protein–protein
interfaces, and GB implicit-solvent models. The Shrake–Rupley algorithm uses a
grid of test points per atom — embarrassingly parallel over atoms. The GPU
implementation tests each atom's points for burial against the other atoms and
accumulates per-atom SASA in parallel. GPU-SASA matters in MM-GBSA workflows
where SASA must be evaluated for *every* trajectory snapshot (thousands of frames).

**The parallel bottleneck:** the burial test. For each atom we generate `P`
(=96) test points and test each against the other `n−1` atoms — an `O(n² · P)`
all-pairs cost that dominates the runtime and is exactly what the GPU
parallelizes: each atom's `P × (n−1)` independent distance comparisons run across
many threads at once, and neighbour coordinates are staged through **shared
memory** so the whole block reuses each global load.

## The algorithm in brief

- **Shrake–Rupley point grid** — place `P` test points on each atom's sphere of
  radius (vdW + probe); here on a **Fibonacci lattice** (uniform, table-free).
- **Burial test** — a test point is *accessible* iff it lies outside every other
  atom's inflated sphere (a squared-distance comparison).
- **Numerical surface integration** — per-atom SASA = (exposed points / `P`) ×
  4π·R²; total SASA = sum over atoms.
- **Buried SASA / interface area** — the difference of total SASA for a complex vs.
  its isolated parts (discussed in THEORY as the real-world extension).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including the LCPO analytic alternative.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/solvent-accessible-surface-area-sasa-on-gpu.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/solvent-accessible-surface-area-sasa-on-gpu.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\solvent-accessible-surface-area-sasa-on-gpu.sln /p:Configuration=Release /p:Platform=x64
```

No extra CUDA library is linked — SASA is pure compute, so only the CUDA runtime
is needed.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/molecule_sample.xyz`, prints the
total SASA and the most-exposed atoms, shows the GPU-vs-CPU agreement check, and
prints a timing line.

## Data

- **Sample (committed):** `data/sample/molecule_sample.xyz` — 27 **synthetic**
  atoms with an engineered exposure pattern, so the demo runs offline and the
  result is interpretable by eye.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` document how to obtain a
  real PDB structure (e.g. `1CRN`) and convert it to the `<element> x y z` format.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PDB protein structures (<https://www.rcsb.org>); ASA
benchmark set for validation; MD trajectory ensembles for SASA time series.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the per-atom exposed-point counts on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they match **exactly** (they are integers from the same shared function), with the
derived areas agreeing to ~1e-9 Å². That agreement is the correctness guarantee.
In the sample you can see the **central atom is buried** (absent from the
most-exposed ranking, ~0 Å²) while the **lone O/N atoms are fully exposed**
(96/96) — the known answer baked into the synthetic geometry.

## Code tour

Read in this order:

1. [`src/sasa_core.h`](src/sasa_core.h) — the shared `__host__ __device__` per-atom
   math (Fibonacci points, burial test, exposed-point count). **The science lives
   here**, compiled identically into CPU and GPU.
2. [`src/main.cu`](src/main.cu) — loads the molecule, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-atom idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (shared-memory tiled) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **FreeSASA** (<https://github.com/mittinatten/freesasa>) — fast open SASA library;
  study its Shrake–Rupley and Lee–Richards implementations and its default Bondi radii.
- **AMBER SASA via pmemd** (<https://ambermd.org>) — SASA inside MM-GBSA; the LCPO
  analytic model.
- **MDTraj SASA** (<https://github.com/mdtraj/mdtraj>) — `shrake_rupley`, a clean
  vectorized reference to compare numbers against.
- **Biopython SASA** (<https://github.com/biopython/biopython>) — `Bio.PDB.SASA`,
  a readable pure-Python Shrake–Rupley.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA threadblocks over atoms (one thread per atom) + **shared-memory tiling** of
neighbour coordinates (the classic all-pairs / N-body pattern), with the burial
decision counted in **registers** and written once per atom (no atomics — outputs
are independent). The deterministic core is an **integer count** of exposed
points, derived to a float area afterward — so CPU and GPU match exactly. See
[docs/PATTERNS.md](../../../docs/PATTERNS.md) §1 (independent jobs) and §2 (the
shared `__host__ __device__` core).

## Exercises

1. **Bigger inputs.** Generate a larger synthetic blob (edit `make_synthetic.py`
   to add more shells) and watch the GPU's relative cost fall as `n` grows — the
   `O(n²)` work finally outweighs launch overhead.
2. **A real protein.** Follow `scripts/download_data.ps1` to fetch `1CRN`, convert
   it, and compare your total SASA to MDTraj's `shrake_rupley` (expect agreement
   to within the point-count discretization).
3. **Finer surface.** Raise `N_SPHERE_POINTS` (e.g. 96 → 960) and observe the SASA
   converge; plot SASA vs. point count. What is the accuracy/runtime trade-off?
4. **A real neighbour list.** Replace the all-pairs inner loop with a uniform-grid
   (cell-list) neighbour search so each point only tests nearby atoms — turning
   `O(n²)` into ~`O(n)`. (Hint: the probe-inflated radius bounds the search.)
5. **LCPO.** Implement the analytic Linear Combination of Pairwise Overlaps model
   (THEORY §real-world) and compare its smooth, differentiable SASA to the
   point-sampled one.

## Limitations & honesty

- **Synthetic sample.** The committed molecule is an engineered test geometry, not
  a real structure; its SASA is a geometric exercise only — **no chemical or
  clinical meaning** (CLAUDE.md §8).
- **All-pairs, not a neighbour list.** For teaching clarity the burial test is
  `O(n²)`; production tools use spatial neighbour lists (Exercise 4). This is fine
  for small/medium molecules but does not scale to whole large proteins as-is.
- **Single radius model.** We use Bondi vdW radii keyed on the element's first
  letter and a single 1.4 Å probe; real tools handle united-atom vs. all-atom
  radii, polar/non-polar decomposition, and per-residue corrections.
- **Point-sampled, not analytic.** Shrake–Rupley discretizes the surface, so SASA
  has a small quantization error (~`4πR²/P` per atom) and is **not differentiable**
  — unsuitable as-is for forces in MD (where LCPO or analytic SES models are used).
- **FP64 throughout.** We use double precision so CPU and GPU match bit-for-bit on
  the geometry; FP32 would be faster but could disagree on grazing burial decisions.
