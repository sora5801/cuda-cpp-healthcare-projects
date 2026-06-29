# 1.27 — MM-GBSA / MM-PBSA Rescoring

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.27`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

**MM-GBSA rescoring** estimates how tightly a drug-like **ligand** binds a
**protein** by averaging a binding-energy formula over many **snapshots** of a
molecular-dynamics trajectory. This project implements a compact, heavily
commented **teaching version**: for each snapshot it sums the van der Waals,
Coulomb, and **Generalized-Born** solvation energies over all receptor–ligand
atom pairs, then averages the per-snapshot ΔG into one binding free-energy
estimate. The snapshots are completely independent, so the GPU evaluates them in
parallel — **one thread per snapshot** — calling the *exact same* energy function
the CPU reference uses, which makes the GPU result verifiable to machine
precision. The committed input is **synthetic** (a small charged pocket and a
ligand that drifts out of it), engineered so the energy visibly climbs toward the
unbound limit — a built-in sanity check. It is study material, **not** a tool for
real affinity prediction.

## What this computes & why the GPU helps

MM-GB(PB)SA computes binding free energies as the MM interaction energy plus solvation free energy (implicit solvent GB or PB), minus entropic terms, from snapshots along an MD trajectory. It is the standard high-throughput rescoring step after docking, offering >10× better accuracy than scoring functions with ~1000× less cost than FEP. GPU-accelerated MD (pmemd.cuda) generates the required trajectory snapshots rapidly; gmx_MMPBSA post-processes GROMACS trajectories. The solvation GB/PB solvers can also be GPU-accelerated.

**The parallel bottleneck:** evaluating the per-snapshot energy. Each snapshot's
energy is an `O(R·L)` sum over receptor–ligand atom pairs, and the snapshots are
**mutually independent** — no frame needs any other. With thousands of frames
(times many candidate ligands in real work), that pair-sum evaluation dominates
the post-MD runtime. The GPU assigns **one thread per snapshot** so all frames are
rescored at once (docs/PATTERNS.md §1, the "independent jobs" pattern). The tiny
ensemble-mean reduction stays on the host.

## The algorithm in brief

Molecular mechanics energy decomposition, Generalized Born (GB) implicit solvent, Poisson-Boltzmann (PB) numerical solver, normal-mode / quasi-harmonic entropy estimation, interaction entropy method, per-residue energy decomposition.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/mm-gbsa-mm-pbsa-rescoring.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/mm-gbsa-mm-pbsa-rescoring.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\mm-gbsa-mm-pbsa-rescoring.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PDB-bind (http://www.pdbbind.org.cn); CASF-2016 (http://www.pdbbind.org.cn/casf.php); ChEMBL activity data (https://www.ebi.ac.uk/chembl/); AMBER MM-GBSA tutorial datasets (https://ambermd.org/tutorials/).

## Expected output

Success looks like `demo/expected_output.txt`:

```
1.27 -- MM-GBSA / MM-PBSA Rescoring
Rescoring 1 complex: receptor=3 atoms, ligand=2 atoms, snapshots=6
per-snapshot dG (kcal/mol):
  frame  0 : dG =     7.2021
  ...
  frame  5 : dG =     7.9640
MM-GBSA dG_bind (ensemble mean) = 7.7731 kcal/mol
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

The program computes the per-snapshot ΔG and the ensemble mean on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts they agree within `1e-6` kcal/mol — that agreement (printed to stderr as
`max_abs_err`) is the correctness guarantee. On the synthetic sample the ligand
unbinds frame by frame, so ΔG climbs toward the bare entropy term (`8.0`); see
[`demo/README.md`](demo/README.md) for how to read it.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

AMBER MMPBSA.py (https://ambermd.org/AmberTools.php) — reference MM-GBSA/PBSA implementation; gmx_MMPBSA (https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) — GROMACS compatibility layer; NAMD MMPBSA (https://www.ks.uiuc.edu/Research/namd/) — NAMD-based MM-PBSA; OpenMM MMGBSA (verify URL) — Python MM-GBSA workflow.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs / one-thread-per-snapshot** (docs/PATTERNS.md §1, exemplified
by flagship `1.12`). Every MD snapshot is rescored by its own GPU thread via a
grid-stride loop; the per-element physics lives in one shared
`__host__ __device__` function (`snapshot_dg()`) so the CPU reference and the GPU
kernel run identical arithmetic (PATTERNS.md §2). No shared memory, no atomics —
each thread writes one independent result. (Production pipelines additionally run
the MD itself on the GPU via `pmemd.cuda`, and can offload the GB/PB solvation
solver to custom CUDA; here we focus on the rescoring step.)

## Exercises

1. **Scale the trajectory.** Run `python scripts/make_synthetic.py --snapshots
   4096`, rerun, and watch the stderr timing: at what `S` does the GPU kernel beat
   the CPU reference? (Regenerate `expected_output.txt` afterward — the per-frame
   values change.)
2. **Move the receptor into `__constant__` memory.** It is read by every thread
   and never written (THEORY §4). Add a fixed-size `__constant__ Atom c_receptor[]`
   path for small receptors and measure the difference. Where does it stop fitting?
3. **Add the nonpolar SA term.** Real MM-GBSA adds `γ·SASA + b`. Approximate each
   atom's exposed surface and add the term; compare the ranking of the frames.
4. **Per-residue decomposition.** Instead of one ΔG scalar, accumulate the pair
   energies into a per-receptor-atom array to find the "hot-spot" atom that
   contributes most to binding.
5. **Block-size sweep.** Try `THREADS_PER_BLOCK` ∈ {64, 128, 256, 512} and plot
   the kernel time; relate what you see to register pressure and occupancy.

## Limitations & honesty

This is a **reduced-scope teaching version** (CLAUDE.md §13), not a real
affinity predictor. Be explicit about what is simplified:

- **Synthetic data.** The committed complex is generated, not a real structure;
  its energies are **not** measured affinities and have **no** predictive or
  clinical validity. It is labeled synthetic everywhere.
- **Rigid receptor, single end-point.** Real MM-GBSA computes
  `ΔG = G_complex − G_receptor − G_ligand` with both partners flexible; we hold
  the receptor rigid and compute only the receptor–ligand cross interaction.
- **GB only, no SA, constant entropy.** We implement the GB polar cross term but
  omit the nonpolar surface-area term, use simple fixed Born radii (not OBC/GBn2),
  do not offer the Poisson-Boltzmann solver, and fold `−T·ΔS` into one constant
  instead of a normal-mode/quasi-harmonic estimate. See [THEORY.md](THEORY.md) §7
  for how production tools (`MMPBSA.py`, `gmx_MMPBSA`, NAMD, OpenMM) do each of
  these properly.
- **Teaching timings, not benchmarks.** The printed millisecond figures are a
  didactic artifact; on this tiny sample the GPU is launch/copy-bound and slower
  than the CPU — the parallel advantage only appears as snapshots scale (§4).
