# 1.28 — Covalent Docking

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.28`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **covalent inhibitor** is a drug that forms a real chemical bond to a protein —
classically to the sulfur of a target **cysteine** (think KRAS-G12C, BTK,
EGFR drugs). Docking one is a two-stage problem: place the reactive *warhead* so
it can reach the cysteine, then **form the bond** and search the rest of the
flexible ligand for its best-fitting pose. This project implements that second
stage as a heavily-commented CUDA program: it scores **46 656** ligand torsion
conformations — one **GPU thread each** — against a small protein pocket, verifies
the GPU energies against a serial CPU reference to machine precision, and reports
the lowest-energy (docked) pose. All geometry and the force field are **synthetic
and didactic**; the point is the GPU pattern, not a real binding prediction.

## What this computes & why the GPU helps

Covalent inhibitors form a permanent or semi-permanent bond with a nucleophilic
residue (usually Cys, Ser, Lys, Tyr). Docking them requires two-stage sampling:
(1) non-covalent pre-reaction pose generation and (2) covalent bond geometry
enforcement with post-reaction scoring. GPU acceleration helps explore the
**expanded conformational space after covalent bond formation**. EGFR/BTK/
KRAS(G12C) covalent drug programs drive industrial interest.

**The parallel bottleneck:** once the warhead is anchored, the rest of the ligand
still has rotatable bonds (torsions). The number of conformations to score grows
**exponentially** with the number of torsions ($G^{N_\tau}$). Each conformation is
scored independently, so the search is embarrassingly parallel — **one thread per
conformation**. The kernel is compute-bound (trig + Lennard-Jones), the ideal
regime for a GPU; its edge over the CPU grows as torsions are added.

## The algorithm in brief

- **Decode** a flat conformation index → three torsion angles (mixed-radix).
- **Forward kinematics** (Rodrigues rotations) → the ligand's 3-D atom positions.
- **Score** = covalent constraint penalty (harmonic bond-length + bond-angle
  springs) + nonbonded energy (Lennard-Jones 12-6 + Coulomb) vs the fixed pocket.
- **Argmin** over all conformations → the docked pose (done deterministically on
  the host).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/covalent-docking.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/covalent-docking.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\covalent-docking.sln /p:Configuration=Release /p:Platform=x64
```

Both `Release|x64` and `Debug|x64` build with zero warnings. No extra CUDA library
is linked — only the CUDA runtime (`cudart_static.lib`).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/covalent_sample.txt`, prints the
docked pose, shows the GPU-vs-CPU agreement check, and prints a timing line on
stderr.

## Data

- **Sample (committed):** `data/sample/covalent_sample.txt` — a tiny, **synthetic**
  offline input so the demo runs with zero downloads.
- **Regenerate / scale:** `python scripts/make_synthetic.py`.
- **Full / real resources:** `scripts/download_data.ps1` / `.sh` print links (they
  never bypass any registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: CovDocker benchmark (2025, verify URL); ChEMBL covalent
inhibitor set (https://www.ebi.ac.uk/chembl/); PDB covalent complex structures
(https://www.rcsb.org); BindingDB covalent entries (https://www.bindingdb.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
1.28 -- Covalent Docking
covalent docking: 46656 conformations (3 torsions x 36 samples)
best pose: id=46401  energy=-2.347011 kcal/mol
best torsions (deg): 330.0 280.0 350.0
warhead-Sgamma bond = 1.810 A (ideal 1.810)
ligand atom[0] = (-0.500, 1.414, 0.000) A
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

The program computes the energy of every conformation on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree within `1e-6 kcal/mol` (the actual error is ~`2e-15`, i.e. machine
precision — the two paths run the same double-precision math). That agreement,
plus recovering a sensible negative-energy minimum, is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/docking.h`](src/docking.h) — the shared `__host__ __device__` physics:
   forward kinematics + the energy terms (the one true math both CPU and GPU run).
2. [`src/main.cu`](src/main.cu) — loads the problem, runs CPU + GPU, verifies,
   reports the docked pose.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the scoring kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O.

## Prior art & further reading

- **AutoDock-GPU** (https://github.com/ccsb-scripps/AutoDock-GPU) — the canonical
  GPU docking engine; has a covalent-docking mode. Learn its Lamarckian-GA search
  and per-individual GPU parallelism.
- **GNINA** (https://github.com/gnina/gnina) — CNN-scored docking with covalent
  options; learn how a learned scoring function replaces hand-tuned terms.
- **Uni-Dock** (https://github.com/dptech-corp/Uni-Dock) — a modern high-throughput
  GPU docking engine, extendable to covalent docking.
- **CovDocker** (arXiv:2506.21085, verify GitHub URL) — a 2025 deep-learning
  covalent-docking benchmark + dataset.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

*Score N independent candidates* (one thread per conformation, grid-stride loop) —
the same pattern as project 1.12 — with the per-element physics in a shared
`__host__ __device__` header so CPU and GPU agree exactly. There is an additional,
distinct **covalent-bond constraint penalty** term in the score (the thing that
makes covalent docking different), and GPU-parallel conformational sampling over
the ligand's torsion (warhead + linker) degrees of freedom. No CUDA library is
needed; the final argmin reduction is deterministic on the host.

## Exercises

1. **Add a torsion.** Bump `N_TORSIONS` to 4 in `src/docking.h` and rebuild. The
   conformation count jumps ×36 to ~1.7M — watch the GPU's relative speed-up grow
   (and update the sample/expected output). This is the curse of dimensionality
   that motivates the GPU.
2. **Break the niceness.** Move one pocket atom to within ~1 Å of the ligand's
   reachable region and observe the energy explode (`r^-12`) **and** the GPU/CPU
   `max_abs_err` blow up to ~`1e4` — exactly the FMA-amplification trap discussed
   in THEORY §5. Then fix it and explain why.
3. **On-device reduction.** Replace the host argmin with a CUB/Thrust
   `reduce`/`min_element` on the device, returning only the best `(id, energy)`.
   Keep the result deterministic — what tie-breaking guarantee do you need?
4. **Stochastic search.** Swap the dense grid for a Monte-Carlo / random-restart
   search using cuRAND (one RNG per thread). How do you keep the *reported* result
   deterministic for the demo while the search is random?
5. **A real warhead angle.** Make `first_dir` a free parameter and add it to the
   covalent-angle penalty, so the search also optimizes the approach geometry.

## Limitations & honesty

- **Synthetic, not real.** The anchor, sulfur, ligand chain, and 6-atom "pocket"
  are made up and labeled synthetic everywhere. The energies are illustrative, not
  predictive; this is **not** a validated force field and makes **no clinical
  claim**.
- **Reduced scope.** Real covalent docking uses stochastic global search over all
  ligand + rigid-body degrees of freedom, full force fields with solvation
  (MM-GBSA) or learned CNN scoring, explicit warhead reaction chemistry, and real
  PDB structures. We fix the anchor (collapsing stage 1), enumerate a dense torsion
  grid, and use a toy LJ + Coulomb + harmonic-constraint score. THEORY §7 spells
  out what production tools add.
- **Grid sampling.** A uniform $36^3$ grid is chosen for determinism and clarity,
  not efficiency; it would not scale to drug-sized ligands (hence Exercise 4).
- **Timing is a teaching artifact**, never a benchmark claim (CLAUDE.md §12).
