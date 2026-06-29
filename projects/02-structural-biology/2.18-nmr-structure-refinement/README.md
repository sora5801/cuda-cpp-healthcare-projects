# 2.18 — NMR Structure Refinement

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.18`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

NMR structure determination requires satisfying distance restraints (NOE: <5 Å), dihedral angle restraints (J-couplings), and RDC (residual dipolar coupling) data via simulated annealing MD. GPU MD accelerates the restrained simulated annealing protocol, especially for large proteins where many restraint evaluations occur per timestep. GPU-accelerated CYANA/XPLOR-NIH can run hundreds of independent SA trajectories simultaneously — essential for ensemble NMR structure determination. Structure validation against chemical shift back-calculation is also GPU-acceleratable.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Simulated annealing MD with NOE/dihedral/RDC restraints, distance geometry embedding, torsion angle dynamics (CYANA), refinement against CSROSETTA chemical shifts, back-calculation of NMR observables.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/nmr-structure-refinement.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/nmr-structure-refinement.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\nmr-structure-refinement.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: BMRB — Biological Magnetic Resonance Bank (https://bmrb.io); PDB NMR-derived structures (https://www.rcsb.org); RECOORD — recalculated NMR structures (verify URL); CASD-NMR automated structure determination benchmarks (verify URL).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

XPLOR-NIH (https://nmr.cit.nih.gov/xplor-nih/) — restrained MD for NMR with GPU support (via NAMD); CYANA (http://www.cyana.org) — torsion angle dynamics for NMR; AMBER NMR refinement (https://ambermd.org) — pmemd.cuda with NMR restraints; ARIA (http://aria.pasteur.fr) — automated NMR assignment and refinement.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Full GPU MD for restrained SA (pmemd.cuda); CUDA kernel for NOE energy and gradient computation; GPU-parallel independent SA replica array via MPI+CUDA; GPU chemical shift back-calculation via ShiftX2-GPU (verify URL). --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
