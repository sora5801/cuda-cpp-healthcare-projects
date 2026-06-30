# 2.12 — Flexible Fitting / MDFF

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.12`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A cryo-EM experiment gives you a 3-D **density map** — a fuzzy cloud showing where
a molecule's atoms probably are — but not the atoms themselves. **MDFF (Molecular
Dynamics Flexible Fitting)** bridges that gap: it takes a starting atomic model
and gently **deforms it until it sits inside the density**, by adding a force that
pushes every atom *uphill along the local density gradient* while a force field
keeps the geometry physically sane. This project implements the heart of that idea
as a reduced-scope, fully-commented CUDA program: a density-derived force computed
by **trilinear interpolation** at each atom, integrated by overdamped steepest
descent, with one **GPU thread per atom**. You watch a deliberately misfitted
27-atom model snap back onto a synthetic density map, and the GPU result is checked
against a plain-C++ reference.

## What this computes & why the GPU helps

Molecular Dynamics Flexible Fitting (MDFF) fits an atomic model into a cryo-EM
density map by adding density-derived forces to MD, deforming the model to match
the experimental map. The density map acts as an external potential; an MD force
field handles sterics and covalent geometry. GPU acceleration enables rapid
convergence for large complexes (ribosomes, viral capsids), and supports fitting
into sub-5 Å maps and interpreting conformational states.

**The parallel bottleneck:** the density-derived force must be evaluated **for
every atom, every timestep**. Each evaluation samples the density (and its
gradient) at the atom's current position via **trilinear interpolation** over the
8 surrounding voxels — a "gather + interpolate" exactly like CT backprojection
(project 4.01). For a ribosome (~10⁵–10⁶ atoms) over thousands of steps that is
billions of independent interpolations. They are all independent given the
(read-only) density map and current positions, so we map **one atom per GPU
thread** and the whole force evaluation parallelizes perfectly. The density map is
uploaded once and shared read-only by every thread on every iteration.

## The algorithm in brief

- **Density-derived force:** `F_dens = w · ∇ρ(x)` — push each atom up the density
  gradient (toward denser regions). `∇ρ` is computed by trilinear interpolation +
  a symmetric finite difference (the "cross-correlation gradient").
- **Restraint (MD-force-field stand-in):** `F_rest = −k·(x − x_ref)` — a harmonic
  restraint that prevents atoms from sliding off to the global density maximum and
  tearing the structure apart. (Production MDFF runs full Langevin MD here.)
- **Integration:** overdamped steepest descent, `x ← x + step·(F_dens + F_rest)`,
  with **Jacobi double-buffering** so every atom updates from the previous step's
  positions (race-free, deterministic).
- **Scoring:** RMSD-to-target and **density cross-correlation** (mean density at
  the atom positions) — the quantity MDFF maximises.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/flexible-fitting-mdff.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/flexible-fitting-mdff.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\flexible-fitting-mdff.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/mdff_problem.txt`, prints the fit
quality (RMSD and cross-correlation, before vs after), shows the GPU-vs-CPU
agreement check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/mdff_problem.txt` — a tiny, offline,
  **synthetic** fitting problem (27-atom lattice misfitted from its target inside
  a 24³ density map) so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (pointers to real EMDB
  maps + PDB models; nothing is downloaded for the demo).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: EMDB reference maps for MDFF (https://www.ebi.ac.uk/emdb/);
EMPIAR raw particle data (https://www.ebi.ac.uk/empiar/); ribosome MDFF benchmarks
(PDB 3J7Y, 4V6X); viral capsid fitting datasets.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program fits the model on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts the two fitted models agree
within a documented `1e-4` tolerance — that agreement is the correctness
guarantee. The deterministic stdout reports the RMSD-to-target **dropping** and
the cross-correlation **rising** from start to finish, which is the fit working.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the problem, runs CPU + GPU, verifies, reports.
2. [`src/mdff.h`](src/mdff.h) — **the heart**: the shared `__host__ __device__`
   trilinear density sampler, gradient, force, and steepest-descent step.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-atom idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper (double-buffered iteration).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + synthetic builder + metrics.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **NAMD MDFF** (<https://www.ks.uiuc.edu/Research/namd/>) — production flexible
  fitting with CUDA MD. Study how the density force is added to a *real* force
  field and how it scales to ribosomes.
- **VMD MDFF plugin** (<https://www.ks.uiuc.edu/Research/vmd/>) — MDFF setup +
  visualization; its "mdff sim" is exactly our Gaussian-blob density simulation.
- **phenix.real_space_refine** (<https://phenix-online.org>) — GPU-accelerated
  real-space refinement; a different optimiser for the same density-fit objective.
- **Coot** (<https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/>) — interactive
  model building into density; the manual counterpart to automated MDFF.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Gather + interpolate** (trilinear density sampling per atom, like CT
backprojection 4.01) wrapped in a **Jacobi / ensemble iteration** with
double-buffered positions (like soft-tissue PBD 10.02 and SEIR ensembles 9.02).
One thread per atom; the density map lives once in global memory, read-only, shared
by all threads. The full production pattern (full GPU MD + cuFFT cross-correlation
in reciprocal space) is described in THEORY "Where this sits in the real world";
this teaching version keeps the density force + a restraint so the whole thing is
readable.

## Exercises

1. **Analytic gradient.** Replace the finite-difference `sample_gradient` in
   `mdff.h` with the closed-form trilinear gradient (the field is piecewise-
   trilinear, so `∂ρ/∂x` is exact). Confirm the fit is unchanged and faster.
2. **Constant memory for parameters.** `MdffParams` is tiny and read by every
   thread — move it into `__constant__` memory and measure any change (cf. the
   constant-memory query in project 1.12).
3. **Texture-memory density.** Bind `rho` to a 3-D CUDA texture and use the
   hardware trilinear filter for `sample_density`. Compare accuracy (texture
   interpolation is 9-bit) and speed.
4. **Tune the restraint.** Sweep `k_rest` and `w_dens`; show how too-weak a
   restraint lets atoms collapse onto the densest blob (over-fitting), and too
   strong a one freezes the misfit.
5. **A real map.** Add a minimal MRC/CCP4 reader so `rho` comes from an actual
   EMDB map and `x0` from a PDB; the kernel does not change.

## Limitations & honesty

- The "molecule" is a **synthetic 27-atom lattice**, not a biomolecule, and the
  density is a simple **Gaussian sum**, not an experimental cryo-EM map.
- The MD force field is replaced by a single **harmonic restraint** — there are no
  bonds, angles, dihedrals, or steric clashes. Real MDFF runs full Langevin MD in
  NAMD/OpenMM; the restraint is a transparent stand-in (THEORY §7).
- Integration is **overdamped steepest descent**, not velocity-Verlet MD with a
  thermostat; there is no temperature, mass, or dynamics.
- Cross-correlation is approximated by the **mean density at atom positions**, not
  the full normalised map–map correlation MDFF optimises.
- This demonstrates the **GPU pattern**, not a validated structure-determination
  pipeline. **Not for clinical use.**
