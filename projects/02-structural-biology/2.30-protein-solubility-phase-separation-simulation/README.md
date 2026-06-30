# 2.30 — Protein Solubility & Phase Separation Simulation

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.30`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). All data here is synthetic._

## Summary

This project simulates, on the GPU, the physics behind **biomolecular
condensates** — the membraneless liquid droplets that intrinsically disordered
proteins (IDPs) like FUS and TDP-43 form by **liquid-liquid phase separation
(LLPS)**. We use a residue-level **coarse-grained** model (one bead per amino
acid) with the **HPS / Ashbaugh-Hatch** "stickiness" potential, integrate the
beads' motion with velocity-Verlet, and watch several short chains coalesce from a
dispersed start into a single dense droplet. A plain-C++ reference runs the same
physics serially; the demo asserts the GPU and CPU trajectories agree to machine
precision. It is a deliberately small, deterministic teaching version of a
research-grade problem.

## What this computes & why the GPU helps

Liquid-liquid phase separation (LLPS) of intrinsically disordered proteins (IDPs)
and RNA-binding proteins underlies the formation of biomolecular condensates
(stress granules, P-bodies, the nucleolus). Simulating LLPS needs systems large
enough to hold both a dense and a dilute phase and long enough for them to
separate — only reachable with GPU coarse-grained MD. FUS, TDP-43, and hnRNPA1
condensate-forming domains have been simulated with HPS or MARTINI models on GPUs.
Applications include predicting condensate-forming mutations and designing
condensate-disrupting drugs.

**The parallel bottleneck:** the cost is the **non-bonded force evaluation** — for
`N` beads it is an all-pairs `O(N²)` sum, redone every one of thousands of time
steps. It is also **embarrassingly parallel**: once positions are fixed, the force
on each bead is independent, so we assign **one GPU thread per bead** and each
thread *gathers* its force from all others. No atomics, no races — the cleanest
parallel pattern. (See [THEORY.md](THEORY.md) §4.)

## The algorithm in brief

- **HPS / Ashbaugh-Hatch non-bonded potential** — a Lennard-Jones core whose
  attractive well is scaled by a per-residue stickiness `λ ∈ [0,1]` (the model's
  one knob for hydrophobicity).
- **Harmonic backbone bonds** between consecutive residues of a chain.
- **Velocity-Verlet** integration (NVE) with **periodic boundaries** and the
  minimum-image convention.
- **Order-parameter clustering for phase detection**: each bead's *local density*
  (neighbours within `r_cut`) distinguishes the dense droplet from the dilute
  background.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-solubility-phase-separation-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-solubility-phase-separation-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-solubility-phase-separation-simulation.sln /p:Configuration=Release /p:Platform=x64
```

Both `Debug|x64` and `Release|x64` build with zero warnings. An optional
`CMakeLists.txt` is provided for Linux/macOS learners (the VS solution is the
required deliverable).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the CMake build)
```

The demo builds if needed, runs on `data/sample/system.txt`, prints the result,
shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/system.txt` — a tiny **synthetic** system
  (6 chains × 6 sticky beads = 36 beads) so the demo runs offline with zero
  downloads.
- **Real-world pointers:** `scripts/download_data.ps1` / `.sh` (LLPS/IDP databases;
  they are sequence/annotation sets, not particle inputs).
- **Synthetic generator:** `scripts/make_synthetic.py` (deterministic, seeded).
- **Provenance, format & license:** see [data/README.md](data/README.md).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
final potential energy = -11.058823
...
  condensed beads (>=4 neighbours)             = 36 of 36
RESULT: PASS (GPU matches CPU within tolerance)
```

The program runs the simulation on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts their final-state
summaries agree within a documented tolerance (energies/checksum `≤ 1e-6`, integer
order parameters exact). The negative potential energy and all-beads-condensed
result mean the chains actually phase-separated into one droplet — the science we
set out to see. That agreement, between two independent implementations, is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the system, runs CPU + GPU, verifies, reports.
2. [`src/hps_model.h`](src/hps_model.h) — the **shared** `__host__ __device__` HPS
   force/energy core (the one place the physics lives; CPU and GPU both call it).
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the System type, the loader, and the trusted serial velocity-Verlet.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-bead idea.
5. [`src/kernels.cu`](src/kernels.cu) — the `force_kernel`/`integrate_kernel` and the host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **LAMMPS + HPS** (https://github.com/lammps/lammps) — large-scale GPU LLPS with
  neighbour lists; the reference for slab-geometry simulations at scale.
- **OpenMM** (https://github.com/openmm/openmm) — GPU MD with a Python API; read
  its custom-force classes to see HPS expressed in practice.
- **CALVADOS 2** (https://github.com/KULL-Centre/CALVADOS) — an improved residue-
  level IDP model; study its λ table and Debye-Hückel electrostatics.
- **GROMACS + MARTINI IDR** (https://github.com/gromacs/gromacs) — GPU CG MD with
  the MARTINI force field.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the essentials didactically and credits the sources
(CLAUDE.md §2).

## CUDA pattern used here

**Independent per-bead "gather" (all-pairs N-body), no atomics.** One thread per
bead reads all other beads to sum its force, writes only its own slot; a second
kernel does the velocity-Verlet update. The per-pair physics lives in a shared
`__host__ __device__` header so the CPU reference and the GPU kernel run byte-
identical math (PATTERNS.md §2). Order parameters (local density) are computed
deterministically for phase detection. This is the same pattern production HPS
codes use, minus the neighbour list, electrostatics, thermostat, and multi-box
ensemble (see THEORY.md §7).

## Exercises

1. **Dissolve the droplet.** Regenerate with weak stickiness
   (`python scripts/make_synthetic.py --lam 0.2`) and rerun. Watch the local
   density collapse — the *soluble* side of the phase boundary. (Update
   `expected_output.txt` for the new input.)
2. **Shared-memory tiling.** Rewrite `force_kernel` so each block cooperatively
   loads a tile of `j`-positions into `__shared__` memory and reuses it across the
   block. Confirm the result is unchanged and measure the speed-up (THEORY §4).
3. **Add a Langevin thermostat.** Give each bead a friction + cuRAND random force
   to sample NVT at fixed temperature. Note that this breaks bit-exact CPU==GPU
   verification — switch to a *statistical* check (matching energy distributions).
4. **Scale it.** Run `--chains 64 --len 12 --box 16` and plot CPU vs GPU time vs
   `N`; find the crossover where the GPU's `O(N²)` parallelism wins.
5. **A real sequence.** Map the FUS low-complexity domain to HPS λ values and
   build a `system.txt` from it; compare its condensation to a polar mutant.

## Limitations & honesty

- **Synthetic data.** The committed system is generated, with a uniform synthetic
  λ = 0.9 — **not** real protein sequence, structure, or measured hydrophobicity.
- **Reduced-scope teaching model.** All-pairs `O(N²)` forces (no neighbour list),
  **NVE** dynamics (no thermostat), **no electrostatics**, a single small box, and
  reduced LJ units. A real LLPS study uses neighbour lists, Langevin/Debye-Hückel,
  hundreds of chains in slab geometry, and a multi-concentration ensemble to draw
  a phase diagram (THEORY §7).
- **Not a predictor.** This demonstrates the *mechanism* and the *GPU pattern*; it
  makes no claim about whether any specific protein phase-separates in a cell.
- **Timing is a teaching artifact**, never a benchmark claim — on this tiny `N`,
  many small kernel launches are launch-bound and the GPU is slower than the CPU.
