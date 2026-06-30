# 2.5 — Coarse-Grained / MARTINI Simulation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.5`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project is a tiny **coarse-grained (CG) molecular-dynamics** simulation in
the spirit of the **MARTINI** force field. In MARTINI, roughly four heavy atoms
are merged into a single "bead", so a whole lipid becomes ~12 interaction sites
instead of ~50 atoms. That coarsening — plus dropping the fast hydrogen
vibrations — is what lets MARTINI reach microsecond-to-millisecond timescales,
about 100× further than all-atom MD. We simulate a small box of two MARTINI-like
bead types (apolar "C" and polar "P") interacting through a Lennard-Jones force
field, integrate Newton's equations with velocity-Verlet under periodic
boundaries, and watch the two species stay **demixed** (oil and water). The
expensive non-bonded pair force is computed on the GPU with **one thread per
bead**.

## What this computes & why the GPU helps

Coarse-grained force fields like MARTINI map ~4 heavy atoms to one interaction
site, enabling microsecond-to-millisecond simulations of large membrane systems
(plasma membranes with dozens of lipid species, viral capsids, ribosomes).
MARTINI 3 CG-MD runs in GROMACS with full GPU acceleration, gaining roughly a
100-fold timescale extension over all-atom MD. Membrane-protein insertion, lipid
scrambling, and vesicle formation become accessible only at CG resolution.

**The parallel bottleneck:** the **non-bonded pair force**. Every bead feels a
force from every other bead within the cutoff. For `N` beads that is `O(N²)` pair
work per timestep (and there are thousands of steps). Crucially, each bead's
total force is an **independent** sum over its partners — no bead's force depends
on another bead's force — so the work maps perfectly onto the GPU: **one thread
per bead**, each thread looping over all partners and writing one force vector.
No two threads write the same output, so there are no atomics and no data races.
The coarse representation also means ~4× fewer particles than all-atom, so the
neighbour lists and (in production) the PME electrostatics are cheaper too.

## The algorithm in brief

- **Lennard-Jones non-bonded force** between every CG bead pair within `rcut`,
  with a per-type-pair well depth (the MARTINI interaction matrix).
- **Minimum-image convention** for periodic boundaries.
- **Velocity-Verlet** integration (half-kick → drift → recompute forces →
  half-kick), the standard symplectic MD integrator.
- **Order parameters**: total energy (a conservation check) and the C/P centroid
  separation (the demixing signal).

This teaching version implements the LJ non-bonded core by hand. The catalog's
fuller feature set — shifted electrostatics, an elastic-network (Gō-MARTINI)
overlay, PME via cuFFT, CG↔atomistic backmapping — is described in
[THEORY.md](THEORY.md) §7. See [THEORY.md](THEORY.md) for the full
science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/coarse-grained-martini-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/coarse-grained-martini-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\coarse-grained-martini-simulation.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/cg_system.txt`, prints the
deterministic result, shows the GPU-vs-CPU agreement check, and prints a timing
line on stderr.

## Data

- **Sample (committed):** `data/sample/cg_system.txt` — a tiny, **synthetic**
  16-bead system (8 C + 8 P) so the demo runs offline with zero downloads.
- **Bigger / different systems:** `python scripts/make_synthetic.py --per-side 4
  --steps 600` (no download).
- **Real MARTINI systems:** `scripts/download_data.ps1` / `.sh` print pointers to
  CHARMM-GUI, the MARTINI force-field repository, `insane.py`, TS2CG, and GROMACS
  — and never bypass any registration.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: CHARMM-GUI MARTINI membrane builder outputs
(<https://charmm-gui.org>); lipid parameter database (<https://cgmartini.nl>);
membrane-active peptide aggregation benchmarks; EMDB viral capsid reference maps
for validation.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program runs the simulation on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) from the *same* initial state, then asserts
the final bead positions agree within `1e-6` (they actually match to ~`1e-11`).
That agreement between two independent implementations is the correctness
guarantee. The stdout also reports the (nearly conserved) total energy and the
C/P demixing separation.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the system, runs CPU + GPU, verifies, reports.
2. [`src/martini.h`](src/martini.h) — the **shared host+device physics**: the LJ
   pair force, minimum image, and the velocity-Verlet update. This is the heart.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-bead idea.
4. [`src/kernels.cu`](src/kernels.cu) — the three kernels and the host time-loop wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + diagnostics.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **GROMACS + MARTINI 3** (<https://github.com/gromacs/gromacs>) — the production
  GPU CG-MD engine; study how it builds Verlet neighbour lists and offloads
  non-bonded + PME to the GPU.
- **MARTINI force-field files** (<https://cgmartini.nl>) — the official bead types
  and the full interaction matrix this toy reduces to a 2×2 `eps` table.
- **TS2CG** (<https://github.com/weria-pezeshkian/TS2CG>) — triangulated-surface →
  CG membrane builder; good for understanding how large CG assemblies are built.
- **`insane.py`** (<https://github.com/Tsjerk/Insane>) — the classic MARTINI
  bilayer/solvent assembly script.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the non-bonded core didactically and credits the
sources (CLAUDE.md §2).

## CUDA pattern used here

**Independent N-body force evaluation** (one thread per bead). Each thread reads
all bead positions/types from global memory, accumulates its bead's Lennard-Jones
force in registers, and writes one force vector — no atomics, no races. A kernel
boundary acts as the global barrier between the *drift* and the *force recompute*
of each velocity-Verlet step. (See [docs/PATTERNS.md](../../../docs/PATTERNS.md)
§1; this is the molecular cousin of the per-trajectory ensemble pattern.) The
catalog also lists **cuFFT for CG PME** — the long-range electrostatics piece —
which this teaching version omits and describes in THEORY §7.

## Exercises

1. **Make the demixing dramatic.** Start the C and P beads *interleaved* (edit
   `make_synthetic.py`) and watch the C/P separation grow as they unmix. Plot the
   separation vs. step (add a per-step print to stderr).
2. **Add a neighbour list.** Replace the `O(N²)` inner loop with a cell list so
   each bead only checks nearby cells — the real production optimization. Verify
   the trajectory is unchanged within tolerance.
3. **Shared-memory tiling.** Stage blocks of partner positions into `__shared__`
   memory (the classic N-body tiling) and measure the speedup; explain why it
   helps (THEORY §4).
4. **Conserve energy harder.** Sweep `dt` and watch the energy drift grow as `dt`
   increases — a hands-on lesson in why MD timesteps are small.
5. **Single vs double precision.** Switch `Vec3` to `float` and quantify how the
   energy conservation and CPU/GPU agreement degrade (THEORY §5).

## Limitations & honesty

- The system is a **synthetic two-type bead box**, not a real lipid membrane.
- The force field is a single shared `sigma` with a 2×2 `eps` matrix — a
  deliberate reduction of MARTINI 3's ~800-level interaction table.
- There are **no bonded terms** (no springs/angles), **no electrostatics/PME**,
  and a **plain cutoff** rather than MARTINI's smoothly shifted potential.
- The all-pairs `O(N²)` force is the simplest, most teachable scheme; production
  codes use neighbour lists for `O(N)` scaling.
- This is **study material**, not a validated molecular model, and not for any
  clinical or scientific production use.
