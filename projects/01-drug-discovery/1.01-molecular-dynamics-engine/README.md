# 1.1 — Molecular Dynamics Engine

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.1`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project is a small but **complete molecular-dynamics (MD) engine**: it takes a
box of atoms, computes the forces between them, and steps them forward in time with
Newton's laws to produce a trajectory. It models the simplest physically meaningful
force field — a single **Lennard-Jones** pair interaction — and integrates it with
the standard **velocity-Verlet** scheme under periodic boundaries. The GPU computes
the expensive all-pairs force sum (one thread per atom, positions tiled through
shared memory); a serial CPU reference runs the identical physics so we can verify
the GPU and watch the integrator **conserve energy** — the hallmark of a correct MD
simulation. It is a **reduced-scope teaching version** of a production engine; see
[THEORY.md §7](THEORY.md) for what GROMACS/OpenMM/NAMD add on top.

## What this computes & why the GPU helps

Classical MD simulates the time evolution of every atom by integrating Newton's
equations of motion. Each timestep must evaluate the **non-bonded forces** between
atom pairs — here the Lennard-Jones term — which is by far the dominant cost.

**The parallel bottleneck:** the force on each atom is an independent sum over all
other atoms, so for `N` atoms the force evaluation is `O(N²)` work *that is fully
data-parallel in the atom index*. The GPU assigns **one thread per atom**; each
thread accumulates its own total force while the block **cooperatively stages atom
positions through shared memory** (the classic n-body tiling trick), cutting global-
memory traffic from ~N² to ~N²/blockSize. Everything else (the velocity/position
updates) is cheap O(N) streaming. This is exactly the step production engines
parallelize first, reducing a day of CPU work to minutes on a modern GPU.

## The algorithm in brief

- **Lennard-Jones 12-6 force/energy** per atom pair, with a cutoff and the
  minimum-image convention for periodic boundaries.
- **Velocity-Verlet** integration (kick → drift → recompute forces → kick): time-
  reversible and symplectic, so it conserves energy over long runs.
- **All-pairs (direct) `O(N²)` force sum** on the GPU with shared-memory tiling.
- **Energy/temperature diagnostics** and a position checksum as verifiable
  observables.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/molecular-dynamics-engine.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/molecular-dynamics-engine.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\molecular-dynamics-engine.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/lj_sample.txt`, prints the result,
shows the GPU-vs-CPU agreement check, and prints a timing line (on stderr).

## Data

- **Sample (committed):** `data/sample/lj_sample.txt` — a tiny **synthetic**
  Lennard-Jones fluid (27 atoms) so the demo runs offline with zero downloads.
- **Generate / scale:** `python scripts/make_synthetic.py [--side 8]`.
- **Pointers to real data:** `scripts/download_data.ps1` / `.sh` (idempotent; print
  the production force-field links, never bypass any registration).
- **Provenance, units, license:** see [data/README.md](data/README.md).

This engine teaches the LJ force field, so the synthetic fluid is the intended
input; real biomolecular MD instead reads CHARMM36m / AMBER ff19SB force fields and
structures (links in `data/README.md`).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program integrates the **same** trajectory on the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts their observables agree
within a documented tolerance — that agreement, together with **energy
conservation** (relative drift ≈ `2.5e-8`), is the correctness guarantee. Sample
result:

```
E0          = -103.417311
E_final     = -103.417308
max |dE|    = 2.534680e-06
rel drift   = 2.450924e-08
RESULT: PASS (GPU matches CPU: dE<=1.0e-06, dchksum<=1.0e-04)
```

## Code tour

Read in this order:

1. [`src/md.h`](src/md.h) — the shared `__host__ __device__` physics: LJ
   force/energy, minimum image, the Verlet helpers. **Start here** — both the CPU
   and GPU call this, which is why their results match.
2. [`src/main.cu`](src/main.cu) — loads the system, runs CPU + GPU, verifies, reports.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial velocity-
   Verlet driver and the file loader.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface and the tiling idea.
5. [`src/kernels.cu`](src/kernels.cu) — the tiled all-pairs force kernel + the
   kick/drift kernels + the device-side integration loop.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, I/O helpers.

## Prior art & further reading

- **GROMACS** (https://github.com/gromacs/gromacs) — production GPU MD; study its
  non-bonded kernels and neighbour search.
- **OpenMM** (https://github.com/openmm/openmm) — clean, documented CUDA platform;
  great for seeing how PME, constraints, and thermostats are organized.
- **NAMD** (https://www.ks.uiuc.edu/Research/namd/) — scalable multi-GPU MD; study
  its domain decomposition.
- **AMBER `pmemd.cuda`** (https://ambermd.org/GPUSupport.php) — highly optimized GPU
  engine for AMBER force fields.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Custom all-pairs force kernel with shared-memory tiling** (the n-body pattern):
one thread per atom, positions staged through shared memory and reused block-wide;
plain O(N) thread-per-atom kernels for the velocity/position updates. Energy is
reduced **deterministically** (per-atom array summed in fixed order on the host) so
stdout is reproducible. Production MD layers cuFFT (PME), Thrust/CUB (neighbour
lists), and NCCL (multi-GPU) on top — out of scope here, described in THEORY §7.

## Exercises

1. **Scale it up.** Generate a 512-atom system (`make_synthetic.py --side 8`) and
   watch the GPU/CPU timing gap on stderr change — the O(N²) GPU win grows with N.
2. **Single vs double precision.** Switch `Vec3` and `md.h` to `float` and observe
   the energy drift worsen; explain why MD needs FP64 (THEORY §5).
3. **Add a neighbour list.** Replace the all-pairs loop with a cell list so only
   pairs within `rcut` are visited, taking the per-step cost from O(N²) to O(N).
4. **Shift the potential at the cutoff.** The raw cutoff makes U discontinuous at
   `rcut`; add the energy/force shift production codes use and see the drift improve.
5. **Add a thermostat.** Implement simple velocity rescaling to hold a target
   temperature, turning the constant-energy run into a constant-temperature one.

## Limitations & honesty

- **Reduced scope on purpose.** One non-bonded term (Lennard-Jones) only — no bonds,
  angles, dihedrals, or electrostatics/PME. A full biomolecular force field is much
  larger (THEORY §7).
- **`O(N²)` direct force sum**, not a neighbour list, so it does not scale to the
  millions of atoms a production run handles. This is a deliberate teaching choice.
- **Synthetic data.** The committed sample is a synthetic LJ fluid, labeled
  synthetic everywhere; it models a noble-gas-like fluid, **not** a biomolecule, and
  carries **no clinical meaning**.
- **Not bit-identical CPU↔GPU.** GPU FMA and summation order differ from the serial
  CPU; we verify to a small, documented physical tolerance and explain why
  (THEORY §5). Timings are teaching artifacts, never benchmark claims.
