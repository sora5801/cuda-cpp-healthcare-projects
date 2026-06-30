# 2.28 — Replica Exchange Solute Tempering (REST2) on GPU

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.28`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Proteins switch between conformations by crossing energy barriers. Plain
molecular dynamics (MD) at body temperature can sit in one basin for
milliseconds to seconds before hopping — far longer than a simulation can
afford. **Replica Exchange with Solute Tempering, version 2 (REST2)** is an
*enhanced sampling* trick that fixes this without the cost of heating a whole
box of water: it heats **only the solute** (the protein/ligand) by *scaling its
slice of the potential energy*, runs a **ladder of replicas** at different
effective solute temperatures, and **periodically swaps configurations** between
neighbouring replicas. The hot replicas hop barriers freely; the swaps funnel
that diversity down to the cold (physical, 300 K) replica.

This project is a **self-contained, exactly-verifiable teaching version**: it
keeps the *real* REST2 mathematics (the three-way energy split, the λ-scaling,
the Metropolis exchange criterion) on a tiny toy solute sampled by Metropolis
**Monte Carlo** instead of full explicit-solvent MD. Each replica runs on **one
GPU thread**; the host coordinates the swaps. You can watch the cold replica,
started trapped in the wrong well, escape to the correct global minimum — the
whole point of REST2 — and confirm the GPU result against a CPU reference.

## What this computes & why the GPU helps

REST2 (Replica Exchange with Solute Tempering version 2) selectively heats only
the solute (protein/ligand) degrees of freedom rather than the whole system,
making replica exchange practical for large solvated systems where heating all
water would be prohibitively expensive. Effective temperature scaling is applied
only to protein-internal and protein-water interactions, while water-water
interactions remain at 300 K. GPU MD runs each replica independently; in
production, NCCL/MPI handles exchange communication between replicas at swap
intervals. Applications include enhanced sampling of protein-ligand binding,
loop conformational changes, and protein folding.

**The parallel bottleneck:** between exchanges, the *M* replicas evolve
**completely independently** — each is its own MD (here, Monte Carlo) trajectory
with no communication. That is the textbook "embarrassing parallelism" map:
replica *r* → GPU thread *r*, each running its full sampling loop in registers.
The only synchronization is the brief exchange handshake between rounds (it
touches just *M* energies). Real REST2 engines (GROMACS, NAMD, OpenMM) are
structured exactly this way: independent GPU integration per replica, a tiny
swap step between rounds.

## The algorithm in brief

- **Scaled (effective) Hamiltonian** per replica: `E_eff = λ·E_pp + √λ·E_pw +
  E_ww`, scaling only solute-internal (`E_pp`) and solute-solvent (`E_pw`) terms.
- **λ-ladder** from the temperatures: `λ_m = T₀/T_m ∈ (0,1]`; cold replica λ = 1.
- **Per-replica sampling**: Metropolis Monte Carlo at the replica's `E_eff`.
- **REST2 Metropolis exchange criterion** between neighbours (the water-water
  term cancels): accept a swap with `min(1, exp(−Δ))`.
- **Even/odd alternation** of swap pairs so configurations migrate the ladder.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/replica-exchange-solute-tempering-rest2-on-gpu.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/replica-exchange-solute-tempering-rest2-on-gpu.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\replica-exchange-solute-tempering-rest2-on-gpu.sln /p:Configuration=Release /p:Platform=x64
```

> This project compiles its device code with **`--fmad=false`** (no fused
> multiply-add contraction) so the GPU's energy arithmetic matches the host
> compiler's. See *Expected output* and THEORY.md §Numerical considerations.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/rest2_config.txt`, prints the
per-replica result, shows the GPU-vs-CPU agreement check, and prints a timing
line on stderr.

## Data

- **Sample (committed):** `data/sample/rest2_config.txt` — a tiny, **synthetic**
  REST2 run configuration (10 numbers) so the demo runs offline with zero
  downloads. Engineered so the cold replica visibly escapes the wrong basin.
- **Regenerate / sweep:** `python scripts/make_synthetic.py [--barrier-h 9 ...]`.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print pointers to the
  real validation sets (no credentials are ever bypassed).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Shaw millisecond folding trajectories for validation;
SAMPL challenges (https://github.com/samplchallenges/SAMPL); GPCRmd REST2
enhanced sampling data (https://gpcrmd.org); chignolin/Trp-cage fast-folder
benchmarks.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program runs the **entire REST2 simulation twice** — once with the GPU sampler
(`src/kernels.cu`), once with a CPU reference (`src/reference_cpu.cpp`), sharing
the same exchange step — and asserts the two agree on **robust statistical
observables** (right-well occupancy and acceptance ratio), not on a bit-identical
trajectory. Why not bit-identical? The Metropolis test calls `exp()`, whose host
and device math libraries differ by ~1 ULP; a single borderline accept can flip
and (because a Monte-Carlo trajectory is chaotic) diverge afterward. We compile
with `--fmad=false` to remove the larger FMA-contraction drift, then verify the
kind of aggregate readout a real REST2 study reports. The headline science: the
**cold replica r0 ends with all 8 beads in the global (right) well**, having
started with 0 (trapped in the left well) — REST2 worked.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the config, runs the REST2 loop on CPU
   and GPU (shared exchange step), verifies, reports.
2. [`src/rest2.h`](src/rest2.h) — the shared `__host__ __device__` physics core:
   energy split, λ-scaling, counter-RNG, one Monte-Carlo sweep, exchange Δ.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-replica idea.
4. [`src/kernels.cu`](src/kernels.cu) — the sampling kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — config loader, λ-ladder, serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

GROMACS + PLUMED REST2 (https://github.com/gromacs/gromacs) — Hamiltonian REMD on
GPU; the reference implementation of solute tempering via per-replica topology
scaling. NAMD REST2 (https://www.ks.uiuc.edu/Research/namd/) — GPU replica
exchange with a `soluteScaling` keyword. OpenMM REST2 via openmmtools
(https://github.com/choderalab/openmmtools) — clean Python `REST2` /
`RESTState` classes worth reading for how the scaling is applied to a `System`.
DESMOND REST2 (Schrödinger, commercial) — GPU REST2 for free-energy
perturbation. The original methods papers: Liu, Kim, Friesner & Berne (2005,
REST) and Wang, Friesner & Berne (2011, REST2 — the √λ cross-term fix).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs — one thread per replica.** Each replica's full sampling loop
runs in a single thread's registers with no inter-thread communication; the host
performs the periodic Metropolis exchange. The shared `__host__ __device__`
physics core (`rest2.h`) guarantees the CPU and GPU run the same math, and a
deterministic **counter-based RNG** (a hash of `(seed, draw-index)`, not a
stateful generator) makes every run reproducible. This is the same structure
production REST2 engines use: independent GPU integration per replica, NCCL/MPI
energy exchange between rounds, a GPU-parallel Metropolis criterion. (Closest
flagship: `9.02` ensemble RK4 — thread-per-trajectory.)

## Exercises

1. **Harder barrier.** Regenerate with `--barrier-h 9` (or raise `--tilt` toward
   0). Watch the cold replica fail to escape with too few replicas, then add
   replicas (`--n-replicas 16`) until it succeeds again. How does swap acceptance
   change down the ladder?
2. **Local energy delta.** `mc_sweep` recomputes the *whole* effective energy
   before and after each single-bead move. Replace it with a **local** energy
   delta (only bead `i`'s well, its two bonds, its solvent term) and confirm the
   results are unchanged — and faster. (See the comment in `rest2.h`.)
3. **HREX vs REST2.** Scale *all* interactions (set `√λ → λ` on the cross term
   and also scale `E_ww`) to recover plain Hamiltonian replica exchange (HREX),
   and compare exchange acceptance for the same ladder. Why is REST2 cheaper for
   big solvent boxes?
4. **Resident-state GPU.** The host re-uploads coordinates each round. Keep the
   replica state resident on the device and perform the swap with a device-side
   pointer exchange; measure the copy time you save.
5. **Acceptance tuning.** Sweep `--step-size` and plot acceptance per replica.
   Find the value that lands the ladder near the classic ~30–40% sweet spot.

## Limitations & honesty

- **Reduced-scope teaching version.** This is **not** explicit-solvent molecular
  dynamics. The "solute" is a toy chain of 8 beads in a tilted 1-D double well,
  sampled by Metropolis Monte Carlo, not Newtonian MD with a force field and a
  thermostat. What is *faithful* is the REST2 machinery itself: the three-way
  energy decomposition, the `λ·E_pp + √λ·E_pw + E_ww` effective Hamiltonian, the
  λ-ladder, and the REST2 exchange criterion. The full method is described in
  THEORY.md §"Where this sits in the real world".
- **Synthetic data.** The committed sample is generated, not measured, and is
  labelled synthetic everywhere. No real protein or trajectory is used.
- **Exchange on the host.** A production GPU REST2 keeps state resident and
  exchanges energies via NCCL/peer copy; here the host owns the swap for clarity.
- **Not bit-identical CPU/GPU.** Verified on robust aggregate observables, not an
  exact trajectory — see *Expected output* and THEORY.md §Numerical considerations.
- **No clinical use.** Educational study material only (CLAUDE.md §8).
