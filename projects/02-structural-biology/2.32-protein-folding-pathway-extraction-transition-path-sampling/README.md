# 2.32 — Protein Folding Pathway Extraction (Transition Path Sampling)

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.32`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._
>
> **Reduced-scope teaching version** (CLAUDE.md §13): this project teaches the
> *structure* of Transition Path Sampling on a **1-D model** of the folding
> free-energy landscape. The full research method (all-atom MD shooting) is
> described in [THEORY.md](THEORY.md) under "Where this sits in the real world".

## Summary

A protein folds by crossing a high free-energy **barrier** between an unfolded
basin and a folded basin — a **rare event** that ordinary molecular dynamics
almost never sees, because the molecule spends nearly all its time *inside* a
basin. **Transition Path Sampling (TPS)** sidesteps the waiting: instead of
simulating one long trajectory and hoping for a crossing, it **shoots** many
short trajectories from points near the barrier and keeps the ones that actually
connect the two basins. This project implements TPS on a deliberately simple 1-D
model — a bead doing Brownian motion on a double-well landscape — so you can see
the whole method (shooting, acceptance, the **committor**) clearly, and watch it
map onto the GPU as an array of thousands of independent "shooters", one per
thread. The GPU and a plain CPU reference run the **identical** shooting moves
and agree on every integer count, exactly.

## What this computes & why the GPU helps

Transition Path Sampling (TPS) harvests rare folding/unfolding events by shooting
from configurations near the transition state and accepting/rejecting
trajectories that connect folded and unfolded basins. GPU MD makes it practical
to run many short shooting moves in parallel. AIMMD (AI-augmented MD) uses
GPU-trained neural networks to identify committor isosurfaces, accelerating TPS
convergence. Applications include protein folding mechanism elucidation, cryptic
pocket opening pathways, and drug unbinding kinetics (τRAMD, WExplore).

**The parallel bottleneck:** each TPS **shooting move** integrates a short
trajectory (here, up to `max_steps` Brownian-dynamics steps per leg) — this is
the expensive inner work, and a converged path ensemble needs *thousands* of
them. Crucially, **every shooting move is independent**: it has its own random
"momenta" (its own RNG stream) and its own trajectory, sharing nothing with the
others. That is the textbook *embarrassingly parallel* pattern: we put **one
shooter on each GPU thread** and run them all at once. (In a real engine each
"shot" is itself a full MD simulation; here each shot is a 1-D Brownian
trajectory, but the *parallel structure is identical*.)

## The algorithm in brief

- **Double-well free-energy landscape** `V(x)` along a 1-D reaction coordinate
  `x` (a folding order parameter): minima at the unfolded basin A and folded
  basin B, separated by a barrier.
- **Overdamped Langevin (Brownian) dynamics**: the bead drifts downhill plus a
  thermal random kick — the kick is the only thing that ever climbs the barrier.
- **TPS shooting move (aimless shooting, simplified)**: from a shooting point
  near the barrier, integrate a **forward** leg and a **backward** leg until each
  reaches a basin; **accept** the path iff the two ends land in *different*
  basins (it connects A and B).
- **Committor analysis**: estimate `p_B(x)` = probability a shot from `x` commits
  to the folded basin; the **transition state** is the isosurface `p_B = 1/2`.
- **GPU pattern**: per-thread RNG + Monte-Carlo histories + **integer atomic**
  tallies (deterministic, matches the CPU exactly).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-folding-pathway-extraction-transition-path-sampling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-folding-pathway-extraction-transition-path-sampling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-folding-pathway-extraction-transition-path-sampling.sln /p:Configuration=Release /p:Platform=x64
```

Both `Debug|x64` and `Release|x64` build with zero warnings on the ratified
toolchain (verified on an RTX 2080, `sm_75`).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if the CMake build is used)
```

The demo builds if needed, runs on `data/sample/tps_params.txt`, prints the
transition-path statistics and the committor curve, shows the GPU-vs-CPU
agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/tps_params.txt` — one line of **synthetic**
  simulation parameters (the double-well landscape + the shooting run), so the
  demo runs with zero downloads.
- **Full dataset:** there is none to download — this is a synthetic 1-D model.
  `scripts/download_data.ps1` / `.sh` print pointers to the *real* inputs a
  research TPS study would use, and `scripts/make_synthetic.py` regenerates the
  parameter file.
- **Provenance & license:** see [data/README.md](data/README.md) (CC0; synthetic).

Catalog dataset notes: Anton/Shaw millisecond trajectories as TPS starting
configurations; GPCRmd pathway datasets (https://gpcrmd.org); folding benchmarks:
Trp-cage, chignolin, WW domain; SAMPL host-guest kinetics challenges.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes the tally on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree **exactly** — every
integer counter (accepted transition paths, the committor histogram) is identical
because both run the same shooting moves and integer atomic adds commute
(tolerance: **exact**, `== 0` mismatches; see THEORY §"How we verify").

Two things to look for in the output:

- The **committor curve `p_B` rises monotonically** from 0 % near the unfolded
  basin to 100 % near the folded basin — the S-curve that defines a good reaction
  coordinate.
- The **transition-state bin is 10**, which sits at x ≈ 0.5 — exactly the
  barrier top. Recovering the barrier top as the `p_B = 0.5` isosurface validates
  the *science*, not just CPU==GPU agreement.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads parameters, runs CPU + GPU, verifies the
   exact integer match, reports the committor curve and transition state.
2. [`src/tps_physics.h`](src/tps_physics.h) — **the heart**: the shared
   `__host__ __device__` RNG, Brownian-dynamics step, and the single
   `run_shot()` shooting move that both CPU and GPU call.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the
   one-shooter-per-thread idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (grid-stride shooters,
   integer atomicAdd tallies) and its host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   (the same `run_shot()`, looped).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, host I/O helpers.

## Prior art & further reading

- **OpenPathSampling** (<https://github.com/openpathsampling/openpathsampling>) —
  the reference TPS framework; runs shooting moves through OpenMM (GPU MD). Study
  its *MoveScheme* / shooting-move abstraction and its committor tooling.
- **WESTPA** (<https://westpa.github.io/westpa/>) — weighted-ensemble sampling
  on GPU MD; the complementary "split walkers across bins" rare-event strategy.
- **HTMD** (<https://github.com/Acellera/htmd>) — GPU-accelerated *adaptive*
  sampling that builds Markov State Models from many short trajectories.
- **AIMMD** (AI-augmented MD) — trains a neural network committor to bias TPS
  toward the transition-state isosurface; the "GPU neural committor" in the
  catalog. (Search "AIMMD transition path sampling".)

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Stochastic Monte-Carlo histories: per-thread RNG + integer atomic scoring** —
the same pattern as the Monte-Carlo dose flagship `5.01`. Each thread runs one
independent shooting move with its own reproducible RNG stream (the catalog's
"embarrassingly parallel independent shooter array"), and tallies its 0/1
outcomes into shared counters with `atomicAdd`. Because the tally is **integer**,
the atomic adds commute → the GPU result is deterministic *and* equals the CPU
reference exactly (PATTERNS.md §1, §3).

## Exercises

1. **Raise the barrier.** Re-run with `--barrier 8.0`
   (`python scripts/make_synthetic.py --barrier 8.0`). Transitions get rarer and
   the accepted-path fraction drops — quantify how the rare-event problem worsens.
2. **Move the transition state.** Set `x0 = 0.65` (asymmetric well via
   `--x0 0.65`). Predict, then check, which committor bin now crosses `p_B = 0.5`.
3. **Sweep the timestep.** Halve `dt` and double `max_steps`. The committor curve
   should be unchanged within statistics — verify the integrator is converged.
4. **Add a second precision.** The shared core uses `double`. Make a `float`
   variant of `bd_step` and measure how much the committor curve shifts — a
   lesson in stochastic-integrator precision.
5. **True backward shooting.** Replace the "independent forward leg" backward
   shot with a genuine time-reversed integration (store and negate the noise) and
   compare acceptance — see THEORY §"Where this sits in the real world".

## Limitations & honesty

This is a **deliberately reduced-scope, synthetic** teaching model:

- **1-D, not 3N-D.** Real folding lives in a 3N-dimensional configuration space;
  we collapse it to a single analytic reaction coordinate with a known
  double-well landscape. There is **no force field, no solvent, no atoms**.
- **Overdamped Langevin, not full MD.** We use Brownian dynamics on a
  free-energy surface, not Newtonian MD with momenta and a thermostat.
- **Simplified shooting move.** The backward leg is an independent forward
  Brownian leg, not a rigorous momentum-reversal — fine for overdamped dynamics
  on a 1-D surface, but not the production acceptance rule.
- **No path storage / Metropolis chain.** Real TPS runs a Markov chain *in path
  space* (each new path is shot from the previous accepted one). We sample
  shooting points across the barrier directly so a single run sweeps the whole
  committor curve.

It teaches the *anatomy* of TPS — shooting, acceptance, committor, the
`p_B = 1/2` transition state — and the GPU pattern, **not** quantitative folding
kinetics. Never use it for any biological or clinical claim.
