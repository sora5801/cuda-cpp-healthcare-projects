# 6.17 — Purkinje System & Conduction System Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.17`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

The His-Purkinje system is the heart's electrical wiring: a branching tree of fast
fibres that carries the activation impulse from the AV node down to the ventricular
muscle. This project simulates that tree as an **ensemble of 1-D excitable cables**
— each solved with the monodomain reaction-diffusion (cable) equation — and reads
off each fibre's **conduction velocity** and the total time to activate the whole
tree. Because every cable is an independent PDE solve, we hand **one GPU thread per
cable**: the exact "ensemble of solvers" pattern, applied to a spatial PDE. It is a
deliberately **reduced-scope teaching model** (a 2-variable ionic model, ~7 cables)
that keeps the physics legible while showing the CUDA pattern that scales to the
~10⁵-segment trees of real hearts.

## What this computes & why the GPU helps

The cardiac conduction system (sinoatrial node, AV node, His bundle, bundle branches, Purkinje fiber network) initiates and coordinates ventricular activation. Simulating the Purkinje tree requires a 1D cable equation solver on a fractal branching network of ~10⁵ segments, coupled at Purkinje-muscle junctions (PMJs) to the 3D ventricular myocardium. GPU parallelism across the large number of independent 1D cable segments accelerates conduction pathway simulations for pacemaker dysfunction and re-entry arrhythmia studies.

**The parallel bottleneck:** the cost is the **per-cable time-marching PDE solve** —
`O(n_steps · n_nodes)` per cable, repeated over every cable in the tree. Those solves
are mutually independent during a beat (their coupling is a cheap `O(N)` graph pass
afterwards), so the tree's total work parallelises across cables with no
communication. On a real ~10⁵-segment tree this is the dominant cost, and giving each
segment its own thread is what makes whole-tree simulation tractable.

## The algorithm in brief

1D cable equation (monodomain) on Purkinje tree, PMJ coupling via gap-junction conductance, Stewart-Zhang Purkinje ionic model, His-Purkinje conduction velocity calibration, tree generation algorithms (L-system or rule-based branching), graph-based conduction delay computation.

Concretely, this teaching version:

- Solves `dV/dt = D ∂²V/∂x² + f(V,w)`, `dw/dt = g(V,w)` per cable with an **explicit
  finite-difference** stencil (3-point Laplacian) + **forward Euler** in time.
- Uses the compact **Aliev-Panfilov** 2-variable excitable membrane model as a
  didactic stand-in for the 20-ODE Stewart-Zhang Purkinje model.
- **Measures conduction velocity** from the front's threshold-crossing times at the
  two ends (His-Purkinje CV calibration).
- Assembles per-cable delays into absolute PMJ activation times via a **graph-based
  conduction-delay** forward pass over the rooted tree.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/purkinje-system-conduction-system-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/purkinje-system-conduction-system-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\purkinje-system-conduction-system-modeling.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
libraries — so it builds out-of-the-box with no path edits.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/purkinje_tree.txt`, prints the
per-cable table, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/purkinje_tree.txt` — a tiny, **synthetic**
  7-cable His-Purkinje tree so the demo runs offline with zero downloads.
- **Regenerate / enlarge:** `python scripts/make_synthetic.py`.
- **Full dataset pointers:** `scripts/download_data.ps1` / `.sh` (idempotent; prints
  links and never bypasses registration).
- **Provenance & license:** see [data/README.md](data/README.md). All data here is
  synthetic and labeled as such.

Catalog dataset notes: openCARP community Purkinje experiments (https://opencarp.org/community/community-experiments); MonoAlg3D_C Purkinje examples (https://github.com/rsachetto/MonoAlg3D_C); NeuroMorpho (morphological analogy for tree datasets) (https://neuromorpho.org); PhysioNet His-bundle electrogram databases (https://physionet.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
6.17 -- Purkinje System & Conduction System Modeling
Purkinje tree: 7 cables, dt=0.010 ms, 6000 steps (60.0 ms)
per-cable (idx parent lenmm D -> CV[mm/ms] PMJ_t[ms] captured):
  c0  p-1   20.0 3.00 ->  2.6247    7.620 yes
  c1  p0    25.0 3.00 ->  2.5853   18.290 yes
  c2  p0    25.0 1.50 ->  1.7973   22.530 yes
  c3  p1    15.0 2.50 ->  2.4430   24.930 yes
  c4  p1    15.0 2.50 ->  2.4430   24.930 yes
  c5  p2    15.0 2.00 ->  2.1614   29.970 yes
  c6  p2    15.0 2.00 ->  2.1614   29.970 yes
tree: 7/7 cables captured; total ventricular activation = 29.970 ms
RESULT: PASS (GPU per-cable steps + CV match CPU; tol=1.0e-09)
```

The program computes each cable on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) using the *same* shared stepper, so the
per-cable activation-step indices match **exactly** and the conduction velocities to
`1e-9` mm/ms. That agreement is the correctness guarantee. Note the CV rises with the
diffusion coefficient `D` (a proxy for fibre diameter) — the physiological
diameter→velocity relationship.

## Code tour

Read in this order:

1. [`src/purkinje.h`](src/purkinje.h) — the shared `__host__ __device__` cable
   physics: the Aliev-Panfilov reaction term and the finite-difference cable stepper
   that measures conduction velocity. **Start here** — it is the heart of the project.
2. [`src/main.cu`](src/main.cu) — loads the tree, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-cable idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline, the
   tree loader, and the graph-delay pass.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — GPU monodomain solver with integrated Purkinje network and PMJ calibration; openCARP (https://git.opencarp.org/openCARP/openCARP) — supports Purkinje cable coupling; Cardioid/LLNL (https://github.com/llnl/cardioid) — includes Purkinje conduction modeling; Chaste (https://github.com/Chaste/Chaste) — 1D cable equation infrastructure.

- **MonoAlg3D_C** — study its GPU monodomain solver and how it calibrates PMJ
  coupling and CV; the production analogue of this project's cable solve.
- **openCARP** — study its Purkinje cable coupling and its `.vtx`/`.elem` mesh format
  if you want to feed a real geometry.
- **Cardioid (LLNL)** — study its large-scale Purkinje conduction modelling.
- **Chaste** — study its 1-D cable-equation infrastructure and unit tests.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble of PDE solvers — one thread per independent 1-D cable** (docs/PATTERNS.md
§1, "same ODE/PDE for many parameter sets", exemplified by flagship 9.02). Each thread
runs the full space×time stepper for its cable in per-thread local memory; no shared
memory, no atomics, no inter-thread communication. The per-element physics lives in a
single `__host__ __device__` header (`purkinje.h`, PATTERNS.md §2) so CPU and GPU
agree exactly.

The catalog also names the *production* pattern — **batched tridiagonal (Thomas)
solvers via cuSPARSE, one thread per Purkinje node, shared-memory tridiagonal
coefficients, and CUDA graphs for the per-beat pattern**. That is the natural upgrade
path (an exercise below); this teaching version keeps the explicit cable solve
readable instead.

## Exercises

1. **Break the CFL limit.** Raise `D` (or `dt`) past `dx²/(2D)` in the sample and watch
   the explicit scheme blow up to NaNs — then derive the stability bound yourself
   (THEORY §Numerical considerations).
2. **Induce conduction block.** Lower a cable's `D` (thin, diseased fibre) or shorten
   the stimulus until the distal end never fires; confirm the demo reports `BLOCK` and
   that the block propagates to that cable's children.
3. **Scale the ensemble.** Extend `make_synthetic.py` to emit a rule-based branching
   tree of thousands of cables and observe the GPU timing overtake the CPU as the
   thread count grows.
4. **Swap the ionic model.** Replace Aliev-Panfilov `pk_reaction()` with a
   FitzHugh-Nagumo or a reduced Stewart-Zhang variant and compare the AP shape and CV.
5. **Go one-thread-per-node.** Reimplement the diffusion step as a batched tridiagonal
   solve (cuSPARSE) with one thread per node and shared-memory coefficients — the
   production mapping — and compare accuracy/speed against this explicit version.

## Limitations & honesty

- **Reduced-scope teaching model.** The ionic kinetics are the 2-variable
  Aliev-Panfilov model, *not* the 20-ODE Stewart-Zhang Purkinje model; the "tree" is 7
  hand-authored cables, not a fractal ~10⁵-segment network.
- **No true PMJ / 3-D coupling.** Cables are solved independently and stitched together
  by a graph-delay pass; there is no bidirectional Purkinje-muscle coupling to a 3-D
  ventricular mesh. Each non-root cable is re-paced at its proximal end rather than
  inheriting the parent's voltage waveform — a simplification that keeps the ensemble
  embarrassingly parallel.
- **Synthetic, uncalibrated data.** The diffusion coefficients are illustrative, chosen
  to produce a clear CV spread; they are not fit to measured His-Purkinje velocities.
- **Explicit solver.** Forward Euler is simple and legible but CFL-limited; production
  codes use implicit/operator-split schemes.
- **Not for clinical use.** This teaches the *shape* of conduction-system behaviour and
  the GPU pattern; it makes no medical claim.
