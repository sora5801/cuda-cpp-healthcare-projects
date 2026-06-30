# 2.29 — Ion Channel Gating & Permeation Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.29`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

An ion channel is a protein pore in a cell membrane that lets specific ions
(K⁺, Na⁺, Cl⁻, Ca²⁺) flow across — the molecular basis of every nerve impulse and
heartbeat, and a major drug target. This project simulates **ion permeation
through a channel under an applied voltage** using **Brownian dynamics (BD)**: we
treat each ion as an overdamped random walker in a 1-D free-energy landscape (a
selectivity-filter barrier) plus an electric driving force, and we count how many
ions cross — that crossing rate **is** the single-channel current a patch-clamp
electrode measures. The work is *embarrassingly parallel* (each ion is
independent), so we give **one GPU thread per ion** and accumulate the results
with integer atomics. The GPU run is verified against a plain-C++ CPU reference
that walks the *identical* trajectories, so the two agree **exactly**.

## What this computes & why the GPU helps

Ion channels (Nav, Kv, CFTR, VGCC) are major drug targets whose gating mechanisms
operate on microsecond-to-millisecond timescales. Full all-atom MD of ion
permeation is enormously expensive; a **reduced Brownian-dynamics** model captures
the essential physics — diffusion over a potential-of-mean-force (PMF) barrier
under an applied field — at a tiny fraction of the cost, which is why BD is a
standard tool for channel conductance (e.g. the BROWNFLEX / GCMC-BD lineage).

**The parallel bottleneck:** statistical convergence. A single ion trajectory is
noisy; a meaningful conductance needs **thousands to millions of independent
trajectories** (and/or very long ones). Those trajectories share nothing, so they
map perfectly onto the GPU: one thread integrates one ion for `n_steps` Brownian
steps, then atomically adds its integer tallies (occupancy histogram + crossing
counts) into shared device buffers. This is the catalog's "GPU-parallel ion
position histogram accumulation" and "applied-field integrator," reduced to a
clean, teachable kernel.

## The algorithm in brief

- **Overdamped Langevin / Brownian dynamics** (Ermak–McCammon update): drift down
  the free-energy gradient + a Gaussian thermal kick each step.
- **Potential of mean force (PMF):** a Gaussian energy barrier at the pore centre
  (the desolvation / selectivity-filter cost) — the catalog's "umbrella-sampling
  PMF along the pore axis," here as a closed-form landscape.
- **Applied-field (voltage-clamp) term:** a linear electrostatic ramp `−qV·z/L`
  that drives the current — the "non-equilibrium MD with applied electric field."
- **Crossing counting / mean-first-passage flavour:** count forward and reverse
  permeations; net forward crossings ∝ conductance.
- **Ion-position histogram:** integer occupancy per z-bin — the probability
  density along the pore, read out directly.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ion-channel-gating-permeation-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ion-channel-gating-permeation-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ion-channel-gating-permeation-simulation.sln /p:Configuration=Release /p:Platform=x64
```

No extra CUDA libraries are linked — only the CUDA runtime — because the RNG is a
hand-rolled, shared `__host__ __device__` splitmix64 (so CPU and GPU draw the same
numbers; cuRAND would not be bit-reproducible against the CPU). `THEORY.md`
explains what cuRAND would buy you and why we did not use it here.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/channel_params.txt`, prints the
deterministic result, shows the GPU-vs-CPU agreement check, and prints a timing
line to stderr.

## Data

- **Sample (committed):** `data/sample/channel_params.txt` — a tiny, **synthetic**
  one-line parameter file (pore length, barrier, charge, voltage, diffusion, step
  counts) so the demo runs offline with zero downloads.
- **Generate variants:** `python scripts/make_synthetic.py --voltage 0 --ions 4096`.
- **Real sources & provenance:** `scripts/download_data.ps1` / `.sh` (print URLs
  for PDB structures, MemProtMD trajectories, Channelpedia electrophysiology) and
  [data/README.md](data/README.md).

Catalog dataset notes: MemProtMD (<https://memprotmd.bioch.ox.ac.uk>); PDB ion
channel structures (<https://www.rcsb.org>); Channelpedia patch-clamp data
(<https://channelpedia.epfl.ch>); GPCRdb (<https://gpcrdb.org>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program runs the simulation on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts the integer tallies — the
occupancy histogram and the forward/reverse crossing counts — agree **exactly**
(tolerance = 0; integer atomics commute, so there is no floating-point slack to
allow). On the committed sample you should see a **net forward current** (forward
crossings > reverse) and a **U-shaped occupancy histogram** that is depleted at
the central barrier — the selectivity-filter bottleneck made visible. See
[demo/README.md](demo/README.md) for how to read those numbers.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the problem, runs CPU + GPU, verifies, reports.
2. [`src/channel_physics.h`](src/channel_physics.h) — **the shared `__host__ __device__` core**: the RNG, the PMF force, and the one true Brownian-dynamics step that both CPU and GPU call (this is *why* they match exactly).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-ion mapping.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (per-thread RNG + integer atomic scoring) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + the file loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **GROMACS + HOLE2** (<https://github.com/gromacs/gromacs>) — production GPU
  membrane MD; HOLE2 measures the pore radius profile along a trajectory. Study
  how a real applied-field MD integrator and pore analysis are organized.
- **NAMD + VMD ion-channel tools** (<https://www.ks.uiuc.edu/Research/vmd/>) —
  the classic applied-field permeation workflow and trajectory analysis.
- **MDAnalysis ion-permeation analysis** (<https://github.com/MDAnalysis/mdanalysis>)
  — how conductance is estimated from a finished trajectory (crossing counting,
  exactly what our `fwd`/`rev` counters mimic).
- **Brownian-dynamics channel codes** (BROWNFLEX, GCMC-BD literature) — the
  reduced-model lineage this project teaches; read for how a PMF table and a
  position-dependent diffusion coefficient replace explicit water.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Stochastic / Monte-Carlo histories** (`docs/PATTERNS.md §1`, exemplified by the
flagship `5.01` Monte-Carlo dose): one GPU thread runs one independent random
history (here, one ion trajectory) with its own reproducible per-thread RNG, then
scores results into shared device tallies with `atomicAdd`. Because every tallied
quantity is an **integer**, the atomic adds commute → the GPU result is
deterministic and matches the CPU bit-for-bit (`docs/PATTERNS.md §3`). The
per-step physics lives in one shared `__host__ __device__` header (`§2`, the
HD-macro idiom) so the CPU reference and the kernel are provably the same math.

## Exercises

1. **Zero-field control.** Run `make_synthetic.py --voltage 0`. Predict, then
   confirm, that the net flux collapses to ≈ 0 (detailed balance: no driving
   force ⇒ no steady current). What residual ±net do you see, and why (finite
   sampling)?
2. **I–V curve.** Sweep `--voltage` over several values, plot net flux vs V. Where
   is it linear (ohmic) and where does the barrier make it nonlinear (rectifying)?
3. **Barrier height vs conductance.** Raise `--barrier` from 2 → 8 kT. The
   crossing rate should fall roughly like the Arrhenius/Kramers factor `exp(−ΔU)`.
   Verify the exponential trend.
4. **Shared-memory privatization.** The kernel hammers the global occupancy
   histogram with `atomicAdd`. Add a per-block histogram in `__shared__` memory,
   reduce it once per block — measure the speed-up while keeping results identical.
5. **FP64 vs FP32.** The physics here is `double`. Switch the Langevin step to
   `float` and see whether the integer tallies still match (they may not — discuss
   why, linking to `THEORY.md` "Numerical considerations").

## Limitations & honesty

- **Reduced 1-D model, not a specific channel.** Real permeation is 3-D through a
  protein with a position-dependent diffusion coefficient, multiple ions
  interacting electrostatically (knock-on conduction), explicit/implicit water,
  and a PMF obtained from umbrella-sampling MD — not a closed-form Gaussian. We
  model a *single* ion at a time (single-file approximation) along one coordinate.
- **Synthetic, dimensionless parameters.** All values are reduced (kT = 1) and
  chosen for a clear demo, **not** fit to any real channel. The output is **not**
  a conductance in picosiemens and carries **no clinical meaning**.
- **Coarse time step.** The committed sample's diffusion step (`√(2Dδt) ≈ 0.4 nm`)
  is large relative to the 3 nm pore — fine for teaching, too coarse for
  quantitative work. Smaller `dt` (more steps) is the right fix; it just runs
  longer.
- **No gating dynamics.** "Gating" (the channel opening/closing conformational
  change) is named in the title but here the pore is always open; we simulate
  *permeation through an open pore*. Modelling the gate is a much larger problem
  (see `THEORY.md` "Where this sits in the real world").
- **Not for clinical use.** Educational study material only.
