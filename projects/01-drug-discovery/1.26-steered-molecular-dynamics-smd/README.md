# 1.26 — Steered Molecular Dynamics (SMD)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.26`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

**Steered Molecular Dynamics (SMD)** pulls a molecule along a chosen coordinate —
classically, dragging a ligand out of its binding pocket — and records the
mechanical **work** done by the pulling force. Pulling fast does *irreversible*
work that exceeds the true free-energy change, but a remarkable result called
**Jarzynski's equality** lets you recover the *equilibrium* free energy ΔG from a
collection of these non-equilibrium pulls. The catch: the estimate is dominated by
rare, low-work trajectories, so you need **many** of them — and each pull is
independent, which is a perfect fit for the GPU. This project implements the
method on a deliberately small 1-D model whose ΔG is known in closed form, runs
thousands of pulls in parallel, and shows Jarzynski's exponential average
recovering the right answer where the naive average fails.

## What this computes & why the GPU helps

SMD applies external forces or velocity constraints to pull a molecule along a
predefined coordinate (e.g., unbinding a ligand from a pocket), enabling
calculation of work profiles and estimation of free energies via Jarzynski's
equality. GPU MD allows many independent SMD trajectories to be run
simultaneously, improving statistical convergence of Jarzynski estimates.
Applications include estimation of drug residence time, rupture force of
protein-ligand bonds, and domain-opening mechanisms. NAMD pioneered GPU SMD;
OpenMM provides Python-scriptable SMD via external forces.

**The parallel bottleneck:** Jarzynski convergence needs a large *ensemble* of
trajectories (the exponential average is controlled by the low-work tail, which
only fills in with many samples). Each trajectory is an independent time
integration — sequential in time, but embarrassingly parallel *across*
trajectories. We map **one GPU thread to one whole trajectory**: the thread runs
the entire pulling simulation in registers and writes a single number (its work
`W_i`). With thousands of pulls in flight at once, the GPU turns a serial sweep
into a single launch (~20–30× faster than the single-threaded reference here, and
the gap widens with ensemble size).

## The algorithm in brief

- **Constant-velocity SMD** with a **harmonic guiding spring**: a "dummy atom"
  moves at constant velocity `v`; a spring of stiffness `k` connects it to the
  reaction coordinate and drags the coordinate along.
- **Overdamped Langevin (Brownian) dynamics** for the coordinate in a fixed
  potential of mean force (PMF), integrated with **Euler–Maruyama**.
- **Non-equilibrium work accumulation**: `W += F_spring · v · dt` along each pull.
- **Jarzynski's equality**: `ΔG = −kT · ln ⟨exp(−W/kT)⟩` over the ensemble,
  contrasted with the (biased) naive mean work and with the analytic ground truth.
- (Mentioned in THEORY for further study: constant-force SMD, fluctuation
  theorems, and umbrella integration as a follow-up free-energy method.)

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/steered-molecular-dynamics-smd.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/steered-molecular-dynamics-smd.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\steered-molecular-dynamics-smd.sln /p:Configuration=Release /p:Platform=x64
```

The project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
libraries — because the RNG is a hand-rolled, host/device-identical splitmix64
stream (so CPU and GPU draw the same random numbers; see THEORY §Numerics).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/smd_config.txt`, prints the
free-energy result, shows the GPU-vs-CPU agreement check, and prints a timing line
to stderr.

## Data

- **Sample (committed):** `data/sample/smd_config.txt` — a single line of 14
  numbers that fully specifies the reduced 1-D model. Synthetic; offline; < 1 KB.
- **Full dataset:** none to download — `scripts/download_data.ps1` / `.sh` print
  pointers to real full-atom SMD material. Larger synthetic ensembles via
  `scripts/make_synthetic.py --n-traj …`.
- **Provenance & license:** see [data/README.md](data/README.md) (per-field table).

Catalog dataset notes: NAMD SMD tutorials (https://www.ks.uiuc.edu/Training/Tutorials/); BindingDB residence time data (https://www.bindingdb.org); PDB force-probe simulation benchmark cases; published SMD studies on ion channels and motor proteins.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The key
line places three estimates side by side:

```
free energy (kJ/mol): naive <W>=-1.8871  Jarzynski dG=-11.3200  true dG=-12.0000
```

The naive mean work is biased high (the second law: `⟨W⟩ ≥ ΔG`), while Jarzynski's
exponential average recovers the true `−12 kJ/mol` to within ~0.7 kJ/mol. The
program computes every trajectory's work on **both** the GPU (`src/kernels.cu`)
and a **CPU reference** (`src/reference_cpu.cpp`) and asserts they agree to
~1e-12 kJ/mol (a tiny FMA/transcendental residue, documented in `main.cu`); that
agreement plus the Jarzynski-vs-truth check is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the config, runs CPU + GPU, verifies (work
   match + Jarzynski recovery), prints the deterministic report.
2. [`src/smd_core.h`](src/smd_core.h) — **the heart**: the shared
   `__host__ __device__` physics (RNG, PMF + force, overdamped-Langevin pull,
   work accumulation, Jarzynski reduction). Both CPU and GPU call this.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the
   one-thread-per-trajectory idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and its host launcher.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — config loader + the trusted
   serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **NAMD** — <https://www.ks.uiuc.edu/Research/namd/> — pioneered production GPU
  SMD; study its tcl-forces / SMD config to see how the spring is applied in a
  real force field.
- **GROMACS pull code** — <https://github.com/gromacs/gromacs> — GPU SMD via
  `pull-coord`; a good reference for reaction-coordinate definitions (distance,
  angle, dihedral, COM-based).
- **OpenMM `CustomExternalForce`** — <https://github.com/openmm/openmm> —
  Python-scriptable SMD; the clearest place to *read* how an external pulling
  force is added to an MD integrator.
- **alchemlyb** — <https://github.com/alchemistry/alchemlyb> — post-processing
  library; study its Jarzynski / BAR estimators and the bias-correction discussion.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble of independent stochastic trajectories** (one thread per trajectory),
with a **per-thread reproducible RNG** for the thermal noise and a **host-side
fixed-order reduction** for the Jarzynski average. This is the
thread-per-trajectory mapping of the SEIR ensemble (flagship `9.02`) and PBPK
(`13.02`), fused with the reproducible per-history RNG of the Monte-Carlo dose
engine (`5.01`). See [docs/PATTERNS.md](../../../docs/PATTERNS.md) §1 ("the same
ODE for many parameter sets" / "stochastic histories") and §2 (the shared
`__host__ __device__` core).

Catalog "CUDA pattern" note (verbatim): *Full GPU MD; custom CUDA force kernel for
harmonic spring SMD; CUDA streams for multiple independent pulling trajectories;
GPU memory for storing work accumulated along path.* (This teaching version
replaces the multi-stream full-atom MD with a single launch over a 1-D model; the
production picture is in THEORY §"Where this sits in the real world".)

## Exercises

1. **Pull faster, watch Jarzynski break.** Edit `data/sample/smd_config.txt` to
   `v_pull = 0.1` (and keep `v·steps·dt = 1`). Dissipation grows, the work tail
   gets harder to sample, and the Jarzynski error rises — re-run and see by how
   much. Then push `--n-traj` up with `make_synthetic.py` and watch it recover.
2. **Add the BAR/Hummer–Szabo bias correction.** The simple Jarzynski estimator
   is biased for finite samples; implement the second-cumulant correction
   `ΔG ≈ ⟨W⟩ − σ_W²/(2kT)` and compare it to the exponential average.
3. **Bin the work into a PMF profile.** Accumulate work vs. spring-center position
   (not just the endpoint) and reconstruct `U(ξ)` along the pull (the
   Hummer–Szabo PMF), then plot it against the analytic `pmf_energy`.
4. **Constant-force SMD.** Replace the moving spring with a constant force and
   measure the *rupture time* distribution instead of work (a residence-time
   proxy). What changes in the kernel?
5. **Profile occupancy.** Each thread holds the full Langevin state in registers.
   Use Nsight Compute to find the register count and achieved occupancy, then try
   `THREADS_PER_BLOCK = 64/128/256` and explain the trade-off (THEORY §GPU map).

## Limitations & honesty

- **Reduced-scope teaching model.** This is **1-D overdamped Langevin** in a fixed
  analytic PMF, *not* full-atom MD. Real SMD integrates Newton's equations for
  every atom in an explicit force field; here the "molecule" is a single
  coordinate and the surroundings are a friction + noise term. The method
  (pull → work → Jarzynski) is faithful; the system is a caricature.
- **Synthetic data, labeled synthetic.** All numbers are pedagogical, chosen so
  the demo is self-checking (the true ΔG is built into the PMF). They are not a
  property of any real ligand or protein.
- **The PMF is *given*, not measured.** In production the PMF is exactly the
  unknown you are trying to estimate; here we hand it to the simulation so we have
  a ground truth. That is why this demo can "pass".
- **Finite-sampling bias is real and visible.** 8192 trajectories is small; the
  Jarzynski estimate carries ~0.5–1 kJ/mol of statistical + bias error (see
  Exercise 2). We verify to a stated tolerance, not to zero.
- **GPU and CPU are not bit-identical.** Over 25000 double-precision steps with
  transcendentals, device and host `libm`/FMA differ at the last bit, accumulating
  to ~1e-13 kJ/mol. We document this and verify to 1e-6, rather than pretending to
  exactness (PATTERNS.md §4).
- **Not for any real decision.** Educational only; no clinical or design use
  (CLAUDE.md §1, §8).
