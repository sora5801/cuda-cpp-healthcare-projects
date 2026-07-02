# 6.10 — Systems-Biology ODE/SDE Network Solver

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.10`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Systems-biology models — gene circuits, signalling cascades, metabolism — are
systems of coupled nonlinear ODEs. Integrating **one** model is fast; the science
lives in solving the **same model thousands of times** for a parameter sweep,
uncertainty quantification, or per-cell heterogeneity. Each solve is independent,
so this is a textbook **batched-ODE** GPU problem: one thread integrates one
trajectory. This project builds that batch solver around the **repressilator**
(Elowitz & Leibler, *Nature* 2000) — a 3-gene ring where each gene represses the
next — and sweeps two parameters (max transcription rate `alpha`, Hill
cooperativity `n`) across a 36-member ensemble, detecting which members settle to
a steady state and which sustain the famous engineered **genetic oscillation**.

## What this computes & why the GPU helps

Gene regulatory networks, signaling cascades, and metabolic models are encoded as
systems of potentially thousands of nonlinear ODEs/SDEs (e.g., SBML models from
BioModels). Integrating a single model is fast, but parameter sweeps, uncertainty
quantification, and multi-cell applications require solving thousands of
independent instances simultaneously — a perfectly GPU-parallel batch problem.
SUNDIALS/CVODE-GPU and libRoadRunner's LLVM JIT backend both target this batch-ODE
pattern.

**The parallel bottleneck:** the ensemble sweep. A single repressilator solve is
6 ODEs × a few thousand RK4 steps — trivial. But an uncertainty-quantification or
sensitivity study needs `10^4`–`10^6` such solves, one per parameter sample, and
those solves share nothing. We map **one ensemble member → one GPU thread**: the
thread runs the entire RK4 time loop in registers (the state is only 6 doubles)
and writes one summary. Thousands of members run concurrently across the GPU's
warps. This is exactly the batch-ODE pattern behind SUNDIALS' CUDA NVector and
libRoadRunner's batch mode.

## The algorithm in brief

- **Model:** the dimensionless repressilator (3 mRNA + 3 protein = 6 ODEs), with
  Hill-function repression `f(p)=1/(1+p^n)` around the ring (see [THEORY.md](THEORY.md)).
- **Integrator:** classical explicit **RK4** (4th order), shared as one
  `__host__ __device__` routine so CPU and GPU run identical arithmetic.
- **Ensemble:** a 2-D parameter sweep `na × nn` = `alpha × n`; member `idx` maps to
  its `(alpha,n)` deterministically.
- **GPU mapping:** one thread per member (`idx = blockIdx.x*blockDim.x+threadIdx.x`),
  no shared memory / atomics / inter-thread comms — embarrassingly parallel.
- **Feature detection:** per member, a numerically-robust two-pass oscillation
  detector (extrema, then hysteretic level-crossing count).
- **SDE variant:** a Chemical-Langevin (CLE) step is included for teaching
  (`grn_cle_step`), off the verified path (see Exercises / THEORY §numerics).
- Reference: CVODE adaptive BDF/Adams multistep, RK45 Dormand-Prince, implicit
  trapezoidal, CLE for SDE, sensitivity equations, SBML parsing/JIT.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/systems-biology-ode-sde-network-solver.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/systems-biology-ode-sde-network-solver.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\systems-biology-ode-sde-network-solver.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`); no extra CUDA
libraries are needed (the RK4 solve is hand-written for teaching).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/ensemble_params.txt` — a tiny, offline
  **synthetic** repressilator sweep config (16 numbers) so the demo runs with
  zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to the real
  model repositories (BioModels, Reactome, BioGRID, VCell). Real models are SBML
  files; parsing them is out of scope for this teaching demo (see data/README.md).
- **Regenerate / resize:** `python scripts/make_synthetic.py --na 64 --nn 64`.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: BioModels Database (EMBL-EBI) — 1000+ curated SBML models
(https://www.ebi.ac.uk/biomodels); Reactome pathways (https://reactome.org);
BioGRID interaction network (https://thebiogrid.org); VCell curated models
(https://vcell.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
6.10 -- Systems-Biology ODE/SDE Network Solver
repressilator ensemble: 36 members (6 alpha x 6 n), 4000 steps @ dt=0.050 (T=200), beta=5.0 alpha0=1.0
sample members (alpha n -> p2_final p2_min p2_max crossings osc):
  m0  :   10.0 1.00 ->    3.3166    3.3166    3.3166   0 0
  m9  :   60.0 2.20 ->    5.6226    2.7848    5.7201  17 1
  m18 :  160.0 1.00 ->   12.6886   12.6886   12.6886   0 0
  m27 :  210.0 2.20 ->    2.2663    2.2477   22.5924  14 1
  m35 :  260.0 3.00 ->    2.6560    1.3408   57.4447  10 1
ensemble: 17/36 members sustain oscillations
RESULT: PASS (GPU ensemble matches CPU within tol=1.0e-09)
```

The program computes each member on the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree: continuous
observables within `1e-9` (both run the same double-precision RK4), and the
integer oscillation flag **exactly**. Notice the science: members with low Hill
coefficient `n=1.0` (m0, m18) collapse to a steady state (`min == max`, 0
crossings), while high-`n`/high-`alpha` members (m9, m27, m35) oscillate. Timing
goes to **stderr** (it varies run-to-run and is not part of the diffed output).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the sweep config, runs CPU + GPU, verifies, reports.
2. [`src/grn.h`](src/grn.h) — the shared `__host__ __device__` core: the repressilator RHS, RK4 step, CLE step, and the two-pass oscillation summary.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) — the ensemble config, `(idx → alpha,n)` map, and serial baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-member idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **SUNDIALS/CVODE GPU** (https://github.com/LLNL/sundials) — LLNL ODE/DAE solver
  with a CUDA NVector and GPU-accelerated batch CVODE. Study its NVector
  abstraction and how a user supplies a CUDA right-hand-side kernel.
- **libRoadRunner** (https://github.com/sys-bio/roadrunner) — high-performance SBML
  ODE integrator with an LLVM JIT backend; learn how SBML → executable RHS works.
- **Tellurium** (https://github.com/sys-bio/tellurium) — Python systems-biology
  platform built on roadrunner; a good place to see the repressilator in Antimony.
- **GillesPy2** (https://github.com/GillesPy2/GillesPy2) — SSA + tau-leaping + CLE
  stochastic solver; the reference for the SDE side of this project.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Batched / ensemble ODE integration (**one thread per ODE system**). The catalog
also mentions "one thread-**block** per system + shared memory for the Jacobian +
cuSPARSE for large sparse Jacobians" — that layout wins only for *large, stiff*
systems needing an implicit linear solve each step. The repressilator is small and
solved explicitly, so **one thread per system** is faster and simpler; THEORY.md
§GPU-mapping explains the trade-off and when to switch. No extra CUDA library is
linked (only the runtime).

## Exercises

1. **Scale the sweep.** Regenerate a `64×64` (4096-member) config with
   `make_synthetic.py --na 64 --nn 64` and watch the GPU's timing advantage over
   the CPU grow — the small `36`-member demo is launch/overhead-bound.
2. **Map the oscillation boundary.** Print the full `oscillates` grid instead of 5
   sample members and find the critical Hill coefficient where oscillations turn
   on. Compare with the linear-stability prediction in THEORY.md §math.
3. **Turn on the SDE.** Use `grn_cle_step` (already in `grn.h`) with a per-thread
   cuRAND generator to integrate the Chemical Langevin Equation; report the mean
   and variance of `p2_final` over many noisy realisations per parameter set. (It
   is off the verified path — see THEORY §numerics for why RNG breaks CPU/GPU parity.)
4. **Adaptive stepping.** Replace fixed-step RK4 with embedded RK45
   (Dormand–Prince) and a per-thread error controller; measure how step counts
   vary across the ensemble (a source of thread divergence).
5. **A real model.** Load an SBML model from BioModels and hand-translate its RHS
   into a new `grn_deriv`-style function — the manual version of what
   libRoadRunner automates.

## Limitations & honesty

- **Reduced-scope teaching version.** We hard-code one small circuit (the
  repressilator) instead of parsing arbitrary SBML; SBML→RHS code generation is a
  whole subsystem (see libRoadRunner) and is intentionally out of scope.
- **Explicit fixed-step RK4 only.** Genuinely stiff metabolic models need an
  implicit/adaptive integrator (CVODE's BDF); THEORY.md §real-world covers this.
- **Synthetic data.** The committed sweep is synthetic and labeled synthetic
  everywhere; nothing here is patient-derived or a clinical claim.
- **The SDE path is not verified.** Only the deterministic ODE is checked GPU-vs-CPU
  (an SDE's RNG stream differs across CPU/GPU, so it cannot be bit-reproducible).
- **Timing is a teaching artifact**, never a benchmark claim (CLAUDE.md §12).
