# 1.25 — Gaussian-Accelerated MD (GaMD)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.25`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._
> **Reduced-scope teaching version:** the GaMD *algorithm* (boost + cumulant
> reweighting) on a 1-D model potential, not a full all-atom force field (see
> [THEORY.md](THEORY.md) §7).

## Summary

Drugs work by changing how proteins move, and the interesting motions — a pocket
opening, a ligand binding, a loop flipping — are **rare events** hidden behind
energy barriers that ordinary molecular dynamics is far too slow to cross.
**Gaussian-accelerated MD (GaMD)** is an *enhanced-sampling* method that adds a
smooth, Gaussian-shaped **boost potential** to flatten those barriers — crucially
**without** having to pick a reaction coordinate in advance — and then recovers the
true free-energy landscape by **reweighting** the boosted run with a 2nd-order
cumulant expansion. This project implements that exact algorithm on the simplest
system that has a barrier (a 1-D double well sampled by Langevin dynamics) and runs
an **ensemble of independent walkers, one per GPU thread**, depositing into a
shared histogram with **deterministic fixed-point atomics** so the GPU result
matches a serial CPU reference bit-for-bit. You watch GaMD let trapped walkers
visit both wells and reconstruct the known free-energy profile.

## What this computes & why the GPU helps

GaMD adds a Gaussian-distributed boost potential to the total potential energy
without predefined collective variables, enabling unconstrained enhanced sampling.
It monitors the system's total potential and boosts when it falls below a
threshold; the boost statistics are then used for free-energy reweighting via a
cumulant expansion. Here that is realized on a model double well: we run many
boosted Langevin **walkers**, tally a reweighted histogram, and reconstruct the
free energy (PMF).

**The parallel bottleneck.** A single trajectory is strictly sequential in time
(step *s+1* needs *s*), so there is no parallelism *within* a walker. The cost
instead comes from needing **many** walkers × **many** steps to converge the
free-energy estimate. Walkers are **mutually independent**, so the GPU runs
thousands of them at once — one thread per walker, each running its whole time loop
in registers. This "ensemble of independent trajectories" is the same GPU pattern
as the SEIR (`9.02`) and PBPK (`13.02`) flagships, plus a per-thread RNG like the
Monte-Carlo dose flagship (`5.01`).

## The algorithm in brief

- **Model potential:** double well `U(x) = u_barrier·(x²−1)²` (two states at `x=±1`,
  barrier at `x=0`).
- **Sampler:** overdamped **Langevin** dynamics (drift downhill + thermal noise).
- **GaMD boost:** `ΔV = ½·k·(E−U)²` for `U<E`, with `k = k0/(V_max−V_min)` — lifts
  deep wells toward the threshold `E`, shrinking barriers everywhere at once.
- **Ensemble:** many independent walkers; each step after burn-in deposits
  `(1, ΔV, ΔV²)` into its histogram bin.
- **Reweighting:** recover the unbiased free energy via the **2nd-order cumulant
  expansion** `ln⟨e^{βΔV}⟩ ≈ β⟨ΔV⟩ + (β²/2)·Var(ΔV)`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gaussian-accelerated-md-gamd.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gaussian-accelerated-md-gamd.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gaussian-accelerated-md-gamd.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`); no extra CUDA
libraries are needed (the RNG is hand-rolled on purpose — see THEORY §4).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, runs the GaMD ensemble on `data/sample/gamd_config.txt`,
prints the reconstructed free-energy profile, shows the exact GPU-vs-CPU agreement
check and the recovered barrier height, and prints a timing line to stderr.

## Data

- **Sample (committed):** `data/sample/gamd_config.txt` — a tiny, **synthetic**
  15-number config (the model/boost/ensemble parameters) so the demo runs offline
  with zero downloads.
- **Regenerate / scale up:** `python scripts/make_synthetic.py [--n-walkers N --steps S …]`.
- **Full / real datasets:** `scripts/download_data.ps1` / `.sh` print where to get
  real all-atom GaMD inputs (they do not auto-download or bypass registration).
- **Provenance & license:** see [data/README.md](data/README.md). No real
  molecular or patient data is used; the sample is synthetic and labeled so.

Catalog dataset notes: AMBER GaMD tutorials (https://www.med.unc.edu/pharm/miaolab/resources/gamd/); GPCRmd (https://gpcrmd.org); D. E. Shaw Research benchmark systems; PDB structures of drug targets (https://www.rcsb.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program runs the ensemble on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and verifies **two** things:

1. the GPU fixed-point histogram equals the CPU histogram **bit-for-bit**
   (tolerance exactly **0** — integer atomics are order-independent); and
2. the reweighted PMF recovers the **known** double-well barrier height to within a
   documented physical tolerance (~0.6 kT), and walkers populated **both** wells.

The printed PMF table shows `F_sim` (reconstructed) tracking `F_true` (analytic
`U(x)`): ~0 at the wells, rising at the barrier. See [demo/README.md](demo/README.md)
for how to read it.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the config, runs CPU + GPU, verifies
   (exact tally + barrier recovery), reports.
2. [`src/gamd.h`](src/gamd.h) — **the heart**: the `__host__ __device__` physics —
   potential, GaMD boost, deterministic RNG, the per-walker loop `run_walker()`,
   the fixed-point tally, and the cumulant reweighting. Shared by CPU and GPU.
3. [`src/kernels.cuh`](src/kernels.cuh) → [`src/kernels.cu`](src/kernels.cu) — the
   one-thread-per-walker kernel + the deterministic device atomic adder.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **AMBER `pmemd.cuda` GaMD** (https://ambermd.org) — the reference GPU GaMD
  engine; study how the boost is folded into the force evaluation and how the boost
  parameters (`iE`, `sigma0P/sigma0D`) are estimated on the fly.
- **NAMD GaMD** (https://www.ks.uiuc.edu/Research/namd/) — GaMD in a second major
  GPU MD engine; useful to compare conventions.
- **MiaoLab GaMD analysis scripts** (https://github.com/MiaoLab20/GaMD) — the
  production reweighting (PyReweighting): cumulant *and* exponential-average
  variants, anharmonicity diagnostics. The real version of this project's §2.
- **OpenMM GaMD plugin** — GaMD in OpenMM (search the OpenMM-org repositories;
  the catalog flags the exact URL as "verify").

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble of independent trajectories — one GPU thread per walker** (PATTERNS.md
§1; exemplars `9.02` SEIR, `13.02` PBPK), with a **per-thread counter-based RNG**
(cf. `5.01`) and a **deterministic fixed-point integer atomic reduction** into the
shared histogram (cf. `11.09`, PATTERNS.md §3 rule 2). The per-element physics
lives in one `__host__ __device__` header so CPU and GPU agree exactly (PATTERNS.md
§2). No external CUDA library is used: the RNG is hand-rolled specifically so the
device and host draw identical streams (THEORY §4).

## Exercises

1. **Crank up the boost.** Set `--k0 1.0` in `make_synthetic.py` and rebuild. The
   walkers cross faster, but watch the recovered barrier *overshoot* the true value
   — see the 2nd-order cumulant bias (THEORY §5) appear. Then sweep `k0` and plot
   recovered-vs-true barrier.
2. **Raise the barrier.** Try `--u-barrier 8`. Confirm that an *unboosted* run
   (`--k0 0.001`) barely samples the barrier region while the boosted run does —
   the enhanced-sampling win, made visible.
3. **Add 3rd-order reweighting.** Extend `gamd.h` to accumulate `ΔV³` and add the
   3rd cumulant to `reweight_pmf_bin`. Does it reduce the strong-boost bias?
4. **Swap in cuRAND.** Replace the hand-rolled RNG with cuRAND in the kernel only.
   You will lose exact CPU==GPU agreement — explain *why*, and switch the verify to
   a statistical (distribution-level) check instead of bit-exact.
5. **2-D potential.** Generalize `U` to a 2-D landscape (e.g. a Müller–Brown
   surface) and a 2-D histogram. The ensemble pattern is unchanged; only the
   per-walker state and bin indexing grow.

## Limitations & honesty

- **Reduced scope (deliberate).** This is a 1-D model, not all-atom MD. There is no
  force field, no thermostat beyond Langevin, no periodic boundaries, no dual
  dihedral+total boost, no on-the-fly boost-parameter estimation, and no LiGaMD.
  THEORY §7 spells out exactly what production GaMD adds. The boost form and the
  cumulant reweighting *are* the real ones.
- **Synthetic everything.** The committed sample is a synthetic config for a model
  system; it represents no specific molecule and implies **no** clinical or
  pharmacological validity.
- **The reweighting is approximate.** The 2nd-order cumulant truncation
  systematically overestimates barriers when the boost is strong (THEORY §5). The
  default sample uses a gentle boost so the recovery is clean; this trade-off is a
  feature of GaMD, taught here on purpose, not hidden.
- **Timing is a teaching artifact**, never a benchmark claim (CLAUDE.md §12). On a
  tiny ensemble the GPU's edge is modest; it grows with `walkers × steps`.
