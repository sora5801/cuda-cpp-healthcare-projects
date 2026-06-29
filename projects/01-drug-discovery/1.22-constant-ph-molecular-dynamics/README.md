# 1.22 — Constant-pH Molecular Dynamics

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.22`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

> **Reduced-scope teaching version** (CLAUDE.md §13). Full constant-pH MD couples
> proton-titration Monte Carlo to a complete GPU molecular-dynamics engine — that
> is research-grade. This project keeps the *transferable ideas* — the pH-coupled
> Metropolis titration, the electrostatic coupling that shifts apparent pKa, and
> the ensemble Monte-Carlo GPU pattern — on a compact analytic energy model with
> **fixed** atom positions. The full picture is described in
> [THEORY.md](THEORY.md) under "Where this sits in the real world".

## Summary

At a fixed pH, each ionizable residue of a protein (Asp, Glu, His, Cys, Lys, …)
flickers between a **protonated** and a **deprotonated** form. Which form it
prefers depends on the pH versus the residue's intrinsic pKa **and** on the
electric field of the other charged residues nearby — so the residues are
*coupled*. This project samples that coupled equilibrium with **Metropolis Monte
Carlo**: it runs a large **ensemble** of independent MC chains — one per
(pH value, replica) — on the GPU, tallies how often each residue is protonated,
and reads off the **titration curve** and the **coupling-shifted pKa** of every
residue. It is a small, self-contained window into why constant-pH simulation
matters for drug design: a residue's real pKa near a binding site can sit far
from its textbook value.

## What this computes & why the GPU helps

Biomolecular simulations normally fix protonation states, ignoring pH-dependent conformational changes critical for drug design (e.g., histidine flips, aspartate protonation near binding sites). Continuous constant-pH MD (CpHMD) in AMBER22 pmemd.cuda couples proton titration MC moves to GPU MD, sampling both conformation and protonation simultaneously. A 400-residue protein at single-pH takes ~1 hour on an RTX 2080 — >1000× faster than CPU. Applications include pKa prediction, pH-dependent drug binding, and ion channel gating.

**The parallel bottleneck (here):** converging a titration curve needs *many*
independent Monte Carlo chains — one per pH value, times many replicas to beat
down Monte Carlo noise — and each chain runs thousands of sweeps. Those chains
are completely independent, so we assign **one GPU thread per chain** and run the
whole ensemble at once. In production CpHMD the dominant cost is instead the
**molecular-dynamics force evaluation** between titration moves (non-bonded /
PME), which is itself massively parallel; here we replace MD with a compact
analytic energy so the *Monte-Carlo ensemble* is the thing the GPU parallelises.

## The algorithm in brief

- **Metropolis Monte Carlo protonation moves** — propose flipping one residue's
  protonation; accept with probability `min(1, exp(-ΔG/kT))`.
- **pH-coupled intrinsic term** — `ΔG_intrinsic = ∓kT·ln(10)·(pKa − pH)`
  (Henderson–Hasselbalch in free-energy form).
- **Electrostatic coupling** — pairwise Coulomb interaction of each residue's
  changing charge with its neighbours, the term that shifts apparent pKa.
- **Ensemble over a pH grid × replicas** — the titration curve and a 50%-crossing
  **pKa estimate** per residue.
- **Integer protonation tallies** — so the GPU's atomic accumulation is exact and
  matches the CPU bit-for-bit.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/constant-ph-molecular-dynamics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/constant-ph-molecular-dynamics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\constant-ph-molecular-dynamics.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/cph_system.txt` — a tiny, **synthetic**
  3-residue system so the demo runs with zero downloads. Regenerate or tweak it
  with `python scripts/make_synthetic.py`.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real pKa
  benchmarks (PKAD, PHMD, DrugBank); nothing is auto-downloaded.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: pKa databases: PKAD (https://compbio.clemson.edu/pkad/), PHMD reference pKa values; Benchmark pKa sets for Asp/Glu/His/Cys/Lys residues; DrugBank compounds with ionizable groups (https://go.drugbank.com).

## Expected output

Success looks like `demo/expected_output.txt`: titration curves (fraction
protonated per residue, integer percent) and the predicted vs intrinsic pKa for
each residue. The program runs the titration on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
their integer protonation tallies are **exactly equal** (tolerance `== 0`): both
sides run the identical RNG-seeded Monte Carlo chains and accumulate integer
counts, so the parallel atomic reduction is order-independent and must match the
serial sum bit-for-bit (PATTERNS.md §4). That exact agreement is the correctness
guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/cph_core.h`](src/cph_core.h) — **the heart**: the shared `__host__
   __device__` RNG, energy function (`delta_G_flip`), and Monte-Carlo `run_chain`
   used identically by CPU and GPU.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-per-chain idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper (atomic tally).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the trusted serial
   baseline, and the `estimate_pKa` curve readout.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

AMBER pmemd.cuda CpHMD (https://ambermd.org/GPUSupport.php) — GPU constant-pH MD; CHARMM CpHMD (https://www.charmm.org) — GBSW implicit solvent titration; OpenMM constant-pH (https://github.com/openmm/openmm) — Python CpH framework; PropKa (https://github.com/jensengroup/propka) — fast pKa prediction for system setup.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble Monte Carlo** (PATTERNS.md §1): one GPU thread runs one independent
Metropolis MC chain — the same idiom as the ensemble-RK4 flagships (`9.02`,
`13.02`) combined with the per-thread-RNG + integer-atomic scoring of the Monte
Carlo dose flagship (`5.01`). The per-element physics lives in one shared
`__host__ __device__` header (`src/cph_core.h`, PATTERNS.md §2) so the CPU
reference and the GPU kernel run **bit-identical** math; per-residue protonation
is tallied as **integer counts** with `atomicAdd`, so the parallel reduction is
order-independent and reproduces the CPU result exactly (PATTERNS.md §3).

Production CpHMD (catalog: AMBER `pmemd.cuda`, replica-exchange across pH via
NCCL/MPI, GB/PME solvent) is described in [THEORY.md](THEORY.md) §"Where this sits
in the real world".

## Exercises

1. **Recover Henderson–Hasselbalch.** Set `coulomb_k 0` in the sample (or run
   `python scripts/make_synthetic.py --coulomb_k 0`). With coupling off, each
   residue's predicted pKa must return to its intrinsic value within Monte Carlo
   noise. Confirm it, and explain why the shifts vanish.
2. **Drive the coupling.** Shrink the spacing (`--spacing 4`) or raise `coulomb_k`
   and watch the ASP/LYS pKa shifts grow. At what point does ASP titrate off the
   bottom of the grid? Relate the shift magnitude to `k·Δq·q/r`.
3. **Tighten the curve.** Increase `replicas` (8 → 64) and `sweeps`. The titration
   percentages should stabilise; quantify the Monte-Carlo standard error.
4. **Add a residue.** Extend `make_synthetic.py` to a 4th titratable site and
   observe how it perturbs the others' pKa. (Watch `CPH_MAX_RESIDUES`.)
5. **Replace the RNG with cuRAND** in a separate kernel and discuss what you lose:
   the CPU can no longer reproduce the exact chain, so verification becomes
   statistical (compare distributions) rather than an exact integer match.

## Limitations & honesty

- **Reduced scope.** This is **not** molecular dynamics. Atom positions are
  **frozen**; the "energy" is a compact analytic surrogate (intrinsic-pKa term +
  pairwise Coulomb on a single effective dielectric), not a force field with
  solvent. Real CpHMD recomputes solvation each move and lets the protein move.
- **Synthetic data.** The committed system is a hand-authored 3-residue toy,
  **not a real protein**, and is labeled synthetic everywhere. The pKa numbers it
  prints are illustrative, **not** predictions for any molecule.
- **Single-dielectric electrostatics.** A constant `epsilon` cannot capture
  desolvation or position-dependent screening; that is why production codes use
  Generalized-Born or explicit-solvent PME.
- **No replica exchange.** We average independent replicas at each pH but do not
  exchange between pH windows (REX-CpHMD), which improves sampling of strongly
  coupled sites.
- **Not for clinical or chemical use.** Educational only (CLAUDE.md §8).
