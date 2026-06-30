# 2.24 — SAXS / SANS Data-Driven Structure Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.24`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Small-angle scattering (SAXS with X-rays, SANS with neutrons) measures the size and
shape of a protein **in solution** as a single 1-D curve, the scattering intensity
`I(q)`. This project implements the workhorse of SAXS-driven structure modeling: the
**forward calculation** that turns a 3-D atomic model into a predicted `I(q)` via the
**Debye formula**, an all-pairs sum over atoms. We compute that profile on the GPU
(one thread per `q` value), verify it against a serial CPU reference, then *use* the
curve — recovering the molecule's radius of gyration from the low-`q` Guinier region
and scoring how well the model fits a (synthetic) experimental curve with a reduced
χ². That forward-model-and-score loop is exactly the inner kernel of conformer
ensemble refinement methods (EROS, BioEn) used to study flexible and intrinsically
disordered proteins.

## What this computes & why the GPU helps

Small-angle X-ray/neutron scattering (SAXS/SANS) provides solution-phase structural
information about proteins and complexes as a 1-D intensity profile `I(q)`. Fitting
atomic or coarse-grained models to SAXS data requires rapid **forward calculation**
of the scattering intensity from 3-D coordinates via the Debye formula — a pairwise
summation over all atoms that is highly GPU-parallelizable. GPU-MD + SAXS ensemble
refinement (EROS, BioEn) samples thousands of conformers and reweights them to match
experimental SAXS; applications include intrinsically disordered protein (IDP)
ensemble characterization.

**The parallel bottleneck:** the **Debye double sum**

```
I(q) = Σ_i Σ_j  f_i f_j · sinc(q · r_ij)
```

is **O(N²)** in the number of atoms `N`, evaluated independently at each of `N_q`
momentum-transfer values. For a real protein (10³–10⁴ atoms) over ~10² `q`-points
this `O(N_q · N²)` sum dominates, and **every `q` is independent** — so we map one
GPU thread to each `q` value, and each thread reduces over all atom pairs in its
registers. No atomics, no inter-thread communication: the cleanest possible GPU map.

## The algorithm in brief

- **Debye scattering formula** — orientationally-averaged `I(q)` as an all-pairs sum
  with the spherically-averaged kernel `sinc(x) = sin(x)/x` (this project's core).
- **Guinier analysis** — recover the radius of gyration `Rg` from the slope of
  `ln I` vs `q²` at low `q` (`ln I ≈ ln I(0) − (Rg²/3) q²`).
- **Model-vs-data fit** — a single least-squares scale factor + reduced χ² against an
  experimental curve (the score an optimizer/ensemble reweighter minimizes).
- **(Discussed, not implemented)** CRYSOL implicit-solvent model & spherical-harmonic
  expansion; SAXS-restrained MD ensemble refinement (EROS/BioEn); maximum-entropy
  reweighting; atomistic vs coarse-grained prediction — see `THEORY.md`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/saxs-sans-data-driven-structure-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/saxs-sans-data-driven-structure-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\saxs-sans-data-driven-structure-modeling.sln /p:Configuration=Release /p:Platform=x64
```

This project links only `cudart_static.lib` (the CUDA runtime) — the Debye kernel is
hand-written, so no extra CUDA math library is needed.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, runs on `data/sample/saxs_sample.txt`, prints the result,
shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/saxs_sample.txt` — a tiny **synthetic** 40-atom
  "protein" plus a SAXS curve computed from it, so the demo runs with zero downloads.
- **Full / real datasets:** `scripts/download_data.ps1` / `.sh` print where to get
  real curves and models (documented, idempotent, no credential bypass).
- **Provenance & license:** see [data/README.md](data/README.md).

Real sources: **SASBDB** (small-angle scattering biological data bank,
<https://www.sasbdb.org>); **PDB / RCSB** depositions (<https://www.rcsb.org>);
**BIOISIS** benchmark (verify URL); or simulated SAXS from MD trajectories.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The program
forward-models `I(q)` on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within a **relative tolerance of
1e-9** — machine precision, because both call the identical Debye routine in
`src/saxs_core.h`. It then prints the normalized profile at a few `q`'s, the
**Guinier `Rg` ≈ 13.8 Å** (vs the synthetic structure's true `Rg` = 13.67 Å), and the
**reduced χ² ≈ 0.8** of the fit. That `Rg` agreement is a *science* check (the curve
really does encode the molecule's size), beyond mere CPU==GPU agreement.

## Code tour

Read in this order:

1. [`src/saxs_core.h`](src/saxs_core.h) — the **one true** per-`q` Debye physics,
   shared verbatim by CPU and GPU (`__host__ __device__`). Start here.
2. [`src/main.cu`](src/main.cu) — loads the model, runs CPU + GPU, verifies, and
   reports the profile / Guinier `Rg` / χ².
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-`q` idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline plus
   the host-side analysis (least-squares scale, χ², Guinier fit, sample loader).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **CRYSOL** (<https://www.embl-hamburg.de/biosaxs/crysol.html>) — the classic
  analytical SAXS calculator; uses a spherical-harmonic expansion **plus an explicit
  hydration shell and excluded-solvent term**. Study it to see what our bare point-atom
  Debye sum omits.
- **FOXS** (<https://modbase.compbio.ucsf.edu/foxs/>) — fast Debye-based SAXS fitting
  with fitted solvent parameters; the closest production analogue to this project.
- **WAXSiS** (verify URL) — explicit-solvent, GPU-accelerated wide-angle scattering
  computed directly from MD; the gold standard for accuracy.
- **MDAnalysis SAXS module** (<https://github.com/MDAnalysis/mdanalysis>) — averaging
  SAXS over a trajectory (the ensemble idea).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + per-thread O(N²) reduction.** One GPU thread owns one `q` value
and computes the entire Debye double sum for that `q` in its registers; the grid
covers all `N_q` values. The atom arrays live in global memory (read by every thread,
never written), stored **structure-of-arrays** (`x[]`,`y[]`,`z[]`,`f[]`) so successive
threads' reads coalesce. Because the whole reduction for a `q` is inside one thread,
the result is **deterministic** and bit-comparable to the CPU. (A shared-memory tiled
variant that reorders the pair sum is described in `THEORY.md` as an exercise.) This
is the same family as flagship `1.12` (Tanimoto) and `12.01` (spectral search), with a
heavier per-thread reduction. See [docs/PATTERNS.md](../../../docs/PATTERNS.md) §1–2.

## Exercises

1. **Shared-memory tiling.** Rewrite `debye_kernel` to stage the atom arrays through
   shared memory in tiles (cooperatively loaded by the block), so the inner loop reads
   on-chip memory. Measure the speed-up for `--atoms 2000`, and explain why GPU==CPU
   now only holds to ~1e-12 (the pair sum is reordered → different FMA rounding).
2. **Real form factors.** Replace the constant per-atom `f` with the standard
   q-dependent atomic form factor `f(q) = Σ aₖ exp(−bₖ q²/16π²) + c` (Cromer–Mann
   coefficients). How does the high-`q` tail change?
3. **Solvent contrast.** Add a crude excluded-solvent term (subtract a Gaussian
   dummy-atom form factor per atom) and watch the profile shift — this is the first
   thing CRYSOL/FOXS add over the bare Debye sum.
4. **Kratky plot.** Compute and print `q²·I(q)` and use it to distinguish a folded
   globular blob from an extended/disordered chain (regenerate data with two clusters).
5. **Ensemble reweighting.** Generate several structures, forward-model each, and find
   the weights that best fit a target curve (a tiny maximum-entropy / least-squares
   problem) — the heart of EROS/BioEn.

## Limitations & honesty

- **Synthetic data, labeled synthetic.** The committed sample is a generated point
  cloud and a Debye-plus-noise curve, **not a real measurement**. No output here says
  anything about a real molecule, and nothing is suitable for clinical use.
- **Point-atom approximation.** We use one constant electron count per atom — no
  `q`-dependent atomic form factor, **no hydration shell, no excluded-solvent term**.
  Real SAXS fitting (CRYSOL/FOXS/WAXSiS) needs all three to match experiment; ours is
  a teaching baseline (see the exercises and `THEORY.md`).
- **No hydrogen/solvent, no instrument effects.** No smearing, no beam profile, no
  background subtraction beyond a single scale factor.
- **Tiny problem size.** At 40 atoms the GPU is *slower* than the CPU here (launch and
  copy overhead dominate) — the timing line is a teaching artifact, not a benchmark.
  The GPU's advantage grows as `N²·N_q`; try the exercises with thousands of atoms.
