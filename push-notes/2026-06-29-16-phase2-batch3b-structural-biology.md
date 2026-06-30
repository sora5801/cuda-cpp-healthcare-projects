# Push 2026-06-29 #16 -- phase2 batch3b structural-biology

> Push-note (CLAUDE.md section 7.1). Second domain-2 batch: the last 3 Beginner + first 3
> Intermediate structural-biology projects, plus a small repo-hygiene fix. Lead-verified.

## 1. Summary

Six more **domain-2 (structural biology)** projects are complete, taking the collection to
**54 -> 60 / 301 (~20%)** and domain 2 to **13/35**. This batch finishes the domain-2
**Beginner tier** (inverse folding, membrane MD, density-map validation) and opens the
**Intermediate tier** (cryo-ET subtomogram averaging, Monte-Carlo structure sampling, cryo-EM
CTF estimation). Cryo-EM/ET is a recurring theme — three of the six lean on **cuFFT** for
3-D/2-D correlation and power spectra. This push also includes a small **repo-hygiene fix**
(a rendered-image `.gitignore` rule) caught during lead verification.

## 2. What changed

Six new fully-implemented projects under `projects/02-structural-biology/`:

- [`2.10` Protein Design / Inverse Folding Inference](../projects/02-structural-biology/2.10-protein-design-inverse-folding-inference)
- [`2.19` Membrane Protein Simulation](../projects/02-structural-biology/2.19-membrane-protein-simulation)
- [`2.22` Electron Density Map Analysis & Model Validation](../projects/02-structural-biology/2.22-electron-density-map-analysis-model-validation)
- [`2.04` Cryo-ET Subtomogram Averaging](../projects/02-structural-biology/2.04-cryo-et-subtomogram-averaging)
- [`2.07` Monte-Carlo Protein Structure Sampling](../projects/02-structural-biology/2.07-monte-carlo-protein-structure-sampling)
- [`2.11` Cryo-EM CTF Estimation & Particle Picking](../projects/02-structural-biology/2.11-cryo-em-ctf-estimation-particle-picking)

Plus a root **`.gitignore`** update (see §6 / §7). `docs/STATUS.md` -> 6 marked **done**
(60/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **2.10 Protein Design / Inverse Folding (ProteinMPNN-style)** — two kernels, one thread
  per residue: an all-pairs O(L^2) **burial count with shared-memory tiling** (the
  message-passing analog) and a per-residue 20-way argmax sequence decode. Shared integer
  scoring core -> bit-exact CPU==GPU; 87% native recovery on the synthetic backbone.
- **2.19 Membrane Protein Simulation** — coarse-grained membrane MD: 3-bead lipids + a
  protein column, velocity-Verlet + Langevin thermostat, one-thread-per-bead LJ/bond force
  gather (shared `membrane.h` core, CPU==GPU ~1e-14). Builds on the CG-MD idea from 2.05.
- **2.22 Electron Density Map Validation** — real-space correlation coefficient (RSCC) +
  **Fourier Shell Correlation** + the cryo-EM resolution at FSC=0.143/0.5, using **cuFFT**
  for the 3-D transform and a naive-DFT CPU reference (RSCC ~1e-15, FSC ~6e-8).
- **2.04 Cryo-ET Subtomogram Averaging** — **cuFFT batched 3-D cross-correlation** alignment
  (conj(FFT(ref)) * FFT(cand)) + rotation/reduce kernels, checked against a direct-correlation
  CPU reference; recovers the planted poses exactly (zero-shift NCC agree ~3e-7).
- **2.07 Monte-Carlo Structure Sampling** — a 2-D **HP lattice protein** with Metropolis MC:
  one thread per replica (ensemble, no atomics), a shared RNG + a precomputed **integer
  Boltzmann table** giving bit-exact CPU==GPU. A clean lattice-MC teaching kernel.
- **2.11 Cryo-EM CTF Estimation** — **cuFFT** 2-D power spectrum + fixed-point atomic radial
  average + a one-thread-per-candidate defocus grid-search (NCC fit). A synthetic micrograph
  with a known 15000 A defocus is recovered at 15300 A; GPU/CPU agree on the defocus index.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic maps/lattices,
labeled synthetic), with production tools (ProteinMPNN, GROMACS, Phenix/RELION, Dynamo,
Rosetta, CTFFIND) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/02-structural-biology/2.04-cryo-et-subtomogram-averaging   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

2.04, 2.11, 2.22 link cuFFT (`.lib` in both `<Link>` sections + `CMakeLists.txt`).

## 5. What to study here

Reading path: **2.07** (lattice MC ensemble) -> **2.10** (tiled all-pairs + argmax decode) ->
**2.19** (CG membrane MD) -> the cuFFT trio **2.22 / 2.04 / 2.11** (FSC validation, 3-D
correlation alignment, power-spectrum CTF fit — three different uses of the same library).
Exercise: in **2.11**, change the true defocus in `make_synthetic.py` and confirm the grid
search still recovers it; in **2.04**, add rotational search steps and watch the alignment
NCC improve.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders + root `.gitignore` changed; no build artifacts.
- ✅ **Caught & fixed during verification:** the 2.08 ray-tracer demo (from batch 3a), when
  run from the repo root by the verify harness, had written a stray `render.pgm` to the root.
  It was never committed (project `git add` is path-scoped, not `-A`). Fixed three ways:
  deleted the stray file, added a netpbm (`*.pgm/*.ppm/*.pbm`) ignore rule to the root
  `.gitignore` **with a `!**/data/sample/**` exception** so committed sample images are
  unaffected, and changed the verify harness to run demos from *inside* each project folder.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (MC/inverse-folding exact; cuFFT trio 1e-4..1e-9;
  membrane 1e-4).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.70–1.19**).
- **Workflow:** 6 agents, ~1.07M agent tokens, 476 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions** with synthetic data (2-D/16^3 maps, HP
  lattice, toy membranes), labeled synthetic; production scale is described in each THEORY.md.
- The `.gitignore` netpbm rule is a safety net; domain-4 (medical imaging) demos that render
  images will rely on it — watch that any *committed* sample images there live under
  `data/sample/` so the exception keeps them tracked.

## 8. Next push preview

Continue **domain-2 Intermediates** (`2.12` MDFF, `2.13` MSA acceleration, `2.14` co-folding,
`2.15` antibody prediction, `2.16` stability prediction, `2.17` allosteric networks, …) in
~6-project batches, then the 4 Advanced (`2.30`, `2.32`, `2.34`, `2.35`). Same workflow,
lead-verified, one push-note per batch.
