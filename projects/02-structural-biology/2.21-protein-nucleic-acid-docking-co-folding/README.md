# 2.21 — Protein-Nucleic Acid Docking & Co-Folding

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.21`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

> **Reduced-scope teaching version (read this first).** The catalog frontier here
> is *co-folding* — predicting a protein-RNA/DNA complex with an all-atom
> diffusion model (AlphaFold3, Boltz-1, RoseTTAFold2NA). That is a multi-GPU,
> multi-week training problem and a black box if dropped in whole. This project
> instead teaches the **rigid-body docking + interface-scoring** loop that those
> systems still rely on (and that classical dockers like ZDOCK/PIPER are built
> from): given a protein and a nucleic-acid fragment, **score a whole grid of
> candidate poses on the GPU and find the best fit.** That is the named catalog
> algorithm *"protein-NA interface scoring,"* implemented exactly and verifiably.
> See **Limitations & honesty** and THEORY §"Where this sits in the real world."

## Summary

We take a rigid **protein** and a rigid **nucleic-acid fragment** ("the ligand")
and ask the most basic docking question: *where, and in what orientation, does
the ligand sit best against the protein surface?* We answer it by brute force —
slide and rotate the ligand over a discrete 6-D grid of **poses** (orientations ×
translations), give each pose an **interface score** (favourable atom contacts +
matched electrostatics − steric clashes), and report the best poses. Every pose
is independent, so the search maps perfectly onto the GPU: **one thread scores
one pose.** Because the whole score is computed in integer fixed-point arithmetic
shared between CPU and GPU, the two agree **exactly** — a clean, checkable result.

## What this computes & why the GPU helps

Protein-RNA and protein-DNA interactions are central to gene regulation, CRISPR
editing, and RNA therapeutics. Predicting a complex requires placing the nucleic
acid against the protein and scoring how well they fit — the **interface scoring**
step. A docker evaluates an enormous number of candidate poses (millions, once
orientations and translations are sampled finely), and *each pose is scored
independently*.

**The parallel bottleneck:** scoring poses. For `P` poses, `Np` protein atoms and
`Nl` ligand atoms, a brute-force search costs `O(P · Np · Nl)` pairwise tests —
the dominant cost, and embarrassingly parallel. We assign **one GPU thread per
pose** (a grid-stride loop so a modest grid covers any `P`). The read-only protein
and ligand atoms are shared by every thread; the tiny rotation set lives in
**constant memory** (broadcast cache). This is the same "score one query vs N
independent items" pattern as projects `1.12` (Tanimoto) and `12.01` (spectral
search) — see `docs/PATTERNS.md` §1.

## The algorithm in brief

- **Pose enumeration.** Orientations = the **24 proper rotations of a cube**
  (integer matrices, entries in {−1,0,+1}); translations = a regular 3-D lattice.
  A flat index `p` decodes to `(rotation, tx, ty, tz)` identically on CPU and GPU.
- **Rigid transform.** Each ligand atom is rotated then translated — pure integer
  math on fixed-point coordinates (no trig, no rounding).
- **Pairwise interface score** (per pose, summed over all atom pairs):
  - `d² < clash_r2` → **−clash penalty** (van-der-Waals overlap),
  - `clash_r2 ≤ d² < contact_r2` → **+contact bonus − elec·(qᵢqⱼ)** (shape +
    electrostatic complementarity in the interface shell),
  - else → 0.
- **Rank** poses by score; report the top-K. The committed sample has a *planted
  native pose* the search recovers as #1.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including the integer-exactness argument.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-nucleic-acid-docking-co-folding.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-nucleic-acid-docking-co-folding.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-nucleic-acid-docking-co-folding.sln /p:Configuration=Release /p:Platform=x64
```

The project links only `cudart_static.lib` (the CUDA runtime) — no extra GPU
libraries are needed, because all the math is hand-written integer arithmetic.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/complex_sample.txt`, prints the
ranked poses, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/complex_sample.txt` — a tiny, **synthetic**,
  human-readable protein-nucleic-acid complex (25 protein atoms, 9 ligand atoms)
  with a *planted native pose*, so the demo runs offline and has a known answer.
- **Full dataset:** real complexes come from the **PDB**; fetch instructions are in
  `scripts/download_data.ps1` / `.sh` (they print links and never bypass any
  registration). `scripts/make_synthetic.py` regenerates the sample.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PDB protein-nucleic acid complexes (https://www.rcsb.org);
RNA structure benchmarks from RNA-Puzzles (https://github.com/RNA-Puzzles);
PDB-NA complex benchmark sets; Rfam RNA family database (https://rfam.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
ranked top-5 of poses, with **pose 312 (rotation 0 = identity, t = (0, 0, 3.5 Å))**
the #1 hit at score 340 — exactly the planted native pose. The final line reads:

```
RESULT: PASS (GPU matches CPU exactly: 648/648 poses agree)
```

The program computes every pose score on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they are **bit-identical**
(integer arithmetic → tolerance 0). That exact agreement is the correctness
guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the complex, runs CPU + GPU, verifies
   (exact), ranks and reports the best poses.
2. [`src/docking_core.h`](src/docking_core.h) — **the one true scoring formula**,
   `__host__ __device__` so CPU and GPU run identical math (PATTERNS.md §2).
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the data model, the pose grid, the file loader, and the trusted serial search.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-pose idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper (constant-memory rotations).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, host I/O.

## Prior art & further reading

- **RoseTTAFold2NA** (https://github.com/uw-ipd/RoseTTAFold2NA) — deep network
  trained on protein-nucleic-acid complexes; learn how the NA token vocabulary and
  templates are handled.
- **Boltz-1** (https://github.com/jwohlwend/boltz) and **AlphaFold3**
  (https://github.com/google-deepmind/alphafold3) — unified all-atom *diffusion*
  co-folding of protein + RNA/DNA + ligand; study the diffusion sampler and the
  atom-level representation our pose search is a toy ancestor of.
- **ZDOCK / PIPER (ClusPro) / HEX** — classical rigid-body dockers that score a
  pose grid with an **FFT** (the algorithmic upgrade our brute-force search begs
  for; see THEORY §"real world"). The closest peers to *this* project.
- **ViennaRNA** (https://github.com/ViennaRNA/ViennaRNA) — RNA secondary structure
  (a CPU preprocessing step in real pipelines; out of scope here).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## Exercises

1. **Finer orientations.** The cube group (24 rotations) is coarse. Replace it with
   a finer SO(3) sampling (e.g. a few hundred quaternions). What breaks the
   integer-exactness guarantee, and how would you re-establish a tolerance? (Hint:
   THEORY §"Numerical considerations".)
2. **The FFT speed-up.** Read how ZDOCK turns the translational scan into one FFT
   per orientation, cutting `O(grid³)` to `O(grid³ log grid)`. Sketch how you'd
   wire `cuFFT` (project `8.03` shows the API) into this code.
3. **Shared-memory tiling.** For large `Np`, stage a tile of protein atoms in
   shared memory and reuse it across the ligand atoms a block scores. Measure the
   bandwidth win (compare to project `6.04`'s tiling discussion).
4. **A better potential.** Replace the 3-shell step potential with a soft,
   distance-dependent score (still integer: tabulate it). Does the native pose
   still win, and by how much margin?
5. **Symmetry.** The sample's charge pattern is deliberately *chiral* so the native
   pose is unique. Make it symmetric (a checkerboard) and watch the top scores
   tie — then explain why, in terms of the cube group.

## Limitations & honesty

- **This is not co-folding.** We dock two *rigid* bodies; we do not fold the RNA or
  flex the protein, and there is no learned model. Real co-folders (AlphaFold3,
  Boltz-1) jointly predict 3-D structure with a diffusion network.
- **Synthetic data.** The committed complex is **synthetic** and engineered to have
  one obvious answer (a charge "lock" and complementary "key"). It is labelled
  synthetic everywhere and implies **no** biological or clinical validity.
- **Toy scoring.** The 3-shell integer potential is a teaching caricature of a real
  force field (no van-der-Waals well shape, no desolvation, formal charges
  quantised to a sign). It is chosen so the CPU and GPU agree *exactly*, which is
  the pedagogical point — not physical accuracy.
- **Brute force.** We enumerate every pose. Production dockers use an **FFT** over
  translations and clustering/refinement; THEORY explains the gap.
- **Coarse orientations.** 24 cube rotations cannot represent an arbitrary
  orientation; a real search uses thousands. We trade resolution for exactness.
