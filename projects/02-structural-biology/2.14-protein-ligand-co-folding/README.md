# 2.14 — Protein-Ligand Co-Folding

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.14`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._
>
> **Reduced-scope teaching version** (CLAUDE.md §13): the full project is a
> learned diffusion transformer (Boltz-1 / AlphaFold3 scale). Here we keep the
> *architecture* — a reverse-diffusion loop whose every step is a self-attention
> pass over a joint protein+ligand token sequence — but replace the trained
> network with a fixed **analytic score**, so the math is fully transparent and
> the GPU result matches a CPU reference to machine precision. The full learned
> model is described in [THEORY.md](THEORY.md) "Where this sits in the real world".

## Summary

"Co-folding" predicts a protein's pocket geometry **and** a ligand's binding pose
**together**, in one generative pass, instead of docking a ligand into a frozen
protein. State-of-the-art tools (Boltz-1, AlphaFold3) do this with a **diffusion
model**: start from random 3-D coordinates and iteratively *denoise* them, where
each denoising step is a full **attention** pass that lets every atom "see" every
other atom. This project builds that exact loop at toy scale: a joint
protein+ligand token sequence, an attention-driven denoiser, and a deterministic
reverse-diffusion sampler that folds a noise cloud back into a planted complex.
You can watch the RMSD-to-native fall from ~1.5 Å to ~0.01 Å over 160 steps, and
read the GPU kernel that makes each step parallel.

## What this computes & why the GPU helps

Co-folding models simultaneously predict protein structure and ligand binding
pose in a single generative process, bypassing separate docking steps. Boltz-1
and AlphaFold3 accept a ligand and a protein sequence as joint inputs to a
diffusion model conditioned on molecular features; GPU inference generates the
complex structure in minutes. **The GPU bottleneck is the diffusion sampling loop
(50–200 denoising steps), each requiring a full attention forward pass over the
joint protein-ligand token sequence.**

**The parallel bottleneck (what we put on the GPU):** the per-step **self-attention**.
For `N` tokens, every step computes all-pairs interactions — `O(N² · d)` work —
and the loop repeats for `T` steps. We map it as **one GPU block per query token**;
the block's threads cooperatively stream over all key tokens doing an online
softmax (max, then exp-weighted aggregation) with a shared-memory reduction —
the shape of FlashAttention. This is where a real co-folding model spends almost
all of its time.

## The algorithm in brief

- **Tokens**: protein backbone (Cα) tokens + ligand atom tokens share one
  sequence (so one attention pass does protein–protein, ligand–ligand **and**
  protein↔ligand cross-attention at once).
- **Forward diffusion (setup)**: add Gaussian noise to the native coordinates to
  get the fully-noised start `x_T`.
- **Reverse diffusion (the loop, `T` steps)**, per token `i`:
  1. **Attention** — geometric (RBF/distance-kernel) self-attention gives weights
     `a_ij`; aggregate a target `x0_hat = Σ_j a_ij · x*_j` (a predicted clean pose).
  2. **Score** — the denoising direction points from the noisy `x_i` toward `x0_hat`.
  3. **DDIM update** — move a fixed fraction of the way; iterate.
- **Verify** — GPU vs CPU agreement (`1e-3`) **and** recovered RMSD-to-native (`<0.5 Å`).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including why this analytic score is a faithful stand-in for a trained one.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-ligand-co-folding.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-ligand-co-folding.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-ligand-co-folding.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — the attention,
softmax, and DDIM math are hand-rolled on purpose so nothing is a black box.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/complex_sample.txt`, prints the
recovered pose and RMSD, shows the GPU-vs-CPU agreement check, and prints a timing
line on stderr. See [demo/README.md](demo/README.md) for how to read the output.

## Data

- **Sample (committed):** `data/sample/complex_sample.txt` — a tiny, **synthetic**
  protein-ligand complex (12 protein + 5 ligand tokens) with a planted native
  pose, so the demo runs offline with a known answer.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print the real benchmark
  sources (documented, idempotent, never bypassing credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Real protein-ligand complex benchmarks: PoseBusters — 428 recent PDB complexes
(https://github.com/maabuu/posebusters); PDBbind v2020 (http://www.pdbbind.org.cn);
Astex Diverse Set — 85 drug-like complexes; CASF cross-docking
(http://www.pdbbind.org.cn/casf.php).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program runs the reverse diffusion on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) from the same noised start, then
asserts (a) the final positions agree within `1e-3` and (b) the recovered
RMSD-to-native falls below `0.5 Å`. The headline line is:

```
RMSD to native: start=1.5503  ->  final=0.0120 (Angstrom)
RESULT: PASS (GPU==CPU within 1e-03; pose folded RMSD<0.5)
```

The five final ligand-atom coordinates each land on their own native position —
the reverse diffusion recovered the planted pose, not just its centroid.

## Code tour

Read in this order:

1. [`src/cofold.h`](src/cofold.h) — **the shared math** (`__host__ __device__`):
   the attention logit, the stable softmax, the DDIM blend, the per-token
   denoising step, and the RMSD metric. This single file is why the CPU and GPU
   agree.
2. [`src/main.cu`](src/main.cu) — loads the complex, builds the noised start,
   runs CPU + GPU, verifies, and prints the deterministic report.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the deterministic
   noise (splitmix64 + Box–Muller), and the serial reverse-diffusion baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the block-per-token
   attention mapping.
5. [`src/kernels.cu`](src/kernels.cu) — the cooperative attention kernel
   (shared-memory online softmax) and the ping-pong reverse-diffusion loop.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, and I/O helpers.

## Prior art & further reading

- **Boltz-1** (https://github.com/jwohlwend/boltz) — open GPU co-folding of
  protein–ligand–nucleic-acid complexes; study its diffusion sampler and the
  joint-token representation.
- **NeuralPLexer3** (https://github.com/zrqiao/NeuralPLexer) — state-specific
  co-folding with CUDA; study how multiple conformational states are sampled.
- **AlphaFold3** (https://github.com/google-deepmind/alphafold3) — the reference
  architecture; study the diffusion module and confidence (pLDDT/iPAE) heads.
- **DiffDock** (https://github.com/gcorso/DiffDock) — diffusion *docking* (ligand
  into a fixed protein); a useful contrast to co-folding.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the *idea* didactically (CLAUDE.md §2).

## CUDA pattern used here

A **per-step attention denoising loop**: the host runs the `T`-step reverse
diffusion (ping-ponging two position buffers); each step launches a kernel that
maps **one block per query token** and cooperatively computes a numerically
stable softmax over all keys via a **shared-memory reduction** — the
FlashAttention shape, taught at toy scale. Production tools use FlashAttention2 +
cuDNN transformer blocks in FP16/BF16 with multi-GPU model parallelism; here we
use FP64 and a hand-rolled kernel for clarity (see PATTERNS.md §1, §2).

## Exercises

1. **Sharpen the attention.** Lower `temp` in the sample (e.g. 0.3) and watch the
   pose tighten; raise it (1.5) and watch atoms collapse toward the centroid.
   Explain the trade-off in terms of the softmax.
2. **Stochastic DDPM sampler.** Add a Gaussian noise term to `ddim_blend` (the
   true DDPM update) and average several runs. Why does stdout stop being
   deterministic, and how would you keep the demo reproducible (hint: seed + fixed
   reduction order, PATTERNS.md §3)?
3. **Tile the keys into shared memory.** The current kernel re-reads key positions
   from global memory; stage a block of keys into shared memory first and measure
   the change. Compare to the lattice-Boltzmann stencil (project 6.04).
4. **Multi-head attention.** Split the position+type features into 2 heads with
   different `temp`, concatenate the aggregated targets. Does the pose improve?
5. **Scale `N`.** Generate a 100-token complex and confirm the GPU loop time grows
   sub-linearly per token relative to the CPU (the O(N²) attention is where the
   GPU starts to win — PATTERNS.md §7).

## Limitations & honesty

- **Reduced scope.** This is **not** a learned model. The "score network" is a
  fixed analytic function of geometry (RBF attention over native targets), chosen
  so the reverse process provably converges to the planted complex and so CPU/GPU
  match exactly. A real co-folding model learns this score from data and
  generalizes to unseen complexes; ours does not generalize at all.
- **Synthetic data.** The sample complex is hand-built and labeled synthetic
  everywhere. It is not a real protein, ligand, or pose, and carries **no clinical
  meaning**. No output here may inform any diagnosis or treatment (CLAUDE.md §8).
- **No confidence head.** Real tools also predict pLDDT/iPAE confidence; we report
  only RMSD-to-native, which we can only do because we *planted* the native pose.
- **Geometric attention caveat.** Distance-kernel attention averages tightly
  clustered tokens toward their centroid; we space the ligand atoms to avoid that
  (THEORY "Numerical considerations"). A learned model sidesteps this with
  expressive features — a real lesson, kept honest here.
- **Timing is a teaching artifact.** At this token count the per-step attention is
  launch-bound, so the GPU loop can be slower than the CPU. That is expected and
  documented (PATTERNS.md §7), not a benchmark claim.
