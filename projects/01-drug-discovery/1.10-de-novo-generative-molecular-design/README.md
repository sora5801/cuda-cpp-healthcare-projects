# 1.10 — De Novo Generative Molecular Design

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.10`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

> **Reduced-scope teaching version.** The full project (transformer/diffusion
> generative nets + RL fine-tuning) is research-grade and needs cuDNN and days of
> multi-GPU training. This version teaches the same *pipeline and GPU pattern*
> with the simplest model that still works end-to-end — a **first-order Markov
> SMILES language model** — generated in parallel and verified exactly against a
> CPU twin. The full picture is in [THEORY.md](THEORY.md) §"Where this sits in the
> real world".

## Summary

This project builds a tiny **de novo molecular generator** and runs it on the
GPU. It "trains" a first-order Markov language model on a handful of synthetic
SMILES strings (counting which character tends to follow which), then **samples
thousands of brand-new molecule strings in parallel** — one GPU thread per
molecule, each with its own reproducible random stream — and **scores** each one
with a toy drug-likeness reward to pick the best. It is a complete,
self-contained illustration of the de-novo loop (learn a distribution → sample
novel structures → score → select the goal-directed winner) and of the GPU
pattern real tools use for RL rollouts, with the GPU result checked **bit-for-bit**
against a plain-C++ reference.

## What this computes & why the GPU helps

Generative models learn the distribution of drug-like molecules and sample novel
structures optimized for multiple properties (potency, selectivity, ADMET,
synthesizability). GPU training is mandatory for the deep-net versions; at
inference, RL fine-tuning generates thousands of candidate molecules per
GPU-second, enabling goal-directed optimization (REINVENT4 combines RL with
curriculum learning on SMILES; DiffSBDD/TargetDiff generate molecules in 3-D
pockets).

**The parallel bottleneck:** *generation + scoring of the candidate batch.*
Sampling each molecule is an independent autoregressive walk, and scoring each is
independent too — so generating `n` candidates is `n` independent jobs. We map
**one GPU thread to one molecule**: thread `i` seeds its own RNG from
`(seed, i)`, samples a full SMILES string, scores it, and writes the result. This
is exactly the "thousands of molecules per GPU-second" rollout the catalog names,
reduced to its teachable core.

## The algorithm in brief

- **Represent** molecules as SMILES character strings over a small fixed alphabet.
- **Train** a first-order Markov model = a `K×K` transition table of integer
  counts (with Laplace +1 smoothing), framed with a START/END sentinel.
- **Sample** each molecule autoregressively: from the current symbol, draw the
  next via **inverse-CDF (roulette-wheel) sampling** over integer weights, until
  the END sentinel or a length cap.
- **Score** each molecule with a toy integer drug-likeness reward; **select** the
  argmax (the goal-directed pick).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/de-novo-generative-molecular-design.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/de-novo-generative-molecular-design.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\de-novo-generative-molecular-design.sln /p:Configuration=Release /p:Platform=x64
```

This project links only `cudart_static.lib` (the CUDA runtime) — no extra CUDA
libraries are needed.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/smiles_corpus_sample.txt`, prints
the generation summary + the best molecule, shows the exact GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/smiles_corpus_sample.txt` — 16 **synthetic**
  toy SMILES strings + a header (`n_train n_generate seed`), so the demo runs
  offline with zero downloads.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print where to get the
  real public corpora (they download nothing — the teaching demo does not need
  them). `scripts/make_synthetic.py` regenerates/enlarges the synthetic sample.
- **Provenance & license:** see [data/README.md](data/README.md). All committed
  data is synthetic; nothing here is a real molecule.

Catalog dataset notes: ChEMBL (CC-BY-SA), ZINC20 (academic), GuacaMol (MIT),
MOSES (MIT).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
1.10 -- De Novo Generative Molecular Design
trained first-order Markov model on 16 SMILES; generated 4096 molecules
drug-like hits (score >= 500): 1289 / 4096
mean reward: 44 milli-units
best molecule: idx=88  SMILES=CNOCO4#2=2Ncc1CCCCCCcOn4Occ1#c4221  score=1500 milli-units
RESULT: PASS (GPU matches CPU exactly: 4096/4096 molecules identical)
```

The program generates the candidate batch on both the **GPU** (`src/kernels.cu`)
and a **CPU reference** (`src/reference_cpu.cpp`) and asserts that every
molecule's score and length match **exactly** (tolerance = 0). Because the RNG,
sampling loop, and scorer are shared host/device code, molecule `i` is
bit-identical on both — so the agreement is exact, not approximate.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the corpus, trains the model, runs CPU +
   GPU, verifies exact agreement, reports the summary + best molecule.
2. [`src/generator.h`](src/generator.h) — **the heart**: the shared
   `__host__ __device__` model, RNG, sampling loop, and scorer (so CPU and GPU
   run identical math).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-
   molecule mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the generation kernel (constant-memory
   model) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   plus the shared corpus-loading and model-training steps.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **REINVENT4** (<https://github.com/MolecularAI/REINVENT4>, Apache-2.0) —
  production SMILES generative model with RL + curriculum learning. *Study how a
  learned transformer replaces our count table and how the RL reward re-weights
  the model.*
- **DiffSBDD** (<https://github.com/arneschneuing/DiffSBDD>) — 3-D
  structure-based diffusion design. *Study generation directly in a binding
  pocket (graphs, not strings).*
- **DiffDock** (<https://github.com/gcorso/DiffDock>) — diffusion pose
  generation. *Study how poses feed the docking-score reward in SBDD pipelines.*
- **DeepChem** (<https://github.com/deepchem/deepchem>) — broad ML
  drug-discovery toolkit. *Study its generative-model and scoring utilities.*

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-thread RNG / "Monte-Carlo histories"** (PATTERNS.md §1; same idiom as
flagship 5.01): one thread independently generates and scores one molecule from
its own reproducible stream. The read-only transition model lives in
**`__constant__` memory** (broadcast cache, like flagship 1.12's query). No
atomics, no shared memory, no synchronisation — embarrassingly parallel; the only
cost is warp divergence from variable molecule lengths. The catalog's deep-net
pattern (cuDNN, FP16, multi-GPU DDP) is described in THEORY §7.

## Exercises

1. **Make a higher-order model.** Extend the table to a *second*-order Markov
   model (`P(s_t | s_{t-2}, s_{t-1})`, a `K²×K` table). Does output quality
   improve? What happens to the constant-memory footprint?
2. **Add a real validity check.** Implement a SMILES bracket/ring-closure
   validator in `score_molecule` and report the fraction of *valid* molecules
   (the standard "validity" metric from MOSES/GuacaMol).
3. **Goal-directed reward shaping.** Change the reward to favour a target
   property (e.g. exactly two oxygens) and watch the mean reward and best
   molecule shift — a one-step taste of RL fine-tuning.
4. **Sweep the block size** (128/256/512) and the molecule count
   (`--n-generate`); plot kernel time vs `n`. Where does the GPU start to win?
5. **Swap the RNG.** Replace `splitmix64` with cuRAND on the device and observe
   that GPU and CPU no longer match bit-for-bit — then add a statistical
   (distribution-level) check instead of the exact one.

## Limitations & honesty

- **Reduced scope.** This is a first-order Markov model, not a neural generative
  model. It captures local character statistics only — it cannot enforce
  long-range constraints (matching ring digits, balanced branches), so many
  generated strings are not valid SMILES. A real RNN/transformer learns validity
  from data; see THEORY §7.
- **The scorer is a toy.** `score_molecule` is a made-up integer proxy, **not**
  QED, SA score, or docking. The "best molecule" it picks is illustrative, not a
  drug candidate, and is typically not even a valid molecule.
- **Synthetic data.** The training corpus is 16 hand-written toy strings, labelled
  synthetic everywhere. No real chemistry is implied.
- **Timing is a teaching artifact**, not a benchmark — the tiny sample is
  launch/copy bound (see the stderr note); the GPU's edge grows with the molecule
  count.
- **Not for clinical use.** Educational only.
