# 1.18 — Fragment / Combinatorial Library Enumeration

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.18`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **combinatorial library** is built by taking a handful of chemical *building
blocks* ("synthons"), sorting them into reactant *slots*, and forming every
possible product by picking one block per slot. A few hundred blocks in three
slots already explode into **billions** of products (Enamine REAL: >6×10⁹). This
project enumerates such a library on the GPU: for every product it computes the
key drug-likeness descriptors (molecular weight, cLogP, polar surface area,
H-bond donors/acceptors) and applies the **Lipinski "Rule of Five" + Veber**
filter, then reports how many products are drug-like. Each product is an
independent job, so the work maps perfectly onto **one GPU thread per product** —
the canonical "embarrassingly parallel" CUDA pattern, here applied to the
real combinatorial-chemistry problem of *virtual library triage*.

## What this computes & why the GPU helps

Fragment-based drug discovery and combinatorial library design require
enumerating billions of reaction products from building blocks in silico. A
single Enamine REAL-like library contains >6B compounds from ~160 reactions and
>130k building blocks. GPU acceleration is applied to (i) SMILES enumeration via
GPU-parallel reaction SMARTS matching, (ii) property calculation (MW, cLogP,
TPSA) for billions of products, and (iii) diversity filtering via GPU fingerprint
clustering.

**The parallel bottleneck:** the *property-and-filter pass over every enumerated
product*. There are `N = s₀ × s₁ × s₂` products and each needs a few dozen
floating-point operations plus a handful of comparisons. The products are
**mutually independent** — product *p* never reads product *q* — so the entire
pass is data-parallel. We give each product its own GPU thread; a `grid-stride`
loop lets a fixed-size grid sweep an arbitrarily large library. This teaching
version uses the **group-contribution (additivity) shortcut**: a product's
descriptors are the *sum* of its building blocks' contributions, so we never have
to assemble the full molecule — exactly why combinatorial pre-filtering is cheap
and GPU-friendly.

## The algorithm in brief

- **Mixed-radix index decode** — turn a flat product index `p ∈ [0, N)` into the
  per-slot building-block indices (an "odometer": slot 0 is the fastest digit).
- **Additive descriptor accumulation** — sum the chosen synthons' MW / cLogP /
  TPSA / HBD / HBA contributions (group-contribution approximation).
- **Lipinski + Veber filtering** — MW ≤ 500, cLogP ≤ 5, HBD ≤ 5, HBA ≤ 10,
  TPSA ≤ 140 Å²; a product passes iff all hold.
- **Deterministic reduction** — count passing products and sum their MW with
  **integer / fixed-point atomics** (so the totals are reproducible and match the
  CPU bit-for-bit), and recover the first few passing product indices.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/fragment-combinatorial-library-enumeration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/fragment-combinatorial-library-enumeration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\fragment-combinatorial-library-enumeration.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/synthons_sample.txt`, prints the
result, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/synthons_sample.txt` — a tiny **synthetic**
  3-slot × 6-block catalog (216 products) so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print the RDKit recipe for
  computing real building-block descriptors (Enamine / ChemSpace catalogs require
  registration — the scripts never bypass it).
- **Provenance & license:** see [data/README.md](data/README.md). The committed
  sample is **synthetic** and labeled as such everywhere.

## Expected output

Success looks like `demo/expected_output.txt`. On the committed sample the program
reports **130 / 216 products (60.2 %)** passing the Lipinski + Veber filter, a
summed passing-MW of **47136.000 g/mol**, and the first eight passing product
indices. The result is computed on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`); because both run identical
integer/fixed-point reductions they must agree **exactly** (count, MW-sum, and
the index list) — that exact agreement is the correctness guarantee, printed as
`RESULT: PASS`.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the catalog, runs CPU + GPU, verifies, reports.
2. [`src/product_core.h`](src/product_core.h) — the shared `__host__ __device__`
   per-product math (decode, accumulate, filter): the heart of CPU↔GPU parity.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (constant-memory tables +
   integer atomics) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **RDKit** (<https://github.com/rdkit/rdkit>) — reaction SMARTS and virtual-library
  tools (`Chem.AllChem.ReactionFromSmarts`, `EnumerateLibrary`); the descriptors we
  approximate additively (`Descriptors.MolWt`, `Crippen.MolLogP`, `CalcTPSA`) are
  RDKit's. Study how it builds *real* products from SMARTS, not just sums.
- **SyntheMol** (<https://github.com/swansonk14/SyntheMol>) — GPU/ML navigation of
  a combinatorial synthesis graph *without* explicit enumeration (the V-Synthes
  idea); contrast its tree search with our brute-force sweep.
- **ASKCOS** (<https://github.com/ASKCOS/ASKCOS>) — reaction-condition prediction;
  learn how synthesizability is judged in practice.
- **SpaceLight / FastROCS** (OpenEye, commercial) — GPU shape screening of
  virtual libraries; the diversity-filtering step our THEORY discusses.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + constant-memory tables + integer-atomic reduction.** One GPU
thread per product (grid-stride loop); the small synthon descriptor tables live in
**constant memory** (broadcast warp-wide); the pass count and fixed-point MW sum
use **integer `atomicAdd`** so the reduction is order-independent and reproducible.
This is the same family as flagship `1.12` (Tanimoto: per-item scoring with a
constant-memory query) and `11.09` (k-means: parallel assign + fixed-point atomic
reduce). The catalog also lists Thrust compaction and GPU k-means diversity
filtering as production extensions — see *Exercises* and THEORY.

## Exercises

1. **Thrust compaction.** Replace the host scan that recovers passing indices with
   `thrust::copy_if` / `thrust::count_if` over the device flag array, so the whole
   pipeline stays on the GPU. (Thrust ships with CUDA; no extra link needed.)
2. **Scale it up.** Run `python scripts/make_synthetic.py --per-slot 40` (64 000
   products) and watch the GPU's relative advantage grow as launch overhead is
   amortized. At what size does the GPU overtake the CPU on your card?
3. **A fourth slot.** Bump `N_SLOTS` to 4 in `product_core.h` (and the loader/
   sample) to model a 4-component reaction. Note how `N` — and the constant-memory
   pressure — change.
4. **Diversity filtering.** After the drug-like set is found, cluster the passing
   products by their descriptor vectors with GPU k-means (cf. flagship `11.09`) and
   keep one representative per cluster — the real "library design" step.
5. **A real filter.** Add a PAINS / rotatable-bond rule. Which of these stay
   additive across the forming bond, and which need the assembled molecule?

## Limitations & honesty

- **Reduced-scope teaching version (CLAUDE.md §13).** Real enumeration forms
  products by **reaction SMARTS matching** — it must know which synthons can react
  and where the new bond forms. We skip that: we assume every slot-combination is a
  valid product and that descriptors are **additive** over building blocks. That
  additivity is a genuine fast pre-filter (cLogP and TPSA are sums of atom/fragment
  contributions), but the true product MW must subtract the atoms lost when a bond
  forms — we fold an approximate correction into the synthon values rather than
  modeling each reaction. The rotatable-bond Veber rule is omitted because it is
  not simply additive across a forming bond.
- **The data is synthetic.** The descriptor numbers are physically plausible but
  **invented**, engineered so a clear fraction of products passes the filter. No
  chemical or clinical conclusion may be drawn (CLAUDE.md §8).
- **Timing is a teaching artifact, not a benchmark.** On 216 products the GPU is
  dominated by launch/copy overhead; the point is the *pattern*, which wins at the
  billion-product scale these libraries actually reach.
