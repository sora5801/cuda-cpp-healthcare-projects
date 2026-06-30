# 3.5 — De Novo Genome Assembly

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.5`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

De novo genome assembly stitches a genome back together from millions of short,
overlapping DNA **reads** — with **no reference** to align to. Its first and most
GPU-friendly step is **all-vs-all read-overlap detection**: deciding which reads
share enough sequence to be neighbours in the assembly graph. This project builds
a **reduced-scope teaching version** of exactly that step: it sketches each read
into a small set of **minimizers** (à la minimap2), then scores **every read
pair** by how many minimizers they share. Each pair is independent, so we hand
each pair to its own GPU thread — the same "embarrassingly parallel" shape as
project 1.12, but over pairs. The output is the **overlap graph** whose connected
components are the draft contigs.

## What this computes & why the GPU helps

De novo assembly reconstructs a genome from raw reads without a reference. The
three GPU-amenable bottlenecks are: (1) **all-vs-all read overlap detection**
(`O(n²)` pairwise comparison), (2) string-graph / De Bruijn graph construction
from k-mers, and (3) consensus polishing of draft contigs. NVIDIA's
GenomeWorks / racon-GPU accelerates polishing (partial-order alignment) ~70× vs.
CPU; the Darwin accelerator showed ~109× GPU speedup for read overlap on PacBio
data. Modern HiFi assemblers (hifiasm) are CPU-centric for the string-graph
phase, but pairwise-overlap kernels are an active GPU insertion point.

**The parallel bottleneck this project targets** is step (1), the all-vs-all
overlap. For `n` reads there are `P = n(n−1)/2` pairs, and scoring a pair (count
shared minimizers) is independent of every other pair. We give **one GPU thread
per pair**: it decodes its `(i,j)` coordinate, fetches each read's sorted
minimizer sketch from flat (CSR) device buffers, and intersects them. The score
is an **integer** (a shared-minimizer count), so the GPU and the CPU reference
agree **bit-for-bit** — there is no floating point anywhere on the scored path.

## The algorithm in brief

- **Minimizer sketch (per read).** Slide a `k`-mer (`k=15`) along the read; in
  each window of `w=5` consecutive k-mers keep the one with the smallest hash.
  Use the **canonical** k-mer (`min(forward, reverse-complement)`) so a read and
  its opposite-strand neighbour pick the same minimizers. Sort + dedup → a small
  sorted set per read.
- **All-vs-all overlap (the GPU kernel).** For every pair `(i,j)`, count shared
  minimizers by a linear **merge-intersection** of the two sorted sets.
- **Overlap graph + layout.** Keep pairs with `shared ≥ 3` as edges; the
  connected components of that graph are the draft **contigs**.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including complexity and the monotonic-deque `O(m)` sketcher.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/de-novo-genome-assembly.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/de-novo-genome-assembly.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\de-novo-genome-assembly.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/reads_sample.fasta`, prints the
**overlap graph** (edges + connected components), shows the **GPU-vs-CPU
agreement** check (exact, tolerance 0), and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/reads_sample.fasta` — 6 **synthetic** reads
  (60 bases each) tiled from a fixed 120-base pseudo-genome so the demo runs with
  zero downloads and has a **known answer** (one contig).
- **Full dataset:** `scripts/download_data.ps1` / `.sh` guide you to the real,
  large benchmark sets (no auto-download — they are gigabytes).
- **Provenance & license:** see [data/README.md](data/README.md). Scale up with
  `python scripts/make_synthetic.py --genome-len 5000 --read-len 500 --step 100`.

Catalog dataset notes: CHM13 telomere-to-telomere human genome (T2T gold standard,
<https://github.com/marbl/CHM13>); GenomeArk (<https://genomeark.github.io/>);
Human Pangenome Reference Consortium (<https://humanpangenome.org/>); SRA PacBio
HiFi & ONT reads (<https://www.ncbi.nlm.nih.gov/sra>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program computes every pair's shared-minimizer count on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree with **tolerance 0** — both call the identical `__host__ __device__`
routine `count_shared_sorted()` in `src/assembly.h`, and integer counts have no
rounding, so the two are exactly equal (`mismatching pairs = 0`). On the
committed sample the graph is **1 component spanning all 6 reads** → one contig.

## Code tour

Read in this order:

1. [`src/assembly.h`](src/assembly.h) — the shared data model and the
   `__host__ __device__` per-pair math (`count_shared_sorted`, `pair_to_ij`,
   `hash32`). **Start here** — it is the CPU/GPU parity core.
2. [`src/main.cu`](src/main.cu) — loads reads, sketches, runs CPU + GPU, verifies,
   prints the overlap graph.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`reference_cpu.cpp`](src/reference_cpu.cpp)
   — FASTA loader, minimizer **sketcher**, and the trusted serial overlap.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-pair idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (grid-stride over pairs) + host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **GenomeWorks / racon-GPU** (<https://github.com/NVIDIA-Genomics-Research/GenomeWorks>)
  — GPU overlap + POA polishing; study its `cudamapper` for the production
  minimizer-index-and-overlap design this project miniaturizes.
- **minimap2** (<https://github.com/lh3/minimap2>) — the reference minimizer
  sketch + overlap/mapping algorithm; read `mm_sketch` to see the real rolling
  hash and the `O(m)` window minimum.
- **hifiasm** (<https://github.com/chhylp123/hifiasm>) — state-of-the-art HiFi
  assembler; learn how overlaps become a string graph and then contigs.
- **Racon CPU** (<https://github.com/lbcb-sci/racon>) — the CPU polishing baseline
  that racon-GPU accelerates (the *consensus* stage, complementary to overlap).

Study these for the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Independent jobs over **pairs** (one thread per read pair) · flat **CSR** layout
(`mins` + `offset`) for ragged per-read sketches in global memory · a **grid-stride
loop** so one grid covers any `P` · a shared `__host__ __device__` core for exact
CPU/GPU parity · **integer** scores ⇒ deterministic, tolerance-0 verification (no
atomics, no shared memory, no reduction). Production tools add a minimizer
**hash-table index** so they avoid the full `O(n²)`; here we keep the explicit
all-vs-all because it is the clearest thing to learn from.

## Exercises

1. **Sketch in `O(m)`.** Replace the `O(m·w)` window-minimum loop in
   `minimizers_of()` with a monotonic deque (the textbook sliding-window
   minimum). Confirm the sketch — and thus every overlap score — is unchanged.
2. **Add sequencing errors.** Regenerate the sample with
   `--error-rate 0.05`; watch shared-minimizer counts drop. How low can
   `MIN_SHARED` go before spurious edges appear? (Re-capture `expected_output.txt`.)
3. **Index instead of all-vs-all.** Build a hash map from minimizer → list of
   reads on the host, and only score pairs that share at least one minimizer.
   Compare the pair count to `n(n−1)/2` — this is how real overlappers scale.
4. **Warp-per-pair.** For long reads (large sketches), assign one *warp* per pair
   and split the merge across lanes with `__shfl`. When does that beat one thread?
5. **Toward contigs.** Extend the layout: order reads within a component by their
   overlap offsets and emit a consensus string (a tiny POA). This is the bridge to
   the *polishing* stage that racon-GPU accelerates.

## Limitations & honesty

- This is a **reduced-scope teaching version**: it implements the **overlap
  detection** stage only. It does **not** build a string/De Bruijn graph, resolve
  repeats, or polish a consensus — those are described in `THEORY.md` §"Where this
  sits in the real world".
- We score overlap by **shared-minimizer count**, a fast proxy. Real overlappers
  additionally chain minimizer *anchors* by position and estimate the overlap
  coordinates/strand; we keep only the count for clarity (positions are an
  exercise).
- The committed data is **synthetic** and labelled synthetic; similarities and the
  recovered "contig" carry **no biological meaning**.
- We compute the full `O(n²)` pairwise comparison on purpose (it is the lesson);
  production tools index minimizers to skip the vast majority of non-overlapping
  pairs. Nothing here is clinically valid.
