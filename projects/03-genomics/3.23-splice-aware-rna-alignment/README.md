# 3.23 — Splice-Aware RNA Alignment

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.23`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

An RNA-seq read comes from a **mature mRNA** whose introns have already been
spliced out — so a single read often spans an **exon-exon junction** and, when
mapped back onto the genome, must **jump over an intron**. A *splice-aware*
aligner (STAR, HISAT2, minimap2 `-ax splice`) does exactly that: it pays one flat
"intron" penalty and emits a CIGAR `N` ("skipped region") operation, rewarded
when the skipped region uses the canonical `GT…AG` splice site. This project
implements the **scientific heart** of that idea as a small, fully verifiable
CUDA program: it aligns a **batch of reads** against a short reference "gene
model" using a **spliced dynamic-programming** recurrence, gives **one GPU block
to each read** (reads are independent), and checks the GPU against a CPU
reference to the exact integer. It is a deliberately **reduced-scope teaching
version** (CLAUDE.md §13) — not a genome-scale aligner — see *Limitations*.

## What this computes & why the GPU helps

Splice-aware aligners (STAR, HISAT2) map RNA-seq reads across exon-exon junctions, requiring the aligner to simultaneously find the best gapped alignment across multi-exon gene models. STAR uses a suffix array for ultra-fast seeding, then extends seeds across splice junctions; HISAT2 uses a graph FM-index encoding known splice sites. GPU acceleration targets the seed-extension step (banded SW across exon pairs) and the loading/querying of the large (28 Gb for STAR human genome) suffix arrays from a GPU-resident or page-locked memory index. For long-read transcriptomics (minimap2 -ax splice), GPU wavefront alignment handles much longer reads across complex splicing.

**The parallel bottleneck this project exploits:** a sequencing run produces
*millions of reads*, and **each read is aligned independently** against the
reference. That across-reads axis is embarrassingly parallel — we give **one
thread block per read**, and thousands of reads' dynamic-programming tables fill
concurrently. (Real aligners *also* parallelise *within* a long read via a
banded/wavefront extension; we keep each short read's DP serial for clarity and
discuss the intra-read wavefront in `THEORY.md`.)

## The algorithm in brief

- A **spliced Smith-Waterman** recurrence: the usual local-alignment moves
  (match/mismatch, read insertion, reference deletion) **plus an intron (`N`)
  move** that lets read base *i* match reference base *j* while a spliced-out
  intron `r[k+1 … j-1]` precedes column *j*.
- **Maximum-parsimony splice scoring:** an intron costs a *flat* `INTRON_OPEN`
  penalty (independent of length), with a **canonical bonus** when the skipped
  region begins with the `GT` donor and ends with the `AG` acceptor dinucleotide
  (the GT-AG rule).
- A **bounded intron band** (`MAX_INTRON`) caps how far back the `N` move scans,
  exactly as real tools cap intron length (e.g. STAR `--alignIntronMax`).
- **CIGAR-with-N traceback** turns a filled table into `12M40N12M`-style strings.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/splice-aware-rna-alignment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/splice-aware-rna-alignment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\splice-aware-rna-alignment.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, aligns the 6 committed sample reads on both CPU and
GPU, prints each read's **CIGAR** + a junction summary, shows the
GPU-vs-CPU agreement check (`RESULT: PASS`), and prints timing to stderr.

## Data

- **Sample (committed):** `data/sample/reads_sample.txt` — a tiny, offline,
  **synthetic** 3-exon gene model + 6 reads, so the demo runs with zero
  downloads. Engineered so junction reads have an interpretable CIGAR.
- **Full / real datasets:** `scripts/download_data.ps1` / `.sh` print pointers
  (they do not auto-download; some sources are large/credentialed).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ENCODE RNA-seq FASTQs (https://www.encodeproject.org/); GENCODE annotation (https://www.gencodegenes.org/); SRA RNA-seq benchmarks (SEQC/MAQC) (https://www.ncbi.nlm.nih.gov/sra); GTEx tissue RNA-seq (https://gtexportal.org/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). Each
read prints its best score, endpoint, intron count, and **CIGAR**:

```
  read  2: len= 24 score= 46 end=(i= 24,j= 82) introns=1  CIGAR=12M40N12M
  read  5: len= 36 score= 68 end=(i= 36,j=148) introns=2  CIGAR=6M40N24M48N6M
```

The program aligns on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree **exactly** — scores, endpoints,
*and every DP-table cell* (tolerance `== 0`, because the DP is integer; see
`docs/PATTERNS.md §4`). That exact agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the batch, runs CPU + GPU, verifies
   (scores/endpoints/cells), tracebacks each CIGAR, reports.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the **shared** `__host__
   __device__` recurrence (`cell_recurrence`, `is_canonical_intron`,
   `intron_score`) — the single source of truth for the math.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the "one block per
   read" mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the batched kernel + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline,
   the loader, and the CIGAR traceback.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

STAR (https://github.com/alexdobin/STAR) — fastest spliced RNA aligner (GPU suffix-array querying target); HISAT2 (https://github.com/DaehwanKimLab/hisat2) — graph-index RNA aligner; minimap2 (https://github.com/lh3/minimap2) — long-read splice-aware (GPU wavefront extension target); AGAThA — GPU-accelerated guided sequence alignment for long-read mapping (verify URL).

- **STAR** — read its Maximal Mappable Prefix seeding + splice-junction
  stitching; that is the suffix-array seeding our reduced version omits.
- **HISAT2** — the graph FM-index that *encodes* known splice sites into the
  index itself (so the aligner "knows" the junctions); contrast with our *de
  novo* GT-AG scoring.
- **minimap2 `-ax splice`** — long-read splice-aware chaining; the closest
  cousin to the wavefront extension `THEORY.md §real-world` describes.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Batched independent jobs** (`docs/PATTERNS.md §1`, the "many small independent
alignments" row): one **thread block per read**, a global-memory DP table per
block, no atomics and no cross-block synchronisation. The per-element physics is
a **shared `__host__ __device__` core** (`docs/PATTERNS.md §2`) so the CPU
reference and the GPU kernel compute bit-identical **integer** scores — enabling
an **exact** (`== 0`) verification. The catalog's full pattern (page-locked
suffix arrays, GPU hash-table splice indices, Thrust seed sorting, multi-sample
streams) is described, not implemented — see *Limitations*.

## Exercises

1. **Affine introns.** Add a tiny per-base intron cost so longer introns are
   gently penalised, and observe whether the double-junction read still prefers
   two canonical introns over one long non-canonical skip.
2. **Non-canonical sites.** Extend `is_canonical_intron` to also reward the minor
   `GC…AG` and `AT…AC` splice classes with a smaller bonus; regenerate a sample
   that uses one and confirm the CIGAR still recovers it.
3. **Intra-read wavefront.** Replace the single-thread-per-block DP with an
   anti-diagonal wavefront (à la project `3.01`) so a *long* read uses the whole
   block; keep the intron `N` move and re-verify against the CPU.
4. **Seeding first.** Add a k-mer seed step (only run the DP in a band around a
   matching seed) — the real STAR speed-up — and measure how much DP work it
   saves.
5. **Splice-site entropy.** Swap the fixed canonical bonus for a small
   position-weight-matrix (maximum-entropy) donor/acceptor score and see how it
   changes borderline junctions.

## Limitations & honesty

- **Reduced scope (labeled).** This aligns short reads against a *short* reference
  gene, not a 3 Gb genome. There is **no** suffix-array / FM-index seeding, no
  chaining, no multi-mapping resolution, no paired-end logic, no quality scores.
  The catalog's full pipeline is described in `THEORY.md §real-world`.
- **Synthetic data.** Everything in `data/sample/` is synthetic and **labeled
  synthetic**; intron interiors are engineered to avoid spurious splice sites so
  the demo is unambiguous. Real reads have errors, non-canonical sites, and
  kilobase introns.
- **Splice-boundary ambiguity is real.** When the bases flanking a junction allow
  an equally-scoring ±1 shift of the boundary, the optimum is genuinely
  degenerate; our deterministic tie-break picks one. Production tools apply extra
  heuristics (annotation, motif priors) here.
- **Timing is a teaching artifact, never a benchmark** (CLAUDE.md §12). A handful
  of short reads barely occupies the GPU; the batched design pays off at millions
  of reads.
- **Not for clinical use.** No diagnostic or therapeutic claim is made.
