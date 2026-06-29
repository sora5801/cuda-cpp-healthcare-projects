# 12.17 — Metagenome-Assembled Genome (MAG) Binning

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.17`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

MAG binning clusters assembled contigs into genome bins representing distinct microbial species, using tetranucleotide frequency (TNF, a 256-dimensional feature vector per contig) and coverage across samples. The binning problem is a clustering problem in 256+N_sample dimensional space; GPU UMAP + GPU clustering (Leiden) of millions of contigs from complex soil or gut metagenomes reduces hours-long CPU pipelines to minutes. Deep learning binners (CONCOCT, SemiBin2) use variational autoencoders or self-supervised contrastive learning whose training and inference are GPU-native.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Tetranucleotide frequency (TNF) 256-dim feature extraction; GPU UMAP dimensionality reduction of contig TNF+coverage; Leiden clustering of contig UMAP graph; variational autoencoder (CONCOCT style) for contiguous-binning; contrastive learning (SemiBin2); checkM completeness/contamination scoring.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/metagenome-assembled-genome-mag-binning.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/metagenome-assembled-genome-mag-binning.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\metagenome-assembled-genome-mag-binning.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: CAMI metagenome benchmarks (https://data.cami-challenge.org/); HMP2 gut metagenomes (https://www.hmpdacc.org/); JGI IMG/M — environmental metagenomes (https://img.jgi.doe.gov/); MGnify metagenome assemblies (https://www.ebi.ac.uk/metagenomics/).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

SemiBin2 (https://github.com/BigDataBiology/SemiBin) — self-supervised contrastive learning binner (GPU-trainable); CONCOCT (https://github.com/BinPro/CONCOCT) — Gaussian mixture model binner; Vamb (https://github.com/RasmussenLab/vamb) — variational autoencoder MAG binner with GPU training; rapids-singlecell UMAP (https://github.com/scverse/rapids_singlecell) — GPU UMAP for contig embedding.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuML UMAP for TNF+coverage contig embedding; cuGraph Leiden for contig clustering; cuDNN for VAE encoder/decoder training; cuDF for contig feature matrix; one CUDA thread per contig coverage computation; multi-GPU gradient reduction for VAE training. -- **Sources used for verification:** [CUDASW++4.0 — BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-024-05965-6) [CUDASW4 GitHub](https://github.com/asbschmidt/CUDASW4) [MMseqs2-GPU — Nature Methods](https://www.nature.com/articles/s41592-025-02819-8) [MMseqs2 GitHub](https://github.com/soedinglab/MMseqs2) [NVIDIA Parabricks Documentation](https://docs.nvidia.com/clara/parabricks/latest/) [Dorado GitHub](https://github.com/nanoporetech/dorado) [f5c GitHub](https://github.com/hasindu2008/f5c) [Remora GitHub](https://github.com/nanoporetech/remora) [GenomeWorks GitHub](https://github.com/NVIDIA-Genomics-Research/GenomeWorks) [racon-GPU GitHub](https://github.com/NVIDIA-Genomics-Research/racon-gpu) [rapids-singlecell GitHub](https://github.com/scverse/rapids_singlecell) [RAPIDS single-cell examples GitHub](https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) [ScaleSC — Bioinformatics Advances](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321287/) [rctd-py GitHub](https://github.com/p-gueguen/rctd-py) [Rapid GPU pangenome layout — SC 2024](https://www.csl.cornell.edu/~zhiruz/pdfs/pangenome-layout-sc2024.pdf) [PGGB GitHub](https://github.com/pangenome/pggb) [CARE GitHub](https://github.com/fkallen/CARE) [MetaCache-GPU preprint](https://arxiv.org/pdf/2106.08150) [GiCOPS GPU proteomics](https://www.nature.com/articles/s41598-023-43033-w) [NovoBench GitHub](https://github.com/jingbo02/NovoBench) [Cas-OFFinder GitHub](https://github.com/snugel/cas-offinder) [fair-esm GitHub](https://github.com/facebookresearch/esm) [Juicer/HiCCUPS GitHub](https://github.com/aidenlab/juicer) [DIA-BERT GPU DIA analysis](https://proteomicsnews.blogspot.com/2025/05/dia-bert-gpu-enabled-dia-analysis.html) [Darwin GPU overlap paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7495891/) [CUDAMPF HMMER GPU — BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0946-4) [CUDA-MEME / mCUDA-MEME](https://cuda-meme.sourceforge.io/homepage.htm) [GPU GWAS-Flow preprint](https://www.biorxiv.org/content/10.1101/783100) [GPU-GWAS GitHub](https://github.com/STRIDES-Codes/GPU-GWAS) [GPU-accelerated methylation — Bioinformatics Advances](https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac088/6855011) [GPU RNA-FISH decoding preprint](https://www.biorxiv.org/content/10.1101/2025.10.10.681751.full.pdf)

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
