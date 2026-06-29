# THEORY — 12.17 Metagenome-Assembled Genome (MAG) Binning

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. Diagrams in Mermaid/ASCII
> are welcome. See [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

<!-- =======================================================================
     The block below is the verbatim catalog deep-dive for this project,
     stamped in by scaffold.py as raw material. Use it to write the sections
     that follow, then DELETE it (or fold it into "The science"). Every
     TODO(theory) below must be completed before the project is "done".
     ======================================================================= -->

<details>
<summary>Catalog deep-dive (raw source material — fold into the sections below, then remove)</summary>

### 12.17 Metagenome-Assembled Genome (MAG) Binning 🟡 · Active R&D
- **Deep dive:** MAG binning clusters assembled contigs into genome bins representing distinct microbial species, using tetranucleotide frequency (TNF, a 256-dimensional feature vector per contig) and coverage across samples. The binning problem is a clustering problem in 256+N_sample dimensional space; GPU UMAP + GPU clustering (Leiden) of millions of contigs from complex soil or gut metagenomes reduces hours-long CPU pipelines to minutes. Deep learning binners (CONCOCT, SemiBin2) use variational autoencoders or self-supervised contrastive learning whose training and inference are GPU-native.
- **Key algorithms:** Tetranucleotide frequency (TNF) 256-dim feature extraction; GPU UMAP dimensionality reduction of contig TNF+coverage; Leiden clustering of contig UMAP graph; variational autoencoder (CONCOCT style) for contiguous-binning; contrastive learning (SemiBin2); checkM completeness/contamination scoring.
- **Datasets:** CAMI metagenome benchmarks (https://data.cami-challenge.org/); HMP2 gut metagenomes (https://www.hmpdacc.org/); JGI IMG/M — environmental metagenomes (https://img.jgi.doe.gov/); MGnify metagenome assemblies (https://www.ebi.ac.uk/metagenomics/).
- **Starter repos/tools:** SemiBin2 (https://github.com/BigDataBiology/SemiBin) — self-supervised contrastive learning binner (GPU-trainable); CONCOCT (https://github.com/BinPro/CONCOCT) — Gaussian mixture model binner; Vamb (https://github.com/RasmussenLab/vamb) — variational autoencoder MAG binner with GPU training; rapids-singlecell UMAP (https://github.com/scverse/rapids_singlecell) — GPU UMAP for contig embedding.
- **CUDA libraries & GPU pattern:** cuML UMAP for TNF+coverage contig embedding; cuGraph Leiden for contig clustering; cuDNN for VAE encoder/decoder training; cuDF for contig feature matrix; one CUDA thread per contig coverage computation; multi-GPU gradient reduction for VAE training.

</details>

---

## 1. The science

TODO(theory): The biology / medicine / physics being modeled — enough for a
reader to understand the *problem* before any math. What real-world question
does computing this answer?

## 2. The math

TODO(theory): The governing equations / formal problem statement, with **every
symbol defined** (units, ranges). State inputs, outputs, and the objective.

## 3. The algorithm

TODO(theory): Step-by-step. Include **complexity analysis**: serial cost vs. the
parallel work/depth. Where is the arithmetic intensity? What is the data-access
pattern?

## 4. The GPU mapping

TODO(theory): How the algorithm becomes **threads / blocks / grids**.
- Thread-to-data mapping (which thread owns which element).
- Launch configuration and the reasoning (block size, grid size).
- Memory hierarchy used and **why**: global / shared / registers / constant /
  texture. Where is the bandwidth bottleneck? What is the occupancy story?
- Which CUDA library (cuBLAS / cuFFT / cuRAND / cuSOLVER / Thrust) does what,
  and what it would take to write that step by hand (no black boxes — §6.1.6).

```
TODO(theory): an ASCII or Mermaid diagram of the grid/block decomposition.
```

## 5. Numerical considerations

TODO(theory): Precision (FP32 vs FP64) and why. Stability. Race conditions and
whether atomics are used. **Determinism**: does the parallel reduction reorder
floating-point sums? If so, say so and quantify the caveat.

## 6. How we verify correctness

TODO(theory): The CPU reference (`src/reference_cpu.cpp`), the **tolerance** and
why that value, and the edge cases checked. Explain why agreement between an
independent serial implementation and the GPU implementation is convincing
evidence of correctness.

## 7. Where this sits in the real world

TODO(theory): How production tools (named in the catalog "Prior art") do this
differently — what they add (scale, accuracy, features) that this teaching
version omits. If this is a 🔴 frontier project shipped as a reduced-scope
teaching version, describe the full approach here.

---

## References

TODO(theory): Papers, docs, and the starter repos from the catalog, with one
line each on what to learn from them.
