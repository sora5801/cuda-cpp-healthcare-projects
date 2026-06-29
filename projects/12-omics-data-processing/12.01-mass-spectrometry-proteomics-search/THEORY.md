# THEORY — 12.1 Mass-Spectrometry Proteomics Search

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

### 12.1 Mass-Spectrometry Proteomics Search 🟢 · Established
- **Deep dive:** Database peptide search correlates each observed MS/MS spectrum against thousands of theoretical peptide spectra from a protein sequence database, the most time-consuming step in proteomics. For a dataset of 100 k spectra against a human tryptic database of 1 M peptides (× 100 modifications), the search space is 10¹¹ comparisons; GPU parallelises scoring of thousands of theoretical spectra simultaneously per observed spectrum. GiCOPS (GPU-accelerated HiCOPS) achieves 1.2–5× speedup over CPU HiCOPS and >10× over older GPU tools like Tempest, using fragment-ion indexing on GPU. MSFragger uses hash-based fragment indexing on CPU but its inner scoring loop is a GPU acceleration target.
- **Key algorithms:** Fragment-ion indexing (hash/sorted lists of b/y-ions); Xcorr / HyperScore spectral dot product; fragment index mass offset search (open search); XCorr normalised cross-correlation; peptide-spectrum match (PSM) q-value estimation (Percolator); precursor mass matching and charge state deconvolution.
- **Datasets:** PRIDE / ProteomeXchange — proteomics data repository (https://www.ebi.ac.uk/pride/); PeptideAtlas — validated human peptide spectral library (https://www.peptideatlas.org/); CPTAC cancer proteomics datasets (https://proteomics.cancer.gov/); MassIVE — mass spectrometry data repository (https://massive.ucsd.edu/).
- **Starter repos/tools:** GiCOPS (https://github.com/pcdslab/gicops) — GPU HPC framework for database peptide search; MSFragger (https://github.com/Nesvilab/MSFragger) — ultra-fast hash-index search (CPU, GPU inner loop target); Tempest — CUDA spectral scoring (verify URL; legacy); OpenMS (https://github.com/OpenMS/OpenMS) — proteomics framework with GPU integration potential.
- **CUDA libraries & GPU pattern:** GPU hash tables for fragment ion indexing; batched dot-product CUDA kernels (one thread per theoretical peptide per observed spectrum); shared-memory spectral vector loading; cuFFT-based cross-correlation; multi-GPU database sharding.

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
