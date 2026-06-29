# THEORY — 12.7 DIA Proteomics Spectral Deconvolution

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

### 12.7 DIA Proteomics Spectral Deconvolution 🟡 · Active R&D
- **Deep dive:** Data-Independent Acquisition (DIA) proteomics (Spectronaut, DIA-NN, FragPipe-DIA) co-isolates and co-fragments all precursors in wide isolation windows, requiring deconvolution of chimeric MS2 spectra containing overlapping fragment ion series. The GPU bottleneck is the inner-loop scoring: for each DIA window, thousands of peptide fragment ion templates must be correlated with the observed chromatographic fragment traces (XIC), a batched sliding-window cross-correlation problem. DIA-BERT (2025) is a GPU-enabled transformer approach treating DIA spectrum sequences analogously to language tokens, enabling improved feature extraction with GPU inference.
- **Key algorithms:** Extracted ion chromatogram (XIC) correlation scoring; deconvolution of chimeric spectra via library matching; Gaussian smoothing of chromatographic peaks; semi-empirical spectral library generation; transformer-based DIA spectrum encoding (DIA-BERT); target-decoy FDR estimation.
- **Datasets:** PRIDE ProteomeXchange DIA datasets (https://www.ebi.ac.uk/pride/); CPTAC DIA cancer proteomics (https://proteomics.cancer.gov/); Proteome profiler benchmark DIA datasets (verify URL); DIA-NN benchmark datasets (https://github.com/vdemichev/DiaNN).
- **Starter repos/tools:** DIA-NN (https://github.com/vdemichev/DiaNN) — fast DIA software (GPU inner-loop target); FragPipe (https://github.com/Nesvilab/FragPipe) — MSFragger-based DIA pipeline; DIA-BERT (https://proteomicsnews.blogspot.com/2025/05/dia-bert-gpu-enabled-dia-analysis.html) — GPU transformer for DIA; Spectronaut (commercial, Biognosys) — industry DIA software.
- **CUDA libraries & GPU pattern:** cuFFT cross-correlation for XIC fragment trace matching; cuDNN transformer for DIA-BERT; batched sliding-window scoring kernels; GPU tensor for precursor×fragment scoring matrix; thrust for peak apex detection; multi-GPU for large clinical DIA cohorts.

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
