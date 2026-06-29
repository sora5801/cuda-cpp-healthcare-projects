# THEORY — 12.2 Metabolomics Spectral Processing

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

### 12.2 Metabolomics Spectral Processing 🟡 · Active R&D
- **Deep dive:** Metabolomics LC-MS/MS produces thousands of spectra per sample that must be denoised, deconvoluted, and matched against spectral libraries (e.g., MassBank, HMDB). Key GPU-amenable steps: (1) denoising via 2D Gaussian filtering on the (m/z, retention-time) ion map, (2) spectral library matching via batched dot-product between observed and reference spectra (identical to proteomics search but with small molecule fragmentation patterns), and (3) isotope deconvolution using the Averagine model for charge-state assignment. GPU batch cross-correlation across tens of thousands of library entries per observed spectrum replaces sequential CPU loops.
- **Key algorithms:** Gaussian kernel smoothing on MS1 ion maps; isotope deconvolution via Averagine model; dot-product spectral library matching; modified cosine similarity for spectral networking (GNPS); mass-defect filtering; retention time alignment via dynamic time warping (DTW).
- **Datasets:** GNPS / MassIVE metabolomics datasets (https://gnps.ucsd.edu/); HMDB — Human Metabolome Database spectral library (https://hmdb.ca/); MetaboLights — metabolomics studies repository (https://www.ebi.ac.uk/metabolights/); MassBank of North America — MS/MS spectral library (https://mona.fiehnlab.ucdavis.edu/).
- **Starter repos/tools:** GNPS (https://gnps.ucsd.edu/) — spectral networking platform (GPU matching target); MZmine3 (https://github.com/mzmine/mzmine3) — open-source LC-MS processing (GPU acceleration integration target); SIRIUS (https://github.com/boecker-lab/sirius) — molecular formula / structure prediction; OpenMS (https://github.com/OpenMS/OpenMS) — LC-MS processing suite.
- **CUDA libraries & GPU pattern:** cuFFT for cross-correlation in spectral library matching; custom 2D Gaussian smoothing CUDA kernels on ion maps; thrust for m/z sorted spectral vector operations; batched cosine similarity via cuBLAS GEMM (spectra as rows of a matrix); GPU-resident library matrix for parallel dot-product.

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
