# THEORY — 12.8 Isotope Pattern Matching & Charge Deconvolution

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

### 12.8 Isotope Pattern Matching & Charge Deconvolution 🟡 · Active R&D
- **Deep dive:** High-resolution mass spectrometry resolves isotope envelopes (the pattern of ¹²C, ¹³C, ²H, ¹⁸O peaks) that report the charge state and monoisotopic mass of each peptide or metabolite. Matching observed isotope patterns against theoretical Averagine distributions (or exact elemental isotope calculations via IsoSpec) across millions of features per LC-MS run is a quadratic search problem. GPU parallelism assigns one thread per candidate mass window, computing the dot product between observed and theoretical isotope patterns simultaneously across thousands of charge states and masses, replacing the sequential CPU sweep.
- **Key algorithms:** Averagine model for average elemental composition; Mercury / IsoSpec exact isotope pattern calculation via Poisson convolution; dot-product / cosine-similarity matching of isotope envelopes; Maximum Likelihood charge state assignment; THRASH deconvolution algorithm; Wavelet transform for isotope detection (IsotopeWavelet).
- **Datasets:** PRIDE ProteomeXchange high-resolution datasets (https://www.ebi.ac.uk/pride/); HMDB high-resolution metabolomics spectra (https://hmdb.ca/); MassBank (https://massbank.eu/); CPTAC iTRAQ/TMT quantitative proteomics (https://proteomics.cancer.gov/).
- **Starter repos/tools:** OpenMS (https://github.com/OpenMS/OpenMS) — comprehensive LC-MS toolkit with GPU integration hooks; IsoSpec (https://github.com/MatteoLacki/IsoSpec) — exact isotope pattern computation; Xtract (Thermo Fisher, proprietary) — charge deconvolution; pyOpenMS (https://github.com/OpenMS/OpenMS) — Python bindings for proteomics.
- **CUDA libraries & GPU pattern:** Batched dot-product CUDA kernels (one warp per candidate m/z window); cuFFT for wavelet-based isotope detection; shared-memory Averagine lookup tables; thrust for peak list sorting and deduplication; cuBLAS GEMM for charge-state × m/z scoring matrix.

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
