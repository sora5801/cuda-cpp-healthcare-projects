# THEORY — 8.2 Spike Sorting

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

### 8.2 Spike Sorting 🟢 · Established
- **Deep dive:** Spike sorting identifies the firing times and cellular identities of individual neurons from raw extracellular voltage traces recorded on multi-electrode arrays (MEAs) or Neuropixels probes (384 channels × 30 kHz). The GPU bottleneck is template-matching: cross-correlating detected waveforms against hundreds of neuron templates across all channels simultaneously. Kilosort4 achieves this via GPU-accelerated template convolution, reducing hours of CPU sorting to minutes and enabling automated curation for large-scale Neuropixels datasets.
- **Key algorithms:** Whitening and common-average reference (CAR) preprocessing, threshold-based spike detection, PCA dimensionality reduction, template-matching (cross-correlation), expectation-maximization (EM) clustering, drift correction via continuous template registration, Gaussian mixture model (GMM) classification.
- **Datasets:** DANDI Archive Neuropixels datasets (https://dandiarchive.org); Allen Brain Observatory Neuropixels visual coding dataset (https://portal.brain-map.org); SpikeInterface benchmark datasets (https://spikeinterface.readthedocs.io); MountainSort benchmark datasets on Zenodo (search zenodo.org "spike sorting benchmark").
- **Starter repos/tools:** Kilosort4 (https://github.com/MouseLand/Kilosort) — GPU template-matching spike sorter, Python, CUDA; MountainSort5 (https://github.com/flatironinstitute/mountainsort5) — Flatiron Institute sorter with GPU preprocessing; SpikeInterface (https://github.com/SpikeInterface/spikeinterface) — unified Python framework wrapping 10+ sorters including GPU ones; Phy (https://github.com/cortex-lab/phy) — manual curation GUI for Kilosort output.
- **CUDA libraries & GPU pattern:** cuFFT for template convolution (FFT-based cross-correlation); cuBLAS for waveform-template matrix multiply; cuSPARSE for sparse cluster assignment; CUDA Thrust for peak-finding in filtered traces; pattern: sliding-window batch convolution with cuFFT, one FFT per channel-template pair in a batched call.

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
