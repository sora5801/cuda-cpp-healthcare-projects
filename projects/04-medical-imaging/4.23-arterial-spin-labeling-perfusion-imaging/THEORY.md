# THEORY — 4.23 Arterial Spin Labeling & Perfusion Imaging

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

### 4.23 Arterial Spin Labeling & Perfusion Imaging 🟡 · Active R&D
- **Deep dive:** Arterial spin labeling (ASL) magnetically labels water protons in arterial blood upstream and images the resulting perfusion-weighted signal difference (labeled minus control). The signal change is only 0.5–1% of background signal, requiring averaging many pairs to achieve adequate SNR; acquisition of dynamic (time-resolved) ASL with 100+ pairs at 2 mm resolution produces datasets where kinetic model fitting (single/multi-delay Buxton model) per voxel is a Bayesian inverse problem amenable to GPU parallelization. Oxford_asl/BASIL uses variational Bayes inference, parallelized across voxels on GPU. 3D multi-delay ASL combined with compressed sensing requires per-timepoint NUFFT reconstruction — same GPU bottleneck as standard CS-MRI.
- **Key algorithms:** Buxton kinetic model (single/multi-delay), pulsed ASL (PASL), pseudo-continuous ASL (pCASL), Bayesian kinetic model fitting (BASIL), variational Bayes per voxel, compressed sensing 3D dynamic ASL, T1 partial-volume correction.
- **Datasets:** HCP ASL data (https://db.humanconnectome.org/); OpenNeuro ASL datasets (https://openneuro.org/ — search "ASL"); ISMRM 2015 ASL challenge data; UK Biobank ASL pilot data.
- **Starter repos/tools:** FSL BASIL (https://fsl.fmrib.ox.ac.uk/fsl/docs/physiological/basil.html) — Bayesian ASL analysis, GPU-parallelizable voxel fits; BART (https://github.com/mrirecon/bart) — dynamic ASL CS reconstruction; ExploreASL (https://github.com/ExploreASL/ExploreASL) — multi-center ASL pipeline; SigPy (https://github.com/mikgroup/sigpy) — dynamic CS-ASL reconstruction.
- **CUDA libraries & GPU pattern:** Per-voxel independent Bayesian fit (one CUDA thread per voxel, Newton-Raphson or variational updates); cuBLAS for kinetic model matrix products; shared memory for model time-course templates; cuFFT for dynamic CS-ASL k-space reconstruction.

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
