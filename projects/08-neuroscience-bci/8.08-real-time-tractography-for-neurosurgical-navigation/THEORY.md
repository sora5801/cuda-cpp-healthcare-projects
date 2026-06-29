# THEORY — 8.8 Real-Time Tractography for Neurosurgical Navigation

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

### 8.8 Real-Time Tractography for Neurosurgical Navigation 🟡 · Active R&D
- **Deep dive:** Diffusion tensor imaging (DTI) tractography traces white matter fiber bundles from seed ROIs by integrating principal diffusion directions through the 3D DTI field (streamline tracking). Intraoperative real-time tractography updates the fiber map as brain shift occurs during surgery, requiring sub-second computation. GPU parallelizes thousands of independent streamline integrations (CUDA: one thread per seed). Probabilistic tractography (FSL BEDPOSTX) samples from diffusion parameter posteriors—thousands of Monte Carlo streamlines per seed—is also GPU-amenable.
- **Key algorithms:** Deterministic streamline tractography (FACT, Runge-Kutta 4th order), probabilistic tractography (FSL BEDPOSTX ball-and-stick model), fiber orientation distribution (FOD) from HARDI (spherical deconvolution), constrained spherical deconvolution (CSD), DSI/Q-ball imaging, anatomical tract atlas registration (MNI-space), curvature-limited streamline termination.
- **Datasets:** Human Connectome Project DT-MRI (https://db.humanconnectome.org); ADNI diffusion MRI (https://adni.loni.usc.edu); ISMRM 2015 Tractography Challenge dataset (verify URL — tractometer.org or ismrm.org); OpenNeuro diffusion MRI datasets (https://openneuro.org).
- **Starter repos/tools:** DIPY (https://github.com/dipy/dipy) — Python DTI/HARDI tractography with GPU acceleration via CuPy; FSL GPU tractography (GPU BEDPOSTX) (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT) — CUDA-accelerated probabilistic tractography; MRtrix3 (https://github.com/MRtrix3/mrtrix3) — constrained spherical deconvolution + tractography; TrackVis/DiffusionTool (verify URL) — surgical navigation-oriented fiber display.
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for parallel streamline integration (one thread per seed, RK4 over DTI field in texture memory); cuBLAS for tensor field operations; cuFFT for spherical harmonic convolution in CSD; pattern: texture-memory DTI field for fast interpolation, warp-level thread divergence handled by fixed-step integration.

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
