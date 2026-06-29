# THEORY — 4.28 GPU-Accelerated DRR Generation for 2D/3D Registration

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

### 4.28 GPU-Accelerated DRR Generation for 2D/3D Registration 🟢 · Established
- **Deep dive:** Digitally reconstructed radiographs (DRRs) simulate X-ray images from 3D CT volumes for 2D/3D registration (aligning daily X-ray portal images to planning CT). Each DRR pixel integrates CT Hounsfield units along a ray path through the volume (Siddon's ray-tracing or tri-linear ray-marching); for a 400×400 DRR from a 512³ CT, ~6.4 × 10⁸ tri-linear interpolations are needed per DRR. Intensity-based 2D/3D registration requires 50–200 DRRs per optimization iteration (~10¹¹ operations total on CPU). GPU texture memory's built-in tri-linear hardware interpolation and embarrassing parallelism (one CUDA thread per DRR pixel) make this an ideal GPU workload, achieving 100×+ speedup.
- **Key algorithms:** Siddon ray-tracing, tri-linear ray-marching (GPU texture), Splatting vs. ray-casting DRR, mutual information / NCC / gradient-magnitude similarity, stochastic gradient descent 2D/3D registration, differentiable DRR (DiffDRR), neural DRR for fast iteration.
- **Datasets:** Gold Atlas prostate CT (https://www.goldenatlasproject.com/ — verify URL); TCIA prostate/lung CTs; AAPM TG-132 test cases; clinical CBCT + kV images (institutional IRB).
- **Starter repos/tools:** Plastimatch (https://plastimatch.org/) — GPU DRR generation tool; CUDA_DigitallyReconstructedRadiographs (https://github.com/fabio86d/CUDA_DigitallyReconstructedRadiographs) — GPU DRR Python library; DiffDRR (https://github.com/eigenvivek/DiffDRR) — differentiable DRR for gradient-based 2D/3D registration; RTK (https://github.com/RTKConsortium/RTK) — GPU ray-casting for DRR.
- **CUDA libraries & GPU pattern:** CUDA 3D texture with hardware tri-linear interpolation (zero-cost); one CUDA thread per output DRR pixel; ray-step loop in kernel; constant memory for projection geometry; multiple CUDA streams for simultaneous multi-view DRR generation.

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
