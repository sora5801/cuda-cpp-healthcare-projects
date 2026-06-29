# THEORY — 2.31 Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction

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

### 2.31 Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction 🟡 · Active R&D

- **Deep dive:** Cryo-ET tilt-series reconstruction requires (1) frame alignment (beam-induced motion), (2) tilt-series alignment (fiducial or fiducial-free), and (3) tomogram reconstruction (weighted back-projection or iterative SART/ASTRA). All three steps are GPU-parallelizable: GPU-accelerated SART iterates over projection angles simultaneously; WBP uses GPU FFT and filter application. IMOD, AreTomo, and etomo handle tilt-series alignment; the ASTRA Toolbox provides GPU iterative reconstruction via CUDA. Cryo-ET remains limited by the missing wedge artifact, which deep learning (IsoNet) corrects post hoc on GPU.
- **Key algorithms:** Weighted back-projection (WBP), SART (simultaneous algebraic reconstruction), AreTomo beam-induced motion correction, fiducial marker alignment, beam-induced motion correction (MotionCor2-TomoTilt), iterative reconstruction convergence.
- **Datasets:** EMPIAR tilt series archives (https://www.ebi.ac.uk/empiar/); EMDB subtomogram averages (https://www.ebi.ac.uk/emdb/); SHREC cryo-ET benchmark (verify URL); in situ ribosome tilt series (EMPIAR-10045).
- **Starter repos/tools:** IMOD (https://bio3d.colorado.edu/imod/) — standard tomographic reconstruction suite; ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU CUDA reconstruction algorithms; AreTomo2 (https://github.com/czimaginginstitute/AreTomo2) — GPU tilt-series alignment; IsoNet (https://github.com/IsoNet-cryoET/IsoNet) — GPU deep learning missing wedge correction.
- **CUDA libraries & GPU pattern:** Custom CUDA WBP kernel over tilt projection angles; cuFFT for filter application in filtered back-projection; GPU SART iteration with CUDA atomic updates; PyTorch CNN for IsoNet missing-wedge correction; multi-GPU for large tomogram reconstruction.

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
