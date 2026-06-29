# THEORY — 4.20 Dual-Energy / Spectral CT Reconstruction

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

### 4.20 Dual-Energy / Spectral CT Reconstruction 🟡 · Active R&D
- **Deep dive:** Dual-energy CT (DECT) acquires sinograms at two X-ray spectra (e.g., 80 kV and 140 kV) to enable material decomposition (separating water vs. iodine basis materials, or bone vs. soft tissue). Material decomposition in projection space requires solving a 2×2 nonlinear system per sinogram bin (~10⁸ bins), each requiring Newton iteration — trivially parallel across bins on GPU. Photon-counting CT (PCCT) extends this to 4–8 energy bins, increasing the system size to 8×8 and multiplying GPU compute by 4× but enabling K-edge imaging of contrast agents. Image-domain decomposition avoids projection-space issues but requires iterative reconstruction at each energy.
- **Key algorithms:** Projection-domain material decomposition (Newton iteration per sinogram bin), image-domain material decomposition, basis-material iterative CT (ADMM), virtual monoenergetic imaging, K-edge subtraction, photon-counting spectral reconstruction, GPU splitting-based DECT ADMM.
- **Datasets:** AAPM Spectral CT challenge datasets (verify URL at aapm.org); MARS photon-counting CT datasets (https://www.marsbioimaging.com/); TCIA DECT collections; simulated DECT from published XCAT phantom.
- **Starter repos/tools:** ASTRA (https://github.com/astra-toolbox/astra-toolbox) — multi-energy projection/backprojection primitives; TIGRE (https://github.com/CERN/TIGRE) — spectral CT reconstruction; ODL (https://github.com/odlgroup/odl) — material decomposition operators; splitting-based GPU DECT paper code (https://arxiv.org/abs/1905.00934 — verify repo link in paper).
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for per-bin Newton iteration (one thread per sinogram bin, 2×2 system solve in registers); cuFFT for spectral filter; shared memory for energy-bin grouped bins; cuBLAS for joint iterative reconstruction across energy channels.

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
