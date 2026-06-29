# THEORY — 5.15 Proton CT & Ion Imaging Reconstruction

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

### 5.15 Proton CT & Ion Imaging Reconstruction 🔴 · Frontier/Theoretical
- **Deep dive:** Proton CT (pCT) measures the residual range of individual protons after traversing a patient, converting to relative stopping power (RSP) maps directly for treatment planning — eliminating the Hounsfield-unit–to–RSP conversion uncertainty in X-ray CT. Each proton's path through tissue is a curved most-likely path (MLP) rather than a straight line; for 10⁸ protons per scan, computing all MLPs and binning them into a sinogram for reconstruction is a massively parallel GPU task. Iterative pCT reconstruction (POCS with RSP constraints, MLSD) requires forward/backprojection along curved proton paths, fundamentally different from X-ray cone-beam and requiring custom GPU geometry kernels. Clinical pCT scanners (IBA, PRaVDA) generate data at 10⁸ proton events/second — GPU is mandatory for any real-time capability.
- **Key algorithms:** Most-likely path (MLP) estimation (Highland formula, Gaussian scattering), list-mode proton CT reconstruction (CSPACS, MLSD), POCS with RSP box constraints, proton trajectory binning for FBP, iterative proton CT with scattering regularization, proton radiography for range verification.
- **Datasets:** PRaVDA proton CT datasets (verify URL); PRIMA proton CT consortium data (verify URL); TOPAS-generated pCT simulation data; ACE collaboration proton CT phantom datasets.
- **Starter repos/tools:** pCT reconstruction code from UCI/Santa Cruz collaboration (verify URL); TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — proton CT simulation; FRED (https://www.fredonline.eu/) — proton transport/range imaging; custom CUDA MLP projection repos (search GitHub "proton CT GPU most likely path").
- **CUDA libraries & GPU pattern:** One CUDA thread per detected proton (massively parallel MLP computation); cuBLAS for scattering covariance matrix updates; thrust sort for proton trajectory binning by projection angle; custom CUDA backprojection along curved MLP geometry; cuRAND for proton beam Monte Carlo sampling.

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
