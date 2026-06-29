# THEORY — 5.7 Brachytherapy Dose & Source Modeling

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

### 5.7 Brachytherapy Dose & Source Modeling 🟢 · Established
- **Deep dive:** Brachytherapy (BT) delivers dose from radioactive sources (Ir-192 HDR, Pd-103, I-125) implanted inside or adjacent to the tumor. TG-43 formalism computes dose analytically from tabulated radial and anisotropy functions per source dwell position; for an HDR plan with 50 dwell positions in a prostate implant, GPU parallelization across (source, voxel) pairs reduces plan calculation from seconds to milliseconds. Beyond TG-43, model-based dose algorithms (MBDCA) — Acuros BT, Monte Carlo — account for tissue heterogeneity and inter-source shielding, requiring the same GPU particle-transport infrastructure as external-beam MC. Real-time BT dose visualization on TRUS/fluoroscopy feed requires GPU latency <100 ms.
- **Key algorithms:** TG-43 dose formalism (radial dose function, anisotropy function), superposition of point-source kernels, MBDCA (model-based dose calculation algorithm), MC for BT (Geant4-TOPAS, EGSnrc BrachyDose), shielding correction for multi-source, real-time dose overlay on TRUS imaging.
- **Datasets:** AAPM TG-43 consensus datasets (radial/anisotropy tables — https://www.aapm.org/pubs/reports/); TCIA prostate BT CT datasets; ESTRO ACROP BT guideline test cases; BrachyView QA data (verify URL).
- **Starter repos/tools:** BrachyDose (via EGSnrc, https://github.com/nrc-cnrc/EGSnrc) — EGSnrc BT MC user code; TOPAS-BrachyDose (https://github.com/topasmc) — Geant4-based BT MC; PyTG43 (https://github.com/GregSal/PyTG43 — verify URL) — Python TG-43 dose calculator; matRad BT module (https://github.com/e0404/matRad) — MATLAB BT dose and optimization.
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for TG-43 dose (grid of threads covering output voxels; inner loop over source dwell positions; tables in constant memory); cuRAND for MC BT photon sampling; texture memory for 2D anisotropy function tables; warp-level reduction for summing source contributions.

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
