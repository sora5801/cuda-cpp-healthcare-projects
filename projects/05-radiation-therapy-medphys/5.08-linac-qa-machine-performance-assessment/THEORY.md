# THEORY — 5.8 Linac QA & Machine Performance Assessment

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

### 5.8 Linac QA & Machine Performance Assessment 🟢 · Established
- **Deep dive:** Linear accelerator (linac) quality assurance measures beam output, flatness, symmetry, and MLC leaf positions from portal dosimetry images or log files. GPU acceleration is applied in three areas: (1) rapid gamma-index computation comparing measured vs. planned dose distributions (3D gamma on a 200³ dose grid requires ~10⁹ distance searches), (2) EPID (electronic portal imaging device) image-based dose reconstruction converting 2D portal images to 3D dose via a GPU MC kernel, and (3) machine learning prediction of machine failures from large log-file datasets (training on GPU). Automated daily QA with immediate GPU-based analysis enables real-time feedback before the treatment session.
- **Key algorithms:** Gamma-index dose comparison (3D, distance-to-agreement + dose-difference), EPID portal dose reconstruction (MC kernel convolution on GPU), MLC leaf-gap analysis, Winston-Lutz test automation, trajectory log analysis, ML anomaly detection on linac logs.
- **Datasets:** AAPM TG-119 IMRT QA test cases; AAPM TG-218 tolerance criteria datasets; TCIA linac log datasets (verify URL); Varian/Elekta log file datasets from published QA studies; OpenMedPhys (https://github.com/jrkerns/awesome-medphys) reference datasets.
- **Starter repos/tools:** Pylinac (https://github.com/jrkerns/pylinac) — Python linac QA automation (image analysis, log files); PRIMO MC linac simulator (https://www.primoproject.net/ — verify URL); Plastimatch (https://plastimatch.org/) — GPU-accelerated gamma index; matRad (https://github.com/e0404/matRad) — plan-vs-measurement comparison.
- **CUDA libraries & GPU pattern:** Custom CUDA for 3D gamma index (each thread manages one reference-dose point, searches neighbor distance sphere in delivered dose volume); texture memory for delivered dose field; cuBLAS for log-file ML feature matrix; warp-level min-reduction for closest distance search.

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
