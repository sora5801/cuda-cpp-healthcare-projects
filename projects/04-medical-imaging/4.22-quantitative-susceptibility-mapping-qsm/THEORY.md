# THEORY — 4.22 Quantitative Susceptibility Mapping (QSM)

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

### 4.22 Quantitative Susceptibility Mapping (QSM) 🟡 · Active R&D
- **Deep dive:** QSM reconstructs tissue magnetic susceptibility (χ) from gradient-echo phase data in a 3D volume. The pipeline involves phase unwrapping (PUROR, ROMEO), background field removal (PDF, SHARP, VSHARP), and dipole inversion (MEDI, TKD, iLSQR, deep learning). The dipole inversion is the computational bottleneck: the forward model in k-space is a multiplication by a dipole kernel (analytically known), but inversion is ill-posed at the magic angle (cone of zero crossing). Iterative MEDI minimization requires O(100) iterations of 3D FFT + gradient updates on a 256³ volume, each costing ~30 ms GPU vs. seconds CPU. Deep learning QSM (QSMnet, xQSM) replaces MEDI with a single GPU network forward pass (<1 s).
- **Key algorithms:** Phase unwrapping (PUROR, ROMEO, BEST path), SHARP/V-SHARP background removal, MEDI (morphology-enabled dipole inversion), TKD (threshold-based k-space division), iterative least-squares (iLSQR), deep learning dipole inversion (QSMnet, xQSM), total-variation regularized inversion.
- **Datasets:** QSM Reconstruction Challenge 2.0 (https://doi.org/10.1101/2020.11.25.397695 — data on Zenodo); HCP 7T multiecho GRE data (https://db.humanconnectome.org/); AHEAD dataset (Amsterdam Ultra-high field Adult lifespan Database); BioBank UKB (https://www.ukbiobank.ac.uk/).
- **Starter repos/tools:** QSMnet (https://github.com/SNU-LIST/QSMnet) — deep learning QSM on GPU; MEDI toolbox (http://pre.weill.cornell.edu/mri/pages/qsm.html — verify URL) — MATLAB MEDI + GPU options; ROMEO (https://github.com/korbinian90/ROMEO) — fast phase unwrapping; STISuite (verify URL) — STI + QSM MATLAB toolbox.
- **CUDA libraries & GPU pattern:** cuFFT for dipole kernel multiplication in k-space per MEDI iteration; custom CUDA gradient/divergence operators for TV regularization; cuBLAS for conjugate gradient solver; memory layout: complex float32 arrays, FFT in-place.

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
