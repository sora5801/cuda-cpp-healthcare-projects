# THEORY — 2.11 Cryo-EM CTF Estimation & Particle Picking

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

### 2.11 Cryo-EM CTF Estimation & Particle Picking 🟡 · Active R&D

- **Deep dive:** Before reconstruction, cryo-EM processing requires estimating the contrast transfer function (CTF) from each micrograph and then detecting protein particle positions (particle picking). CTF estimation fits a parametric model to power spectra computed via GPU FFT. Particle picking using template matching requires cross-correlation of the micrograph with reference projections — an O(N·M) operation over image patches and reference orientations, naturally GPU-parallelized. Deep learning pickers (TOPAZ, crYOLO) run GPU CNN inference on tiled micrographs. Both stages process thousands of micrographs in real time.
- **Key algorithms:** CTF power spectrum estimation (Thon rings fitting), 2D cross-correlation template matching, GPU-batched FFT for power spectra, CNN-based particle detection (YOLO/TOPAZ), active learning picker improvement.
- **Datasets:** EMPIAR micrograph archives (https://www.ebi.ac.uk/empiar/); EMPIAR-10025 (β-galactosidase), EMPIAR-10064 (80S ribosome); curated picking benchmarks from CryoBench (verify URL); RELION tutorial datasets (https://relion.readthedocs.io).
- **Starter repos/tools:** RELION CtfFind/MotionCor2 interface (https://github.com/3dem/relion) — GPU CTF + motion correction; TOPAZ (https://github.com/tbepler/topaz) — deep learning particle picker with GPU; crYOLO (https://cryolo.readthedocs.io) — YOLO-based GPU particle detector; CTFFIND4 (https://grigoriefflab.umassmed.edu/ctffind4) — GPU-accelerated CTF estimation.
- **CUDA libraries & GPU pattern:** cuFFT for micrograph power spectrum; CUDA 2D cross-correlation for template matching; custom CUDA FFT-based NCC; PyTorch GPU CNN for TOPAZ/crYOLO inference; multi-stream processing for parallel micrograph batches.

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
