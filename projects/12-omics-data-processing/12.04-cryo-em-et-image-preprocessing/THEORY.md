# THEORY — 12.4 Cryo-EM / ET Image Preprocessing

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

### 12.4 Cryo-EM / ET Image Preprocessing 🟢 · Established
- **Deep dive:** Cryo-electron microscopy produces thousands of noisy micrographs (4k×4k pixels) that must be motion-corrected, CTF-estimated, particle-picked, and 2D/3D classified before structure determination. CryoSPARC and RELION both natively use CUDA for all major processing steps: motion correction via cross-correlation in Fourier space (cuFFT), CTF estimation via Thon ring fitting on GPU, particle picking via neural network (Topaz, crYOLO), and 3D refinement via GPU-accelerated back-projection and real-space expectation-maximisation. A single H100 processes hundreds of micrographs per minute end-to-end, enabling real-time feedback during cryo-EM sessions.
- **Key algorithms:** Fourier-space cross-correlation for frame alignment (MotionCor2); CTF fitting via Thon ring power spectrum (CTFFIND); 2D class averaging (RELION E-M); 3D gold-standard FSC refinement; CNN particle picking (Topaz); back-projection 3D reconstruction; Wiener filter CTF correction.
- **Datasets:** EMDB — Electron Microscopy Data Bank, raw micrographs and maps (https://www.ebi.ac.uk/emdb/); EMPIAR — raw cryo-EM micrograph repository (https://www.ebi.ac.uk/empiar/); wwPDB cryo-EM entries (https://www.rcsb.org/); CryoSPARC demo datasets (https://cryosparc.com/download).
- **Starter repos/tools:** CryoSPARC (https://cryosparc.com/) — fully GPU-native cryo-EM pipeline, particle picking through 3D refinement; RELION4 (https://github.com/3dem/relion) — GPU-accelerated 3D classification and refinement; Topaz (https://github.com/tbepler/topaz) — GPU CNN particle picker; MotionCor2 (verify URL — Zheng lab UCSF) — GPU frame alignment.
- **CUDA libraries & GPU pattern:** cuFFT for Fourier-domain frame alignment and CTF power spectrum; cuDNN for CNN particle picking; custom back-projection CUDA kernels; atomic operations for back-projection accumulation; multi-GPU 3D refinement with gradient averaging.

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
