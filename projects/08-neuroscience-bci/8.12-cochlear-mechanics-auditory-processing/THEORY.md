# THEORY — 8.12 Cochlear Mechanics & Auditory Processing

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

### 8.12 Cochlear Mechanics & Auditory Processing 🟡 · Active R&D
- **Deep dive:** The cochlea performs mechanical frequency decomposition via basilar membrane (BM) traveling waves, transforming sound to a tonotopic neural code via inner hair cells (IHCs) and auditory nerve fibers (ANFs). GPU simulation of a 3D BM model (finite element) or active cochlear model (outer hair cell electromotility — prestin) with coupled fluid mechanics and IHC/ANF spike generation supports hearing prosthesis design, audiogram prediction, and noise-induced hearing loss modeling.
- **Key algorithms:** 1D/2D/3D basilar membrane wave equation (FEM/FD), fluid-structure interaction for perilymph-BM coupling, outer hair cell electromotility (Prestin ODE), inner hair cell transducer (MET channel), auditory nerve fiber spike model (Zilany-Bruce), gammatone filterbank (frequency-domain equivalent), cochlear implant electrode models.
- **Datasets:** NH Hearing database (verify URL at nhlibrary.org); Auditory Model Toolbox benchmark datasets (https://amtoolbox.org); PhysioNet auditory brainstem response datasets (https://physionet.org); cochlear implant stimulation datasets from Cochlear Ltd (proprietary; verify institutional access).
- **Starter repos/tools:** CoNNear cochlea (https://github.com/HearingTechnology/CoNNear_cochlea) — PyTorch DNN cochlear mechanics model for real-time inference; mrkrd/cochlea (https://github.com/mrkrd/cochlea) — Python inner ear models interfacing NEURON/Brian; Auditory Model Toolbox (https://amtoolbox.org) — MATLAB/Octave/Python cochlear models; NEST simulator (https://github.com/nest/nest-simulator) — ANF population spiking.
- **CUDA libraries & GPU pattern:** cuFFT for gammatone filterbank (bank of FIR filters via FFT convolution); custom CUDA FEM kernel for 1D BM wave equation (Thomas tridiagonal along BM length); batched ODE for ANF spike generation (one thread per fiber); pattern: frequency-band-parallel GPU computation, each warp handles one characteristic frequency band.

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
