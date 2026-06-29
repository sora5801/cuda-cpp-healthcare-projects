# THEORY — 8.5 Neural Mass / Whole-Brain Dynamics Models

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

### 8.5 Neural Mass / Whole-Brain Dynamics Models 🟡 · Active R&D
- **Deep dive:** Neural mass models (Wilson-Cowan, Jansen-Rit, Kuramoto oscillators, Stuart-Landau) approximate the mean firing rate of cortical regions, coupled by structural connectivity matrices from diffusion tractography. The Virtual Brain (TVB) simulates 84–360 cortical + subcortical regions, each with an ODE system of 2–8 state variables, coupled via a time-delayed connectivity matrix (50–100 ms conduction delays). GPU parallelism is exploited both for region-level ODE integration and for ensemble simulations fitting personalized connectomes.
- **Key algorithms:** Wilson-Cowan / Jansen-Rit neural mass ODEs, Kuramoto phase oscillator network, Stuart-Landau Hopf normal form, delay differential equations (DDE) with ring buffer, structural connectivity eigenspectrum analysis, Bayesian parameter inference for connectome fitting, graph-theoretic network analysis.
- **Datasets:** Human Connectome Project structural connectivity (https://db.humanconnectome.org); TVB compatible connectome datasets (https://www.thevirtualbrain.org/tvb/zwei/client-area); OpenNeuro fMRI for BOLD signal comparison (https://openneuro.org); ADNI structural MRI for patient-specific connectomes (https://adni.loni.usc.edu).
- **Starter repos/tools:** The Virtual Brain (https://github.com/the-virtual-brain/tvb-root) — whole-brain neural mass simulator with GPU via Numba/CUDA backends; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — multiscale NEURON network with structural connectivity import; Brian2 (https://github.com/brian-team/brian2) — network ODE with Brian2CUDA; MOOSE (https://github.com/BhallaLab/moose-core) — compartmental neural mass implementation.
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for per-region ODE with ring-buffer delay lookup; cuBLAS for connectivity matrix-vector multiply (coupling term); CUDA Thrust for eigenvalue analysis; pattern: one CUDA thread per region, delay ring-buffer in shared memory, connectivity matrix in texture memory for caching.

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
