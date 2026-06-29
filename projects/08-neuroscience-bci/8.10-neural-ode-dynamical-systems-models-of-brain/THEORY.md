# THEORY — 8.10 Neural ODE / Dynamical Systems Models of Brain

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

### 8.10 Neural ODE / Dynamical Systems Models of Brain 🔴 · Frontier/Theoretical
- **Deep dive:** Neural ODEs parameterize the time derivative of hidden neural state as a neural network, enabling continuous-time models of brain dynamics that can be fit to irregular-interval neural recordings and extrapolated to unseen time points. Applied to whole-brain fMRI or calcium imaging, they learn latent dynamical manifolds underlying cognition. Adjoint sensitivity (checkpointed backpropagation through the ODE solver) is memory-intensive and GPU-critical; the adjoint method requires storing only a constant number of activations regardless of integration depth.
- **Key algorithms:** Neural ODE (Runge-Kutta adjoint), augmented neural ODE (ANODE), latent ODE / SDE (VAE + neural ODE), flow matching, score-based generative modeling for neural trajectories, continuous normalizing flow (CNF), Gaussian process ODE, reservoir computing (echo state networks).
- **Datasets:** Human Connectome Project resting-state fMRI (https://db.humanconnectome.org); DANDI electrophysiology (https://dandiarchive.org); Allen Brain Observatory calcium imaging (https://portal.brain-map.org); NLB Neural Latents Benchmark (https://neurallatents.github.io).
- **Starter repos/tools:** torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU neural ODE with adjoint backpropagation; torchsde (https://github.com/google-research/torchsde) — stochastic differential equation neural models on GPU; LFADS (https://github.com/google-research/google-research/tree/master/lfads) — RNN-based latent factor analysis; Diffrax (https://github.com/patrick-kidger/diffrax) — JAX-based GPU ODE/SDE solver suite.
- **CUDA libraries & GPU pattern:** cuDNN for neural network RHS evaluation; checkpointed adjoint via custom CUDA memory management; cuRAND for SDE noise sampling; pattern: time-reversed adjoint integration with activations recomputed on-the-fly, CUDA graph for repeated-pattern ODE step.

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
