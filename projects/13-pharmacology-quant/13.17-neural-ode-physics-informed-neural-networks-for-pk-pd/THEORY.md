# THEORY — 13.17 Neural-ODE & Physics-Informed Neural Networks for PK/PD

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

### 13.17 Neural-ODE & Physics-Informed Neural Networks for PK/PD 🔴 · Frontier/Theoretical

- **Deep dive:** Replaces explicit pharmacokinetic ODEs with neural networks embedded within differential equations (Neural ODEs) or constrains neural architectures to satisfy ODE physics (Physics-Informed Neural Networks, PINNs). This allows learning latent pharmacokinetic dynamics from sparse clinical observations without specifying a mechanistic compartmental model. The GPU bottleneck is differentiating through the ODE solver (adjoint sensitivity method) for backpropagation, implemented in torchdiffeq. For PINNs, the collocation loss (residual of the ODE at sample points) is evaluated in batches on GPU. Recent Latent Neural-ODE approaches (arXiv:2602.03215) model-informed precision dosing with 15% fewer AEs than standard dosing.
- **Key algorithms:** Neural ODE (Chen et al. 2018), adjoint sensitivity for backprop through ODE, Physics-Informed Neural Networks (PINNs), Universal Differential Equations (UDEs), Latent ODE with VAE encoder, Gaussian process ODE priors, Fourier Neural Operators for PDE-based dosing, symbolic regression to recover interpretable ODE from data.
- **Datasets:**
  - Latent Neural-ODE precision dosing dataset (https://arxiv.org/abs/2602.03215) — model-informed dosing with neural ODE
  - MIMIC-IV ICU PK data — vancomycin/aminoglycoside time series (https://physionet.org/content/mimiciv/)
  - Published population PK datasets (vancomycin, busulfan) from PharmPK listserv (verify URL)
  - Synthetic NLME benchmark datasets from Monolix/NONMEM validation suites (verify URL)
- **Starter repos/tools:**
  - torchdiffeq (https://github.com/rtqichen/torchdiffeq) — Neural ODE with GPU-accelerated adjoint sensitivity
  - DiffEqFlux.jl (https://github.com/SciML/DiffEqFlux.jl) — Universal Differential Equations in Julia with GPU
  - DeepXDE (https://github.com/lululxvi/deepxde) — GPU PINN framework for PDE/ODE-constrained learning
  - SciMLBenchmarks (https://github.com/SciML/SciMLBenchmarks.jl) — benchmarks for neural ODE solvers
- **CUDA libraries & GPU pattern:** torchdiffeq adjoint ODE solver on GPU via PyTorch CUDA, cuBLAS for neural ODE network forward pass, JAX XLA for JIT-compiled PINN training; pattern: batched neural ODE integration with GPU-resident adjoint sensitivity gradients.

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
