# THEORY — 14.11 Differentiable Simulation for Biomedicine

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

### 14.11 Differentiable Simulation for Biomedicine 🟡 · Active R&D

- **Deep dive:** Differentiable physics simulators propagate gradients through the entire simulation (FEM, CFD, rigid-body dynamics, particle dynamics), enabling gradient-based optimization of simulator parameters, boundary conditions, or material properties against experimental observations. NVIDIA Warp achieves up to 669× CPU speedup for GPU-differentiable simulation with seamless PyTorch/JAX integration. In biomedicine, differentiable FEM tunes patient-specific tissue stiffness maps by fitting simulated deformation to intraoperative imaging; differentiable CFD optimizes catheter shape to minimize hemolysis; differentiable pharmacokinetic ODE systems fit drug absorption parameters from sparse clinical data. DiffXPBD extends differentiable position-based dynamics to compliant constraint systems.
- **Key algorithms:** Reverse-mode automatic differentiation through simulation (adjoint method), differentiable PBD (DiffXPBD), differentiable FEM (Warp/JAX), differentiable Lagrangian particle dynamics (MPM), physics-informed loss functions, gradient-based material parameter identification.
- **Datasets:** Patient-specific tissue deformation datasets from intraoperative US (Hamlyn); Cardiovascular 4D Flow MRI (HeartFlow); Warp Tutorial Benchmarks (https://github.com/NVIDIA/warp); DeepMind MuJoCo Warp benchmarks (https://github.com/google-deepmind/mujoco).
- **Starter repos/tools:** NVIDIA Warp (https://github.com/NVIDIA/warp) — Python GPU differentiable physics engine (JAX/PyTorch integration); DiffTaichi (https://github.com/taichi-dev/taichi) — differentiable GPU simulation via Taichi lang; JAX MD (https://github.com/google/jax-md) — differentiable molecular dynamics; FEniCSx + UFL (https://github.com/FEniCS/dolfinx) — differentiable FEM (adjoint via dolfin-adjoint).
- **CUDA libraries & GPU pattern:** CUDA with reverse-mode AD (Warp's gradient tape), cuDNN for neural-network coupling in hybrid sim-ML pipelines, Tensor Cores for mixed-precision Jacobian accumulation; pattern: simulation forward pass on GPU → gradient tape records operations → backward pass propagates gradients through PDE/ODE → gradient-based optimizer updates material parameters → iterate.

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
