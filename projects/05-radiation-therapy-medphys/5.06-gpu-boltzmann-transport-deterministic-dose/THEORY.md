# THEORY — 5.6 GPU Boltzmann Transport (Deterministic Dose)

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

### 5.6 GPU Boltzmann Transport (Deterministic Dose) 🟡 · Active R&D
- **Deep dive:** The linear Boltzmann transport equation (LBTE) describes radiation transport deterministically: it tracks the fluence distribution of particles as a function of position, direction, and energy without stochastic noise. Solving it on a clinical 6-DoF phase-space grid (x, y, z, θ, φ, E) discretized at clinical resolution yields a system with ~10⁹–10¹⁰ unknowns; iterative solvers (source iteration, diffusion synthetic acceleration) require GPU to be tractable. Acuros XB (Varian Eclipse) implements a GPU-accelerated LBTE solver that outperforms superposition-convolution in heterogeneous tissue. The 3D_RZ geometry and electron transport coupling make Boltzmann dose accurate in lung, bone/tissue interfaces where MC is preferred but slow.
- **Key algorithms:** Discrete ordinates (Sₙ) method, source iteration (SI), diffusion synthetic acceleration (DSA), multi-group energy discretization, linear discontinuous spatial FEM, Legendre polynomial scattering expansion, Acuros XB algorithm, coupled photon-electron LBTE.
- **Datasets:** AAPM TG-105 lung benchmark; IROC heterogeneity phantom datasets; IAEA photon cross-section library; Acuros XB validation datasets from Varian white papers (publicly documented).
- **Starter repos/tools:** OpenMC (https://github.com/openmc-dev/openmc) — open MC but with deterministic capabilities; Attila (commercial) and Denovo (https://github.com/ORNL-CEES/Exnihilo — verify URL) — deterministic transport; AHOTN (analytical and hybrid ordinates) codes (verify URL); GPU-accelerated Sₙ solvers in nuclear engineering literature (search "GPU Sn transport CUDA").
- **CUDA libraries & GPU pattern:** cuSPARSE for angular flux sweep (upwind differencing); cuFFT not applicable; custom CUDA kernel for inner transport sweep (spatial + angular decomposition); GPU memory: angular flux tensor in global memory, scattering source in shared memory; wavefront parallelism across spatial cells.

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
