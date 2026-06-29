# THEORY — 6.12 Metabolic Flux / Constraint-Based Modeling

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

### 6.12 Metabolic Flux / Constraint-Based Modeling 🟢 · Established
- **Deep dive:** Flux balance analysis (FBA) finds optimal metabolic fluxes by solving a linear program (LP) constrained by stoichiometry, thermodynamics, and enzyme capacity on genome-scale metabolic models (GEMs) with 3 000–8 000 reactions. GPU parallelism enters through solving thousands of LP instances in parallel (e.g., for all conditions in a drug screen, or all single-gene knockouts in an essentiality screen). Mixed-integer programming (MILP) variants for gap-filling and thermodynamic FBA benefit from GPU-accelerated interior-point methods.
- **Key algorithms:** Flux balance analysis (FBA), flux variability analysis (FVA), parsimonious FBA (pFBA), thermodynamic FBA (tFBA), MILP gap-filling, minimal cut sets, COBRA toolbox algorithms, interior-point LP (revised simplex), shadow price / sensitivity analysis.
- **Datasets:** Recon3D — human genome-scale metabolic model (https://github.com/SBRG/Recon3D); HMDB — Human Metabolome Database (https://hmdb.ca); Reactome (https://reactome.org); BiGG Models Database — curated GEMs (http://bigg.ucsd.edu).
- **Starter repos/tools:** COBRApy (https://github.com/opencobra/cobrapy) — Python FBA/FVA with multiple LP/MILP solver backends; Recon3D model files (https://github.com/SBRG/Recon3D); Virtual Metabolic Human (https://vmh.life) — interactive Recon3D portal; SUNDIALS (https://github.com/LLNL/sundials) — for dynamic FBA ODE integration.
- **CUDA libraries & GPU pattern:** cuSOLVER dense LP factor (batch small LP); custom CUDA interior-point primal-dual kernel for LP batches; ArrayFire (https://github.com/arrayfire/arrayfire) for dense matrix batches; pattern: one LP per CUDA block, shared memory for constraint matrix, warp-level reduction for objective gradient.

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
