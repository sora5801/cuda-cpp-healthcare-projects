# THEORY — 1.5 Free Energy Perturbation / Thermodynamic Integration

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

### 1.5 Free Energy Perturbation / Thermodynamic Integration 🟢 · Established

- **Deep dive:** FEP and TI compute binding free energy differences (ΔΔG) between two ligands by running MD along an alchemical λ-pathway that slowly transforms one molecule into another. Each λ-window requires independent GPU MD trajectories; the collection of windows is trivially parallel across GPUs. The critical computational cost is the length and number of λ-windows required for convergence (typically 12–24 windows × 2–5 ns each). GPU-accelerated pmemd.cuda and NAMD-FEP achieve >10× speedup over CPU, reducing multi-day calculations to hours on a single A100. Relative FEP (RBFE) is now a standard tool in lead optimization pipelines at major pharmaceutical companies.
- **Key algorithms:** Alchemical λ-coupling, soft-core potentials (Beutler/Zacharias), multi-state Bennett acceptance ratio (MBAR), thermodynamic integration quadrature, replica exchange with solute tempering (REST2), overlap matrix assessment.
- **Datasets:** Merck FEP benchmark set — 8 targets with experimental ΔΔG (available via OpenFE; https://github.com/OpenFreeEnergy/openfe); FEP+ validation set (Schrodinger, verify URL); PDB-bind — experimental binding affinities (http://www.pdbbind.org.cn); ChEMBL activity data for target families (https://www.ebi.ac.uk/chembl/).
- **Starter repos/tools:** OpenFE (https://github.com/OpenFreeEnergy/openfe) — open FEP toolkit supporting GROMACS and OpenMM backends; GROMACS FEP (https://github.com/gromacs/gromacs) — GPU-accelerated FEP with MBAR post-processing via alchemlyb; OpenMMTools (https://github.com/choderalab/openmmtools) — alchemical replica exchange on GPU via OpenMM; AMBER pmemd.cuda TI (https://ambermd.org/GPUSupport.php) — softcore TI on NVIDIA GPUs.
- **CUDA libraries & GPU pattern:** Full MD engine on GPU (cuFFT PME + custom force kernels); embarrassingly parallel λ-window array across multiple GPUs; NCCL for REMD communication; CPU post-processing via alchemlyb/MBAR.

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
