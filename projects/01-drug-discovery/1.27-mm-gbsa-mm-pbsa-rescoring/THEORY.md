# THEORY — 1.27 MM-GBSA / MM-PBSA Rescoring

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

### 1.27 MM-GBSA / MM-PBSA Rescoring 🟢 · Established

- **Deep dive:** MM-GB(PB)SA computes binding free energies as the MM interaction energy plus solvation free energy (implicit solvent GB or PB), minus entropic terms, from snapshots along an MD trajectory. It is the standard high-throughput rescoring step after docking, offering >10× better accuracy than scoring functions with ~1000× less cost than FEP. GPU-accelerated MD (pmemd.cuda) generates the required trajectory snapshots rapidly; gmx_MMPBSA post-processes GROMACS trajectories. The solvation GB/PB solvers can also be GPU-accelerated.
- **Key algorithms:** Molecular mechanics energy decomposition, Generalized Born (GB) implicit solvent, Poisson-Boltzmann (PB) numerical solver, normal-mode / quasi-harmonic entropy estimation, interaction entropy method, per-residue energy decomposition.
- **Datasets:** PDB-bind (http://www.pdbbind.org.cn); CASF-2016 (http://www.pdbbind.org.cn/casf.php); ChEMBL activity data (https://www.ebi.ac.uk/chembl/); AMBER MM-GBSA tutorial datasets (https://ambermd.org/tutorials/).
- **Starter repos/tools:** AMBER MMPBSA.py (https://ambermd.org/AmberTools.php) — reference MM-GBSA/PBSA implementation; gmx_MMPBSA (https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) — GROMACS compatibility layer; NAMD MMPBSA (https://www.ks.uiuc.edu/Research/namd/) — NAMD-based MM-PBSA; OpenMM MMGBSA (verify URL) — Python MM-GBSA workflow.
- **CUDA libraries & GPU pattern:** GPU MD for trajectory generation (pmemd.cuda); CPU MMPBSA.py for post-processing (GPU PB solver possible via custom CUDA); GPU-parallel evaluation of snapshots via embarrassingly parallel CUDA stream array.

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
