# THEORY — 1.28 Covalent Docking

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

### 1.28 Covalent Docking 🟡 · Active R&D

- **Deep dive:** Covalent inhibitors form a permanent or semi-permanent bond with a nucleophilic residue (usually Cys, Ser, Lys, Tyr). Docking them requires two-stage sampling: (1) non-covalent pre-reaction pose generation (as in standard docking) and (2) covalent bond geometry enforcement with post-reaction scoring. GPU acceleration helps explore the expanded conformational space after covalent bond formation. Methods include CovDock (Schrodinger), AutoDock-GPU covalent option, and emerging DL methods (CovDocker, 2025). EGFR/BTK/KRAS(G12C) covalent drug programs drive industrial interest.
- **Key algorithms:** Two-stage covalent docking protocol, warhead reactive group enumeration, covalent bond geometry constraint, MM-GBSA rescoring of covalent complexes, covalent pharmacophore matching.
- **Datasets:** CovDocker benchmark (2025, verify URL); ChEMBL covalent inhibitor set (https://www.ebi.ac.uk/chembl/); PDB covalent complex structures (https://www.rcsb.org); BindingDB covalent entries (https://www.bindingdb.org).
- **Starter repos/tools:** AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — supports covalent docking mode; GNINA (https://github.com/gnina/gnina) — CNN-scored docking with covalent options; Uni-Dock (https://github.com/dptech-corp/Uni-Dock) — GPU docking extendable to covalent; CovDocker (arxiv 2506.21085, verify GitHub URL) — DL covalent docking benchmark.
- **CUDA libraries & GPU pattern:** Same as standard docking GPU pattern; additional CUDA kernel for covalent bond constraint penalty; GPU-parallel conformational sampling of warhead + linker degrees of freedom.

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
