# THEORY — 1.8 Semi-Empirical & Tight-Binding Quantum Methods

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

### 1.8 Semi-Empirical & Tight-Binding Quantum Methods 🟢 · Established

- **Deep dive:** Semi-empirical methods (PM7, GFN2-xTB) approximate quantum mechanics at 100–10000× lower cost than DFT by parameterizing integral expressions with empirical data. They bridge the gap between force fields and full DFT, enabling geometry optimization and reactivity screening of drug-like molecules at scale. GPU implementations parallelize the sparse Hamiltonian construction and diagonalization over molecule batches — thousands of small molecules can be optimized simultaneously on one GPU. XTB is critical for conformer ranking, tautomer enumeration, and QM-based ADMET calculation in modern drug discovery pipelines.
- **Key algorithms:** MNDO/AM1/PM6/PM7 Hamiltonians, GFN1/GFN2-xTB (extended tight-binding), DFTB+ (density functional tight binding), diagonalization via cuSOLVER, GPU-batched molecular calculations.
- **Datasets:** ANI-1 — 20M DFT energy calculations on 57k molecules (https://github.com/isayev/ANI1); QM9 (https://doi.org/10.6084/m9.figshare.978904); GMTKN55 — benchmark thermochemistry and kinetics set (https://www.chemie.uni-bonn.de/grimme/de/software/gmtkn); COMPAS — computational database of polycyclic aromatic systems (verify URL).
- **Starter repos/tools:** xtb (https://github.com/grimme-lab/xtb) — GFN2-xTB reference implementation (CPU-only but used as GPU backend reference); DFTB+ (https://github.com/dftbplus/dftbplus) — GPU-accelerated DFTB via ELPA library; GFN-FF / xTB-IFF (https://github.com/grimme-lab) — force field from tight binding; TBLite (https://github.com/tblite/tblite) — lightweight tight-binding library.
- **CUDA libraries & GPU pattern:** cuSOLVER for batch matrix diagonalization; cuBLAS for Hamiltonian density-matrix products; custom CUDA kernels for two-center integral batches; stream concurrency to overlap compute and data transfer for molecule batches.

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
