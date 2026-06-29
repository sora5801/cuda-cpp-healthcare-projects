# THEORY — 1.4 Ultra-Large Virtual Screening

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

### 1.4 Ultra-Large Virtual Screening 🟢 · Established

- **Deep dive:** Modern make-on-demand chemical libraries (Enamine REAL: >6 billion compounds, ZINC: ~2 billion) make exhaustive docking computationally prohibitive with CPU resources. GPU-accelerated docking allows screening of billions of compounds by batching thousands of ligands simultaneously on a single GPU. Additionally, ML surrogate models trained on docked subsets (active learning / Bayesian optimization) dramatically reduce the number of full docking evaluations required. Specialized tools like HASTEN and REINVENT combine GPU docking with ML to achieve 90% recall of true top-1000 hits after evaluating only 1% of the library. The Summit supercomputer campaign against COVID-19 targets docked >1 billion compounds using AutoDock-GPU.
- **Key algorithms:** GPU-batched LGA/BFGS docking, Bayesian active learning, surrogate-model filtering (random forest, GNN), pharmacophore pre-filtering, shape screening pre-filter, Lipinski/ADMET filter cascades.
- **Datasets:** Enamine REAL library — >6B synthesizable compounds (https://enamine.net/compound-collections/real-compounds); ZINC20 — free virtual screening database (https://zinc20.docking.org); ChEMBL — bioactivity reference (https://www.ebi.ac.uk/chembl/); ExCAPE-DB — large-scale public chemogenomics dataset (https://solr.ideaconsult.net/search/excape/).
- **Starter repos/tools:** AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — core CUDA docking engine used in billion-compound campaigns; Uni-Dock (https://github.com/dptech-corp/Uni-Dock) — high-throughput GPU docking with batch input; DiffDock (https://github.com/gcorso/DiffDock) — diffusion model for blind docking of large libraries; gpusimilarity (https://github.com/schrodinger/gpusimilarity) — GPU fingerprint similarity for rapid pre-screening.
- **CUDA libraries & GPU pattern:** Texture memory for grid lookups; warp-parallel GA evaluation; multiple ligands co-resident in GPU memory; NVLink multi-GPU for campaign-scale throughput; thrust for top-K selection.

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
