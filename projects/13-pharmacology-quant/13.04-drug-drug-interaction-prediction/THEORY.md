# THEORY — 13.4 Drug-Drug Interaction Prediction

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

### 13.4 Drug-Drug Interaction Prediction 🟡 · Active R&D

- **Deep dive:** Predicts pharmacokinetic drug-drug interactions (PK-DDI) caused by CYP enzyme inhibition/induction, transporter competition, and protein binding displacement; also predicts pharmacodynamic DDI from synergistic/antagonistic receptor effects. Graph neural networks encode drug molecular structure; bipartite interaction graphs model shared enzyme substrates. GPU parallelism across large drug-pair combination spaces is essential — the DrugBank DDI graph has ~250k interaction edges from 2.4k drugs, but virtual screening explores millions of hypothetical pairs. Static mechanistic models (R-value, AUC ratio prediction) are solved in batched parallel ODEs on GPU for all pairs simultaneously.
- **Key algorithms:** GNN on drug molecular graphs with edge-level DDI prediction, DeepDDI (sequence-based DDI), TransE/RotatE knowledge graph embedding for DDI, R-value static mechanistic model, AUC ratio DDI prediction, CYP inhibition ODE models, PBPK-embedded DDI simulation.
- **Datasets:**
  - DrugBank DDI — 250k+ drug interaction records with mechanism (https://www.drugbank.com/)
  - TWOSIDES — 3.7M adverse event pairs from spontaneous reports (verify URL; originally published by Tatonetti lab)
  - OFFSIDES — off-label adverse effects dataset (verify URL; Tatonetti lab)
  - FDA Adverse Event Reporting System (FAERS) (https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
- **Starter repos/tools:**
  - DeepDDI (https://github.com/NCIBI/DeepDDI) — deep learning DDI prediction from drug SMILES
  - SkipGNN (verify URL) — graph neural network for DDI on drug interaction graphs
  - TorchDrug (https://github.com/DeepGraphLearning/torchdrug) — GPU molecular GNN framework applicable to DDI
  - STITCH — chemical-protein interactions database (http://stitch.embl.de/) with downloadable interaction files
- **CUDA libraries & GPU pattern:** DGL/PyG sparse message passing on drug interaction graphs, cuBLAS for PBPK ODE Jacobians, custom CUDA DDI scoring kernels; pattern: batch-parallel DDI pair scoring over millions of drug combinations on GPU.

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
