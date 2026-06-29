# THEORY — 13.7 Adverse-Event & Pharmacovigilance Signal Detection

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

### 13.7 Adverse-Event & Pharmacovigilance Signal Detection 🟡 · Active R&D

- **Deep dive:** Detects unexpected drug safety signals from spontaneous reporting systems (FAERS, EudraVigilance) by applying disproportionality analysis and machine learning over millions of case reports. Reporting Odds Ratio (ROR) and Information Component (IC) calculations across all drug-AE pairs are parallelisable on GPU as batched sparse contingency table computations. Deep learning NLP models (BioBERT, ClinicalBERT) applied to FAERS narrative free-text are GPU-bound transformer inference. Longitudinal signal monitoring with Bayesian information component (multi-item gamma Poisson shrinker, MGPS) across a drug×AE matrix of 10⁶+ pairs requires GPU-resident sparse tensor operations.
- **Key algorithms:** Reporting Odds Ratio (ROR), Proportional Reporting Ratio (PRR), Multi-item Gamma Poisson Shrinker (MGPS), Bayesian Confidence Propagation Neural Network (BCPNN), NLP-based signal extraction (BERT NER on adverse event text), longitudinal CUSUM signal monitoring, graph-based drug-AE network analysis.
- **Datasets:**
  - FDA FAERS (Adverse Event Reporting System) — 25M+ individual case safety reports (https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
  - EudraVigilance — EMA adverse event reporting database (https://www.adrreports.eu/)
  - WHO VigiAccess — global drug adverse reaction database (https://www.vigiaccess.org/)
  - SIDER — side-effect data from drug package inserts (http://sideeffects.embl.de/)
- **Starter repos/tools:**
  - PhViD (https://cran.r-project.org/web/packages/PhViD/) — R pharmacovigilance disproportionality package
  - pyVigilance (verify URL) — Python FDA FAERS signal detection package
  - BioBERT (https://github.com/dmis-lab/biobert) — GPU-pretrained biomedical BERT for FAERS NLP
  - OpenVigil 2.1 (http://openvigil.pharmacology.uni-kiel.de/) — web-based pharmacovigilance signal detection tool
- **CUDA libraries & GPU pattern:** cuSPARSE for sparse drug-AE contingency matrix, cuBLAS for MGPS matrix operations, cuDNN for BERT-based NLP inference; pattern: batch-parallel disproportionality computation across all drug-AE pairs on GPU.

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
