# THEORY — 11.11 CHO Cell & Mammalian Bioprocess Digital Twin

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

### 11.11 CHO Cell & Mammalian Bioprocess Digital Twin 🟡 · Active R&D

- **Deep dive:** Chinese Hamster Ovary (CHO) cell fed-batch cultures for monoclonal antibody production exhibit complex interplay of metabolism, glycosylation, dissolved oxygen, and pH dynamics that are expensive to characterize experimentally. GPU-accelerated hybrid digital twins (Nature npj 2026) couple ODE kinetic models with genome-scale FBA on GPU, with LSTM networks trained on GPU correcting model-plant mismatch online. Bayesian parameter estimation with HMC (GPU-accelerated via NumPyro/JAX) fits hundreds of kinetic parameters to multi-omics fed-batch data in hours. Real-time digital twins receive PAT (process analytical technology) sensor streams and predict glycoform distributions ahead of time for automated feeding control.
- **Key algorithms:** Hybrid mechanistic-ML (ODE + LSTM), genome-scale metabolic modeling (FBA, GEM reduction), Bayesian HMC parameter estimation, Gaussian process regression for process uncertainty, PLS/PCA for spectroscopic soft sensing, dynamic FBA (dFBA).
- **Datasets:** CHO Fed-Batch Time-Course Metabolomics (BioRxiv 2025, Zenodo) — 12 cultures with 80+ metabolite time profiles; BioNumbers Database — CHO-specific growth/uptake rates (https://bionumbers.hms.harvard.edu/); BioModels Database — published CHO kinetic models (https://www.ebi.ac.uk/biomodels/); JGI/DBTBS gene expression compendium for CHO pathway analysis (verify URL).
- **Starter repos/tools:** COBRApy (https://github.com/opencobra/cobrapy) — GEM FBA for CHO; NumPyro (https://github.com/pyro-ppl/numpyro) — GPU Bayesian HMC for kinetic parameter estimation; PyTorch LSTM (https://pytorch.org/) — hybrid ODE-LSTM digital twin training; Pyomo (https://github.com/Pyomo/pyomo) — algebraic modeling for dynamic FBA optimization.
- **CUDA libraries & GPU pattern:** cuDNN for LSTM training/inference, JAX GPU backend for HMC MCMC, CUDA batch LP for parallel FBA across time points; pattern: online PAT sensor feed → GPU LSTM state update → GPU GEM FBA at current metabolite concentrations → kinetic ODE integration → feeding strategy MPC → compare to lab measurements → Bayesian posterior update.

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
