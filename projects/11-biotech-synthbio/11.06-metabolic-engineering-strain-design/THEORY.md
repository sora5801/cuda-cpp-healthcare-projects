# THEORY — 11.6 Metabolic Engineering & Strain Design

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

### 11.6 Metabolic Engineering & Strain Design 🟡 · Active R&D

- **Deep dive:** Metabolic engineering seeks genetic modifications (gene knockouts, overexpression, heterologous pathway insertion) that maximize desired metabolite production. GPU acceleration enables genome-scale flux-balance analysis (FBA) to be solved for millions of genetic perturbation combinations in parallel — each FBA is an independent LP problem — dramatically outpacing CPU batch FBA. Constraint-based strain design algorithms (OptKnock, MOMA) search exponentially large combinatorial spaces, tractable only with GPU parallelism. Kinetic whole-pathway models (ODEs with hundreds of reactions) can be fitted to multi-omics data using GPU-accelerated Bayesian MCMC (NUTS/HMC).
- **Key algorithms:** Flux Balance Analysis (LP, GPU batch), Dynamic FBA, OptKnock / RobustKnock strain design, ensemble kinetic modeling (EKM), Bayesian MCMC parameter estimation (NUTS/HMC), genome-scale metabolic network reduction (data-driven, 2025).
- **Datasets:** BiGG Models — 108 genome-scale metabolic models (https://bigg.ucsd.edu/); KEGG Metabolic Pathways (https://www.kegg.jp/kegg/pathway.html); MetaboLights — metabolomics raw data (https://www.ebi.ac.uk/metabolights/); CHO-GEM Genome-Scale Model — CHO cell metabolic network (verify URL via Zenodo/BioModels).
- **Starter repos/tools:** COBRApy (https://github.com/opencobra/cobrapy) — FBA/FVA in Python; cameo (https://github.com/biosustain/cameo) — strain design algorithms including OptKnock; MICOM (https://github.com/micom-dev/micom) — microbiome community FBA; GPU-FBA (verify URL, search "GPU flux balance analysis CUDA") — CUDA batch LP solver for parallel strain enumeration.
- **CUDA libraries & GPU pattern:** CUDA LP solver (per-combination parallel simplex/interior-point), cuBLAS for stoichiometric matrix operations, Thrust for parallel combinatorial enumeration; pattern: stoichiometric matrix resident on GPU → one thread block per genetic perturbation combination → parallel FBA solve → objective value reduction → top strains ranked.

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
