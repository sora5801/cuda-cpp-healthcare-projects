# THEORY — 11.10 Antibody Developability Prediction & Optimization

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

### 11.10 Antibody Developability Prediction & Optimization 🟡 · Active R&D

- **Deep dive:** Even potent antibodies fail if they aggregate, have high viscosity, polyreact with off-targets, or are immunogenic — properties collectively called developability. Predicting all six key developability flags (pI, hydrophobicity, aggregation propensity, poly-specificity, expression level, immunogenicity) from sequence alone via GPU-trained BERT-style models enables early-stage winnowing of design libraries with millions of variants. Multi-property Pareto optimization across affinity and developability runs on GPU via multi-objective Bayesian optimization over learned surrogate surfaces.
- **Key algorithms:** Protein LLM fine-tuning for developability regression, multi-objective Bayesian optimization (qParEGO), aggregation prediction (camsol/spatial aggregation propensity), immunogenicity prediction (T-cell epitope presentation MHC-II), expression-level prediction from sequence.
- **Datasets:** SAFit dataset — self-association from AstraZeneca (verify URL via Bioinformatics journal); TAP dataset — Therapeutic Antibody Profiler developability (https://opig.stats.ox.ac.uk/webapps/oas/tap); OAS (https://opig.stats.ox.ac.uk/webapps/oas/oas) — natural antibody sequence space for pre-training; CoV-AbDab (https://opig.stats.ox.ac.uk/webapps/covabdab/) — experimental affinity + neutralization data.
- **Starter repos/tools:** Therapeutic Antibody Profiler (TAP) (https://opig.stats.ox.ac.uk/webapps/oas/tap) — web server + scoring functions; AbLang (https://github.com/oxpig/AbLang) — antibody language model pre-training; AntiFold (https://github.com/oxpig/AntiFold) — GPU antibody inverse folding for sequence redesign; ANARCI (https://github.com/oxpig/ANARCI) — antibody numbering for feature alignment.
- **CUDA libraries & GPU pattern:** cuDNN for Transformer LLM inference over antibody sequence batches, Flash Attention for variable-length CDR context, CUDA kernels for parallel developability feature computation; pattern: million-variant library → batch GPU LLM embedding → multi-property regression → GPU Pareto front computation → top candidates advance to wet-lab synthesis.

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
