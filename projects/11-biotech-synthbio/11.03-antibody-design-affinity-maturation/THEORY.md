# THEORY — 11.3 Antibody Design & Affinity Maturation

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

### 11.3 Antibody Design & Affinity Maturation 🟡 · Active R&D

- **Deep dive:** Antibody engineering spans CDR-loop design, affinity maturation, and developability optimization — each requiring GPU inference over large sequence/structure spaces. RFdiffusion-Antibody (Baker Lab, 2025) generates novel CDR-H3 loops conditioned on antigen epitopes via SE(3)-equivariant diffusion on GPU. Affinity maturation via flow matching (AffinityFlow, 2025) guides sequence trajectories toward high-affinity regions on GPU. Structure-aware inverse folding (AbMPNN) redesigns CDR sequences while preserving Fv geometry. The AbBiBench benchmark (2025) standardizes evaluation across 10+ affinity maturation methods. The FDA approved 13 new monoclonal antibodies in 2024, underlining the industrial importance of accelerated in silico design.
- **Key algorithms:** SE(3)-equivariant diffusion (RFdiffusion), flow matching for affinity maturation (AffinityFlow), inverse folding (AbMPNN/ProteinMPNN), language-model-guided combinatorial optimization (LLM + genetic algorithm + simulated annealing), ΔΔG binding affinity prediction (Rosetta flex_ddg, FoldX), multi-objective developability scoring.
- **Datasets:** SAbDab — Structural Antibody Database, 10000+ Fv structures (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); AbBiBench Benchmark — standardized affinity maturation evaluation (https://arxiv.org/abs/2506.04235); OAS — Observed Antibody Space, 2B+ sequences (https://opig.stats.ox.ac.uk/webapps/oas/oas); CoV-AbDab — SARS-CoV-2 antibody database (https://opig.stats.ox.ac.uk/webapps/covabdab/).
- **Starter repos/tools:** RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — CDR design via SE(3) diffusion (RFdiffusion2 available 2025); ABodyBuilder3 (https://github.com/oxpig/ABDB) — GPU antibody structure prediction; ImmuneBuilder (https://github.com/oxpig/ImmuneBuilder) — GPU-fast Fv structure modeling; AbMPNN/ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU CDR sequence design.
- **CUDA libraries & GPU pattern:** Flash Attention for long CDR+antigen context, cuDNN Transformer inference for LLM-based sequence scoring, CUDA kernels for parallel ΔΔG evaluation; pattern: antigen epitope input → RFdiffusion GPU generates CDR scaffold ensemble → AbMPNN scores/redesigns sequences in batch → GPU ΔΔG filter → developability scoring → top candidates to wet lab.

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
