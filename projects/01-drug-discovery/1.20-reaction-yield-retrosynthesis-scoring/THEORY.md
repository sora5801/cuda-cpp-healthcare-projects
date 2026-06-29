# THEORY — 1.20 Reaction Yield / Retrosynthesis Scoring

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

### 1.20 Reaction Yield / Retrosynthesis Scoring 🟡 · Active R&D

- **Deep dive:** Computational retrosynthesis decomposes a target molecule into commercially available building blocks via known reactions, enabling synthesizability assessment in generative design. GPU-trained transformer models (Molecular Transformer, Chemformer) predict reaction products and retrosynthetic routes over large training sets of reaction SMILES. GPU inference scores millions of candidate synthetic routes per second. Reaction yield prediction (using GNN or transformer on reaction SMILES + conditions) guides experimental prioritization. Integration with generative models creates closed-loop synthesis-aware drug design.
- **Key algorithms:** Transformer on augmented SMILES (reaction SMILES), sequence-to-sequence models, Monte Carlo tree search (MCTS) for retrosynthesis planning, graph-to-graph transformations, reaction center prediction, graph neural network on reaction graphs.
- **Datasets:** USPTO-50k — 50k atom-mapped reactions (https://github.com/connorcoley/rexgen_direct); Reaxys/CAS reaction databases (commercial); Open Reaction Database (ORD) — open-access reaction data (https://open-reaction-database.org); USPTO-MIT — 479k reactions (https://github.com/wengong-jin/nips17-rexgen).
- **Starter repos/tools:** Molecular Transformer (https://github.com/pschwllr/MolecularTransformer) — GPU transformer for reaction prediction; AiZynthFinder (https://github.com/MolecularAI/aizynthfinder) — GPU-accelerated retrosynthesis planning; ASKCOS (https://github.com/ASKCOS/ASKCOS) — synthesis planning platform; Chemformer (https://github.com/MolecularAI/Chemformer) — pre-trained BART-based reaction model.
- **CUDA libraries & GPU pattern:** cuDNN transformer attention kernels; FP16 mixed precision for large SMILES vocabularies; GPU-batched beam search decoding; MCTS rollouts in parallel on GPU with batched transformer scoring.

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
