# THEORY — 2.16 ΔΔG Stability Prediction

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

### 2.16 ΔΔG Stability Prediction 🟡 · Active R&D

- **Deep dive:** Predicting the thermodynamic stability change upon single amino acid mutation (ΔΔG) is critical for protein engineering, antibody optimization, and understanding disease variants. ML approaches train on experimental ΔΔG datasets (Protherm, Megascale) using structural features (ProteinMPNN ddG, ThermoMPNN), sequence language models (ESM-1v, EVmutation), or structure-sequence joint models. GPU training on millions of mutation datapoints and GPU inference for saturation mutagenesis scanning (all 20 AA × every position) makes library-scale ΔΔG feasible.
- **Key algorithms:** ProteinMPNN fixed-backbone energy decomposition, ESM-1v zero-shot log-likelihood scoring, Rosetta ddG monomer protocol (FoldX, Cartesian ddG), GNN per-residue embedding, saturation mutagenesis scanning.
- **Datasets:** Protherm database — >25k experimental ΔΔG values (https://www.abren.net/protherm/); Megascale dataset — 2.5M thermodynamic stability measurements (https://github.com/Rocklin-Lab/cdna-display-proteolysis-datasets); ProteinGym substitutions benchmark (https://github.com/OATML-Markslab/ProteinGym); S669 curated stability benchmark (verify URL).
- **Starter repos/tools:** ThermoMPNN (https://github.com/Kuhlman-Lab/ThermoMPNN) — GPU ΔΔG prediction from ProteinMPNN; ProteinMPNN-ddG (https://github.com/PeptoneLtd/proteinmpnn_ddg) — saturation mutagenesis ΔΔG; ESM-1v (https://github.com/facebookresearch/esm) — zero-shot stability from language model; FoldX (https://foldxsuite.crg.eu) — fast empirical ΔΔG.
- **CUDA libraries & GPU pattern:** GPU GNN inference for per-residue stability; batched language model forward passes (cuDNN attention); GPU saturation mutagenesis via batched masked prediction; PyTorch Distributed for large-scale training.

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
