# THEORY — 1.10 De Novo Generative Molecular Design

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

### 1.10 De Novo Generative Molecular Design 🟡 · Active R&D

- **Deep dive:** Generative models learn the distribution of drug-like molecules and sample novel structures optimized for multiple properties (potency, selectivity, ADMET, synthesizability). GPU training is mandatory: large transformer/RNN/diffusion models over SMILES strings or 3D molecular graphs require days on multi-GPU nodes. At inference, reinforcement learning (RL) fine-tuning generates thousands of candidate molecules per GPU-second, enabling goal-directed optimization. REINVENT4 combines RL with curriculum learning on SMILES; diffusion-based methods (DiffSBDD, TargetDiff) generate molecules directly in 3D protein binding pockets.
- **Key algorithms:** Variational autoencoders (VAE), transformer language models on SMILES/SELFIES, graph generative models, denoising diffusion probabilistic models (DDPM), reinforcement learning with REINFORCE/PPO, scoring functions (docking, QED, SA score).
- **Datasets:** ChEMBL — 2M+ bioactive molecules (https://www.ebi.ac.uk/chembl/); ZINC20 — 1.4B purchasable compounds (https://zinc20.docking.org); GuacaMol benchmark — distribution learning and goal-directed generation benchmarks (https://github.com/BenevolentAI/guacamol); MOSES — molecular generation benchmarks (https://github.com/molecularsets/moses).
- **Starter repos/tools:** REINVENT4 (https://github.com/MolecularAI/REINVENT4) — production SMILES generative model with RL, Apache 2.0 license; DiffSBDD (https://github.com/arneschneuing/DiffSBDD) — 3D structure-based diffusion design; DiffDock (https://github.com/gcorso/DiffDock) — diffusion model for pose generation used in SBDD pipelines; DeepChem (https://github.com/deepchem/deepchem) — broad ML drug discovery toolkit including generative models.
- **CUDA libraries & GPU pattern:** cuDNN for transformer/RNN layers; custom CUDA scatter/gather for molecular graph message passing; multi-GPU DDP training; FP16 mixed precision via torch.amp; GPU-batched scoring function evaluation during RL rollouts.

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
