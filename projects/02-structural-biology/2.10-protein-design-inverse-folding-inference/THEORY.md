# THEORY — 2.10 Protein Design / Inverse Folding Inference

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

### 2.10 Protein Design / Inverse Folding Inference 🟢 · Established

- **Deep dive:** Inverse folding (sequence design) asks: given a backbone structure, what amino acid sequences will fold into it? ProteinMPNN uses a graph neural network that processes backbone geometry (Cα coordinates, virtual Cβ, backbone dihedrals) to autoregressively decode sequences. GPU inference generates diverse sequences at ~1–2 seconds per protein per 100 residues. Wet-lab validation shows 50–55% native sequence recovery. Integration with structure prediction (RFdiffusion for backbone generation → ProteinMPNN for sequence → AlphaFold2 for validation) creates a fully GPU-accelerated computational protein design pipeline.
- **Key algorithms:** Autoregressive sequence decoding on protein graph, message-passing GNN over backbone geometry, tied decoding for symmetric oligomers, temperature-controlled sampling, order-agnostic decoding, LigandMPNN for small-molecule-aware design.
- **Datasets:** CATH protein structure database — 500k+ domain structures (https://www.cathdb.info); PDB training set for ProteinMPNN (https://www.rcsb.org); ProteinGym benchmark — mutational fitness (https://github.com/OATML-Markslab/ProteinGym); CAMEO validation (https://www.cameo3d.org).
- **Starter repos/tools:** ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — official GPU inverse folding model (Baker Lab); LigandMPNN (https://github.com/dauparas/LigandMPNN) — inverse folding with ligand context; ESM-IF1 (https://github.com/facebookresearch/esm) — ESM inverse folding model; RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — backbone diffusion for de novo design coupled with ProteinMPNN.
- **CUDA libraries & GPU pattern:** PyTorch Geometric CUDA GNN layers for protein graph; GPU autoregressive decoding with kv-cache; FP16 mixed precision; batched sequence generation across multiple backbone inputs; cuDNN for backbone encoder.

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
