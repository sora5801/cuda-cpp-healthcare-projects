# THEORY — 1.9 ML Interatomic Potentials (Neural Network Potentials)

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

### 1.9 ML Interatomic Potentials (Neural Network Potentials) 🟢 · Established

- **Deep dive:** Neural network potentials (NNPs) learn the potential energy surface from ab initio data, reproducing DFT accuracy at near-classical MD speed. Architectures range from atom-centered symmetry functions (ANI) to equivariant message-passing networks (NequIP, MACE, SchNet). GPU acceleration is essential: each forward pass involves neighborhood construction, message passing over all atomic pairs within a cutoff, and backpropagation for forces. On an A100, a 500-atom protein+ligand system runs at ~10 ns/day — 1000× slower than classical FF but 100× faster than DFT, enabling reactive drug-target simulations previously impossible.
- **Key algorithms:** Atom-centered symmetry functions (ACSF/BEHLER), equivariant neural networks (E(3)-equivariant / SE(3)), message-passing neural networks (MPNN/SchNet/DimeNet), MACE (multi-ACE), NequIP, neural achitecture via PyTorch Geometric.
- **Datasets:** ANI-1ccx — CCSD(T) energies on 500k conformers of drug-like molecules (https://github.com/isayev/ANI1ccx_dataset); SPICE — quantum chemistry dataset for ML potentials covering drug-like molecules and proteins (https://github.com/openmm/spice-dataset); rMD17 — revised MD17 benchmark (https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038); OE62 — 62k organic molecules with DFT energetics (verify URL).
- **Starter repos/tools:** TorchANI (https://github.com/aiqm/torchani) — PyTorch ANI NNP with CUDA acceleration and OpenMM integration; TorchMD-Net (https://github.com/torchmd/torchmd-net) — equivariant NNPs with GPU-optimized neighbor list; MACE (https://github.com/ACEsuit/mace) — fast equivariant NNP with GPU kernels; NequIP (https://github.com/mir-group/nequip) — E(3)-equivariant network for accurate NNPs.
- **CUDA libraries & GPU pattern:** PyTorch CUDA autograd for force computation via backpropagation; custom CUDA kernels for neighbor list construction with periodic boundaries; torch.compile/TorchScript for inference optimization; multi-GPU via DDP for training.

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
