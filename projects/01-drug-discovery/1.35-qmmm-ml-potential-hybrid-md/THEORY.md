# THEORY — 1.35 QMMM/ML Potential Hybrid MD

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

### 1.35 QMMM/ML Potential Hybrid MD 🔴 · Frontier/Theoretical

- **Deep dive:** The next frontier beyond QM/MM is using ML potentials trained on QM data to replace the expensive QM region — enabling microsecond reactive MD at QM accuracy. GPU-accelerated equivariant NNPs (MACE, NequIP) can serve as drop-in QM replacements in an MM environment. This hybrid NNP/MM approach runs fully on GPU: the NNP forward pass and MM evaluation occur in overlapping CUDA streams. Challenges include training data coverage for reactive intermediates and accurate long-range electrostatics across the QM-ML/MM boundary.
- **Key algorithms:** NNP/MM coupling, link-atom boundary treatment, active learning for reactive system NNP training, δ-ML correction to DFT, equivariant NNP with long-range electrostatic correction.
- **Datasets:** ANI-1ccx reactive extensions (verify URL); DFT reaction pathway datasets from QM/MM studies; Transition1x — 10M DFT calculations along reaction paths (https://zenodo.org/record/5781475); SPICE dataset (https://github.com/openmm/spice-dataset).
- **Starter repos/tools:** TorchMD-Net (https://github.com/torchmd/torchmd-net) — equivariant NNP with MM coupling; MACE (https://github.com/ACEsuit/mace) — fast NNP for hybrid ML/MM; OpenMM-ML (https://github.com/openmm/openmm-ml) — NNP/MM interface for OpenMM; NNPOps (https://github.com/openmm/NNPOps) — CUDA-optimized NNP primitives.
- **CUDA libraries & GPU pattern:** CUDA MACE kernels for equivariant message passing; OpenMM CUDA platform for MM region; CUDA streams for async NNP+MM; PyTorch autograd for NNP force gradients; cuBLAS for spherical harmonic transforms.

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
