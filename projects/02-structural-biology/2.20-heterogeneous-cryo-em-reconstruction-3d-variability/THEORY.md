# THEORY — 2.20 Heterogeneous Cryo-EM Reconstruction (3D Variability)

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

### 2.20 Heterogeneous Cryo-EM Reconstruction (3D Variability) 🟡 · Active R&D

- **Deep dive:** Real protein complexes adopt multiple conformational states simultaneously. Heterogeneous reconstruction methods disentangle these states from particle images. CryoDRGN uses a variational autoencoder (VAE) with an amortized encoder that maps each particle image to a latent code representing its conformation, and a decoder that generates the 3D density from the latent code via a coordinate MLP. GPU training is essential: a cryoDRGN run on 100k particles requires hours on A100. 3DVA (cryoSPARC) uses PCA-like linear subspace methods. Applications reveal continuous flexibility in ribosomes, GPCR complexes, and viral assembly intermediates.
- **Key algorithms:** Variational autoencoder (VAE) with image encoder and volume decoder, coordinate-based implicit neural representation (NeRF/MLP decoder), 3D variability analysis (PCA on volume subspace), pose estimation EM, Fourier-slice theorem in the network, manifold learning.
- **Datasets:** EMPIAR-10180 (spliceosome), EMPIAR-10076 (80S ribosome), EMPIAR-10028 (TRPV1) (all at https://www.ebi.ac.uk/empiar/); cryoDRGN benchmark datasets (https://github.com/ml-struct-bio/cryodrgn); simulated heterogeneous datasets from IgG/spike protein.
- **Starter repos/tools:** CryoDRGN (https://github.com/ml-struct-bio/cryodrgn) — GPU VAE for heterogeneous reconstruction; cryoSPARC 3DVA (https://cryosparc.com) — GPU linear 3D variability analysis; Recovar (verify URL) — GPU regularized covariance heterogeneous reconstruction; DrgnAI (verify URL) — neural 3D reconstruction with pose optimization.
- **CUDA libraries & GPU pattern:** PyTorch CUDA for VAE encoder/decoder; FlashAttention for particle image attention layers; GPU Fourier-slice theorem evaluation via differentiable nufft; cuFFT for power spectrum during training; FP16 mixed precision.

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
