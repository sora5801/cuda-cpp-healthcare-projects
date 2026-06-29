# THEORY — 14.9 Multi-Physics Tumor / Treatment Digital Twin

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

### 14.9 Multi-Physics Tumor / Treatment Digital Twin 🔴 · Frontier/Theoretical

- **Deep dive:** A cancer digital twin couples tumor growth (reaction-diffusion PDE for cell density + nutrient + oxygen), mechanical deformation of surrounding tissue (FEM), vascular remodeling (angiogenesis ODE), drug pharmacokinetics (PKPD ODE), immunological response, and radiation damage (LQ model), all personalized from serial multimodal imaging. GPU parallelism tackles the stiff multi-physics coupling: the reaction-diffusion grid (512³ voxels), the FEM mesh (500K elements), and the vascular graph (10⁴ vessel segments) each run on separate GPU streams, synchronized at each time step. Multi-GPU inverse problem fitting of all biophysical parameters to longitudinal MRI + ctDNA data is the frontline computational challenge. The field saw publication of physics-informed ML digital twins for prostate cancer (PSA-driven) in Nature npj Digital Medicine 2025.
- **Key algorithms:** Anisotropic tumor-growth reaction-diffusion PDE (Fisher-Kolmogorov), vascular angiogenesis ODE (VEGF-driven), linear-quadratic (LQ) radiation damage model, pharmacokinetic two-compartment model, Bayesian ensemble Kalman filter for parameter assimilation, adjoint-based sensitivity for PDE inversion.
- **Datasets:** TCIA (The Cancer Imaging Archive) — multimodal tumor imaging (https://www.cancerimagingarchive.net/); TCGA (The Cancer Genome Atlas) — multi-omics tumor data (https://www.cancer.gov/tcga); ISPY2 — breast cancer treatment response imaging trial (https://www.ispy2.org/); NSCLC-Radiomics (Lung1) — CT + survival on 422 patients (https://www.cancerimagingarchive.net/).
- **Starter repos/tools:** CHASTE (https://github.com/Chaste/Chaste) — cancer multiscale + vascular simulation; OpenCMISS-Iron (https://github.com/OpenCMISS/iron) — GPU FEM for tumor-tissue mechanics; NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) — PINN surrogates for tumor growth; TumorFEM (verify URL, search "tumor digital twin FEM GitHub") — patient-specific tumor mechanical FEM.
- **CUDA libraries & GPU pattern:** CUDA 3D stencil kernels for reaction-diffusion PDE, cuSPARSE for FEM tissue mechanics, cuSolver for vascular pressure-flow network, multi-GPU NCCL for coupled physics domains; pattern: patient MRI → tumor/tissue segmentation → multi-physics GPU simulation → synthetic MRI generation → Bayesian parameter assimilation → treatment prediction.

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
