# THEORY — 12.16 GPU-Accelerated Hi-C Contact Loop Calling

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

### 12.16 GPU-Accelerated Hi-C Contact Loop Calling 🟡 · Active R&D
- **Deep dive:** Hi-C loop calling (HiCCUPS) identifies chromatin loops as enriched point interactions above a 2D background, estimated by a sliding Donut kernel convolution over the contact map. At 5 kb resolution, a human contact map is ~600 k × 600 k (sparse, stored as pairs); the Donut convolution at each potential loop pixel is a GPU embarrassingly parallel 2D operation. NVIDIA's original HiCCUPS paper used a GPU implementation as the default, making this one of the earliest established GPU genomics tools. Recent deep-learning loop callers (Peakachu) apply CNNs to local contact map patches, each patch independently inferrable on GPU.
- **Key algorithms:** Donut kernel background estimation (2D convolution on sparse contact map); Poisson enrichment scoring per pixel; multi-resolution peak merging; FDR control for loop calls; Peakachu CNN local patch classification; Fit-Hi-C probability model; anchor pair exhaustive scoring.
- **Datasets:** 4DN Hi-C datasets (https://data.4dnucleome.org/); GEO GSE63525 (Rao 2014) — original HiCCUPS benchmark (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525); ENCODE Hi-C (https://www.encodeproject.org/); 3D Genome Browser datasets (http://3dgenome.fsm.northwestern.edu/).
- **Starter repos/tools:** Juicer / HiCCUPS (https://github.com/aidenlab/juicer) — GPU loop caller, original CUDA implementation; Peakachu (https://github.com/tariks/peakachu) — CNN-based loop caller (GPU inference); Higashi (https://github.com/ma-compbio/Higashi) — single-cell Hi-C GPU model; MUSTACHE (https://github.com/ay-lab/mustache) — multi-scale Hi-C loop caller.
- **CUDA libraries & GPU pattern:** Custom 2D convolution kernels for Donut background; cuSPARSE for sparse contact matrix operations; cuDNN for CNN local-patch loop classification; thrust for sparse pixel sorting; GPU-resident contact map tiles in texture memory.

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
