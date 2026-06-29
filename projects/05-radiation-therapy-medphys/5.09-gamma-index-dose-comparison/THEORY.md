# THEORY — 5.9 Gamma-Index Dose Comparison

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

### 5.9 Gamma-Index Dose Comparison 🟢 · Established
- **Deep dive:** The gamma index (γ) at each reference point searches for the minimum normalized Euclidean distance in combined dose-difference / distance-to-agreement (DTA) space over all evaluated points: γ(r_ref) = min_r √[(Δd/Δd_crit)² + (Δr/DTA_crit)²]. For 3D clinical distributions at 2 mm DTA and 2% dose criterion, the exhaustive search over a 200³ evaluation grid from each of 200³ reference points is O(N⁶) naively, reduced to O(N³ × K) by limiting the search radius. GPU parallelizes this: one thread per reference point, searches a kernel of neighbor evaluated points; with shared-memory tiling this achieves 100–1,000× speedup over CPU, enabling sub-second 3D gamma on clinical GPUs. This is critical for patient-specific IMRT/VMAT pre-treatment verification.
- **Key algorithms:** 3D gamma index exhaustive search (distance-limited), fast gamma approximations (1D cross-plane), GPU kernel tiling for shared-memory neighbour caching, global gamma pass-rate statistics, normalized agreement testing (NAT), χ (chi) factor dose comparison.
- **Datasets:** AAPM TG-218 patient-specific IMRT QA reference data; plan+measurement DICOM pairs from departmental QA systems; IROC-Houston phantom dose datasets; linac EPID measurement datasets.
- **Starter repos/tools:** Pymedphys (https://github.com/pymedphys/pymedphys) — Python gamma index, DICOM dose tools; Plastimatch (https://plastimatch.org/) — GPU gamma-index C++ library; gamma-index GPU (https://pubmed.ncbi.nlm.nih.gov/21317484/ — verify GitHub from paper) — UCSD GPU gamma; OpenGATE (https://github.com/OpenGATE/opengate) — includes dose comparison utilities.
- **CUDA libraries & GPU pattern:** One CUDA thread per reference point; shared-memory tile of evaluated dose grid (tiled by distance radius); minimum reduction in registers; atomic min for tie-breaking; cuBLAS for vectorized pass/fail statistics across patient cohort.

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
