# THEORY — 5.12 FLASH Radiotherapy GPU Modeling

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

### 5.12 FLASH Radiotherapy GPU Modeling 🔴 · Frontier/Theoretical
- **Deep dive:** FLASH-RT delivers doses at ultra-high dose rates (>40 Gy/s, typically >10⁴ Gy/s for electrons, >100 Gy/s for protons) in millisecond pulses, sparing normal tissue while maintaining tumor control. Modeling the FLASH effect requires coupled radiation-chemistry simulation: (1) GPU MC particle transport to compute local dose deposition patterns, (2) GPU track-structure to generate initial radical (OH•, H₂O₂, e⁻ₐq) distributions, and (3) GPU diffusion-reaction kinetics to simulate oxygen depletion and radical recombination in tissue. The MPEXS2.1-DNA code implements GPU water radiolysis under UHDR. Biological effect modeling requires stochastic ODE integration over microscopic reaction networks — a GPU-parallel task across millions of spatial positions.
- **Key algorithms:** GPU MC particle transport at UHDR pulse structure, water radiolysis reaction-diffusion (Gillespie SSA on GPU), oxygen depletion kinetics, stochastic diffusion-reaction (MPEXS2.1-DNA), LET-dependent radical yield models, oxygen enhancement ratio (OER) map computation, pulse-by-pulse dose accumulation.
- **Datasets:** FLASH-RT experimental dosimetry from CERN/CLEAR, UCLouvain, Stanford FLASH programs (verify access); AAPM FLASH-RT working group benchmark datasets (verify URL); published oxygen tension measurements in tumors; GEANT4-DNA radiolysis validation datasets.
- **Starter repos/tools:** MPEXS2.1-DNA (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12551771/ — verify GitHub URL from paper) — GPU water radiolysis for UHDR; GATE 10 (https://github.com/OpenGATE/opengate) — FLASH macro-dose MC; TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — FLASH dosimetry extensions; Geant4-DNA (https://github.com/Geant4/geant4) — micro-kinetics for FLASH effect modeling.
- **CUDA libraries & GPU pattern:** Custom CUDA diffusion-reaction kernel (per-spatial-voxel Gillespie SSA, one thread block per µm³ tissue voxel); cuRAND for stochastic reaction channel selection; shared memory for local species concentration array; CUDA streams for pipelining pulse-by-pulse dose transport and chemistry; atomic ops for species count updates.

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
