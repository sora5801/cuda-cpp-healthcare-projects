# THEORY — 5.3 Proton & Heavy-Ion Therapy Dose

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

### 5.3 Proton & Heavy-Ion Therapy Dose 🟡 · Active R&D
- **Deep dive:** Proton and carbon-ion beams deposit dose with a sharp Bragg peak distal to the target, enabling sparing of surrounding normal tissue. Analytical dose engines (pencil-beam algorithm, PBA) convolve pencil-beam kernels with CT stopping-power maps; GPU parallelizes the per-spot convolution across the ~10⁴ spots in a plan, reducing a full plan from minutes to seconds. Full Monte Carlo (FRED, TOPAS, GATE) simulates hadronic physics including nuclear fragmentation (dominant for carbon ions), requiring GPU for clinical throughput. Range uncertainty (due to CT Hounsfield-unit–to–stopping-power conversion) is managed by robust optimization over 3 mm / 3.5% scenarios, multiplying GPU compute requirements.
- **Key algorithms:** Pencil-beam algorithm (PBA), analytical Bragg-peak model, GPU MC (FRED, MOQUI, gPMC), nuclear fragmentation transport (Geant4-TOPAS), LET (linear energy transfer) calculation, RBE (relative biological effectiveness) weighting, multi-field optimization, robust proton optimization.
- **Datasets:** TOPAS/GATE benchmark proton beam data; clinical proton CT datasets (develop via institution); TCIA proton treatment response datasets; POPI model for proton treatment planning (https://www.creatis.insa-lyon.fr/rio/popi-model).
- **Starter repos/tools:** FRED (https://www.fredonline.eu/) — GPU fast MC for ions, clinical-grade, DICOM-RT input; MOQUI (https://github.com/mghro/moquimc) — GPU proton MC for quick dose recalculation (MGH, open source); OpenTOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — open fork of TOPAS, Geant4-based proton MC; matRad (https://github.com/e0404/matRad) — analytic proton dose engine with GPU-parallel spot convolution.
- **CUDA libraries & GPU pattern:** Custom CUDA for per-spot pencil-beam convolution (one thread per spot × voxel pair); cuFFT for convolution in k-space; texture memory for CT stopping-power map; cuRAND for MC sampling; CUDA atomic adds for dose histogram accumulation.

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
