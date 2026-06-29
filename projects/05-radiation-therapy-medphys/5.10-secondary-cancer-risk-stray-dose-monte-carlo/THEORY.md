# THEORY — 5.10 Secondary Cancer Risk & Stray-Dose Monte Carlo

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

### 5.10 Secondary Cancer Risk & Stray-Dose Monte Carlo 🔴 · Frontier/Theoretical
- **Deep dive:** Radiotherapy delivers dose not only to the target but also to distant organs via stray radiation (leakage, scatter, neutrons from proton therapy nuclear interactions), creating secondary cancer risk. Stray-dose is ~3–4 orders of magnitude lower than target dose, requiring 10¹¹–10¹²+ particle histories per calculation for statistical precision — intractable even on GPU without variance reduction (splitting, forced detection, geometry importance). GPU-based stray-dose MC requires importance sampling and photon-electron transport over the full body habitus beyond the treated field, rarely implemented in commercial systems. Secondary neutron fluence from proton therapy high-Z nozzle elements requires hadronic physics in Geant4/TOPAS, adding GPU parallelization complexity.
- **Key algorithms:** Forced detection variance reduction, splitting/Russian roulette, photonuclear interaction cross-sections, hadronic interaction model (INCL, BERT) for secondary neutrons, whole-body geometric phantom integration (ICRP110 voxel phantoms), Lifetime Risk Model (BEIR VII) convolution with dose distribution.
- **Datasets:** ICRP 110 voxel phantoms (adult male/female, https://www.icrp.org/publication.asp?id=ICRP%20Publication%20110); NIST photon cross-section databases (https://www.nist.gov/pml/xcom-photon-cross-sections); secondary dose measurements from literature; TCIA proton therapy planning CTs.
- **Starter repos/tools:** TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — full hadronic transport, stray-dose extensions; GATE 10 (https://github.com/OpenGATE/opengate) — neutron transport, out-of-field dose scoring; EGSnrc (https://github.com/nrc-cnrc/EGSnrc) — photon/electron with advanced variance reduction; PHITS (https://phits.jaea.go.jp/ — verify URL) — hadronic + neutron transport for radiation protection.
- **CUDA libraries & GPU pattern:** Custom CUDA hadronic transport kernel (one thread per particle, nested interaction sampling loop); constant memory for cross-section tables; variance reduction handled per-thread (splitting → thread forking via particle stack on GPU); cuRAND for correlated sampling sequences.

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
