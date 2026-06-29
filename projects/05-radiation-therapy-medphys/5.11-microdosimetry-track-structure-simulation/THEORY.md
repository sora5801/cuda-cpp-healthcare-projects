# THEORY — 5.11 Microdosimetry & Track-Structure Simulation

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

### 5.11 Microdosimetry & Track-Structure Simulation 🔴 · Frontier/Theoretical
- **Deep dive:** Microdosimetry and nanodosimetry characterize the stochastic distribution of energy deposition events in microscopic volumes (µm–nm scale), relevant for predicting DNA damage and biological effectiveness. Track-structure codes (Geant4-DNA, MPEXS-DNA) simulate every electron interaction step-by-step, requiring liquid water cross-sections down to sub-eV energies; a single proton track produces ~10⁵ secondary interactions. GPU parallelization across simultaneous primary particle tracks (one thread per track) achieves 50–70× speedup. Applications include carbon-ion RBE calculation, targeted radionuclide dosimetry (alpha emitters), and predicting clustered DNA damage yields from mixed radiation fields.
- **Key algorithms:** Event-by-event track structure (Geant4-DNA cross-sections), step-by-step condensed random walk, DNA damage scoring (DSB, SSB, base damage), diffusion-reaction chemistry simulation (radiolysis), nanodosimeter simulation, LET spectrum calculation, biological effectiveness prediction.
- **Datasets:** Geant4-DNA physics validation data (https://geant4-dna.in2p3.fr/); NIST electron stopping powers (https://www.nist.gov/pml/estar); AAPM/NCRP microdosimetry benchmark datasets; published DNA damage yield datasets from radiobiology experiments.
- **Starter repos/tools:** Geant4-DNA (https://geant4-dna.in2p3.fr/ — part of Geant4, https://github.com/Geant4/geant4) — standard track-structure code; MPEXS-DNA (CUDA GPU version, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6850505/ — verify GitHub) — GPU microdosimetry and radiolysis; TOPAS-nBio (https://github.com/topas-nbio/TOPAS-nBio) — nano-biological extension of TOPAS; PARTRAC (verify URL) — track structure specialized for DNA damage.
- **CUDA libraries & GPU pattern:** Custom CUDA per-track simulation (one warp per track, reaction lookup in constant memory); divergence minimized by sorting tracks by interaction type before step; cuRAND Philox generator for per-track random sequences; atomic adds to DNA damage histogram; shared memory for cross-section table of current material step.

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
