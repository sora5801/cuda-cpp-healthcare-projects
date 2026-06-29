# THEORY — 5.1 Monte Carlo Dose Calculation

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

### 5.1 Monte Carlo Dose Calculation 🟡 · Active R&D
- **Deep dive:** Monte Carlo (MC) simulation tracks individual particle histories through patient CT geometry, sampling physics interactions (Compton scatter, pair production, photoelectric effect) stochastically. Clinical accuracy requires ~10⁸–10⁹ particle histories; on CPU (e.g., EGSnrc, MCNP), a single prostate plan takes hours. GPU MC exploits the independence of particle histories: each CUDA thread tracks one particle, with warp-level divergence managed by sorting particles by material. GPU codes (DPM-GPU, gDPM, Acuros, FRED) achieve 100× speedups over single-CPU. The primary GPU challenge is divergent execution paths when different threads take different interaction branches and managing the CT voxel geometry lookup efficiently in constant/texture memory.
- **Key algorithms:** Condensed-history electron transport, class-II MC (Berger/ICRU), photon interaction sampling (Klein-Nishina, photoelectric), bremsstrahlung production, Russian roulette / splitting variance reduction, GPU divergence management (particle sorting by material), macro-MC for ultra-fast TPS dose.
- **Datasets:** IAEA benchmark photon beam data (https://www.iaea.org/resources/databases/iaea-photon-electron-interaction-data-library); AAPM TG-119 IMRT QA phantom dataset; clinical patient CT + plan DICOM from departmental archives (IRB-required); CIRS anthropomorphic phantom CT datasets.
- **Starter repos/tools:** EGSnrc (https://github.com/nrc-cnrc/EGSnrc) — reference CPU MC for photon/electron, GPU extensions in literature; GATE 10 (https://github.com/OpenGATE/opengate) — Python-based Geant4 wrapper, GPU-capable via Geant4 MT; gDPM / DPM-GPU (verify URL, published by Ma et al.) — GPU photon/electron MC dose; FRED (https://www.fredonline.eu/) — GPU MC for proton/ion therapy (verify URL); MC-GPU (https://github.com/adler-j/GPUMC) — CUDA GPU photon MC, open source.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for particle transport loop (one thread per particle history); physics tables in constant/texture memory; warp-divergence reduction via material sorting before interaction step; atomic adds to dose voxel array; batch history generation via cuRAND.

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
