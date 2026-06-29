# THEORY — 2.24 SAXS / SANS Data-Driven Structure Modeling

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

### 2.24 SAXS / SANS Data-Driven Structure Modeling 🟡 · Active R&D

- **Deep dive:** Small-angle X-ray/neutron scattering (SAXS/SANS) provides solution-phase structural information about proteins and complexes as a 1D intensity profile I(q). Fitting atomic or CG models to SAXS data requires rapid forward calculation of the scattering intensity from 3D coordinates via Debye formula or spherical harmonic expansion — a pairwise summation over all atoms that is GPU-parallelizable. GPU-MD + SAXS ensemble refinement (EROS, BioEn) samples thousands of conformers and reweights to match experimental SAXS. Applications include intrinsically disordered protein (IDP) ensemble characterization.
- **Key algorithms:** Debye scattering formula (O(N²) GPU-parallel), CRYSOL implicit solvent scattering model, spherical harmonic expansion for SAXS, SAXS-restrained MD ensemble refinement (EROS/BioEn), maximum entropy reweighting, atomistic vs CG SAXS prediction.
- **Datasets:** SASBDB — small-angle scattering biological data bank (https://www.sasbdb.org); PDB-SAXS depositions (https://www.rcsb.org); BIOISIS benchmark (verify URL); simulated SAXS from MD trajectories.
- **Starter repos/tools:** CRYSOL (https://www.embl-hamburg.de/biosaxs/crysol.html) — analytical SAXS computation; FOXS (https://modbase.compbio.ucsf.edu/foxs/) — fast SAXS fitting; WAXSiS (verify URL) — GPU-accelerated wide-angle scattering; MDAnalysis SAXS module (https://github.com/MDAnalysis/mdanalysis) — trajectory SAXS averaging.
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for O(N²) Debye summation over atom pairs; GPU partial sum reduction for form factors; cuBLAS for spherical harmonic coefficients; GPU-parallel ensemble member scoring.

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
