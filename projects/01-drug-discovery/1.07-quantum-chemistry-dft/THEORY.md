# THEORY — 1.7 Quantum Chemistry / DFT

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

### 1.7 Quantum Chemistry / DFT 🟢 · Established

- **Deep dive:** Density Functional Theory (DFT) calculates electronic structure by solving the Kohn-Sham equations self-consistently on a basis set (plane waves or Gaussians). The dominant cost is the construction of the Fock/Kohn-Sham matrix via electron repulsion integrals (ERIs) — an O(N^4) bottleneck that GPUs reduce substantially by computing integrals in batches. TeraChem pioneered GPU-accelerated DFT and can achieve 100× speedup over single-CPU codes. Applications in drug discovery include geometry optimization of drug fragments, calculation of electrostatic potential maps for pharmacophore generation, and QM-derived force field parameterization.
- **Key algorithms:** Kohn-Sham SCF, B3LYP/ωB97X-D exchange-correlation functionals, resolution-of-identity (RI) approximation for ERIs, DIIS convergence acceleration, plane-wave pseudopotential (PW-PP), linear-scaling DFT.
- **Datasets:** QM9 — DFT-computed properties of 134k organic molecules (https://doi.org/10.6084/m9.figshare.978904); ANI-1ccx — CCSD(T)-level energies for diverse organic molecules (https://github.com/isayev/ANI1ccx_dataset); PubChemQC — DFT calculations for ~3M PubChem molecules (http://pubchemqc.riken.jp); CSD — Cambridge Structural Database for crystal structures (https://www.ccdc.cam.ac.uk).
- **Starter repos/tools:** TeraChem (https://www.petachem.com) — GPU-native DFT, commercial but widely cited; PySCF (https://github.com/pyscf/pyscf) — pure Python quantum chemistry with GPU4PySCF extension; CP2K (https://github.com/cp2k/cp2k) — GPU-accelerated mixed Gaussian/plane-wave DFT; NWChem (https://github.com/nwchemgit/nwchem) — parallel quantum chemistry with GPU-accelerated modules.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for ERI computation (two-electron integrals in shared memory); cuBLAS for matrix diagonalization; cuFFT for plane-wave FFT; warp-level parallelism over shell pairs.

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
