# THEORY — 1.23 QM/MM Molecular Dynamics

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

### 1.23 QM/MM Molecular Dynamics 🟡 · Active R&D

- **Deep dive:** Hybrid quantum mechanics/molecular mechanics (QM/MM) partitions a system into a reactive QM region (drug + key residues, 50–200 atoms) treated at DFT/semi-empirical level and a larger MM region. GPU acceleration applies to both the QM Hamiltonian (via TeraChem/GPU-DFT) and the MM dynamics (via AMBER/GROMACS). The critical bottleneck is the QM/MM electrostatic coupling and QM Hamiltonian evaluation at every MD step. Open-source GPU QM/MM is available via AMBER+QUICK (GPU-accelerated DFT engine). Applications include enzyme catalysis mechanism, covalent drug reactivity, and proton transfer pathways.
- **Key algorithms:** ONIOM/link-atom QM/MM coupling, electrostatic embedding, DFT-based QM region (B3LYP/PBE), GFN2-xTB semi-empirical QM, AIMD in QM region with Verlet MM, adaptive QM/MM for large reactive systems.
- **Datasets:** QM/MM benchmark from SAMPL challenges (verify URL); enzyme reaction databases (BRENDA, https://www.brenda-enzymes.org); crystal structures of enzyme-drug complexes from PDB (https://www.rcsb.org); RCSB ligand validation data (https://www.rcsb.org).
- **Starter repos/tools:** AMBER+QUICK (https://github.com/merzlab/QUICK) — GPU-accelerated DFT for QM/MM with AMBER; TeraChem-TCPB (https://www.petachem.com) — GPU DFT server for QM/MM with NAMD/AMBER; OpenMM+PySCF QM/MM (https://github.com/openmm/openmm) — Python QM/MM interface; cp2k (https://github.com/cp2k/cp2k) — GPU-accelerated QM/MM for periodic systems.
- **CUDA libraries & GPU pattern:** GPU ERI computation for QM Hamiltonian via TeraChem/QUICK CUDA kernels; MM region on GPU (pmemd.cuda); asynchronous GPU-CPU communication for QM/MM coupling; CUDA streams for overlapping QM and MM compute.

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
