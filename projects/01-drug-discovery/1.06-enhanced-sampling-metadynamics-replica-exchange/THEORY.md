# THEORY — 1.6 Enhanced Sampling — Metadynamics & Replica Exchange

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

### 1.6 Enhanced Sampling — Metadynamics & Replica Exchange 🟢 · Established

- **Deep dive:** Standard MD cannot cross large free energy barriers on accessible timescales. Enhanced sampling methods accelerate conformational exploration by adding history-dependent bias potentials (metadynamics) or by running multiple copies at different temperatures/Hamiltonians (REMD/HREX). PLUMED plugs into GROMACS, NAMD, OpenMM, and LAMMPS to implement CVs and bias on the fly. GPU MD trajectories feed the bias engine with minimal overhead. Well-tempered metadynamics ensures convergence of the free energy surface (FES) and is widely used for drug binding pathway elucidation. GPU-MetaD (2025) achieves full-lifecycle GPU acceleration for ML potential metadynamics with systems up to 1.3M atoms.
- **Key algorithms:** Well-tempered metadynamics, funnel metadynamics, Hamiltonian replica exchange (HREX), temperature REMD (T-REMD), replica exchange with solute tempering (REST2), collective variable (CV) on-the-fly evaluation, free energy surface estimation via reweighting.
- **Datasets:** PLUMED-NEST — repository of published metadynamics/enhanced sampling input files (https://www.plumed-nest.org); GPCRmd trajectory archive (https://gpcrmd.org); D. E. Shaw millisecond MD datasets (available via RCSB); benchmark FES for alanine dipeptide / chignolin (commonly used test systems).
- **Starter repos/tools:** PLUMED (https://github.com/plumed/plumed2) — plugin for collective variables and enhanced sampling, GPU-compatible via host MD engine; GROMACS + PLUMED (https://github.com/gromacs/gromacs) — standard GPU metadynamics stack; OpenPathSampling (https://github.com/openpathsampling/openpathsampling) — transition path sampling framework; HTMD (https://github.com/Acellera/htmd) — high-throughput MD with adaptive sampling on GPU clusters.
- **CUDA libraries & GPU pattern:** Bias potential evaluation on CPU via PLUMED with negligible overhead; full MD force/integration on GPU; multi-walker metadynamics uses MPI + NCCL across GPUs; GPU kernels for on-the-fly CV computation in GPU-MetaD.

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
