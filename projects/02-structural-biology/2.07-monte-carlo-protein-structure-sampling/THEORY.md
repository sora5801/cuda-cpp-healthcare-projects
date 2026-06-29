# THEORY — 2.7 Monte Carlo Protein Structure Sampling

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

### 2.7 Monte Carlo Protein Structure Sampling 🟡 · Active R&D

- **Deep dive:** Monte Carlo (MC) methods sample protein conformational space by proposing random moves (backbone/sidechain dihedral rotations, rigid-body domain motions) and accepting/rejecting via Metropolis criterion. GPU acceleration is applied to (i) batch scoring of many independent MC walkers in parallel and (ii) GPU-accelerated energy evaluation for each trial move. Rosetta's protein design/folding MC engine has been partially GPU-accelerated. Parallel tempering MC scales to GPU arrays via independent temperature replicas. Applications include loop modeling, sidechain packing, and protein-ligand pose sampling.
- **Key algorithms:** Metropolis-Hastings MC, parallel tempering, fragment-based backbone moves (Rosetta), rotamer library sidechain packing (Dunbrack), basin hopping, simulated annealing, energy function evaluation (Rosetta or AMBER).
- **Datasets:** CASP protein structure benchmarks (https://predictioncenter.org); PDB structures for folding benchmarks (https://www.rcsb.org); Dunbrack rotamer library (https://dunbrack.fccc.edu/bbdep2010/); CAMEO continuous benchmarking (https://www.cameo3d.org).
- **Starter repos/tools:** Rosetta (https://github.com/RosettaCommons/rosetta) — protein MC sampling (GPU extensions experimental); FoldX (https://foldxsuite.crg.eu) — fast energy evaluation for MC design; OpenMM MC (https://github.com/openmm/openmm) — Python MC on GPU via custom integrators; ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU sequence design complementary to MC backbone sampling.
- **CUDA libraries & GPU pattern:** GPU-parallel scoring of independent MC replica arrays; CUDA kernels for energy evaluation (Lennard-Jones + torsion); cuRAND for GPU random number generation; warp-level acceptance ratio evaluation.

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
