# THEORY — 14.4 Quantum-Classical Hybrid Drug Design

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

### 14.4 Quantum-Classical Hybrid Drug Design 🔴 · Frontier/Theoretical

- **Deep dive:** Quantum computers can solve electronic structure problems for drug-binding active sites more accurately than classical DFT, but current NISQ devices are noisy and limited to ~50–100 qubits. Hybrid quantum-classical algorithms (VQE for Hamiltonian ground-state energies, QAOA for docking optimization) run the quantum circuit on the QPU and the classical optimization loop on GPU clusters, with GPU accelerating the many-shot Pauli expectation-value estimation. AWS Quantum Computing Exploration for Drug Discovery (2024) demonstrates VQE-based protein folding in small fragments. GPU simultaneously handles the classical molecular mechanics components of QM/MM with GPU-accelerated DFT (CP2K, Psi4 on GPU). The practical near-term use case is 20–50 atom active-site electronic structure for tight binding-affinity ranking.
- **Key algorithms:** Variational Quantum Eigensolver (VQE), Quantum Approximate Optimization Algorithm (QAOA), GPU-accelerated density functional theory (DFT, B3LYP/PBE), QM/MM with quantum active site, orbital-free embedding, GPU-accelerated tensor network contraction for quantum state simulation.
- **Datasets:** PDBbind refined binding affinity dataset (http://www.pdbbind.org.cn/); ChEMBL (https://www.ebi.ac.uk/chembl/) for classical ML baseline; QM9 (GPU DFT benchmark, 134 K small molecules, https://paperswithcode.com/dataset/qm9); AWS Quantum Drug Discovery Benchmark (https://github.com/aws-solutions-library-samples/quantum-computing-exploration-for-drug-discovery-on-aws).
- **Starter repos/tools:** Qiskit (https://github.com/Qiskit/qiskit) — VQE/QAOA with GPU-accelerated statevector simulator (cuStateVec); PennyLane (https://github.com/PennyLaneAI/pennylane) — differentiable quantum ML with GPU backend; Psi4 (https://github.com/psi4/psi4) — GPU-accelerated QM (CUDA DFT integrals); CP2K (https://github.com/cp2k/cp2k) — GPU QM/MM with CUDA backend.
- **CUDA libraries & GPU pattern:** cuStateVec (NVIDIA cuQuantum) for GPU quantum circuit simulation, cuDNN for NN-guided ansatz optimization, CUDA DFT integral kernels (Psi4/CP2K); pattern: drug-protein complex → GPU DFT for electronic Hamiltonian → Pauli decomposition → VQE on GPU statevector sim (or QPU) → binding ΔG estimate → classical optimizer updates ansatz parameters.

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
