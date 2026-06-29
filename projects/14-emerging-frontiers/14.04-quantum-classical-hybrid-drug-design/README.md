# 14.4 — Quantum-Classical Hybrid Drug Design

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.4`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

Quantum computers can solve electronic structure problems for drug-binding active sites more accurately than classical DFT, but current NISQ devices are noisy and limited to ~50–100 qubits. Hybrid quantum-classical algorithms (VQE for Hamiltonian ground-state energies, QAOA for docking optimization) run the quantum circuit on the QPU and the classical optimization loop on GPU clusters, with GPU accelerating the many-shot Pauli expectation-value estimation. AWS Quantum Computing Exploration for Drug Discovery (2024) demonstrates VQE-based protein folding in small fragments. GPU simultaneously handles the classical molecular mechanics components of QM/MM with GPU-accelerated DFT (CP2K, Psi4 on GPU). The practical near-term use case is 20–50 atom active-site electronic structure for tight binding-affinity ranking.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Variational Quantum Eigensolver (VQE), Quantum Approximate Optimization Algorithm (QAOA), GPU-accelerated density functional theory (DFT, B3LYP/PBE), QM/MM with quantum active site, orbital-free embedding, GPU-accelerated tensor network contraction for quantum state simulation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/quantum-classical-hybrid-drug-design.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/quantum-classical-hybrid-drug-design.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\quantum-classical-hybrid-drug-design.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PDBbind refined binding affinity dataset (http://www.pdbbind.org.cn/); ChEMBL (https://www.ebi.ac.uk/chembl/) for classical ML baseline; QM9 (GPU DFT benchmark, 134 K small molecules, https://paperswithcode.com/dataset/qm9); AWS Quantum Drug Discovery Benchmark (https://github.com/aws-solutions-library-samples/quantum-computing-exploration-for-drug-discovery-on-aws).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

Qiskit (https://github.com/Qiskit/qiskit) — VQE/QAOA with GPU-accelerated statevector simulator (cuStateVec); PennyLane (https://github.com/PennyLaneAI/pennylane) — differentiable quantum ML with GPU backend; Psi4 (https://github.com/psi4/psi4) — GPU-accelerated QM (CUDA DFT integrals); CP2K (https://github.com/cp2k/cp2k) — GPU QM/MM with CUDA backend.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuStateVec (NVIDIA cuQuantum) for GPU quantum circuit simulation, cuDNN for NN-guided ansatz optimization, CUDA DFT integral kernels (Psi4/CP2K); pattern: drug-protein complex → GPU DFT for electronic Hamiltonian → Pauli decomposition → VQE on GPU statevector sim (or QPU) → binding ΔG estimate → classical optimizer updates ansatz parameters. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
