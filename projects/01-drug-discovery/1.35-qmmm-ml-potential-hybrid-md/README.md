# 1.35 — QMMM/ML Potential Hybrid MD

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.35`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._
>
> **⚠️ Reduced-scope teaching version (CLAUDE.md §13).** The full research method
> needs a *trained* equivariant neural-network potential (MACE/NequIP) + PyTorch
> autograd + a production force field. This project keeps every **structural**
> idea of that pipeline and shrinks only the scale: the ML potential is a tiny
> Behler–Parrinello-style network with **fixed, synthetic surrogate weights**
> (clearly labeled), differentiated **analytically** for forces. The energies are
> not physical quantities for any real molecule.

## Summary

A **hybrid neural-network-potential / molecular-mechanics (NNP/MM)** molecular-
dynamics engine, in miniature. We model a 1-D chain of atoms whose *reactive
center* is described by a machine-learned potential (here, a small surrogate
neural network standing in for a QM-accurate model) and whose *environment* is a
classical Lennard-Jones force field. The two regions are stitched together at a
**link atom** by **mechanical embedding**. We then run an **ensemble** of short
velocity-Verlet trajectories — one per GPU thread — that differ by a small
perturbation of the link atom, mimicking the *active-learning* sweeps used to
train reactive NNPs. The GPU result is verified against a serial CPU reference.

## What this computes & why the GPU helps

The next frontier beyond QM/MM is using ML potentials trained on QM data to
replace the expensive QM region — enabling microsecond reactive MD at QM
accuracy. GPU-accelerated equivariant NNPs (MACE, NequIP) serve as drop-in QM
replacements in an MM environment; the hybrid runs fully on GPU with the NNP
forward pass and MM evaluation overlapping in CUDA streams.

**The parallel bottleneck.** Two axes are parallel: (1) *within* one force
evaluation, every atom's descriptor and MLP forward/backward pass is independent;
(2) *across* the workload, every short trajectory in an active-learning ensemble
is independent. This teaching version exploits **axis (2)**: it gives **each
trajectory its own GPU thread** (the ensemble pattern, docs/PATTERNS.md §1), the
same mapping used by flagships 9.02 (SEIR) and 13.02 (PBPK). The per-step physics
is shared `__host__ __device__` code, so the GPU reproduces the CPU exactly.

## The algorithm in brief

- **Descriptor:** Behler–Parrinello radial *symmetry functions* encode each ML
  atom's local environment as a fixed-length vector `G` (Gaussians of neighbor
  distances inside a smooth cutoff).
- **NNP energy:** a 1-hidden-layer MLP (`tanh`) maps `G` → per-atom energy; the
  ML region energy is the sum over ML atoms.
- **Forces:** the MLP is differentiated **analytically** (`dE/dG`, chain rule
  through `dG/dr`) — the explicit version of what autograd does for a real NNP.
- **MM region + coupling:** Lennard-Jones 12-6 over MM–MM and ML–MM pairs
  (mechanical embedding); ML–ML pairs are governed by the NNP instead.
- **Link-atom boundary:** an explicit atom caps the ML region (the standard
  QM/MM trick), so the ML neighborhood is well-defined.
- **Integrator:** velocity-Verlet (symplectic, time-reversible) — stable energy.
- **Ensemble / active learning:** M trajectories differ by a deterministic
  link-atom perturbation; one GPU thread runs each to completion.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including the analytic force gradient.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/qmmm-ml-potential-hybrid-md.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/qmmm-ml-potential-hybrid-md.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\qmmm-ml-potential-hybrid-md.sln /p:Configuration=Release /p:Platform=x64
```

No extra CUDA libraries are linked (only `cudart`): all kernels are hand-written
so nothing is a black box.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/ensemble_params.txt`, prints the
ensemble summary, shows the GPU-vs-CPU agreement check, and prints a timing line
on stderr.

## Data

- **Sample (committed):** `data/sample/ensemble_params.txt` — a tiny **run
  config** (`M dt steps amp`) so the demo runs offline with zero downloads. The
  physical system is defined in code (`src/nnpmm.h`), not in this file.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to the real
  QM/DFT training sets (you would need these to train a *genuine* NNP).
- **Provenance & license:** see [data/README.md](data/README.md). All data here is
  **synthetic** and labeled as such.

Catalog dataset notes: ANI-1ccx reactive extensions (verify URL); DFT reaction
pathway datasets from QM/MM studies; Transition1x — 10M DFT calculations along
reaction paths (https://zenodo.org/record/5781475); SPICE dataset
(https://github.com/openmm/spice-dataset).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program runs the ensemble on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts the per-member summaries agree
within tolerance `1e-6` (observed worst diff `~4e-13`). It also reports a physical
sanity check — **energy conservation** per trajectory — because a symplectic
integrator keeps total energy bounded; that validates the *dynamics*, not just
CPU==GPU agreement.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the config, runs CPU + GPU, verifies, reports.
2. [`src/nnpmm.h`](src/nnpmm.h) — **the heart**: the shared `__host__ __device__`
   physics (descriptor → MLP → analytic force → Verlet → trajectory driver).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the ensemble mapping.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per trajectory) + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, host I/O helpers.

## Prior art & further reading

- **[MACE](https://github.com/ACEsuit/mace)** — fast higher-order equivariant
  message-passing NNP; the modern choice for hybrid ML/MM. Study its message
  construction to see what our toy descriptor abstracts.
- **[TorchMD-Net](https://github.com/torchmd/torchmd-net)** — equivariant NNPs
  with MM coupling; good reference for the NNP/MM interface.
- **[OpenMM-ML](https://github.com/openmm/openmm-ml)** — the production NNP/MM
  interface for OpenMM; read it for how mechanical vs. electrostatic embedding is
  actually wired up.
- **[NNPOps](https://github.com/openmm/NNPOps)** — CUDA-optimized NNP primitives
  (neighbor lists, symmetry functions); the real version of our descriptor loop.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble / independent jobs** (docs/PATTERNS.md §1): one GPU thread runs one
full MD trajectory in registers and writes a single summary — no shared memory,
no atomics, no inter-thread communication. The per-step physics is a shared
`__host__ __device__` core (PATTERNS.md §2) so CPU and GPU run identical math.

Catalog's stated production pattern (the *full* method, beyond this teaching
scope): CUDA MACE kernels for equivariant message passing; OpenMM CUDA platform
for the MM region; CUDA streams for async NNP+MM overlap; PyTorch autograd for
NNP force gradients; cuBLAS for spherical-harmonic transforms.

## Exercises

1. **Shrink the timestep for stiff configs.** Member `m0` (link atom shoved into
   the LJ wall) conserves energy worst. Regenerate the config with
   `python scripts/make_synthetic.py --dt 0.001 --steps 1500` and watch the
   `worst energy conservation` figure drop. Why does the symplectic integrator
   still not blow up at the larger `dt`?
2. **Go 3-D.** Replace the scalar `x[i]` with `float3`/`double3` and update the
   descriptor/force loops. The descriptor and chain-rule math are unchanged; only
   `dr/dx` becomes a unit vector. (This is the single biggest step toward realism.)
3. **Add a second symmetry-function family.** Add angular (3-body) symmetry
   functions so the descriptor "sees" bond angles, not just distances — the other
   half of a real Behler–Parrinello descriptor.
4. **Two CUDA streams.** Split the force evaluation into an NNP kernel and an MM
   kernel and overlap them in separate streams (`cudaStreamCreate`), the way the
   real hybrid overlaps the NNP forward pass with MM — measure the overlap.
5. **δ-ML correction.** Make the NNP a *correction* on top of a cheap baseline
   (e.g. a harmonic bond), as in δ-ML/Δ-learning, and verify forces still match.

## Limitations & honesty

- **Surrogate, not trained.** The NNP weights are fixed synthetic constants, not
  a model trained on QM data. The network architecture (descriptor → MLP →
  analytic force) is faithful; the *parameters* are made up and labeled so.
- **1-D and tiny.** Eight atoms on a line. Real systems are 3-D with thousands of
  atoms, neighbor lists, and periodic boundaries. The math generalizes directly
  (Exercise 2); the scale does not fit a teaching file.
- **Mechanical embedding only.** No electrostatics across the boundary
  (electrostatic/polarizable embedding and long-range PME are the hard part of
  real QM/MM-ML — see THEORY §real-world).
- **Energies are unitless and arbitrary.** Nothing here is a physical energy,
  rate, or property of any molecule. Educational only — never a chemical or
  clinical claim.
