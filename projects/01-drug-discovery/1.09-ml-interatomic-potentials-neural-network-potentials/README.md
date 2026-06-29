# 1.9 — ML Interatomic Potentials (Neural Network Potentials)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.9`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **neural network potential (NNP)** predicts a molecule's energy from the
positions of its atoms — fast enough for molecular dynamics, but trained to be far
more accurate than a classical force field. This project builds a small,
**Behler-Parrinello-style** NNP and runs it on the GPU: for each atom it computes
a set of **atom-centered symmetry functions** (a rotation/translation-invariant
"fingerprint" of the atom's local environment), feeds that fingerprint through a
tiny per-atom neural network to get an **atomic energy** `E_i`, and sums the atomic
energies into a total energy `E = Σ E_i`. Because each atom's energy depends only
on its neighbors within a cutoff, the atoms are independent jobs — one GPU thread
per atom. It is a deliberately **reduced-scope teaching version** (one element,
radial descriptors only, fixed weights); `THEORY.md` explains exactly what
production NNPs (ANI, NequIP, MACE) add.

## What this computes & why the GPU helps

Neural network potentials (NNPs) learn the potential energy surface from ab initio
data, reproducing near-DFT accuracy at near-classical-MD speed. Architectures range
from atom-centered symmetry functions (ANI) to equivariant message-passing networks
(NequIP, MACE, SchNet). GPU acceleration is essential: each forward pass involves
neighborhood construction, an evaluation over all atomic pairs within a cutoff, and
(in training/MD) backpropagation for forces. On an A100 a ~500-atom protein+ligand
system runs at ~10 ns/day — ~1000× slower than a classical force field but ~100×
faster than DFT, enabling reactive drug-target simulations previously impossible.

**The parallel bottleneck:** the energy is `E = Σ_i E_i`, and computing each `E_i`
means (a) scanning atom `i`'s neighbors to build its descriptor and (b) running a
small MLP forward. That work is **identical in form for every atom and independent
across atoms**, so it maps perfectly onto "one thread per atom" (this project's
GPU kernel). At protein scale (thousands of atoms × thousands of MD steps) this is
the dominant cost, and doing all atoms at once is exactly where the GPU wins.

## The algorithm in brief

- **Atom-centered symmetry functions (ACSF / Behler G2):** for atom `i`, a vector of
  radial descriptors `G2_s(i) = Σ_{j≠i, r<Rc} exp(−η(r_ij − Rs_s)²)·fc(r_ij)` — one
  smooth Gaussian "shell" per center `Rs_s`, tapered to zero at the cutoff by the
  cosine cutoff `fc`.
- **Per-atom MLP:** a small fully-connected network `desc → tanh → tanh → E_i`
  (here 8→16→16→1) maps the descriptor to that atom's energy contribution.
- **Sum:** `E = Σ_i E_i` (the additive-atomic-energy ansatz that makes NNPs scale).
- **Where production NNPs go further:** angular (three-body) symmetry functions,
  multiple element types, equivariant message passing (NequIP/MACE), and analytic
  **forces** via autograd. Covered in `THEORY.md`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ml-interatomic-potentials-neural-network-potentials.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ml-interatomic-potentials-neural-network-potentials.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ml-interatomic-potentials-neural-network-potentials.sln /p:Configuration=Release /p:Platform=x64
```

The project links only `cudart_static.lib` (no extra CUDA libraries — the
descriptor and MLP are hand-written so the math is fully visible).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/water_cluster.xyzc`, prints the
per-atom and total energies, shows the GPU-vs-CPU agreement check, and prints a
timing line.

## Data

- **Sample (committed):** `data/sample/water_cluster.xyzc` — a tiny **synthetic**
  24-atom cluster so the demo runs with zero downloads.
- **Generate / resize:** `python scripts/make_synthetic.py --mols 64`.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print the URLs and fetch
  guidance (they never bypass any registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: ANI-1ccx — CCSD(T) energies on 500k conformers of drug-like
molecules (<https://github.com/isayev/ANI1ccx_dataset>); SPICE — quantum-chemistry
dataset for ML potentials covering drug-like molecules and proteins
(<https://github.com/openmm/spice-dataset>); rMD17 — revised MD17 benchmark
(<https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038>);
OE62 — 62k organic molecules with DFT energetics (verify URL).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
first few per-atom energies, the total energy, and `RESULT: PASS`. The program
computes the energies on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-9` — that agreement is
the correctness guarantee. Both paths call the **same** `__host__ __device__`
physics in `src/nnp.h`, so the measured difference is only floating-point round-off
(~`1e-15` here; see the stderr `[verify]` line and `THEORY.md §Numerics`).

## Code tour

Read in this order:

1. [`src/nnp.h`](src/nnp.h) — **the shared physics** (`__host__ __device__`): the
   cutoff function, the radial descriptor, and the per-atom MLP. Both CPU and GPU
   call these, so start here.
2. [`src/main.cu`](src/main.cu) — loads the structure, builds the model, runs CPU +
   GPU, verifies, reports (deterministic stdout / timing on stderr).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-atom
   idea (constant-memory model).
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline +
   the model/weight builder + the file loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **TorchANI** (<https://github.com/aiqm/torchani>) — PyTorch ANI NNP with CUDA
  acceleration and OpenMM integration. Study how ACSF descriptors and per-element
  networks are organized; our `nnp.h` is a one-element, radial-only cousin.
- **TorchMD-Net** (<https://github.com/torchmd/torchmd-net>) — equivariant NNPs with
  a GPU-optimized neighbor list. Study the neighbor-list construction we replace
  with a brute-force scan.
- **MACE** (<https://github.com/ACEsuit/mace>) — fast equivariant NNP with GPU
  kernels; the state of the art for accuracy-per-parameter. Study the
  higher-body equivariant features that go beyond our radial descriptors.
- **NequIP** (<https://github.com/mir-group/nequip>) — E(3)-equivariant network;
  study why equivariance buys data efficiency.
- Foundational papers: Behler & Parrinello, *PRL* 98, 146401 (2007); Behler,
  *J. Chem. Phys.* 134, 074106 (2011) (symmetry functions); Smith et al., *Chem.
  Sci.* 8, 3192 (2017) (ANI-1).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs + constant-memory model** (PATTERNS.md row "score one query vs N
items, each independent"). One GPU thread owns one atom and computes its `E_i`
end to end (descriptor + MLP). The read-only model — the ACSF hyperparameters and
the MLP weights — lives in **constant memory**, broadcast to a whole warp from one
address (the same trick the 1.12 Tanimoto flagship uses for its query). The final
sum is done on the host in a fixed atom order so the total is **deterministic** and
matches the CPU exactly (PATTERNS.md §3). Production NNPs add a neighbor-list kernel
and PyTorch CUDA **autograd** for forces; this teaching version keeps the energy
forward pass fully hand-written so nothing is a black box.

## Exercises

1. **Add angular descriptors.** Implement a Behler G4/G5 three-body symmetry
   function (it depends on the angle `θ_jik`) and extend `N_DESC`. Watch the
   descriptor distinguish environments that the radial-only version cannot.
2. **Compute forces.** Forces are `F = −∇E`. Implement an analytic gradient (or a
   finite-difference check) and verify `F` against a central difference of `E`.
3. **Two element types.** Give H and O separate networks and a type array; route
   each atom to its element's MLP (this is exactly how ANI handles multiple species).
4. **Neighbor list.** Replace the O(n²) brute-force neighbor scan with a cell list
   so the cost becomes O(n); measure the crossover atom count where it wins.
5. **Block-size / precision sweep.** Try FP32 vs FP64 and 64/128/256 threads/block;
   record the GPU time on stderr and the effect on the GPU-vs-CPU error.

## Limitations & honesty

- **Reduced scope (on purpose).** Single atom type, **radial-only** descriptors, a
  small **fixed** MLP whose weights are *manufactured, not trained*. The printed
  energies are in arbitrary synthetic units and have **no chemical meaning** — the
  real lesson is the descriptor → per-atom-MLP → sum *pipeline* and its GPU mapping.
- **Synthetic data, labeled everywhere.** `data/sample/water_cluster.xyzc` is a
  generated cluster, not a real molecule (see `data/README.md`).
- **No forces, no training, no periodic boundaries.** Production NNPs need analytic
  forces (autograd), training on quantum-chemistry data, and PBC for condensed
  phases — all described in `THEORY.md §Where this sits in the real world`.
- **O(n²) neighbor search** here for clarity; real codes use cell lists.
- **Not for any chemical or clinical use.** Educational study material only.
