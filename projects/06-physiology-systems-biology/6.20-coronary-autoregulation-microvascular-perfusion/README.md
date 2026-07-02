# 6.20 — Coronary Autoregulation & Microvascular Perfusion

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.20`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project simulates blood flow through a small **coronary microvascular network** — a graph of
cylindrical vessel *segments* meeting at *junction nodes* — and lets the arterioles **autoregulate** their
radii to keep perfusion near a metabolic set-point. Physically, each segment obeys **Poiseuille's law**
(flow ∝ pressure-drop × conductance, conductance ∝ radius⁴), and conservation of flow at every interior
node turns the whole network into a **sparse, symmetric, positive-definite linear system** for the nodal
pressures. We solve that system with **Conjugate Gradient (CG)**, whose inner loop is a **sparse
matrix-vector product (SpMV)** run on the GPU with **one thread per node**. On top of the flow solve we run
an outer **autoregulation** loop and compute a virtual **Fractional Flow Reserve (FFR)** across a modeled
stenosis — the clinical read-out this class of model targets.

## What this computes & why the GPU helps

Coronary blood flow is regulated by metabolic (adenosine), myogenic, and neural mechanisms operating across
scales from capillaries (5 µm) to epicardial arteries (4 mm). GPU simulation of a microvascular network with
10⁴–10⁶ vessel segments requires solving a large sparse linear system (network Poiseuille flow) coupled to
oxygen transport (convection-diffusion along each segment) and auto-regulatory feedback ODEs. Real-time
coronary perfusion models support fractional flow reserve (FFR) virtual assessment for stenosis evaluation.

**The parallel bottleneck:** the cost is dominated by repeatedly solving `L p = b` for the nodal pressures
`p` — once per autoregulation step, and once per CG iteration inside that we do a **sparse SpMV** over the
whole network. For a real network (10⁴–10⁶ segments) that SpMV is a massively parallel, memory-bound
operation: every node's row is independent, so we assign **one GPU thread per node** and let thousands of
rows compute simultaneously. This is exactly the workload cuSPARSE's SpMV accelerates; we hand-roll the
CSR-SpMV here so nothing is a black box.

## The algorithm in brief

- **Poiseuille conductance** per segment: `G = π r⁴ / (8 μ L)`, with a **Fåhræus–Lindqvist** hematocrit/
  radius-dependent viscosity `μ` (see `src/coronary.h`).
- **Graph-Laplacian assembly**: flow conservation at interior nodes → symmetric SPD system `L p = b`;
  Dirichlet (fixed-pressure) inlet/outlet nodes are *eliminated* into `b` to keep `L` SPD.
- **Conjugate Gradient**: iterative SPD solver; each iteration = one **CSR-SpMV** + two dot-products + three
  AXPYs. Deterministic block reductions keep the result reproducible.
- **Autoregulation ODE (surrogate)**: proportional radius feedback toward a metabolic target flow, clamped
  to a physiological band; because `G ∝ r⁴`, small radius changes have large flow effects.
- **Virtual FFR**: `(P_distal − P_venous) / (P_aortic − P_venous)` across the modeled stenosis; `< 0.80`
  flags a flow-limiting lesion.

Full derivations and complexity analysis are in **[THEORY.md](THEORY.md)**.

## Build

Requires **Visual Studio 2026** (v145 toolset, *Desktop development with C++*) and **CUDA Toolkit 13.3**
(see [`docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md)).

1. Open `build/coronary-autoregulation-microvascular-perfusion.sln` in Visual Studio 2026.
2. Select the **`Release`** configuration and **`x64`** platform.
3. **Build ▸ Build Solution** (`Ctrl+Shift+B`). The executable lands in
   `build/x64/Release/coronary-autoregulation-microvascular-perfusion.exe`.

Command line (Developer PowerShell), for the exact toolchain used here:

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" `
  build\coronary-autoregulation-microvascular-perfusion.sln /p:Configuration=Release /p:Platform=x64 /m
```

Linux/macOS learners can use the optional CMake build (`cmake -S . -B build/cmake && cmake --build build/cmake`).

## Run the demo

One command builds (if needed), runs on the committed sample, and checks the output:

```powershell
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1
```

(Bash: `./demo/run_demo.sh`.) See [`demo/README.md`](demo/README.md) for what each line means.

## Data

The committed sample `data/sample/coronary_network.txt` is a **tiny, fully synthetic** coronary network:
8 nodes, 8 segments, with a deliberately narrow **stenosis** on the branch to one territory and a low-flow
collateral cross-link. It is engineered so the result is interpretable (the stenosed branch's virtual FFR
comes out flow-limiting). Regenerate it with `python scripts/make_synthetic.py`. It is **not patient data**
and carries no clinical validity — see [`data/README.md`](data/README.md). Pointers to real datasets
(UK Biobank coronary CTA, PhysioNet pressure/flow, the Vascular Model Repository, MICCAI challenges) and how
to fetch them are printed by `scripts/download_data.ps1` / `.sh` (they never bypass credentialed access).

## Expected output

Running the demo prints a deterministic report to **stdout** and timing/diagnostics to **stderr**. The
headline lines (see `demo/expected_output.txt`) are the per-node pressures, the pre-autoregulation vs.
regulated inlet perfusion, and the virtual FFR:

```
nodal pressures (mmHg):
 P[0]=100.0000 P[1]=57.7066 P[2]=41.1765 P[3]=42.5168 P[4]=29.4118 P[5]=27.5056 P[6]=20.0000 P[7]=20.0000
inlet perfusion (pre-autoreg)  = 8.4317e+10 (um^3/s)
inlet perfusion (regulated)    = 613442857.9655 (um^3/s)
virtual FFR (Pd/Pa)    = 0.2815  [flow-limiting (<0.80)]
RESULT: PASS (GPU pressures match CPU within tol=1.0e-06 mmHg)
```

**How correctness is checked:** the program solves the network on **both** the CPU (a plain serial CG,
`reference_cpu.cpp`) and the **GPU** (CSR-SpMV CG, `kernels.cu`), then asserts the two pressure fields agree
within `1e-6 mmHg`. Because both call the identical per-vessel physics in `coronary.h`, they agree to
~`5e-14 mmHg` in practice; the tolerance covers the tiny FMA/summation-order divergence between the CPU's
edge loop and the GPU's CSR rows (see THEORY §Numerical considerations).

## Code tour

Read in this order:

1. **`src/main.cu`** — the 5-step flow: load → CPU solve → GPU solve → verify → deterministic report (FFR,
   perfusion). Start here.
2. **`src/coronary.h`** — the *one* shared `__host__ __device__` physics core: conductance (r⁴ law),
   Fåhræus–Lindqvist viscosity, and the autoregulation radius update. Used by both CPU and GPU so their math
   is identical.
3. **`src/reference_cpu.cpp`** — the serial ground truth: graph-Laplacian assembly, matrix-free SpMV, CG,
   and the autoregulation outer loop.
4. **`src/kernels.cu`** — the GPU solve: CSR assembly on the host, then the `csr_spmv`, dot-product /
   reduction, and AXPY kernels that make up Conjugate Gradient; the `solve_gpu` driver ties them together.
5. **`src/kernels.cuh`** — the host-facing declaration + the GPU idea in prose.

## Prior art & further reading

The catalog names these starter tools; study them, don't copy them:

- **[SimVascular / svFSI](https://github.com/SimVascular/svFSI)** — patient-specific cardiovascular flow;
  learn how coronary outlets use *structured-tree* boundary conditions (our lumped venous outlet is the toy
  version of that).
- **[HemeLB](https://github.com/hemelb-codes/hemelb)** — sparse lattice-Boltzmann for vascular beds; a
  different (mesoscopic) route to the same flow field, and a lesson in scaling sparse solvers on GPUs.
- **[APBS](https://github.com/Electrostatics/apbs)** — Poisson–Boltzmann electrostatics; the *same* sparse
  SPD linear-solve machinery (CG on a Laplacian) reused in a completely different domain — a good analogy
  for oxygen transport as a diffusion problem.
- **[OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-dev)** — full 3-D coronary CFD; shows what our 1-D
  network model abstracts away (and when you'd need the heavy machinery).

## Exercises

1. **Oxygen transport.** Add a per-segment convection–diffusion equation for O₂ (a second sparse solve or a
   marching update along each segment), and colour nodes by O₂ saturation. This is the catalog's Green's-
   function tissue-transport extension.
2. **Sweep the stenosis.** Vary the stenosis radius in `make_synthetic.py` from 4 → 20 µm and plot virtual
   FFR vs. radius. Where does it cross the 0.80 clinical cut-point?
3. **Jacobi preconditioner.** Add diagonal (Jacobi) preconditioning to the CG (`M⁻¹ = 1/diag(L)`) and count
   how many iterations you save on a larger network. See THEORY §GPU mapping.
4. **Scale it up.** Generate a large random binary tree (10⁴–10⁵ segments) and compare CPU vs. GPU wall time
   — this is where the GPU's SpMV parallelism finally pays off (the tiny sample is launch-bound).
5. **Swap in cuSPARSE.** Replace the hand-rolled `csr_spmv` with `cusparseSpMV` and confirm identical
   results; note the extra setup (descriptors, buffer sizing). THEORY §Where this sits shows the calls.

## Limitations & honesty

- The network is **synthetic and tiny** (8 nodes) — enough to teach the method and make the demo instant,
  but not a real coronary tree. The GPU is *slower* than the CPU here because it is launch-bound; its edge
  appears only at 10⁴–10⁶ segments (stated plainly in the stderr timing line).
- The autoregulation rule is a **deterministic proportional-feedback surrogate**, not the coupled
  metabolic (adenosine/O₂) + myogenic ODE system of the real coronary bed (described in THEORY §Where this
  sits). Viscosity uses a **simplified** Fåhræus–Lindqvist ramp, not the full Pries in-vivo fit.
- Units are a **scaled teaching set** chosen to keep pressures in the physiological 0–100 mmHg range; the
  absolute flow magnitudes are illustrative, not physiological.
- **No clinical claims.** The virtual FFR here is a didactic calculation on synthetic data and must never be
  used for diagnosis or treatment.
