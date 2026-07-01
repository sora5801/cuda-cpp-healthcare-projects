# 5.6 — GPU Boltzmann Transport (Deterministic Dose)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.6`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project solves the **linear Boltzmann transport equation (LBTE)**
*deterministically* — with no Monte-Carlo randomness and no statistical noise —
to compute how radiation fluence (and an absorbed-dose proxy) is distributed
through a layered slab of tissue. It uses the **discrete-ordinates (Sₙ)** method:
discretize the beam's directions into a handful of ordinates, sweep each ordinate
across the slab with an upwind **diamond-difference** update, and repeat via
**source iteration** until the scalar flux converges. To keep it readable, this
is a **reduced-scope teaching version** — 1-D, single-energy, isotropic
scattering — that still contains every concept a production engine like **Acuros
XB** uses. The GPU parallelizes across the independent ordinates; a plain CPU
reference computes the same thing so you can trust the GPU answer.

## What this computes & why the GPU helps

The linear Boltzmann transport equation (LBTE) describes radiation transport
deterministically: it tracks the fluence distribution of particles as a function
of position, direction, and energy without stochastic noise. On a clinical 6-DoF
phase-space grid `(x,y,z,θ,φ,E)` this is ~10⁹–10¹⁰ unknowns, so iterative solvers
(source iteration, diffusion synthetic acceleration) need a GPU to be tractable.
**Acuros XB** (Varian Eclipse) implements a GPU-accelerated LBTE solver that
beats superposition-convolution in heterogeneous tissue — lung and bone/tissue
interfaces where Monte Carlo is accurate but slow.

**The parallel bottleneck:** the inner **transport sweep** — evaluating the
angular flux for every direction over every spatial cell, once per source
iteration — is the dominant cost. The directions (ordinates) are mutually
independent within an iteration, so we map **one GPU thread per ordinate**; each
thread sweeps the whole slab for its direction. A separate, fixed-order reduction
turns the per-ordinate fluxes into the scalar flux (see [THEORY.md](THEORY.md)
§4).

## The algorithm in brief

- **Discrete ordinates (Sₙ)** — Gauss-Legendre quadrature `{μ_n, w_n}` replaces
  the angular integral.
- **Transport sweep** — per direction, integrate the 1-D transport ODE cell by
  cell (upwind), using the **diamond-difference** closure.
- **Source iteration (SI)** — lag the scattering source on the previous scalar
  flux; sweep all ordinates; re-form the scalar flux; repeat to convergence.
- **Absorbed-dose proxy** — `D ∝ Σ_a · φ` from the converged flux.
- *(Described but out of scope here: DSA acceleration, multi-group energy,
  Legendre-anisotropic scattering, coupled photon-electron transport, LD-FEM.)*

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-boltzmann-transport-deterministic-dose.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-boltzmann-transport-deterministic-dose.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-boltzmann-transport-deterministic-dose.sln /p:Configuration=Release /p:Platform=x64
```

Only the CUDA runtime (`cudart_static.lib`) is linked — no extra CUDA libraries.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/slab_problem.txt`, prints the
per-cell flux/dose profile, shows the GPU-vs-CPU agreement check, and prints a
timing line to stderr.

## Data

- **Sample (committed):** `data/sample/slab_problem.txt` — a tiny, **synthetic**
  "tissue / lung / tissue" slab with a source band, so the demo runs offline with
  zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print where to obtain the
  real references (they do not redistribute credentialed data).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: AAPM TG-105 lung benchmark; IROC heterogeneity phantom
datasets; IAEA photon cross-section library; Acuros XB validation datasets from
Varian white papers (publicly documented).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a
table of `scalar_flux` and `dose_proxy` per cell, ending in
`RESULT: PASS (GPU flux matches CPU within tol=1.0e-11)`. The program computes the
flux on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-11` (they agree to
~`5e-17` in practice). The flux peaks in the source band and the dose proxy
visibly drops in the low-density "lung" layer — the physics this method exists to
capture.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the slab, runs CPU + GPU, verifies, reports.
2. [`src/boltzmann_sn.h`](src/boltzmann_sn.h) — **the physics**: the shared
   `__host__ __device__` diamond-difference per-cell update and the single-ordinate
   sweep (used identically by CPU and GPU).
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`reference_cpu.cpp`](src/reference_cpu.cpp)
   — the `SlabProblem`, the Gauss-Legendre quadrature, and the serial source
   iteration baseline.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the ordinate-parallel idea.
5. [`src/kernels.cu`](src/kernels.cu) — the sweep kernel, the deterministic
   reduction kernel, and the host source-iteration driver.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **OpenMC** (`https://github.com/openmc-dev/openmc`) — primarily Monte Carlo;
  study it to contrast the stochastic and deterministic philosophies.
- **Denovo / Exnihilo** (ORNL, `https://github.com/ORNL-CEES/Exnihilo`) —
  production 3-D deterministic transport; learn its sweep + DSA architecture.
- **Attila** (commercial) — the deterministic engine lineage behind medical LBTE.
- **AHOTN / arbitrarily-high-order transport nodal** codes — higher-order spatial
  schemes than our diamond difference.
- **"GPU Sn transport CUDA"** literature — wavefront-parallel sweeps on the GPU.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Independent-jobs parallelism across **ordinates** (one thread sweeps one
direction), then a **fixed-order reduction** (one thread per cell) to form the
scalar flux — deterministic, no atomics. The angular-contribution tensor lives in
global memory; each sweep carries its edge flux in a register. The catalog's
cuSPARSE upwind sweep and shared-memory scattering source are the ≥2-D
optimizations, described in [THEORY.md](THEORY.md) §4/§7 but not needed at this
teaching scale.

## Exercises

1. **Raise the S_N order.** Regenerate the sample with `--nord 16` (or 32) and
   watch the flux profile converge; how much does the answer move from S₈ to S₁₆?
2. **Sweep the scattering ratio.** Edit a cell's `sigma_s` toward `sigma_t`
   (c → 1) and observe the source-iteration count climb — then read the DSA
   section (§7) to see how production codes fix it.
3. **Thicken the lung layer** (edit `make_synthetic.py`) and confirm the dose
   proxy dips further there while the flux stays smooth — the tissue/lung effect.
4. **Add a reflective boundary.** Change one face BC so outgoing flux re-enters
   (mirror), and check the flux no longer decays to ~0 there.
5. **Move the reduction on-device.** Replace the host convergence copy with an
   on-GPU L∞ reduction (Thrust/CUB) so the whole iteration stays on the device.

## Limitations & honesty

- **Reduced scope on purpose.** 1-D slab, one energy group, isotropic scattering,
  vacuum boundaries, diamond-difference spatial closure. Real dose engines add
  3-D geometry, multi-group energy, anisotropic (Legendre) scattering, coupled
  photon→electron transport, LD-FEM, and DSA acceleration (see THEORY §7).
- **Synthetic data.** The cross-sections in `data/sample/` are illustrative, not
  measured tissue values; nothing here is calibrated to a real beam or patient.
- **"Dose" is a proxy.** We report `Σ_a·φ` (energy-deposition-rate density), which
  is proportional to absorbed dose only for a fixed particle energy and unit
  density — enough to teach the shape, not a clinical dose.
- **Timing is a teaching artifact.** On a 24-cell/S₈ slab the GPU is launch-bound
  and slower than the CPU; the point is the pattern, and the GPU's edge grows with
  `ncell × nord` and with multi-D meshes. Never a benchmark claim.
- **Not for clinical use.** No output may inform diagnosis or treatment.
