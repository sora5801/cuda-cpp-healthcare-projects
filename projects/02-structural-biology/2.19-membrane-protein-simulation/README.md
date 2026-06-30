# 2.19 — Membrane Protein Simulation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey) ![scope](https://img.shields.io/badge/scope-reduced--scope%20teaching%20version-orange)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.19`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project is a tiny **coarse-grained molecular-dynamics (MD) simulation of a
lipid bilayer with an embedded protein** — the first stage of how real membrane
proteins are simulated. Each lipid is reduced to three beads (one polar **head**
+ two oily **tails**) wired by springs; a short column of **protein** beads sits
in the membrane. The beads attract and repel via a Lennard-Jones potential whose
well depths encode the **hydrophobic effect** (tails love tails), so a flat
double layer holds itself together while we shake the system with a thermostat.
We integrate the equations of motion with **velocity-Verlet + a Langevin
thermostat**, watch the bilayer equilibrate, and — the teaching payoff — run the
exact same physics on a **CPU reference** and a **GPU kernel** and prove they
agree. This is a deliberately **reduced-scope** model (CLAUDE.md §13): it teaches
the GPU MD pattern without the all-atom force field, PME electrostatics, or
microsecond runtimes a production membrane study needs (those are described in
[THEORY.md](THEORY.md)).

## What this computes & why the GPU helps

Membrane proteins (GPCRs, ion channels, transporters, integrins) are embedded in
lipid bilayers and represent >50% of current drug targets. Explicit membrane MD
requires building asymmetric bilayers with physiological lipid compositions and
running microsecond simulations to sample conformational changes. CHARMM-GUI
automates system building; GPU GROMACS/NAMD runs production simulations. Key
challenges include equilibrating the membrane (~100 ns), maintaining bilayer
asymmetry, and capturing slow conformational transitions. GPU-accelerated
CG-MARTINI pre-equilibration (1–10 μs) followed by backmapping to all-atom
provides a common pipeline — and the **CG pre-equilibration step is exactly what
this project models in miniature**.

**The parallel bottleneck:** the cost of MD is the per-step **force evaluation** —
every bead feels every other bead within a cutoff, which is O(N²) pair work per
step (production codes cut it to ~O(N) with neighbour lists + PME). Crucially,
during one step the force on bead *i* is **independent** of the force on bead *j*,
so we give **each bead its own GPU thread**, which loops over the others and sums
its own force. Thousands of beads → thousands of threads. The integration
(Verlet half-kicks + Langevin) is a second independent per-bead pass. This is the
"independent per-item job" pattern (`docs/PATTERNS.md` §1), the same one the
`1.12` Tanimoto and `12.01` spectral-search flagships use.

## The algorithm in brief

- **Coarse-grained model:** 3-bead lipids (HEAD + 2×TAIL) + protein beads; bead
  types set the Lennard-Jones well depths (a 3×3 ε matrix).
- **Forces:** truncated **Lennard-Jones** non-bonded pairs (minimum-image in the
  periodic membrane plane) + **harmonic bonds** along each lipid/protein chain.
- **Integration:** **velocity-Verlet** (symplectic, energy-conserving) split into
  two half-kicks around a force recompute.
- **Thermostat:** **Langevin** dynamics (friction + a deterministic random kick
  tuned by the fluctuation-dissipation theorem) to hold temperature — the NVT
  ensemble of a membrane equilibration.
- **Observables:** **bilayer thickness** (head-to-head separation) and **total
  potential energy**, the numbers you watch to know a membrane is intact and
  equilibrated.

The catalog also lists the *production* algorithms this teaching version stands
in for — CHARMM36 lipid force field, POPE/POPC/cholesterol assembly, a
semi-isotropic barostat (NPT-xy), PME for the charged bilayer, CG→AA backmapping,
and k-means clustering of gate states — all discussed in
[THEORY.md](THEORY.md) "Where this sits in the real world".

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/membrane-protein-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/membrane-protein-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\membrane-protein-simulation.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
library — because the reduced-scope model hand-rolls its own forces and
integrator (that is the point: nothing is a black box).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/membrane_sample.txt`, prints the
before/after membrane state, shows the GPU-vs-CPU agreement check, and prints a
timing line.

## Data

- **Sample (committed):** `data/sample/membrane_sample.txt` — a tiny, **synthetic**
  parameter file so the demo runs offline with zero downloads. The bilayer
  geometry itself is *generated in code* (`build_system()`); the file holds only
  the run parameters (lipid/protein counts, box, LJ matrix, dt, thermostat,
  seed).
- **"Full" dataset:** `scripts/download_data.ps1` / `.sh` — this project needs no
  download; the script only points at the real membrane databases for further
  study (it never bypasses any registration).
- **Provenance & license:** see [data/README.md](data/README.md). Everything is
  synthetic and labeled as such.

Real-world resources (for the curious, not used here): MemProtMD
(<https://memprotmd.bioch.ox.ac.uk>), GPCRdb (<https://gpcrdb.org>), CGMD
Platform benchmark systems
(<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7765266/>), OPM
(<https://opm.phar.umich.edu>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program runs the **same** MD on the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree within a documented
`1e-4` tolerance on every bead's final position and velocity — that agreement is
the correctness guarantee. In practice the two paths agree to ~`1e-14`
(round-off), because they call the **same** double-precision physics
(`src/membrane.h`) in the **same** order; the `1e-4` tolerance is the honest
ceiling that also covers fused-multiply-add divergence on other GPUs (see
[THEORY.md](THEORY.md) "Numerical considerations"). The report also shows the
bilayer **thickness** (stays near its built value → membrane intact) and the
**potential energy** (drops as the membrane relaxes into its attractive wells).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — the 5-step shape: load → build → run CPU+GPU →
   verify → report.
2. [`src/membrane.h`](src/membrane.h) — **start here for the science**: the shared
   `__host__ __device__` physics (LJ + bond forces, Verlet, Langevin,
   deterministic RNG) used by *both* CPU and GPU.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-bead idea.
4. [`src/kernels.cu`](src/kernels.cu) — the three kernels (forces / kick-drift /
   kick) and the host time loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the deterministic
   bilayer builder, and the trusted serial MD loop.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, I/O helpers.

## Prior art & further reading

- **CHARMM-GUI Membrane Builder** (<https://charmm-gui.org>) — the standard tool
  that places lipids around a protein and assigns a force field. Study it to see
  what "building a system" really entails (asymmetry, solvation, ions).
- **GROMACS** (<https://github.com/gromacs/gromacs>) — production GPU membrane MD;
  read its neighbour-list and PME design to see how the O(N²) here becomes O(N).
- **HTMD** (<https://github.com/Acellera/htmd>) — a GPU-accelerated membrane
  protein pipeline (system setup → simulation → analysis) worth studying end to
  end.
- **packmol-memgen** (AMBER membrane builder) — an alternative system builder;
  compare its packing approach to CHARMM-GUI's.
- **MARTINI force field** (<http://cgmartini.nl>) — the coarse-grained model this
  project imitates; read it to understand bead types and the 4-to-1 mapping.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the ideas didactically and credits the sources
(CLAUDE.md §2).

## CUDA pattern used here

**One thread per bead, independent force gather + Verlet integration**
(`docs/PATTERNS.md` §1, exemplified by `1.12`). Each `compute_forces_kernel`
thread sums the truncated-LJ force on its bead from all others plus its bonded
springs — the O(N) inner loop that, across N threads, is the O(N²) all-pairs
evaluation. Two more per-bead kernels do the Verlet half-kicks with the Langevin
thermostat. No atomics and no races: every kernel reads the shared state and
writes only its own bead's slot. The per-pair/per-bead math lives in a single
`__host__ __device__` header (`docs/PATTERNS.md` §2) so the CPU reference and GPU
kernels are byte-for-byte identical and verification is near-exact. (The
catalog's production pattern — semi-isotropic barostat, cuFFT-based PME for the
charged bilayer, multi-GPU z-decomposition, heterogeneous neighbour lists — is
the *next* step up, described in `THEORY.md`.)

## Exercises

1. **Melt the membrane.** Raise `temp` (kT) in `data/sample/membrane_sample.txt`
   and watch the bilayer thickness collapse as the lipids disorder. At what kT
   does the membrane stop being a membrane?
2. **Kill the hydrophobic effect.** Set `eTT` (tail–tail well depth) equal to
   `eHT` and rerun. The driving force for the bilayer disappears — what happens
   to the energy and thickness?
3. **Bigger patch, real speed-up.** Regenerate with
   `python scripts/make_synthetic.py --n-lipids 200 --steps 200` and compare the
   CPU vs GPU timings (stderr). The GPU's edge grows with bead count because the
   O(N²) work finally outweighs launch overhead.
4. **Add a neighbour list.** The all-pairs loop is the obvious O(N²) bottleneck.
   Sketch (or implement) a cell list so each bead only checks nearby cells —
   exactly the optimization GROMACS makes. (THEORY §"The algorithm" gives the
   complexity.)
5. **A second observable.** Add an *area-per-lipid* metric (box area / lipids per
   leaflet) to the report, the other canonical membrane equilibration diagnostic.

## Limitations & honesty

This is a **reduced-scope teaching model**, not a validated membrane simulation,
and **must not** be used for any scientific or clinical purpose. Specifically:

- **Coarse-grained & generic:** beads are not real lipids; the 3-bead lipid and
  generic LJ ε matrix are pedagogical, not parameterized against experiment (a
  real run uses CHARMM36 or calibrated MARTINI beads).
- **No electrostatics / PME:** charged head groups and the long-range Coulomb
  forces (handled by Particle-Mesh Ewald in production) are omitted entirely.
- **No solvent, no pressure coupling:** there is no explicit water and no
  barostat; the box is fixed and z is a free slab. Real membrane MD runs NPT with
  a semi-isotropic barostat to set the correct area-per-lipid.
- **Tiny & short:** dozens of beads for a few hundred steps, in reduced units —
  versus 10⁵–10⁶ atoms for microseconds. The numbers are illustrative, not
  physical predictions.
- **O(N²) forces:** we loop all pairs (skipping past the cutoff) for clarity;
  production codes use neighbour/cell lists for O(N). See Exercise 4.
- **Synthetic data:** every input is synthetic and labeled so; nothing here is a
  real structure or patient-derived. No diagnostic or therapeutic claim is made.
