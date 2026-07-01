# 6.2 — Whole-Heart Digital Twin

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.2`
>
> **Reduced-scope teaching version** (CLAUDE.md §13). The full twin is a 3-D
> finite-element PDE model on a patient mesh; here we ship a spatially-lumped
> (0-D) closed-loop heart that teaches every ingredient — EP, contraction,
> circulation, and the ensemble inference loop — on the GPU.
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **cardiac digital twin** is a personalized heart simulation, calibrated so its
outputs match a patient's clinical measurements, then used to answer "what if"
questions. This project builds a minimal but *complete* closed-loop twin in 0-D:
a **FitzHugh-Nagumo** excitable cell (electrophysiology) drives a **time-varying
elastance** ventricle (contraction) that pumps blood into a **3-element
Windkessel** arterial load (circulation). It runs an **ensemble** of virtual
hearts spanning a range of **contractility**, computes each one's pressure–volume
summary (stroke volume, ejection fraction, peak pressures), and performs the
**twin-fit** step: picking the contractility whose stroke volume best matches a
clinical target. Each virtual heart is an independent forward solve, so the whole
ensemble maps onto the GPU as **one thread per heart**.

## What this computes & why the GPU helps

Building a real twin means running **thousands to millions of forward heart
simulations** while adjusting parameters until the model matches data (the
"inference step"). This project isolates that loop: it forward-simulates an
ensemble of hearts and scans for the best-fitting contractility.

**The parallel bottleneck:** the *forward simulation*. Each heart requires
`beats · (bcl_ms/dt_ms)` sequential RK4 steps (48 000 here) — but the ensemble's
members are completely **independent**, so the total work `n · steps` parallelizes
perfectly across GPU threads while the per-thread critical path stays just
`steps`. This is the batched-ensemble pattern the catalog names ("batched forward
solves across ensemble members for parameter inference"), the same GPU idiom as
the SEIR (`9.02`) and PBPK (`13.02`) flagships.

## The algorithm in brief

- **FitzHugh-Nagumo EP** — a 2-variable excitable-cell ODE paced periodically.
- **Time-varying-elastance mechanics** — activation ramps ventricular stiffness,
  giving the end-systolic pressure–volume relation `P_lv = E(t)·(V − V0)`.
- **Diode valves** — mitral filling and aortic ejection flows gated by pressure.
- **3-element Windkessel circulation** — arterial pressure from a compliance +
  resistance reservoir charged by ejected flow.
- **RK4 integration** — 4th-order Runge-Kutta, shared host/device for exact parity.
- **Ensemble parameter scan** — 1-D grid search over contractility to fit stroke
  volume (a miniature stand-in for ensemble-Kalman / adjoint inference).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/whole-heart-digital-twin.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/whole-heart-digital-twin.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\whole-heart-digital-twin.sln /p:Configuration=Release /p:Platform=x64
```

Both `Debug|x64` and `Release|x64` build with zero warnings.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/heart_ensemble.txt`, prints the
per-member pressure–volume table and the twin-fit result, shows the GPU-vs-CPU
agreement check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/heart_ensemble.txt` — a one-line
  **synthetic** ensemble config (a contractility sweep + a target stroke volume)
  so the demo runs with zero downloads.
- **Regenerate/resize:** `python scripts/make_synthetic.py --n 256`.
- **Real-world sources:** `scripts/download_data.ps1` / `.sh` print links to the
  imaging datasets a full twin is built from (none are auto-fetched).
- **Provenance & license:** see [data/README.md](data/README.md).

Real-world datasets (study only): UK Biobank Cardiac MRI (https://www.ukbiobank.ac.uk);
Zenodo Synthetic Biventricular Heart Meshes (https://zenodo.org/records/4506930);
Visible Human Project (https://www.nlm.nih.gov/research/visible/visible_human.html);
ACDC MICCAI (https://www.creatis.insa-lyon.fr/Challenge/acdc/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a table
of 12 virtual hearts whose stroke volume and ejection fraction rise with
contractility, a `twin-fit` line selecting the best-matching member, and
`RESULT: PASS`. The program computes every result on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree within a `1e-9` tolerance — that agreement is the correctness
guarantee. Because both paths share the same double-precision physics
(`src/heart.h`), the observed worst difference is ~`5.7e-14`.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the ensemble, runs CPU + GPU, verifies,
   reports the table + twin-fit.
2. [`src/heart.h`](src/heart.h) — **the physics**: the shared `__host__ __device__`
   FHN + elastance + Windkessel model and the RK4 integrator (start here for the "why").
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp)
   — the ensemble config, the `idx → E_max` map, and the trusted serial loop.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-heart idea.
5. [`src/kernels.cu`](src/kernels.cu) — the ensemble kernel and host wrapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O.

## Prior art & further reading

- [openCARP](https://git.opencarp.org/openCARP/openCARP) — the EP engine used in
  many published twins; study its monodomain solver and ionic cell models.
- [simcardems](https://github.com/ComputationalPhysiology/simcardems) — FEniCS
  electromechanics coupling; how EP and finite-element mechanics join.
- [TorchCor](https://github.com/sagebei/torchcor) — PyTorch GPU cardiac EP FEM;
  the model for *differentiable* twin fitting (gradient-based inference).
- [Awesome-Cardiac-Digital-Twins](https://github.com/lileitech/Awesome-Cardiac-Digital-Twins)
  — curated index of datasets, methods, and papers.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the *concepts* didactically (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble ODE integration — one thread per trajectory** (PATTERNS.md §1). Each
thread runs the full multi-beat RK4 loop for one virtual heart entirely in
registers and writes one summary; there is **no shared memory, no atomics, no
sparse solver**. (The catalog lists cuSPARSE/cuSOLVER/cuBLAS because the *full*
FEM twin inverts a large sparse matrix each timestep — our 0-D model has no
spatial coupling, so a transparent hand-written kernel is both sufficient and more
instructive. See THEORY §4 and §7.)

## Exercises

1. **Sweep a second parameter.** Add a peripheral-resistance (`Rp`) axis to make a
   2-D ensemble (afterload × contractility) — extend `member_params` to a 2-D
   index map like flagship `9.02`'s `member_params`.
2. **Better inference.** Replace the grid search with a bisection or Newton step on
   `SV(E_max) − SV*` (the function is monotone), and report how many forward solves
   it takes versus the exhaustive scan.
3. **Report the P–V loop.** Have a thread also write the (V, P_lv) trace of its
   final beat so you can plot the pressure–volume loop and read stroke work as its
   enclosed area.
4. **Scale it.** Run `--n 100000` and watch the GPU-vs-CPU timing gap widen — the
   ensemble pattern's whole point (a *teaching* timing, not a benchmark).
5. **Add an atrium.** Make `P_venous` a second dynamic compartment to see how
   preload changes stroke volume (Frank-Starling).

## Limitations & honesty

- **0-D, not 3-D.** There is **no spatial PDE**: no wave propagation, no mesh, no
  fibers. This is a lumped model — the biggest simplification versus a real twin
  (THEORY §7 describes the full FEM version).
- **Phenomenological EP.** FitzHugh-Nagumo captures the *shape* of an action
  potential but is not a biophysical ionic model (ten Tusscher, O'Hara-Rudy). Its
  `v` variable is treated as a normalized activation, not a millivolt.
- **Synthetic parameters.** Every number is made up and labeled synthetic; nothing
  here derives from a real patient. Peak LV pressures at the high-contractility end
  of the sweep exceed physiological values — expected for a lumped model with fixed
  filling and no ventricular-arterial matching, and a good prompt for exercise 5.
- **Not for clinical use.** Outputs are a teaching artifact. No diagnostic or
  therapeutic claim is made or implied.
