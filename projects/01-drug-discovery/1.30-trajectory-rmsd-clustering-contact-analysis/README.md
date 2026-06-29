# 1.30 — Trajectory RMSD, Clustering & Contact Analysis

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.30`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A molecular-dynamics (MD) **trajectory** is a movie of a molecule: a sequence of
*frames*, each a 3-D snapshot of every atom. After the simulation we ask, frame
by frame, **"how far has the structure moved?"** This project computes two
classic per-frame metrics on the GPU — the optimal-superposition **RMSD** to a
reference structure (via the **Kabsch / QCP** algorithm) and the **fraction of
native contacts Q** — then groups frames into conformational states with a simple
**RMSD clustering** histogram. Every frame is independent of every other, so the
work maps onto **one GPU thread per frame**: the same "embarrassingly parallel"
shape as a fingerprint search, but each job is a small dense linear-algebra
computation instead of a popcount.

## What this computes & why the GPU helps

Post-MD analysis of multi-microsecond trajectories generates terabytes of
coordinate data. The headline operation, **RMSD after optimal superposition**,
must align every frame to a reference before measuring deviation — removing rigid
translation and rotation so only *real* conformational change is counted. The
classical Kabsch route forms a 3×3 covariance per frame and takes its SVD; we use
the equivalent **QCP** (Quaternion Characteristic Polynomial) closed form, which
replaces the SVD with the **largest eigenvalue of a 4×4 matrix** — cheaper, and
trivial to make deterministic and identical on CPU and GPU. **Contact analysis**
(which atom pairs are close) and the **native-contact fraction Q** are an O(N²)
per-frame sweep, also independent across frames.

**The parallel bottleneck** is the **per-frame RMSD + contact computation**: with
F frames it is `O(F · (N + N²))` and F can reach millions. Every frame is data-
independent, so we parallelize across the frame dimension F (one thread per
frame), keep the shared reference structure in **constant memory**, and compute
in **double precision** so the GPU and the CPU reference agree to ~machine
epsilon.

## The algorithm in brief

- **Kabsch / QCP RMSD:** center both structures, build the 3×3 cross-covariance
  `M`, form the 4×4 key matrix `K(M)`, and find its largest eigenvalue
  `λ_max`; then `RMSD = sqrt((G − 2·λ_max)/N)`. No SVD required.
- **Native contacts & Q:** a contact is an atom pair within a cutoff (skipping
  near-neighbours along the chain). `Q(frame)` is the fraction of the
  *reference's* contacts still present in that frame — a 1→0 folding coordinate.
- **RMSD clustering:** bin frames into RMSD shells (a deterministic, GROMOS-style
  fixed-radius idea) to reveal metastable conformational states.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/trajectory-rmsd-clustering-contact-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/trajectory-rmsd-clustering-contact-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\trajectory-rmsd-clustering-contact-analysis.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the per-frame RMSD/Q
table and the RMSD-cluster histogram, shows the **GPU-vs-CPU agreement** check,
and prints a timing line.

## Data

- **Sample (committed):** `data/sample/trajectory_sample.txt` — a tiny
  **synthetic** 12-frame × 16-atom trajectory. Frame 0 is a compact helix (the
  reference); later frames progressively unfold through three metastable states,
  so the RMSD grows and Q decays in an interpretable, known pattern.
- **Full dataset:** real trajectories from **MDCATH / GPCRmd / MDDB / the PDB** —
  see `scripts/download_data.ps1` / `.sh` and [data/README.md](data/README.md).
- For a larger synthetic trajectory: `python scripts/make_synthetic.py --frames 100000`.

## Expected output

`demo/expected_output.txt` holds the deterministic stdout. The program computes
all per-frame metrics on the **GPU** (`src/kernels.cu`) and on a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-9`. Because both sides
call the **same `__host__ __device__` math** in `src/rmsd_core.h` in double
precision, the residual is only the FP-ordering noise of the covariance sum
(`max_abs_err ≈ 5e-14`, far inside the tolerance). You should see RMSD climb from
`0.0000` (frame 0) to `~10.76` (frame 11), `Q` fall `1.0 → 0.52 → 0.0`, and the
cluster histogram cleanly separate three populated RMSD shells (the three states).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the trajectory, runs CPU + GPU, verifies, prints the table + clusters.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model (`Trajectory`, `FrameMetrics`) + loader/reference prototypes.
3. [`src/rmsd_core.h`](src/rmsd_core.h) — **the one true per-frame math** (`kabsch_rmsd`, contacts), shared `__host__ __device__` by CPU and GPU.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-frame / constant-memory idea.
5. [`src/kernels.cu`](src/kernels.cu) — the kernel (grid-stride, constant-memory reference) + host wrapper.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + the clustering reduction.

## Prior art & further reading

- **MDTraj** (<https://github.com/mdtraj/mdtraj>) — GPU-accelerated RMSD and trajectory analysis; its RMSD uses exactly the QCP method implemented here.
- **RAPIDS cuML** (<https://github.com/rapidsai/cuml>) — GPU clustering (k-means/DBSCAN) for Markov-state-model construction from trajectory features.
- **MDAnalysis** (<https://github.com/MDAnalysis/mdanalysis>) — general trajectory analysis (RMSD, contacts, RDF) with some GPU support.
- **HTMD** (<https://github.com/Acellera/htmd>) — adaptive MD with GPU-accelerated analysis pipelines.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

One **thread per frame** (independent jobs, PATTERNS.md §1) · shared reference in
**constant memory** (broadcast) · grid-stride loop over frames · a shared
`__host__ __device__` math core so CPU and GPU run **byte-identical** arithmetic
(PATTERNS.md §2) · **double precision** with ~machine-epsilon verification
(PATTERNS.md §4) · deterministic stdout / timing on stderr (PATTERNS.md §3).

## Exercises

1. **Pairwise RMSD matrix.** Compute the full `F×F` RMSD matrix (every frame vs.
   every other) — an `O(F²)` job that maps one thread to one `(i,j)` pair. This is
   what real GROMOS/DBSCAN clustering consumes; the catalog calls it out as the
   true bottleneck. Where does it stop fitting in memory?
2. **Real GROMOS clustering.** Replace the 1-D RMSD-shell histogram with the
   GROMOS fixed-radius rule on that pairwise matrix (repeatedly pick the frame
   with the most neighbours within `r`, remove its cluster, repeat).
3. **Contact maps.** Emit the full per-frame `N×N` contact matrix and average it
   over the trajectory to get a contact *probability* map (a standard figure).
4. **One warp per frame.** When `N` grows, assign a *warp* (not a thread) per
   frame and reduce the covariance with `__shfl_down_sync`. When does that win?
5. **FP32 vs FP64.** Switch the coordinates and `rmsd_core.h` math to `float` and
   watch the CPU/GPU `max_abs_err` grow — a concrete lesson in why RMSD libraries
   accumulate the covariance in double precision.

## Limitations & honesty

- The sample is **synthetic**; the helix-unfolding trajectory carries **no
  physical or biological meaning** beyond the geometry we impose to make RMSD and
  Q interpretable. It is labeled synthetic everywhere.
- This is a **reduced-scope teaching version**: a *fixed* atom count (`N_ATOMS=16`,
  a compile-time constant) and a *1-D RMSD-shell* clustering stand-in for true
  pairwise-RMSD GROMOS/DBSCAN clustering (Exercise 1–2). Real tools also do
  H-bond, RDF and S² order parameters (named in the catalog) — those are deferred
  to THEORY's "real world" section.
- We load the whole trajectory into device memory; production tools stream frames
  from disk, where the real bottleneck is **trajectory-file I/O bandwidth**, not
  the arithmetic.
- Timing is a **teaching artifact, not a benchmark** — at this tiny size launch /
  copy overhead dominates; the GPU's advantage appears at millions of frames.
