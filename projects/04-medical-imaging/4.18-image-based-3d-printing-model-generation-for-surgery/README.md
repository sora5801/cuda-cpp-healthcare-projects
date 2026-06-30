# 4.18 — Image-Based 3D Printing / Model Generation for Surgery

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.18`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Surgeons rehearse complex operations on a **physical 3-D print of the patient's
own anatomy** — a skull, a vertebra, an aortic root — printed from that patient's
CT or MRI scan. The pipeline that turns voxels into a printable model has one
geometric heart: **isosurface extraction**, i.e. converting a 3-D scalar volume
(intensities) into a **triangle mesh** that traces a chosen threshold (e.g. the
bone density). An STL file — what every 3-D printer eats — *is* exactly such a
triangle list. This project implements that step, **Marching Cubes**, on the GPU
and checks it against a serial CPU reference. The committed sample is a synthetic
sphere volume, so the extracted mesh has a known analytic surface area to
cross-check against.

## What this computes & why the GPU helps

Patient-specific anatomical models for surgical rehearsal require segmenting CT/MRI volumes (GPU CNN inference), smoothing and decimating meshes (GPU geometry processing), and generating printable STL/OBJ files. For a full torso CT at 0.5 mm isotropic resolution the input volume is ~10⁹ voxels; running marching cubes on GPU (NVIDIA CUB-accelerated or CUDA-native) reduces the surface extraction step from minutes to seconds. Multi-material prints (bone, soft tissue, vessels) require multi-label segmentation and per-label mesh Boolean operations — all benefiting from GPU parallelism. Finite-element simulation for patient-specific implant design (titanium plates, aortic stents) additionally uses GPU FEM solvers.

**The parallel bottleneck:** the volume is chopped into `(nx-1)·(ny-1)·(nz-1)`
little cubes ("cells"). Each cell's triangles depend **only on its own 8 corner
values**, so every cell is an independent job — one GPU thread per cell. A
clinical 512³ CT has ~1.3×10⁸ cells; marching them serially takes minutes, in
parallel it takes milliseconds. The one twist is that each cell emits a
**different number of triangles (0–5)**, so the output is ragged; we solve that
with the classic **count → prefix-sum → write** stream-compaction idiom.

## The algorithm in brief

- **Classify** each cube: build an 8-bit index from "which of the 8 corners are
  inside the surface (value ≥ iso)".
- **Look up** that index in the Lorensen–Cline `TRI_TABLE` to get the list of
  triangles (as triples of cube *edges*).
- **Interpolate** each triangle vertex along its edge: `t = (iso−v_a)/(v_b−v_a)`,
  vertex `= p_a + t·(p_b−p_a)` — linear, and run with identical FP32 ops on CPU
  and GPU so the meshes match.
- **Compact** the ragged per-cell output with an exclusive prefix sum (here a
  hand-rolled, deterministic two-level scan) so each cell knows where to write.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/image-based-3d-printing-model-generation-for-surgery.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/image-based-3d-printing-model-generation-for-surgery.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\image-based-3d-printing-model-generation-for-surgery.sln /p:Configuration=Release /p:Platform=x64
```

No extra CUDA libraries are linked — the scan is hand-rolled, so only the CUDA
runtime (`cudart`) is needed.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/volume_sample.txt` — a tiny, **synthetic**
  17³ scalar volume whose `value = radius − distance` field has a known sphere
  isosurface. Runs offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions for the
  real clinical collections (they require registration; the scripts never bypass
  credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: TCIA body CT collections; OsteoArthritis Initiative (OAI) for knee models (https://nda.nih.gov/oai/); VerSe vertebral segmentation dataset (https://github.com/anjany/verse); TotalSegmentator dataset (https://zenodo.org/record/6802614).

## Expected output

Success looks like `demo/expected_output.txt`: a 17³ sphere volume yields **1352
triangles**, a surface area of **448.42 mm²** (vs the analytic sphere
`4πr² = 452.39 mm²` — marching cubes slightly under-estimates curved area, a real
and expected artifact), a symmetric bounding box `[−6, 6]³ mm`, a stable mesh
checksum, and `RESULT: PASS`. The program computes the mesh on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they agree vertex-by-vertex within `1e-3 mm` — that agreement is the correctness
guarantee. On this build the agreement is in fact **exact** (`max_vertex_err = 0`).

## Code tour

Read in this order:

1. [`src/mc_core.h`](src/mc_core.h) — **start here.** The shared `__host__ __device__`
   marching-cubes core: the lookup tables, cube classification, and edge
   interpolation. Both the CPU and GPU call this, which is why their meshes match.
2. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the trusted serial baseline + the volume loader + the mesh metrics.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface and the count → scan →
   generate strategy.
4. [`src/kernels.cu`](src/kernels.cu) — the kernels, the hand-rolled exclusive
   prefix sum, and the host wrapper.
5. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

3D Slicer (https://github.com/Slicer/Slicer) — GPU-accelerated volume rendering, segmentation, STL export via SlicerRT; VTK (https://vtk.org/) — GPU-accelerated marching cubes and mesh operations; TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — fast GPU segmentation for print-ready model prep; OpenVDB (https://www.openvdb.org/) — GPU sparse volume processing for complex anatomies.

The original algorithm is Lorensen & Cline, *"Marching Cubes: A High Resolution
3D Surface Construction Algorithm"*, SIGGRAPH 1987 — the `TRI_TABLE`/edge tables
in `mc_core.h` are its connectivity lookup. Study these to learn the production
approach; **do not copy code wholesale** — reimplement didactically and credit the
source (CLAUDE.md §2).

## CUDA pattern used here

Catalog: *"CUDA marching cubes (thrust scan for compact output); cuBLAS for FEM
stiffness assembly; GPU ray-casting; custom CUDA for Laplacian smoothing;
cuSPARSE for FEM linear system."*

This teaching version implements the **marching-cubes + compaction** core — the
isosurface extraction that produces the printable mesh. The compaction is the
`count → exclusive-prefix-sum → scatter` idiom (PATTERNS.md §1, "ragged output"),
here with a **hand-rolled deterministic scan** so it is a no-black-box teaching
artifact; THEORY.md notes the production `thrust::exclusive_scan` / `cub::DeviceScan`
equivalent and the FEM/smoothing extensions, which are out of scope for a Beginner
teaching project.

## Exercises

1. **STL writer.** Add a function that writes `mesh_gpu` to an ASCII `.stl`
   (per-triangle facet normal + 3 vertices) and open it in a slicer / 3-D viewer.
2. **Change the iso-value.** Re-run with `iso = 2.0` (a smaller sphere). How do
   the triangle count and surface area change? Does the analytic check still hold?
3. **A different shape.** Edit `make_synthetic.py` to write a torus implicit field
   `value = r_minor − |(sqrt(x²+y²) − R_major), z|`. Does the genus-1 surface
   extract cleanly?
4. **Use the library.** Replace the hand-rolled scan in `kernels.cu` with
   `thrust::exclusive_scan` and confirm the mesh is byte-identical (you will need
   `/Zc:preprocessor` on the CUDA host compiler — see THEORY.md §"Where this sits").
5. **Bigger volume.** Bump `--n` to 65 and observe where the GPU starts winning on
   time. At what size does the single-level scan's `n_blocks ≤ 256` guard trip,
   and how would you make the scan recursive (THEORY.md §"GPU mapping")?

## Limitations & honesty

- **Reduced-scope teaching version.** This implements isosurface extraction only —
  the geometric heart of the print pipeline. The catalog's CNN **segmentation**,
  **mesh smoothing/decimation**, multi-material **Boolean ops**, and **FEM** for
  implant design are described in THEORY.md but not implemented; a real model goes
  segment → mesh → smooth → repair → support-generate before printing.
- **Synthetic data.** The sample is a mathematically-generated sphere, clearly
  labeled synthetic — **not** patient data, with no clinical meaning.
- **Classic Marching Cubes has ambiguous cases.** This uses the original
  Lorensen–Cline table, which can produce topologically inconsistent surfaces on
  certain saddle configurations; production tools use *Marching Cubes 33* or
  *dual contouring*. For a smooth blob like our sphere it is exact enough.
- **The single-level scan** assumes `n_blocks ≤ 256` (≤ 65 536 cells). It fails
  loudly above that; the recursive generalization is described, not coded.
- **Not for clinical use.** Study material only (CLAUDE.md §8).
