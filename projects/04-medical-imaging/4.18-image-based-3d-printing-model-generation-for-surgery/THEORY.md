# THEORY — 4.18 Image-Based 3D Printing / Model Generation for Surgery

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## The science

A surgeon planning a difficult operation — separating conjoined vasculature,
fitting a custom titanium cranial plate, rehearsing a tumor resection — benefits
enormously from holding a **physical replica of that specific patient's anatomy**.
Those replicas are 3-D printed directly from the patient's CT or MRI scan. The
clinical pipeline is:

```
CT/MRI volume ──► segment ──► extract surface ──► smooth/repair ──► STL ──► print
 (voxels)        (label the     (voxels → a        (clean mesh)     (triangle  (FDM/SLA
                 anatomy)        triangle mesh)                       list)      printer)
```

A medical CT is a 3-D grid of **Hounsfield-unit** intensities: air ≈ −1000,
soft tissue ≈ 0–80, cortical bone ≈ +1000. To print "the bone", you pick an
**iso-value** (threshold) somewhere in between and ask: *where is the surface at
which intensity equals that threshold?* That surface — the **isosurface** — is
the boundary of the printed object. Converting the scalar grid into a watertight
triangle mesh of the isosurface is the geometric heart of the whole pipeline, and
the algorithm that does it is **Marching Cubes** (Lorensen & Cline, 1987).

This project implements that extraction step. The other pipeline stages
(segmentation, smoothing, FEM for implant stress) are described under
"Where this sits in the real world"; here we focus on the one piece that is a
clean, classic, embarrassingly-parallel GPU lesson.

## The math

We are given a scalar field sampled on a regular grid,
`f : ℤ³ → ℝ`, with `f(i,j,k)` the intensity at grid point `(i,j,k)`, and a chosen
iso-value `c`. We want the surface `S = { x ∈ ℝ³ : f(x) = c }`, approximated as a
triangle mesh.

**Per-cube decomposition.** Partition the grid into cubic **cells**, each spanning
8 neighbouring samples. Define each corner as *inside* if `f ≥ c`, *outside*
otherwise. With 8 corners there are `2⁸ = 256` possible inside/outside patterns.
For a given pattern, the surface enters and leaves the cube through a fixed set of
the 12 **edges** — and the way those edge-crossings connect into triangles is the
same for every cube with that pattern. That combinatorial fact is what makes a
**lookup table** possible.

**Edge interpolation.** On a crossed edge from corner `a` (value `v_a`) to corner
`b` (value `v_b`), the surface (where `f = c`) sits, under a linear model of `f`
along the edge, at parameter

```
t = (c − v_a) / (v_b − v_a),     vertex  p = p_a + t · (p_b − p_a).
```

This is the only floating-point arithmetic in the whole algorithm, and running it
with **identical FP32 operations** on CPU and GPU is what makes their meshes match
exactly (see "How we verify").

**Surface area** of the resulting mesh is `Σ ½ |(b−a) × (c−a)|` over triangles.
For our synthetic sphere of radius `r`, the analytic area is `4πr²`; marching
cubes' piecewise-flat triangles slightly **under**-estimate a curved surface,
which the demo reports — a concrete numerical lesson, not a bug.

## The algorithm

```
for each cell (ci,cj,ck):
    gather the 8 corner values v[0..7] and world positions p[0..7]
    cube_index = Σ_c (v[c] ≥ c) << c             # 8-bit pattern, 0..255
    for each triangle in TRI_TABLE[cube_index]:  # 0..5 triangles, -1 terminated
        for each of its 3 edges e:
            vertex = interp_edge(e, p, v, c)     # linear, the formula above
        emit triangle
```

- **Complexity:** `O(N_cells)` work, with a tiny constant (≤ 5 triangles/cell).
  For an `nx·ny·nz` volume, `N_cells = (nx−1)(ny−1)(nz−1)`. Serial time is linear
  but the constant × hundreds of millions of cells is minutes on a CPU.
- **Independence:** a cell reads only its own 8 corners and writes only its own
  triangles. There is **no dependence between cells** — the property that makes
  the GPU mapping trivial.
- **The one complication — ragged output.** Cells emit *different* triangle
  counts, so we cannot pre-assign output slots by `cell_index × constant`. The
  serial CPU just `push_back`s into a growing vector; the parallel GPU needs a
  **prefix sum** to lay the output out (next section).

## GPU mapping

The kernel structure is the canonical **count → scan → scatter** stream-compaction
idiom (PATTERNS.md §1, "ragged output"):

```mermaid
flowchart LR
  V[volume in global mem] --> C[count_kernel<br/>1 thread / cell<br/>counts[cell] = #tris]
  C --> S[exclusive prefix sum<br/>counts -> offsets]
  S --> G[generate_kernel<br/>1 thread / cell<br/>write tris at offsets[cell]]
  G --> M[compacted mesh]
```

1. **`count_kernel`** — thread `cell = blockIdx.x·blockDim.x + threadIdx.x` owns one
   cube. It loads 8 corners (8 global reads), classifies, and writes one int,
   `counts[cell] ∈ {0..5}`. Each thread writes a private location → **no atomics,
   no races**. Block size 256 (`BLOCK`), a 1-D grid over the flattened cell list.

2. **Exclusive prefix sum** of `counts` → `offsets`, where
   `offsets[i] = counts[0]+…+counts[i-1]` is the index of cell `i`'s first
   triangle, and `offsets[last]+counts[last]` is the total mesh size. We
   **hand-roll** this (rather than call `thrust::exclusive_scan`) to keep it a
   no-black-box artifact, using the classic two-level scheme:
   - `block_scan_kernel`: each block scans its ≤256 elements in **shared memory**
     with a Hillis–Steele inclusive scan (`log₂256 = 8` doubling steps), converts
     to exclusive, and records its block total in `block_sums[blockIdx]`.
   - `scan_block_sums_kernel`: a single block exclusively scans `block_sums`.
   - `add_block_offsets_kernel`: adds each block's running offset back into its
     elements, stitching the per-block scans into one global scan.

   Because every add is on **integers** (associative), the scan equals the serial
   prefix sum bit-for-bit, on every run and machine (PATTERNS.md §3) — the source
   of the mesh's deterministic ordering. *Production* code would call
   `thrust::exclusive_scan` or `cub::DeviceScan`, which run this same algorithm
   (work-efficient Blelloch) tuned for all sizes.

3. **`generate_kernel`** — same launch shape. Thread `cell` re-classifies its cube
   (cheaper than stashing state between passes) and writes its triangles starting
   at `offsets[cell]`. Since offsets ascend in `cell`, the global write order
   equals the CPU's serial sweep order → meshes are directly comparable.

**Memory hierarchy used:** the volume and the mesh live in **global** memory
(bandwidth-bound streaming, the right place for big arrays). The lookup tables
(`TRI_TABLE`, `EDGE_VERTS`) are small `const` arrays the compiler places in
constant/global and the cache serves cheaply; every thread indexes them by its own
`cube_index`. The scan uses **shared** memory for the per-block cooperative
reduction. Per-cell corner values/positions live in **registers** (small fixed-size
local arrays).

**Thread-to-data mapping:** `thread (blockIdx.x, threadIdx.x) → cell index
i = blockIdx.x·blockDim.x + threadIdx.x`; `cell_to_ijk(i)` recovers `(ci,cj,ck)`
with `ci = i % CX`, `cj = (i/CX) % CY`, `ck = i/(CX·CY)` — the exact inverse of the
CPU's `cz`-outer / `cx`-inner loop nesting, so the two orderings coincide.

**Occupancy / bandwidth note:** the kernels are light on arithmetic and heavy on
the 8 corner loads, so they are **bandwidth-bound**; 256-thread blocks give good
occupancy on sm_75–sm_89. On the tiny committed sample the GPU is *launch-bound*
(five kernel launches dominate), so it is slower than the CPU — honest, and exactly
the teaching point in PATTERNS.md §7. The GPU's advantage grows with volume size.

## Numerical considerations

- **Precision:** vertices are FP32 (`float`) because that is what STL and printers
  use and what the GPU is fastest at. Area/checksum reductions accumulate in
  `double` so the headline numbers are stable regardless of triangle count.
- **CPU/GPU parity:** the per-element math (`classify_cube`, `interp_edge`) lives in
  one `__host__ __device__` header (`mc_core.h`), so both sides run the identical
  float ops. On this build the meshes are in fact **bit-identical**
  (`max_vertex_err = 0`); we still *verify* to a documented `1e-3 mm` tolerance to
  absorb any future FMA-contraction divergence (PATTERNS.md §4).
- **Boundary convention:** "inside" is `value ≥ iso` (not `>`), chosen so the
  `value == iso` corner case is classified identically on both back ends — no
  host/device disagreement at the threshold.
- **Degenerate edge:** if `v_a == v_b` the interpolation denominator is zero; both
  sides fall back to the edge midpoint (`t = 0.5`), again identically.
- **Determinism:** the scan reduces with **integer** adds (commutative and exact),
  so offsets — and therefore the mesh order and the printed checksum — are
  reproducible. Floating sums via `atomicAdd` would *not* be (PATTERNS.md §3); we
  avoid them entirely.

## How we verify correctness

Two independent checks:

1. **CPU vs GPU, vertex-by-vertex.** `marching_cubes_cpu` and `marching_cubes_gpu`
   produce meshes in the same order; `max_vertex_err` is the largest
   per-coordinate difference, and a triangle-count mismatch returns `+∞` so a
   topology bug can never masquerade as agreement. Tolerance `1e-3 mm` (observed:
   `0`). This proves the parallel pipeline reproduces the trusted serial one.
2. **Against an analytic ground truth.** The committed sample is a sphere implicit
   field, so the extracted surface area (`448.42 mm²`) is checked against
   `4πr² = 452.39 mm²`. The ~0.9 % shortfall is the expected piecewise-flat
   under-estimate of a curved surface — validating the *science*, not just that two
   implementations agree. The symmetric bounding box `[−6,6]³ mm` confirms the
   sphere is where it should be.

Edge cases exercised by the sample: empty cells (outside the sphere) emit 0
triangles; the ragged last block is guarded (`if (cell >= n_cells) return`).

## Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). A clinical
image-to-print pipeline adds, around this extraction core:

- **Segmentation.** Thresholding alone leaks (a bone iso also catches dense
  contrast in vessels). Production tools (**TotalSegmentator**, nnU-Net) run a 3-D
  CNN to *label* each voxel by organ before meshing — itself a heavy GPU workload.
- **Mesh post-processing.** Raw MC output is staircased and noisy; **Laplacian /
  Taubin smoothing** (a per-vertex neighbour average — a GPU stencil over the mesh
  graph) and **decimation** reduce triangle count and artefacts. Multi-material
  prints (bone + vessel + tumor) need per-label meshes and **Boolean operations**.
- **Repair & supports.** Watertightness, manifoldness, and FDM **support-structure**
  generation come before slicing.
- **FEM for implants.** Custom titanium plates / aortic stents are stress-tested
  with **finite-element** solvers (stiffness-matrix assembly via cuBLAS, sparse
  solve via cuSPARSE) — the catalog's other listed GPU uses.
- **Library MC.** Real systems call **VTK**'s `vtkFlyingEdges3D` / `vtkMarchingCubes`
  or NVIDIA **CUB**-accelerated MC, and **OpenVDB** for sparse volumes. The
  algorithm here is the same; production versions add *Marching Cubes 33* (resolves
  the ambiguous-saddle cases the classic table can mis-stitch) or **dual contouring**
  (sharper features), and a fully recursive multi-block scan for arbitrarily large
  volumes — our single-level scan caps at `n_blocks ≤ 256` (≤ 65 536 cells) and
  fails loudly above that.

**A note on the toolchain (Exercise 4):** swapping the hand-rolled scan for
`thrust::exclusive_scan` requires passing `/Zc:preprocessor` to the CUDA host
compiler on VS 2026 + CUDA 13.3, because the Thrust/CCCL headers reject MSVC's
traditional preprocessor. We hand-roll partly to dodge that toolchain friction and
partly because the scan is the single most transferable primitive a learner takes
from this project.
