# Demo — 4.18 Image-Based 3D Printing / Model Generation for Surgery

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/volume_sample.txt` — a synthetic 17³
   sphere volume.
3. **Extract** the iso-surface mesh with Marching Cubes on **both** the GPU
   (`kernels.cu`, the count → scan → generate pipeline) and a serial **CPU
   reference** (`reference_cpu.cpp`), and **verify** they agree vertex-by-vertex —
   printing a clear `PASS`/`FAIL`.
4. **Cross-check the science**: the extracted surface area vs the analytic sphere
   area `4πr²` (on stderr).
5. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing, the source path, the science cross-check, and the
  numeric error (which vary run to run), so it is shown but never diffed.

## Expected result

```
4.18 -- Image-Based 3D Printing / Model Generation for Surgery
volume: 17 x 17 x 17 samples, spacing 1.000 mm, iso = 0.000
cells marched: 4096
triangles extracted: 1352
surface area: 448.4199 mm^2
bbox min: (-6.0000, -6.0000, -6.0000) mm
bbox max: (6.0000, 6.0000, 6.0000) mm
mesh checksum: 36600.281
RESULT: PASS (GPU mesh matches CPU within tol=1.0e-03 mm)
```

### How to read it

- **1352 triangles** trace the sphere's surface — this is the mesh an STL would
  store and a printer would build.
- **448.42 mm²** is the mesh area; the stderr `[science]` line compares it to
  `4πr² = 452.39 mm²` (r = 6 mm). The ~0.9 % shortfall is Marching Cubes'
  piecewise-flat under-estimate of a curved surface — expected, not a bug.
- **bbox `[−6, 6]³ mm`** confirms the radius-6 sphere is centred and correctly
  sized.
- **mesh checksum** is an order-independent fingerprint of every vertex; it changes
  if any vertex moves, making this output a real regression guard.
- **`RESULT: PASS`** means the GPU mesh and CPU mesh are identical within `1e-3 mm`
  (in fact exactly, `max_vertex_err = 0`). That agreement is the correctness proof.
