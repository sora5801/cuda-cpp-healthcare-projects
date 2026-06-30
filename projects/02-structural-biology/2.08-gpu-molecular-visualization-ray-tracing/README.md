# 2.8 — GPU Molecular Visualization & Ray Tracing

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.8`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). The molecule
> rendered here is **synthetic**, not a real structure._

## Summary

This project is a tiny **GPU ray tracer for molecules**. It reads a list of atoms
(each a sphere: 3-D centre, van-der-Waals radius, colour) and renders a greyscale
image of them in the **space-filling / VDW** style used by VMD and PyMOL. Every
output pixel shoots a ray, finds the nearest atom it hits, and shades that surface
point with **ambient occlusion** (soft "darken the crevices" shading that makes
3-D shape readable), a directional light, and a hard shadow. Rendering is
embarrassingly parallel — each pixel is independent — so we give **one GPU thread
per pixel**. It is the conceptual core of how real molecular viewers ray-trace
millions of atoms in real time, stripped down so you can read every line.

## What this computes & why the GPU helps

Interactive visualization of molecular structures means rendering many atoms as
shaded spheres (VDW surfaces) with ambient occlusion and shadows, ideally at
interactive frame rates. Production tools (VMD with CUDA/OptiX) ray-trace
multi-million-atom systems at >30 fps; the per-ray math they run — ray/sphere
intersection, occlusion sampling, shading — is exactly what we implement here on
a small synthetic molecule.

**The parallel bottleneck:** the renderer casts, per pixel, one primary ray + `N`
ambient-occlusion rays + one shadow ray, and (brute force) tests each against
every atom. For our demo that is ~100 million ray/sphere tests for a single
200×200 frame. Those pixels are **completely independent** — no pixel reads
another's result — which is the textbook **per-output-pixel gather** GPU pattern
(docs/PATTERNS.md §1, same shape as flagship 4.01 CT backprojection). A 2-D grid
of threads renders the whole frame at once.

## The algorithm in brief

- **Primary ray per pixel** (orthographic camera looking down −z).
- **Ray/sphere intersection** — solve the quadratic `|o+t·d−c|²=r²` for the
  nearest hit; the nearest atom over all spheres wins.
- **Surface normal** `n = (p − c)/|p − c|` at the hit point.
- **Ambient occlusion** — cast `N` rays into the hemisphere about `n` using a
  **deterministic** Hammersley sample set; AO = fraction that escape un-blocked.
- **Direct light + hard shadow** — Lambert `max(0, n·ℓ)`, zeroed if a shadow ray
  toward the light is blocked.
- **Quantize** the luminance to a 0–255 byte; write the pixel.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation (with the constant-memory and determinism reasoning).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-molecular-visualization-ray-tracing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-molecular-visualization-ray-tracing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-molecular-visualization-ray-tracing.sln /p:Configuration=Release /p:Platform=x64
```

> The project compiles device code with `--fmad=false` (`<FMAD>false</FMAD>` in
> the `.vcxproj`) to minimize CPU-vs-GPU divergence — see THEORY §5–6.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, renders the committed sample on **both** the CPU and
GPU, verifies they agree within tolerance, prints an **ASCII-art preview** of the
molecule plus a whole-frame checksum, and shows a timing line. It also writes
`render.pgm` — open it in any image viewer to see the actual picture.

## Data

- **Sample (committed):** `data/sample/molecule_sample.scene` — a tiny **synthetic**
  5-turn helix (77 atoms) so the demo runs offline with zero downloads.
- **Regenerate / resize:** `python scripts/make_synthetic.py --turns 6 --width 320 --height 320`.
- **Real structures:** `scripts/download_data.ps1` / `.sh` print pointers (PDB,
  EMDB, GPCRmd, CHARMM-GUI); they download nothing automatically by design.
- **Provenance, format & license:** see [data/README.md](data/README.md).

The scene format is one header line (`n_atoms width height ao_samples`) then one
`x y z radius color` line per atom (Å). Converting a real PDB into this format is
an Exercise below.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt) (the
title line, image stats, an FNV-1a checksum, the ASCII preview, and
`RESULT: PASS`). The program renders the image on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) — both
calling the *same* `shade_pixel()` — and asserts the two byte images agree within
a documented tolerance (`≤ 8` grey levels on `≤ 0.1 %` of pixels; in practice 2
pixels of 40 000 on the reference GPU). That agreement is the correctness
guarantee; THEORY §6 explains why a handful of silhouette-edge pixels differ.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the scene, renders CPU + GPU, verifies, reports.
2. [`src/render_core.h`](src/render_core.h) — **the shared per-pixel physics** (ray/sphere,
   AO, shading), the `__host__ __device__` core both back-ends call.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the per-pixel thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel + host wrapper (scene in constant memory).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — scene loader + the trusted serial render.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **VMD** (<https://www.ks.uiuc.edu/Research/vmd/>) — the reference GPU molecular
  ray tracer (CUDA/OptiX, Tachyon); learn how AO + a hardware BVH scale to
  millions of atoms.
- **PyMOL** (<https://github.com/schrodinger/pymol-open-source>) — GPU molecular
  graphics; learn the representations (sticks, cartoons, surfaces).
- **OVITO** (<https://www.ovito.org>) — GPU scientific visualization for MD; clean
  sphere/cylinder rendering at scale.
- **Mol\*** (<https://github.com/molstar/molstar>) — WebGL viewer; the same ideas
  via a rasterizer + screen-space AO in the browser.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-output-pixel gather** — a 2-D thread grid over the image, one thread per
pixel, each independently gathering over the (constant-memory) scene. No shared
memory, no atomics. The scene sits in **constant memory** for cheap broadcast
reads; the output image is written once, coalesced, to global memory. (Production
tools replace our brute-force trace with an OptiX hardware BVH — see THEORY §7.)

## Exercises

1. **Anti-aliasing.** Cast `k×k` jittered primary rays per pixel and average them.
   Watch the silhouette-edge CPU/GPU mismatch (THEORY §6) shrink toward zero — and
   the image get visibly smoother.
2. **RGB colour.** Extend `cpk_luma` to a `(r,g,b)` palette and render three
   channels; write a PPM (`P6`) instead of PGM. The per-channel math is identical.
3. **Real molecules.** Write a small PDB → `.scene` converter (parse `ATOM`/`HETATM`
   records, map element → VDW radius + CPK colour id) and render `1UBQ` from RCSB.
4. **A BVH.** Replace the brute-force `trace_nearest`/`occluded` with a uniform
   grid or BVH so each ray touches `O(log n)` atoms; render a 100k-atom scene and
   measure the speed-up.
5. **Perspective camera + cylinders.** Add a perspective ray generator and draw
   bonds as cylinders (ball-and-stick), then add a second light.

## Limitations & honesty

- **Synthetic data.** The committed molecule is a mathematically generated helix,
  **not** a real protein. It is labelled synthetic everywhere; nothing here is a
  patient- or specimen-derived structure, and no image implies any biological or
  clinical conclusion.
- **Reduced scope on purpose.** One sample per pixel (so silhouettes alias), one
  light, Lambert + AO shading, an orthographic camera, VDW spheres only, and a
  **brute-force** `O(n)`-per-ray trace. Production renderers add super-sampling, a
  perspective camera, a hardware BVH, multiple representations (surfaces, ribbons),
  and volume rendering for cryo-EM maps — see THEORY §7.
- **Not bit-exact across math libraries.** The GPU and CPU images differ by a few
  silhouette-edge pixels because the host and device `cosf`/`sinf`/`sqrtf` round
  differently; we verify to an honest, physically-negligible tolerance rather than
  claim bit-exactness (THEORY §6, docs/PATTERNS.md §4).
- **Timing is a teaching artifact, never a benchmark claim** (CLAUDE.md §12).
