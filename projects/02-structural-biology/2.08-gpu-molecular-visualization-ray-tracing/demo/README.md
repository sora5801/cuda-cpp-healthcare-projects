# Demo — 2.8 GPU Molecular Visualization & Ray Tracing

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Render** the committed synthetic molecule (`data/sample/molecule_sample.scene`)
   **twice** — once on the CPU reference, once on the GPU — by casting a ray
   through every pixel and shading the nearest atom with ambient occlusion + a
   directional light + a hard shadow.
3. **Verify** the GPU image against the CPU image and print a clear `PASS`/`FAIL`.
4. **Time** the render (CUDA events vs CPU wall clock) — a *teaching artifact*,
   not a benchmark claim.
5. **Save** the rendered image as `render.pgm` (open it in any image viewer to
   see the molecule).

The program splits its output deliberately (docs/PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic — image stats, the whole-frame
  checksum, an **ASCII-art preview** of the render, and the `PASS`/`FAIL` line —
  all computed from the **CPU** image (which is identical on every machine). It
  is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the GPU-vs-CPU agreement detail (which can vary
  by a pixel or two across GPUs), so it is shown but never diffed.

## What you are looking at

The ASCII preview is a 32×16 down-sampled thumbnail of the 200×200 render. The
synthetic molecule is a 5-turn helix; you can see the two bright lobes (the
oxygen "caps", rendered with `#`/`%`/`@`) and the darker grooves between turns
where **ambient occlusion** correctly darkens the crevices — that AO shading is
exactly what makes a molecular render read as 3-D.

## Expected result

```
2.8 -- GPU Molecular Visualization & Ray Tracing
orthographic VDW ray trace: 77 atoms -> 200x200 image, AO=32 samples/pixel
lit pixels = 16603 / 40000 (41.5% of frame is molecule)
image checksum (FNV-1a) = 9d18640c
ASCII preview (downsampled 200x200 -> 32x16):
  |                                |
  |            ..    ..            |
  |          .==+=..-=++-          |
  |      ... .-+++=:-===- .:.      |
  |     :==++===+++-=+++:-==+**=.  |
  |     :--=+++=---:----=++==#%%#  |
  |    ..::====-      :-===::=*+:  |
  |   :===++++-        :--=====-   |
  |   :----===-        :=+++===-   |
  |  .+#*=:=+++-      :++**=...    |
  |  =*##*:===-=++--+=-=+++===     |
  |   :===---:-+++-=++=::--==-     |
  |      ... .----::-===- ...      |
  |          .::::..:---:          |
  |            ..     .            |
  |                                |
RESULT: PASS (GPU render matches CPU reference within tolerance)
```

> The `[verify]` line on **stderr** shows the GPU's own checksum and the exact
> pixel agreement, e.g. `mismatched pixels=2/40000  max byte diff=2`. A couple of
> silhouette-edge pixels can differ by a few grey levels because the host and
> device math libraries round `cosf`/`sinf`/`sqrtf` differently in the last bit;
> this is expected and documented in [`../THEORY.md`](../THEORY.md) ("How we
> verify correctness"). The tolerance is `<=8` grey levels on `<=0.1%` of pixels.
