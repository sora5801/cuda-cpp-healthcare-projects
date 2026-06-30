# Demo — 4.31 Virtual Colonoscopy & CT Colonography

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/colon_volume_sample.txt` (a tiny
   synthetic CT volume of an air-filled colon with one planted polyp).
3. **Render** one virtual-colonoscopy fly-through frame on the **GPU** (volume
   ray-casting) and again on the **CPU reference**, and **verify** they agree.
4. **Show** a small ASCII preview of the rendered frame and a `PASS`/`FAIL`.
5. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (integer pixel counts, rounded floats,
  the ASCII preview) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric max-error (which vary run to run),
  so it is shown but never diffed.

## Reading the output

- `wall-hit pixels` — how many rays hit the colon wall (almost all; only a few
  escape straight down the open lumen).
- `polyp-region mean brightness ≈ 0.7331` — the **known answer**. The planted polyp
  bulges into the lumen; its convex surface catches the headlamp and reads ~2×
  brighter than the smooth wall there (≈ 0.36 without the polyp). The renderer
  recovering this brightening is the science check.
- The **ASCII preview** is a downsampled view of the frame: bright `#`/`%`/`@` are
  lit wall, spaces/`.` are the dark deep lumen / background. You should see a round
  lumen with the polyp as a bright `#@@%` cluster in the upper-center.

## Expected result

```
4.31 -- Virtual Colonoscopy & CT Colonography
CT colonography fly-through: volume 32x32x48 -> frame 48x48 (iso=0.50)
wall-hit pixels = 2296 / 2304 (0.997)
mean intensity = 0.6041
max intensity = 1.0000 at (px,py)=(27,8)
polyp-region mean brightness = 0.7331
ascii preview (24x12, '@'=bright wall, ' '=dark lumen/background):
  |%######*****************|
  |#####******++++++++++***|
  |####****+++++#@@%*++++++|
  |###****++++#*###*+++++++|
  |##****++==-+#@@%#+==++++|
  |##***+++=-:-+***+===++++|
  |##***+++=-:: :-=++=+++++|
  |##****++==-:--======++++|
  |###****+++========++++++|
  |####****++++++++++++++++|
  |#####*****++++++++++++**|
  |%######*****************|
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

The exact `stdout` above is what `expected_output.txt` contains; the demo passes
when it matches and the GPU image equals the CPU reference within `1e-3`.
