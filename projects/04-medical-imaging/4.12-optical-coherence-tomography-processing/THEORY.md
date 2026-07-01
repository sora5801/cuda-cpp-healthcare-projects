# THEORY — 4.12 Optical Coherence Tomography Processing

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Optical Coherence Tomography (OCT)** is to light what B-mode ultrasound is to
sound: it produces cross-sectional images of tissue micro-structure, but at
**micron** axial resolution (roughly 100× finer than clinical ultrasound). It is
the workhorse of ophthalmology — every retina clinic uses it to image the ~10
retinal layers, detect fluid (edema), and stage macular degeneration and diabetic
retinopathy — and it is increasingly used in cardiology (intravascular OCT) and
dermatology.

OCT works by **low-coherence interferometry**. Broadband (many-wavelength) light
is split into a *sample arm* (into the tissue) and a *reference arm* (to a mirror).
Light back-scattered from a reflector at depth `z` in the tissue travels an extra
`2z` and, when recombined with the reference light, interferes. Because the source
is broadband, this interference encodes depth.

There are two flavors:

- **Time-domain OCT (TD-OCT):** physically scan the reference mirror to find each
  depth. Slow (one depth at a time).
- **Spectral-domain OCT (SD-OCT):** keep the mirror fixed and read the entire
  interference **spectrum** at once with a spectrometer (or a swept laser). A
  single spectrum contains **all depths simultaneously** — you recover them with a
  Fourier transform. This is the modern standard and what we implement.

The key insight: **a reflector at depth `z` produces a spectral interference
fringe whose frequency is proportional to `z`.** So the spectrum is a sum of
cosines, one per reflector, and its Fourier transform is the depth profile — the
**A-scan**. Sweep the beam laterally, collect one spectrum per position, and the
stack of A-scans is a 2-D image — the **B-scan**.

Two physical nuisances must be handled for a sharp image:

- **Dispersion mismatch.** The sample and reference arms contain slightly different
  amounts of dispersive material (glass, water, tissue). Dispersion makes optical
  phase a *non-linear* function of wavenumber, blurring every depth peak. It is
  corrected numerically (a phase multiply) — the OCT-specific step this project
  highlights.
- **Spectral leakage.** The spectrum is a finite window; FFT-ing a hard-truncated
  window sprays each reflector's energy into side lobes. A smooth apodization
  window (Hann) suppresses them.

## 2. The math

**Symbols.** Let a single A-scan's raw spectrum be `I[i]`, `i = 0 … N-1`, sampled
in wavenumber `k` (we assume uniform-in-`k` sampling; see §7 for resampling).
Define the normalized bin coordinate `κ = i/N ∈ [0,1)` and the band-centred
coordinate `k_c = (i − (N−1)/2)/N ∈ [−½, ½)`.

**Forward model (what the instrument measures).** For reflectors `r` at depths
`z_r` with reflectivities `R_r`, plus a DC/background term `B`:

```
I[i] = B + Σ_r R_r · cos( 2π z_r κ + φ(k_i) )        (spectral interferogram)
φ(k) = a₂ k_c² + a₃ k_c³                              (dispersion phase error)
```

`a₂, a₃` are the 2nd/3rd-order dispersion coefficients (unitless in this
normalized `k`). `z_r` is in units of FFT bins (a reflector at depth-bin `z` lands
at output bin `z`).

**Reconstruction (what we compute).**

1. **DC removal:** `s[i] = I[i] − mean(I)` (kills the `B` offset and the strong
   zero-frequency term).
2. **Window:** `s[i] ← s[i] · w[i]`, `w[i] = ½(1 − cos(2π i/(N−1)))` (Hann).
3. **Dispersion compensation:** form the complex sample
   `x[i] = s[i] · e^{−i φ(k_i)}` — multiplying by `e^{−iφ}` cancels the phase error.
4. **DFT:** `A[z] = Σ_{i=0}^{N-1} x[i] · e^{−2πi z i / N}`, `z = 0 … N/2−1`.
5. **Image:** `P[z] = |A[z]|²`, normalized per A-scan to its peak; displayed as
   `10·log₁₀(P/P_max)` dB.

We keep only `z ∈ [0, N/2)`: a real-input signal has a Hermitian-symmetric FFT, so
the upper half is a mirror image (the "complex-conjugate artifact" in OCT). The
per-sample steps 1–3 live in `src/oct_core.h::preprocess_sample`; step 4 is the FFT
(cuFFT on the GPU, naive DFT on the CPU); step 5 is `power_norm_kernel` / the CPU
tail.

## 3. The algorithm

Per A-scan the reconstruction is: one pass to compute the mean (DC), one pass to
preprocess (`O(N)`), one length-`N` FFT, one pass to form and normalize the
magnitude (`O(N)`). The FFT dominates.

**Complexity (per A-scan, `A` A-scans, length `N`):**

| Step | Serial cost | Notes |
|---|---|---|
| DC + preprocess | `O(N)` | trivially parallel over samples |
| **Transform (naive DFT)** | **`O(N²)`** | the CPU reference — obviously correct, slow |
| **Transform (FFT)** | **`O(N log N)`** | cuFFT — the algorithmic win |
| magnitude + normalize | `O(N)` | a reduction (max) per A-scan |

Whole B-scan: naive DFT is `O(A·N²)`; FFT is `O(A·N log N)`. For a clinical
`2048×2048` B-scan that is `~8.6×10⁹` vs `~4.6×10⁷` multiply-adds — a ~180×
algorithmic factor *before* any parallelism. The A-scans are **independent**, so on
top of the algorithmic win the GPU parallelizes across all `A` of them.

**Data-access pattern.** Row-major `raw[a·N + i]`: consecutive threads read
consecutive `i` → **coalesced** global loads. Arithmetic intensity is low (a few
flops per element loaded), so the preprocessing kernels are memory-bandwidth-bound;
the FFT is where the compute lives, and cuFFT is heavily optimized for it.

## 4. The GPU mapping

The pipeline is **custom kernel → library → custom kernel**:

```
   raw spectra [A×N]  (device, float)
          │
   ┌──────▼───────┐   dc_kernel: 1 thread / A-scan → mean  (dc[A], double)
   │  DC (mean)   │
   └──────┬───────┘
   ┌──────▼───────────────┐  preprocess_kernel: 1 thread / SAMPLE (A·N threads)
   │ DC-remove + Hann +   │  reads raw[t], dc[a]; calls SHARED preprocess_sample()
   │ dispersion phase     │  writes float2 x[t]  → the cuFFT input
   └──────┬───────────────┘
   ┌──────▼───────┐   cufftExecC2C(plan, x, x, FORWARD)   ← LIBRARY, batched:
   │  batched FFT │   ONE call = all A FFTs of length N (input/output stride N)
   └──────┬───────┘
   ┌──────▼───────────────┐  power_norm_kernel: 1 thread / A-scan
   │ |A|² + per-A-scan max │  reads x[0..N/2), writes normalized image[A×N/2]
   └──────┬───────────────┘
   image [A×(N/2)]  (device, double)  → copied to host
```

**Thread-to-data maps.**
- `preprocess_kernel`: thread `t = blockIdx.x·blockDim.x + threadIdx.x` owns global
  sample `t`; its A-scan is `a = t / N`, its in-scan index `i = t % N`. Grid
  `⌈A·N / 256⌉` blocks × 256 threads.
- `dc_kernel` / `power_norm_kernel`: thread `a` owns A-scan `a`. Grid `⌈A / 256⌉` ×
  256. These deliberately keep the per-A-scan **reduction** (sum for the mean, max
  for the normalization) *inside one thread* so it is order-fixed and deterministic
  (see §5) — no cross-thread float atomics.

**Launch config reasoning.** 256 threads/block is a warp multiple (32) that gives 8
warps to hide memory latency and leaves many blocks resident for occupancy on
`sm_75…sm_89`. The preprocessing kernel is bandwidth-bound, so occupancy (to hide
DRAM latency) matters more than registers here.

**Memory hierarchy.** All buffers live in **global** memory; the access is
coalesced and streamed once, so shared memory buys little for the preprocessing
(no data reuse across threads). cuFFT internally uses **shared memory** and
registers for its radix butterflies — that is exactly the well-tuned complexity we
*don't* want to hand-roll. Per-A-scan scalars (`dc`, `peak`) live in **registers**.

**The library, not a black box.** `cufftPlan1d(&plan, N, CUFFT_C2C, A)` builds a
plan for `A` contiguous length-`N` complex FFTs (input/output stride `N` — exactly
our `[A×N]` buffer). `cufftExecC2C(..., CUFFT_FORWARD)` computes
`A_a[z] = Σ_i x_a[i] e^{−2πi z i/N}` for every A-scan — the same sum the CPU does by
hand. **Hand-rolling it** means a Cooley–Tukey radix-2/4 kernel with bit-reversal,
precomputed twiddle factors, and shared-memory staging per stage — correct and
educational, but a whole project of its own, and slower than the vendor library.
So we *use* cuFFT and *explain* it (PATTERNS.md §5). The custom kernels do the parts
cuFFT cannot: OCT-specific dispersion compensation and the display magnitude.

## 5. Numerical considerations

- **Precision split.** The shared per-sample math (`oct_core.h`) is **double**, so
  the CPU reference and the GPU preprocessing are bit-identical *before* the FFT.
  cuFFT here is **single precision** (`cufftComplex` = `float2`) — the realistic
  choice for real-time OCT, where FP32 throughput and memory bandwidth matter. So
  the only CPU/GPU divergence is FFT-vs-DFT rounding.
- **FFT reorders additions.** A radix FFT sums the `N` terms in a *different order*
  than the naive DFT's left-to-right loop, and floating-point addition is not
  associative — so `|A[z]|` differs at the ~`1e-6…1e-7` level (single precision,
  `N=256`). This is expected and bounded, not a bug (PATTERNS.md §4).
- **Determinism.** stdout must be byte-identical every run. Two safeguards: (a) the
  **deterministic result we print is the integer peak-depth index** (argmax),
  which is immune to sub-ULP wobble as long as the peak is well-separated from its
  neighbors — it is, by construction of the sample; (b) the per-A-scan reductions
  (mean, max) run **inside a single thread**, so no float `atomicAdd` reorders
  them. Timings and the raw float error go to **stderr** (shown, not diffed).
- **No race conditions.** Every output element is written by exactly one thread;
  no shared accumulators, no atomics.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an **independent** reconstruction: same preprocessing
(shared header), but a transparent `O(N²)` naive DFT instead of cuFFT. `main.cu`
runs both and checks:

1. **Exact (integer):** the per-A-scan peak-depth index matches CPU↔GPU with `== 0`
   tolerance. An argmax over a deterministic profile is order-independent, so this
   is a true bit-for-bit check of *where* each reflector is — the physically
   meaningful output.
2. **Tolerance (float):** `max |image_cpu − image_gpu| ≤ 2e-4` over the whole
   normalized image. On the committed sample the worst difference is ~`1.3e-7`,
   comfortably inside the bound, confirming cuFFT reproduces the DFT to
   single-precision rounding.

**A second, stronger check — the science, not just CPU==GPU.** The synthetic sample
places a bright reflector at a *known* depth arc (`8 → 14 → 8` across the field).
The demo **recovers that arc** as the peak depths, and the ASCII B-scan shows the
curved layer — so we are validating that the reconstruction finds the *right
depths*, not merely that two implementations agree. Turning dispersion compensation
off (Exercise 1) visibly broadens the peaks, demonstrating the correction works.

Why is agreement convincing? An independent serial implementation and a parallel
GPU+library implementation share no code paths except the (tiny, audited) shared
per-sample math; agreeing to rounding on an integer argmax *and* a full image is
strong evidence both are correct.

## 7. Where this sits in the real world

Production OCT engines (Heidelberg Spectralis, Zeiss Cirrus, Thorlabs/Bioptigen
SDKs, and open stacks) add several stages this teaching version omits — described
here per CLAUDE.md §13:

- **k-space resampling (NUFFT).** Real spectrometers sample uniformly in
  **wavelength λ**, but the FFT needs uniform **wavenumber** `k = 2π/λ`. Production
  code interpolates (spline / linear / NUFFT) from λ-grid to k-grid before the FFT;
  we assume already-uniform-in-k spectra. (Exercise 5 adds this.)
- **Numerical dispersion calibration.** We inject a known `(a₂,a₃)`; real systems
  *estimate* them by maximizing image sharpness over a calibration scan.
- **Layer segmentation.** The catalog's downstream task: 3-D **graph-cut** across ~8
  retinal boundaries (Iowa Reference Algorithms) or a **U-Net / U-NetRT** CNN
  (TensorRT-optimized, ~3.5 ms/B-scan). Fluid detection adds another CNN.
- **Doppler / OCT-angiography.** Phase differences between repeated B-scans map
  blood flow (OCTA-500 dataset).
- **Real-time pipelining.** Surgical-guidance OCT overlaps spectrometer acquisition
  with GPU reconstruction using **CUDA streams** and pinned host memory to hit ~100
  B-scans/s; here we time a single B-scan as a teaching artifact only.

The reconstruction core we build — DC/window/dispersion + batched FFT + log-mag — is
exactly the front end every one of those systems runs before any AI or measurement.

---

## References

- **Fercher et al., "Optical coherence tomography — principles and applications,"
  Rep. Prog. Phys. (2003)** — the canonical SD-OCT theory (interferometry → FT).
- **Wojtkowski et al., "Ultrahigh-resolution … dispersion compensation," Opt.
  Express (2004)** — numerical dispersion compensation, the phase-multiply we use.
- **cuFFT documentation** (<https://docs.nvidia.com/cuda/cufft/>) — batched
  `cufftPlan1d` / `cufftExecC2C`, the exact API this project drives.
- **Iowa Reference Algorithms** (<https://www.iibi.uiowa.edu/content/shared-software-Iowa-reference-algorithms>)
  — graph-based retinal-layer segmentation (the downstream step).
- **OCT-Marker** (<https://github.com/neurodial/OCT-Marker>) — B-scan / layer-label
  data model, useful when wiring real datasets.
- **NVIDIA cuFFT samples** — real-time OCT reconstruction demos mirroring this
  batched-FFT pattern.
