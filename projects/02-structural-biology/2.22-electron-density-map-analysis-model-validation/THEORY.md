# THEORY — 2.22 Electron Density Map Analysis & Model Validation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A protein structure is "solved" by reconstructing a 3-D map of **electron
density** — literally, how many electrons sit near each point in space, because
both X-rays (crystallography) and electrons (cryo-EM) scatter off the electron
cloud of the molecule. The map is a cube of real numbers `ρ(x,y,z)` sampled on a
grid. The biologist then builds an **atomic model** (the chain of amino acids) so
that the atoms land where the density is high.

Two questions must be answered before anyone trusts the result:

1. **How good is the map?** A cryo-EM reconstruction is computed from tens of
   thousands of noisy particle images. The standard trick is to split the data in
   half, reconstruct **two independent "half-maps,"** and ask: *at what spatial
   frequency do the two half-maps stop agreeing?* That frequency is the
   **resolution**. Beyond it, the "detail" is just noise that happens to differ
   between the halves.

2. **Does the model fit the density?** For each residue (or the whole map) you
   correlate the model-derived density with the experimental density. A low
   correlation flags a mis-built region.

The first question is answered by **Fourier Shell Correlation (FSC)**; the second
by the **Real-Space Correlation Coefficient (RSCC)**. Both are computed by every
deposition pipeline (wwPDB OneDep) and every refinement package (Phenix, CCP4).
This project computes both and validates a GPU implementation against a CPU one.

> **Why "before deposition"?** EMDB/PDB entries carry a validation report. A map
> whose claimed resolution is not supported by its FSC, or a model with poor RSCC,
> gets caught here. None of this is diagnostic of a patient — it is quality
> control on a scientific measurement.

## 2. The math

Let `a(x)` and `b(x)` be two real maps sampled on the same `n × n × n` grid, with
voxel index `x = (x,y,z)`, `0 ≤ x,y,z < n`, and physical voxel size `δ` Å.

**Real-space correlation coefficient (RSCC).** Pearson correlation over all
`N = n³` voxels:

```
        N·Σ a·b − (Σa)(Σb)
RSCC = ───────────────────────────────────────────
       √[ (N·Σa² − (Σa)²) · (N·Σb² − (Σb)²) ]
```

`RSCC ∈ [−1, 1]`; `1` = identical up to scale/offset, `0` = uncorrelated. We
accumulate the five sums `Σa, Σb, Σa², Σb², Σab` in one pass and close with the
formula above (`pearson_from_sums` in `map_core.h`).

**Discrete Fourier transform.** The 3-D DFT maps the real cube into reciprocal
space (spatial frequencies `k = (kx,ky,kz)`):

```
F(k) = Σ_x  f(x) · exp(−2πi (kx·x + ky·y + kz·z) / n)
```

Each axis index runs `0 … n−1`; the **signed** frequency of bin `i` is
`fftfreq(i) = i` for `i ≤ n/2`, else `i − n` (NumPy convention). The distance to
the origin `|k| = √(kx² + ky² + kz²)` is the spatial frequency in *cycles per
box*; a frequency of `s` cycles/box corresponds to a real-space period
(**resolution**) of `n·δ / s` Å.

**Fourier Shell Correlation (FSC).** Group reciprocal space into spherical
**shells** `s = round(|k|)`. With `F₁ = DFT(a)`, `F₂ = DFT(b)`:

```
                Re( Σ_{k ∈ shell s} F₁(k) · conj F₂(k) )
FSC(s) = ───────────────────────────────────────────────────────
          √[ Σ_{k ∈ s} |F₁(k)|²  ·  Σ_{k ∈ s} |F₂(k)|² ]
```

`FSC(s) ∈ [−1, 1]` is the normalized cross-correlation of the two transforms in
that frequency band. The three per-voxel accumulands —
`cross = Re(F₁·conj F₂) = re₁re₂ + im₁im₂`, `p₁ = |F₁|²`, `p₂ = |F₂|²` — are in
`fsc_accumulate` (`map_core.h`).

**Resolution.** The reported resolution is the finest shell `s*` that stays at or
above a threshold before FSC first crosses it:

- **0.143** for two independent half-maps (the *gold-standard* criterion; Rosenthal
  & Henderson 2003).
- **0.5** for map-vs-model FSC.

Convert: `resolution = n·δ / s*` Å.

## 3. The algorithm

```
INPUT  : two n³ maps a, b ; voxel size δ
1. RSCC : one pass over N voxels accumulating Σa,Σb,Σa²,Σb²,Σab → Pearson r
2. FFT  : F1 = DFT3(a), F2 = DFT3(b)                 (the expensive step)
3. BIN  : for every voxel, s = round(|k|); add cross,|F1|²,|F2|² to shell s
4. FSC  : FSC[s] = cross[s] / √(p1[s]·p2[s])
5. RES  : s* = last shell with FSC ≥ threshold ; resolution = n·δ / s*
OUTPUT : RSCC, FSC curve, resolution @ {0.143, 0.5}
```

**Complexity.** Steps 1, 3, 4, 5 are all `O(N) = O(n³)`. Step 2 is the cost
driver:

| FFT method | per map | why |
|---|---|---|
| **Naive DFT** (CPU reference) | `O(n⁴)` (separable: `n³` outputs × `n` work × 3 axes) | transparently correct, but `n=64` ≈ 10⁹ ops |
| **FFT** (cuFFT) | `O(n³ log n)` | radix decomposition along each axis |

For `n = 256` the naive DFT is ~`256` × more work than the FFT *per axis* — the
difference between "instant" and "overnight." That is why FSC software always uses
an FFT, and why the GPU's job here is essentially *the FFT.*

The data-access pattern: steps 1 and 3 are a single streaming pass over the cube
(memory-bandwidth bound, low arithmetic intensity). The FFT is bandwidth-bound too
but with a non-trivial access pattern that cuFFT has tuned for the memory
hierarchy.

## 4. The GPU mapping

Three GPU phases, then a small host finish:

**(a) RSCC reduction — `rscc_partials_kernel`.**
- Thread map: a **grid-stride loop**. Thread `g = blockIdx.x·blockDim.x +
  threadIdx.x` accumulates voxels `g, g+stride, g+2·stride, …` into five private
  `double` registers, so any `N` is covered by any grid size.
- Block size **256** (multiple of the 32-lane warp; 8 warps to hide latency).
- Each block tree-reduces its 256 threads' partials in **shared memory** (5 ×
  256 doubles), then thread 0 writes one partial per sum to global memory at
  `blockIdx.x`. No atomics — one output slot per block.
- The host sums the ≤1024 block-partials in a fixed order (deterministic).

**(b) FFT — cuFFT, not a black box (`fft_map_gpu`).**
- `cufftPlan3d(&plan, n, n, n, CUFFT_C2C)` builds a plan for one length-`n×n×n`
  complex FFT in C-order (z slowest, x fastest — matching our map layout).
- `cufftExecC2C(..., CUFFT_FORWARD)` computes exactly the triple sum `F(k)` from
  §2 for all `n³` bins, in `O(n³ log n)`. We feed the real map as complex
  (imag = 0) and use the **full** transform so the GPU produces the identical
  cube the CPU's `dft3d` does.
- **To hand-roll this** you would implement a mixed-radix FFT along each axis with
  bit-reversal permutation and precomputed twiddle factors, tile into shared
  memory while dodging bank conflicts, and tune per `n` — precisely the work cuFFT
  has already done and validated. That is the lesson: *use the library, but know
  what it computes* (CLAUDE.md §6.1.6).

**(c) Widen — `extract_complex_kernel`.** One thread per bin copies cuFFT's
`float2` (`cufftComplex`) output into a portable `Cplx` (double) so the host can
bin in the same precision as the CPU.

**(d) Shell binning — host.** With the spectra in `Cplx[]`, the host loops voxels,
computes `s = shell_index(kx,ky,kz)`, and calls the **shared** `fsc_accumulate`
— byte-identical to the CPU path.

```
              n³ density voxels
                     │
        ┌────────────┼─────────────┐
   (a) RSCC      (b) cuFFT      (b) cuFFT
   block-reduce   FFT(a)=F1      FFT(b)=F2
        │             │              │
   host sum 5      (c) extract   (c) extract
   partials → r     float2→Cplx   float2→Cplx
                     └──────┬───────┘
                      (d) host shell-bin
                  s=round|k|: cross,|F1|²,|F2|²
                            │
                   FSC[s] = cross/√(p1·p2) → resolution
```

**Why finish on the host (the honest part).** A parallel float reduction sums in a
nondeterministic order, so its low bits change run to run. The FFT is the heavy
`O(n³ log n)` work and belongs on the GPU; the `O(n³)` shell/RSCC *finishing* sums
are cheap, so we do them deterministically on the host to keep stdout
byte-identical (PATTERNS.md §3). At real map scale you would keep the reduction on
the GPU using **fixed-point** accumulation (integer atomics commute → deterministic
*and* exact), the same trick flagships 5.01 and 11.09 use.

## 5. Numerical considerations

- **Precision.** cuFFT here is **single precision** (`cufftComplex` = `float2`);
  the CPU DFT and all reductions are **double**. So the FFT values differ from the
  exact DFT by ~`1e-6` relative, which propagates into a per-shell FSC difference
  of ~`1e-7…1e-5`. This is *real and worth teaching*: single-precision FFT is the
  norm in cryo-EM for speed/memory, and the resolution it reports is unaffected.
- **RSCC** uses the *same* double formula on the *same* maps on both sides, so CPU
  and GPU agree to ~machine precision (~`1e-15` here).
- **Determinism.** stdout prints the CPU's fully-deterministic double values; the
  GPU's block-reduction is made deterministic by the fixed-order host sum of
  per-block partials. The GPU-vs-CPU error goes to **stderr** (shown, not diffed).
- **Stability.** The single-pass Pearson formula can lose precision if the maps
  have a huge constant offset (catastrophic cancellation in `N·Σa² − (Σa)²`). Our
  maps are mean-near-zero, so this is fine; a two-pass (subtract-the-mean)
  formulation is the robust alternative and a good exercise.
- **Empty/degenerate shells.** `fsc_from_sums` and `pearson_from_sums` guard the
  zero-denominator case (a shell with no power, or a flat map) by returning 0.

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp`) computes RSCC with the same five-sum
formula and FSC via a **naive separable 3-D DFT** — the textbook transform written
out as explicit `cos`/`sin` sums. It is `O(n⁴)` and obviously correct, which is the
entire point of a reference: when the fast GPU path (cuFFT) agrees with the slow
transparent one, we trust the fast one.

Because the per-voxel math lives in **one shared `__host__ __device__` header**
(`map_core.h`), the CPU and GPU accumulate the *same* quantities in the *same*
order; only the FFT engine differs. So the comparison isolates exactly the thing
we want to trust — the FFT.

**Tolerances** (PATTERNS.md §4):

| Quantity | Tolerance | Why |
|---|---|---|
| RSCC | `≤ 1e-9` (abs) | identical double formula on identical data |
| FSC curve | `≤ 1e-4` (abs, per shell) | single-precision cuFFT vs double DFT; physically negligible for a correlation in [−1,1] |

On the committed sample the measured errors are RSCC `~1e-15` and worst FSC
`~6e-8` — comfortably inside the tolerances, and printed on stderr so the gap is
honest rather than hidden. A second, *scientific* check: the recovered resolution
(8.0 Å) matches where we deliberately injected the high-frequency noise in
`make_synthetic.py` — so the pipeline recovers a known answer, not just CPU==GPU.

## 7. Where this sits in the real world

Production validation differs in scale and breadth, not in principle:

- **Real map formats.** EMDB maps are MRC/CCP4 (a 1024-byte header + float32
  voxels, possibly non-cubic, with an axis-order field). Tools parse this with
  GEMMI / mrcfile; we use a plain text cube so the loader is a few readable lines.
- **Masking & corrections.** Real FSC is computed inside a soft **mask** around the
  molecule (solvent voxels dilute the correlation), and high-resolution FSC is
  *phase-randomization-corrected* to remove mask-induced correlation. We omit
  masking for clarity.
- **Local resolution.** MonoRes/ResMap slide a window across the map and compute a
  *local* FSC per window — many small independent FFTs, the ideal GPU batch
  (exercise 5).
- **Crystallography.** There the validation set also includes **difference maps**
  (`Fo−Fc`, `2Fo−Fc`) — you compute model structure factors `Fcalc` from the atoms,
  combine with observed `Fobs` and experimental phases, inverse-FFT to a real-space
  difference map, and inspect peaks (missing/extra atoms). The global agreement is
  the **R-factor** `R = Σ||Fobs|−|Fcalc|| / Σ|Fobs|` and its cross-validated twin
  **R-free** (computed on a held-out reflection set to detect over-fitting). All of
  this is one or more FFTs over reciprocal space — exactly the cuFFT pattern here,
  just with `Fobs`/`Fcalc` instead of two maps. We describe but do not implement it
  (it needs experimental reflections + a model).

This project is a faithful, verifiable *core* of map validation; the omissions are
about input plumbing and statistical corrections, not the underlying transform.

---

## References

- **Rosenthal & Henderson (2003)**, *J. Mol. Biol.* — the **FSC = 0.143**
  gold-standard resolution criterion for half-maps. The reason shell 4 → 8 Å here.
- **van Heel & Schatz (2005)**, *J. Struct. Biol.* — the **half-bit** FSC
  criterion; basis for exercise 3.
- **Afonine et al., Phenix `phenix.map_model_cc`** — how production RSCC/map-model
  FSC is computed (masking, per-residue scores).
- **CCP4 `EMDA` / `Servalcat`** — FSC, map-model validation, difference maps.
- **GEMMI docs** (https://gemmi.readthedocs.io) — MRC/CCP4 map header layout, for
  reading real EMDB maps (exercise 2).
- **NVIDIA cuFFT documentation** — the `cufftPlan3d` / `cufftExecC2C` semantics,
  data layout, and the R2C half-spectrum used in exercise 4.
