# THEORY — 2.2 Protein-Protein Docking

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Most of biology is proteins touching other proteins. A hormone binds its
receptor; an antibody clamps onto a virus; an enzyme is switched off by a
regulatory partner; a drug blocks a protein-protein interface. To understand or
to *design* any of these, you need the **structure of the complex**: how do the
two proteins sit relative to each other?

Experiments (X-ray, cryo-EM) solve some complexes, but there are vastly more
pairs than solved structures. **Docking** predicts the complex computationally:
given the two individual structures, find the relative placement that forms a
good interface.

The simplest and oldest computational model is **rigid-body shape
complementarity**. Treat each protein as a fixed solid object. A good complex is
one where the two surfaces are *complementary* — bumps fit into hollows, with a
large contact area and no interpenetration. This is the "lock and key" picture.
It ignores flexibility and chemistry, but it is remarkably powerful as a first
filter, and it is exactly what the FFT-docking servers ZDOCK and ClusPro compute
at their core. That core is what this project builds.

## 2. The math

### Shape functions on a grid

Lay down a cubic grid of `N×N×N` voxels (cells), `spacing` Å on a side. Turn each
protein into a real-valued **shape function** on that grid. The classical
Katchalski-Katzir / ZDOCK choice is two-valued:

- a voxel deep **inside** a protein (all neighbors also inside) → `core = +1`;
- a voxel on the **surface** shell → `skin = −ρ` (a negative penalty, here `ρ=9`);
- empty space → `0`.

Call the receptor's shape function `R(x)` and the ligand's `L(x)`, where
`x = (x,y,z)` is a voxel.

### The docking score: a cross-correlation

Slide the ligand by an integer translation `t` and score the overlap by summing
the product of the two shape functions:

```
S(t) = Σ_x  R(x) · L(x − t)
```

Why this scores complementarity:

- `core·core` (both `+1`) → `+1`: the ligand's interior overlaps the receptor's
  interior. A little of this is good (close contact); a lot means the bodies are
  passing *through* each other.
- `core·skin` and `skin·skin`: a ligand voxel landing on the receptor's surface
  shell hits `−ρ`. Deep overlap therefore racks up a large **negative** penalty
  (`skin·core = −ρ`), which is what stops the optimizer from simply burying one
  protein inside the other. The net effect: the best `t` puts the two **surfaces
  in contact** with maximal buried area and minimal interpenetration.

`S(t)` as a function of `t` is precisely the **cross-correlation** of `R` and `L`.
We want the translation that maximizes it:

```
t* = argmax_t  S(t)
```

### The correlation theorem (why an FFT)

Computing `S(t)` for one `t` costs `O(Ng)` (a sum over all `Ng = N³` voxels), and
there are `Ng` translations, so the brute force is `O(Ng²)`. The **correlation
theorem** says a correlation becomes a *pointwise product* in Fourier space:

```
S = IFFT( FFT(R) · conj(FFT(L)) )
```

Each FFT is `O(Ng log Ng)`, the pointwise product is `O(Ng)`, so the entire grid
of scores costs `O(Ng log Ng)` — for `Ng = 10⁶` that is the difference between
`10¹²` and `~2·10⁷` operations.

**Which spectrum to conjugate sets the sign of the shift.** A plain product
`FFT(R)·FFT(L)` gives a *convolution* `Σ_x R(x)L(t−x)`. Conjugating one factor
flips it to a correlation. Conjugating the **ligand** (`conj(FFT(L))`) yields
exactly our convention `S(t)=Σ_x R(x)L(x−t)`; conjugating the receptor would give
the mirror image `S(−t)`. We verified the ligand-conjugate choice bit-for-bit
against the brute-force reference (see `src/kernels.cu` and §6).

A subtlety: the FFT correlation is **circular** (periodic) — it treats the grid
as a torus, so `L` wraps around the edges. Our CPU reference uses the identical
wrap (the `wrap()` helper in `reference_cpu.h`), so the two match exactly. Real
docking pads the grid with zeros so the wrap-around never reaches real density.

## 3. The algorithm

```
1. load receptor + ligand atom coordinates                       (host I/O)
2. choose a common grid origin (center the receptor)             (host)
3. voxelize receptor -> R   (occupancy ball per atom, then core/skin)   O(n_atoms + Ng)
4. voxelize ligand   -> L   (same rule, same frame)                     O(n_atoms + Ng)
5. CPU reference: S_cpu(t) = Σ_x R(x) L(wrap(x-t))   direct sum         O(Ng^2)*
6. GPU: S_gpu = IFFT( FFT(R) · conj(FFT(L)) ) / Ng   via cuFFT          O(Ng log Ng)
7. verify S_gpu ≈ S_cpu (grid) and argmax matches (pose)               O(Ng)
8. report best translation + score
```

`*` Step 5 skips empty receptor voxels (`R(x)=0`), so in practice it is
`O(Ng · (occupied voxels))`, but the asymptotics are quadratic — which is exactly
why the FFT exists. Step 5 is the slow, transparently-correct baseline; we only
run it at the small teaching grid size.

**Complexity comparison** (the headline lesson):

| Route | Work | Sample (`N=32`, `Ng=32768`) |
|---|---|---|
| Brute-force correlation | `O(Ng²)` | ~1.3 s (CPU) |
| FFT correlation (cuFFT) | `O(Ng log Ng)` | ~0.5 ms (GPU) |

## 4. The GPU mapping

The heavy lifting is **three 3D FFTs**, which we delegate to **cuFFT** — the
"use a library without a black box" pattern (PATTERNS.md §1). The data layout and
the two custom kernels:

### cuFFT, explained (not hidden)

- `cufftPlan3d(&plan, nz, ny, nx, CUFFT_R2C)` — a plan for a 3D **real-to-complex**
  forward FFT of an `nz×ny×nx` grid. Because a real signal's spectrum is
  Hermitian-symmetric, cuFFT stores only the non-redundant half along the fastest
  (x) axis: the complex output is `nz × ny × (nx/2+1)` `cufftComplex`. For each
  frequency `k` it computes `X[k] = Σ_n x[n] exp(−2πi k·n/N)` — the same sum the
  brute-force DFT would do, in `O(Ng log Ng)`.
- `cufftExecR2C` runs it; we do this twice (receptor, ligand).
- `cufftPlan3d(..., CUFFT_C2R)` + `cufftExecC2R` is the **inverse** transform of
  the spectrum product, giving the real score grid back.

To hand-roll this you would write a mixed-radix Cooley-Tukey FFT with bit-reversal
and twiddle-factor caching in three passes — hundreds of lines, easy to get wrong,
and slower than cuFFT's tuned kernels. Hence the library.

### The two custom kernels

Both are embarrassingly parallel "one thread per element" maps (the most
fundamental CUDA pattern), launched with 256-thread blocks:

```
spectral_correlate_kernel   one thread per complex bin k:
                            P[k] = Rf[k] * conj(Lf[k])
                            (the correlation theorem's pointwise product)

scale_kernel                one thread per real voxel i:
                            S[i] *= 1/Ng
                            (cuFFT's forward+inverse pair is unnormalized,
                             so the round-trip multiplies by Ng = N^3)
```

Thread-to-data map: `i = blockIdx.x * blockDim.x + threadIdx.x`, guarded by
`if (i < n)` for the ragged last block. No shared memory, no atomics — each output
depends only on its own input, so the work is pure independent register math
(memory-bandwidth bound).

### Memory & dataflow

```
host R,L (N^3 floats)
      │  cudaMemcpy H2D
      ▼
 d_real ──cufftExecR2C──▶ d_Rf  (receptor spectrum, N*N*(N/2+1) complex)
 d_real ──cufftExecR2C──▶ d_Lf  (ligand spectrum)
                              │
        spectral_correlate_kernel:  d_Lf ← d_Rf · conj(d_Lf)
                              │
 d_real ◀──cufftExecC2R──── d_Lf  (real score grid)
        scale_kernel:  d_real *= 1/Ng
      │  cudaMemcpy D2H
      ▼
 host score grid  ──argmax──▶ best translation
```

We reuse one real buffer (`d_real`) for both inputs and the output, and write the
spectrum product back into `d_Lf`, so total device memory is one real grid + two
half-complex grids.

**Why voxelize on the host?** Voxelization is cheap (`O(n_atoms)`) and
order-sensitive (core/skin classification). Doing it once on the host with the
*shared* rule (`build_shape_grid` in `reference_cpu.cpp`) guarantees the CPU and
GPU consume **byte-identical** grids, so the only source of CPU/GPU divergence is
the FFT's floating-point round-off — which is the honest thing we want to measure.

## 5. Numerical considerations

- **Precision.** cuFFT here is **single precision** (`cufftReal`/`cufftComplex`),
  while the CPU reference accumulates in **double**. The shape grids hold small
  integers, and the CPU correlation is an exact sum of integer products — so the
  CPU score grid is *exactly* integer-valued. The GPU score differs only by the
  FFT's accumulated round-off.
- **How big is the round-off?** A length-`Ng` FFT accumulates `~log(Ng)` rounding
  steps; the relative error is `~ε·√(log Ng)` with `ε ≈ 6×10⁻⁸` for FP32. On the
  sample (peak score ≈ `7.7×10⁴`) the worst observed voxel error is ≈ `0.05`, a
  **relative** error of `~6×10⁻⁷` — right at that floor. We set the verification
  tolerance to an absolute `0.5` (`~1×10⁻⁵` relative): comfortably above the
  round-off, far below the integer spacing of the scores, so it cannot hide a real
  bug. This is the honest "machine-precision-ish" tolerance class of PATTERNS.md §4.
- **Determinism.** cuFFT's result is deterministic for a fixed plan/size/GPU, and
  our two kernels do no cross-thread reduction (no `atomicAdd`), so there is **no
  floating-point reordering** — stdout is byte-identical every run. The reported
  *score* in stdout is taken from the **CPU** grid (exactly `77294.0000`) so the
  printed value does not depend on FP32 FFT rounding and stays stable across GPUs;
  timings (which vary) go to stderr.
- **The argmax is robust.** Even though individual voxels differ by `~1e-7`
  relative, the *location* of the maximum is determined by gaps of order the score
  itself, so the best translation is identical on CPU and GPU (and matches the
  known answer).

## 6. How we verify correctness

Two independent checks, both in `src/main.cu`:

1. **Grid agreement.** The cuFFT score grid must match the brute-force CPU grid
   at *every* voxel within the `0.5` absolute tolerance above. Two completely
   different implementations (a direct `O(Ng²)` triple loop vs. three FFTs and a
   spectral multiply) agreeing voxel-by-voxel is strong evidence the FFT route —
   including the tricky conjugation sign and the `1/Ng` normalization — is right.
2. **Pose agreement + known answer.** The `argmax` of both grids must be the same
   translation, and (for the synthetic sample) it must equal the **known answer**
   baked into the data. The ligand is the receptor displaced by `D=(3,2,−1)`
   voxels, so by Cauchy-Schwarz the autocorrelation has its unique maximum at
   `t = −D = (−3,−2,1)`; recovering exactly that confirms the whole pipeline
   (voxelization frame, correlation convention, argmax) end-to-end.

Edge cases covered: the ragged last thread block (guarded in both kernels), the
periodic wrap shared by CPU and GPU, and the negative-index `floor` in
`world_to_voxel`.

## 7. Where this sits in the real world

This project is the **translational inner loop** of FFT docking. Production tools
add, in roughly this order:

- **Rotational search.** The real problem is 6D (3 translation + 3 rotation). ZDOCK
  /ClusPro run the FFT correlation above for **thousands of ligand orientations**
  (e.g. a 6°–15° Euler grid → 10³–10⁴ rotations), each an independent FFT search —
  massively parallel, and exactly why GPUs help. A neat alternative expands the
  shape into **spherical harmonics** so rotations become cheap multiplications
  (Hex, PIPER's fast variants), turning the 6D search into FFTs over both
  translation *and* rotation.
- **Richer scoring.** Beyond shape: electrostatics (a second correlation of charge
  grids, as in ZDOCK), desolvation/statistical pair potentials (PIPER's DARS),
  and knowledge-based terms. Each extra term is another grid correlated by the
  same FFT machinery and summed in.
- **Pose ranking & clustering.** Thousands of candidate poses are clustered;
  ClusPro ranks by cluster *size* (a populated funnel beats a single high score).
- **Flexibility & refinement.** Rigid poses are relaxed with GPU molecular dynamics
  / energy minimization (HADDOCK) to model induced fit.
- **Deep learning.** The newest methods skip the exhaustive grid entirely:
  DiffDock-PP runs an **equivariant diffusion** model over the pose manifold, and
  AlphaFold-Multimer / RoseTTAFold2NA **co-fold** the complex from sequence and
  coevolution signal — no FFT correlation at all. The FFT method remains valuable
  as a fast, interpretable, assumption-light baseline and pose generator.

---

## References

- **Katchalski-Katzir et al. (1992)**, *PNAS* — the original FFT shape-correlation
  docking; the core/skin grid and correlation idea implemented here.
- **Chen, Li & Weng (2003), ZDOCK**, *Proteins* — pairwise shape + electrostatics +
  desolvation FFT docking; the grid scoring model this project simplifies.
- **Kozakov et al. (2017), ClusPro/PIPER**, *Nat. Protoc.* — the production FFT
  docking server and its cluster-based ranking. <https://cluspro.bu.edu>
- **cuFFT documentation** — the R2C/C2R layout and plan API used in `kernels.cu`.
  <https://docs.nvidia.com/cuda/cufft/>
- **Ketata et al. (2023), DiffDock-PP** — the diffusion-model successor to grid
  docking. <https://github.com/ketatam/DiffDock-PP>
- **Honorato et al., HADDOCK** — data-driven docking with MD refinement.
  <https://wenmr.science.uu.nl/haddock2.4/>
