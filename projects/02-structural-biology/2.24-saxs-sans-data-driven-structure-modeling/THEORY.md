# THEORY — 2.24 SAXS / SANS Data-Driven Structure Modeling

> The deep didactic explanation (the "why"). Written for a sharp student who knows
> C++ but is new to CUDA and new to this domain. See [README.md](README.md) for the
> quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A crystal structure tells you what a protein looks like when it is frozen into a
lattice. But proteins do their work **in solution**, where they tumble, breathe, and
(for intrinsically disordered proteins, IDPs) never adopt a single shape at all.
**Small-angle scattering** is the premier experimental method for studying that
solution state.

The experiment is conceptually simple. Shine a collimated beam of **X-rays** (SAXS)
or **neutrons** (SANS) through a dilute protein solution. Most of the beam passes
straight through, but a little is **elastically scattered** by the electrons (X-rays)
or nuclei (neutrons) of the molecules. Because the molecules are randomly oriented,
the scattered intensity depends only on the **magnitude** of the momentum transfer,

```
q = (4π / λ) · sin(θ)      [units: 1/Å]
```

where `λ` is the beam wavelength and `2θ` is the scattering angle. The detector
records a **1-D curve `I(q)`** — that is the entire experimental dataset. "Small
angle" means small `q`, which by the reciprocity of Fourier optics probes **large**
real-space distances (the overall size and shape of the molecule, ~10–100 Å), not
atomic detail.

From `I(q)` you can read, with increasing model dependence:

- the **radius of gyration `Rg`** (overall size) from the lowest-`q` "Guinier" region;
- the **molecular shape** (globular vs. extended vs. disordered) from the mid-`q`
  decay and the Kratky plot `q²I(q)`;
- and, by **fitting candidate atomic models**, which 3-D structure (or *ensemble* of
  structures) is consistent with the data.

That last step is the focus here. To fit a model you must be able to **predict** the
SAXS curve a given set of atomic coordinates would produce — the *forward problem* —
fast, because model fitting and ensemble refinement evaluate it thousands of times.

## 2. The math

### 2.1 The Debye formula

Treat the molecule as `N` point scatterers at positions `r_i` with scattering
strengths `f_i` (for X-rays, roughly the number of electrons on atom `i`). The
amplitude scattered in a direction with momentum-transfer **vector** `q` is

```
A(q) = Σ_i f_i · exp(i q·r_i)          and      I(q) = |A(q)|² = Σ_i Σ_j f_i f_j exp(i q·(r_i − r_j)).
```

The solution is isotropic, so we must **average over all orientations** of `q` on the
sphere `|q| = q`. The orientational average of a plane wave is the famous spherical
Bessel result

```
⟨ exp(i q·r_ij) ⟩_orientation = sin(q r_ij) / (q r_ij) = sinc(q r_ij),
```

with `r_ij = |r_i − r_j|`. Substituting gives the **Debye formula** (Debye, 1915):

```
            N   N
   I(q) =  Σ   Σ   f_i f_j · sinc(q r_ij),     sinc(x) = sin(x)/x,  sinc(0) = 1.
           i=1 j=1
```

This is the one equation the whole project computes. Symbols:

| symbol | units | meaning |
|---|---|---|
| `q` | 1/Å | momentum-transfer magnitude (the curve's x-axis) |
| `r_ij` | Å | distance between atoms `i` and `j` |
| `f_i` | electrons | scattering strength of atom `i` (point-atom approx: constant) |
| `I(q)` | electrons² | predicted intensity at `q` |

Two useful exact facts: at `q→0`, `sinc→1`, so `I(0) = (Σ_i f_i)²` (the forward
intensity is the squared total scattering — proportional to molecular weight²); and
the curve is **symmetric in `i,j`**, so `I(q) = Σ_i f_i² + 2 Σ_{i<j} f_i f_j sinc(q r_ij)`
(diagonal + twice the upper triangle), which halves the work.

### 2.2 Guinier's law (recovering `Rg`)

Expanding the Debye sum for small `q` (`sinc(x) ≈ 1 − x²/6`) yields **Guinier's
approximation**, valid for `q·Rg ≲ 1.3`:

```
   ln I(q) ≈ ln I(0) − (Rg² / 3) · q²,
```

where `Rg² = Σ_i f_i |r_i − r_cm|² / Σ_i f_i` is the scattering-weighted **radius of
gyration**. So a straight-line fit of `ln I` against `q²` over the low-`q` points has
slope `−Rg²/3`, giving `Rg = sqrt(−3·slope)`. This is how the demo turns the computed
curve back into a single structural number it can check against the known geometry.

### 2.3 Fitting a model to data

Experimental intensities are on an arbitrary scale, so we fit a single positive scale
`c` minimizing the χ² against the data `I_exp(q)` with error bars `σ(q)`:

```
   χ²(c) = Σ_k ( (c·I_model(q_k) − I_exp(q_k)) / σ(q_k) )²,
```

solved in closed form by `c = Σ(I_model I_exp/σ²) / Σ(I_model²/σ²)`. The **reduced**
χ² (χ²/N_q) is ≈ 1 when the model fits to within the noise. Minimizing this score over
conformations (or over *weights* of an ensemble) is the modeling problem; here we
just evaluate it for one model.

## 3. The algorithm

```
load model (atoms r_i, f_i; q grid; experimental curve)
for each q_k:                                  # N_q independent outputs
    I_model[k] = Σ_i f_i²                       #   diagonal (self) terms
               + 2 Σ_{i<j} f_i f_j sinc(q_k r_ij)   #   pair terms
c    = best_scale(I_model, I_exp, σ)           # closed-form least squares
chi2 = reduced_chi_square(I_model, c, ...)     # goodness of fit
Rg   = guinier_fit(ln I_model vs q²)           # recover the size
```

**Complexity.** Each `q` costs `O(N²)` distance+sinc evaluations; the full profile is
**`O(N_q · N²)`** time. Serial depth is the same. The parallel version has depth
`O(N²)` (one thread's reduction) and total work `O(N_q · N²)` spread over `N_q`
threads. The analysis steps are all `O(N_q)` and negligible.

**Arithmetic intensity.** Very high: each pair term reads a handful of coordinates
from memory but does a `sqrt`, a `sin`, and several multiplies — this is a
**compute-bound** kernel, not memory-bound, which is ideal for the GPU (its
weakness, memory bandwidth, is not the limiter; its strength, raw FLOP/s and fast
transcendental units, is).

## 4. The GPU mapping

**Thread-to-data mapping.** One thread per `q` value:
`k = blockIdx.x·blockDim.x + threadIdx.x`. Thread `k` loops over **all** atom pairs
and writes a single `I_model[k]`. The entire `O(N²)` reduction lives in that thread's
registers — there is **no shared accumulator, no atomics, no inter-thread sync**.

```
   q grid:   q0  q1  q2  q3  q4  q5  q6  q7  ...  q[N_q-1]
              |   |   |   |   |   |   |   |          |
   threads:  t0  t1  t2  t3  t4  t5  t6  t7  ...   t[N_q-1]
              |
              +--> thread t_k:  acc = Σ_i f_i²
                                for i<j:  acc += 2 f_i f_j sinc(q_k * |r_i - r_j|)
                                I_out[k] = acc          (one global write)

   grid  = ceil(N_q / 128) blocks      block = 128 threads (4 warps)
```

**Launch configuration.** `THREADS_PER_BLOCK = 128`. Each thread is register-heavy
(it holds the running sum and pulls coordinates in a tight loop), so 128 (4 warps per
block) balances *enough* warps to hide the global-memory latency of the atom reads
against register pressure that would otherwise cut occupancy. The grid is the ceiling
division `(N_q + 127)/128`, and a bounds guard `if (k >= N_q) return;` protects the
ragged last block.

**Memory hierarchy.**
- **Global memory** holds the atom arrays `x,y,z,f` (one element per atom). They are
  read by every thread but never written during the launch.
- **Structure-of-arrays (SoA)** layout (`x[]`, `y[]`, `z[]`, `f[]`, separate arrays)
  rather than array-of-structs: when the 128 threads of a block march through atom
  index `j` together, they read `x[j], x[j+1], …` from *contiguous* addresses →
  **coalesced** loads (one memory transaction serves a whole warp). An array of
  `{x,y,z,f}` structs would scatter those reads across a 16-byte stride and waste
  bandwidth.
- **Registers** hold the running double-precision accumulator and the loop scalars —
  the reduction never touches shared memory.

**Why not constant memory for the atoms?** Flagships `1.12`/`12.01` put the *query* in
constant memory because it is small and broadcast to all threads. Here the shared
operand is the *whole atom set* (potentially 10⁴ atoms = 320 KB), which exceeds the
64 KB constant bank — so the atoms stay in global memory.

**No CUDA math library.** The Debye sum is a custom kernel; we link only the CUDA
runtime. The catalog mentions cuBLAS "for spherical harmonic coefficients" — that
belongs to the *alternative* CRYSOL-style algorithm (§7), which expands the amplitude
in spherical harmonics and would use BLAS to contract coefficient vectors. We chose
the **direct Debye sum** instead because it is the most transparent forward model and
maps to the simplest, most teachable GPU pattern. Writing the spherical-harmonic path
by hand would mean computing `Y_lm` on a Lebedev angular grid, projecting each atom's
contribution, and summing `|A_lm|²` — a worthwhile but much larger exercise.

**A faster variant (left as an exercise).** Stage the atom arrays through **shared
memory** in tiles: each block cooperatively loads a tile of atoms into shared memory,
every thread reuses that on-chip tile for its inner loop, then the block advances to
the next tile. This is the classic N-body tiling and cuts global-memory traffic by the
tile size. The catch: it reorders the pair summation, so GPU and CPU then agree only to
~`1e-12` instead of machine precision (PATTERNS.md §4). We ship the un-tiled,
exactly-matching kernel as the teaching baseline.

## 5. Numerical considerations

- **Precision: FP64 throughout.** `I(q)` spans many orders of magnitude — the `q≈0`
  forward intensity is `(Σf)²` (here ~58 000) while the high-`q` tail is hundreds.
  A double accumulator preserves the small high-`q` structure that single precision
  would drown in rounding. The cost (≈1/2 the FP throughput on consumer GPUs) is
  irrelevant for a teaching-scale problem and worth the clarity.
- **The `sinc(0)` singularity.** `sin(x)/x` is `0/0` at `x=0` (the self terms, and any
  coincident atoms). We branch on `|x| < 1e-8` and return the Taylor value
  `1 − x²/6`, which is accurate to ~`1e-12` there. The branch uses the **same
  constant on CPU and GPU** so the two paths stay in lockstep.
- **Determinism.** Each `q`'s reduction is done by a **single thread in a fixed
  order** — there is no atomic accumulation across threads, so there is no
  floating-point reordering between runs. stdout is therefore byte-identical every
  run (PATTERNS.md §3); timings (which vary) go to stderr.
- **No atomics, no races.** Threads write disjoint outputs `I_out[k]`; the atom arrays
  are read-only during the launch.

## 6. How we verify correctness

The trusted baseline is `debye_profile_cpu()` in `src/reference_cpu.cpp`: a plain
serial loop over `q` with **no parallelism**. Crucially, both it and the GPU kernel
call the **identical** function `debye_intensity_at_q()` from `src/saxs_core.h` — the
shared `__host__ __device__` core idiom (PATTERNS.md §2). So the two paths execute the
same operations in the same order; the only possible discrepancy is whether the host
compiler and `nvcc` contract a multiply-add into an FMA differently, which is bounded
by a few ULP per term.

**Tolerance:** GPU-vs-CPU **max relative error ≤ 1e-9** (we observe ~`7e-16`, i.e.
machine precision). Relative, because `I(q)` spans decades. This is the
"≈ machine precision for short double-precision computations" regime of PATTERNS.md §4.

**A second, stronger (science) check:** the demo recovers the Guinier `Rg` from the
*computed* curve and compares it to the synthetic structure's **true geometric `Rg`**
(13.815 Å recovered vs. 13.671 Å true). Agreement there validates that the whole
pipeline — Debye sum, Guinier fit — measures real geometry, not just that two code
paths agree. The reduced χ² ≈ 0.8 (near 1) confirms the model fits the noisy synthetic
"experiment" to within its error bars, as it must, since the data was generated from
the model plus 1% noise.

## 7. Where this sits in the real world

Our forward model is a **bare point-atom Debye sum**. Production SAXS tools add three
things that matter for fitting real data:

1. **`q`-dependent atomic form factors.** Real atoms are electron clouds, not points;
   `f_i(q)` falls off with `q` (Cromer–Mann Gaussians). We use a constant `f_i`.
2. **Excluded solvent.** The protein displaces buffer; the *contrast* (protein minus
   solvent electron density) is what scatters. CRYSOL/FOXS subtract a dummy-atom
   excluded-volume term, with a fitted parameter.
3. **The hydration shell.** A ~3 Å layer of ordered water around the protein is denser
   than bulk and contributes measurably; CRYSOL fits its contrast, WAXSiS models the
   water explicitly from MD.

Algorithmically, **CRYSOL** avoids the `O(N²)` sum by expanding the scattering
amplitude in **spherical harmonics** (`O(N · L²)` for a chosen `L`), which is faster
for large `N` and is where the catalog's "cuBLAS for spherical-harmonic coefficients"
would enter. **FOXS** keeps a Debye-style sum but with the solvent terms and clever
distance binning. **WAXSiS** computes scattering from explicit-solvent MD frames for
the highest accuracy. And the *modeling* layer — **EROS/BioEn**, maximum-entropy
reweighting — wraps this forward model in an optimizer that finds the **ensemble of
conformations** (not a single structure) whose averaged curve fits the data, the key
tool for flexible proteins and IDPs. Everything in this project is the innermost,
most-called kernel of that stack, written to be understood.

---

## References

- **P. Debye (1915)**, *Zerstreuung von Röntgenstrahlen* — the original derivation of
  the orientationally-averaged scattering sum.
- **Svergun, Barberato & Koch (1995)**, *CRYSOL* — the spherical-harmonic SAXS
  calculator with implicit solvent + hydration shell; the reference production method.
- **Schneidman-Duhovny et al. (2013)**, *FOXS* — fast Debye-based fitting; closest to
  this project's approach. <https://modbase.compbio.ucsf.edu/foxs/>
- **Chen & Hub (2015)**, *WAXSiS* — explicit-solvent, GPU-accelerated scattering from
  MD; gold standard for accuracy.
- **Bottaro & Lindorff-Larsen; Hummer & Köfinger (BioEn)** — maximum-entropy ensemble
  reweighting against SAXS, the modeling layer above the forward model.
- **SASBDB** — open repository of experimental SAXS/SANS data + models:
  <https://www.sasbdb.org>.
- **NVIDIA, "Fast N-Body Simulation with CUDA"** (GPU Gems 3, ch. 31) — the shared-
  memory tiling pattern for all-pairs sums referenced in §4's exercise.
