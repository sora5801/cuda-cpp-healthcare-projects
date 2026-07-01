# THEORY — 6.18 ECG Forward Problem & Body-Surface Potential Mapping

> Read this alongside the code. The reading order is `src/main.cu` →
> `src/kernels.cuh` → `src/kernels.cu` → `src/reference_cpu.cpp`, with the shared
> physics in `src/ecg_core.h`. This document explains *why* those files do what
> they do, from the biology down to the CUDA launch configuration.

---

## The science

An electrocardiogram (ECG/EKG) records tiny voltages on the skin that are produced
by the heart's electrical activity. During each heartbeat, a wave of
**depolarization** sweeps through cardiac muscle. At the cellular level this is a
flow of ionic current across cell membranes; at the tissue level it looks like a
distribution of **current sources and sinks** moving through the heart.

The body's tissues (blood, muscle, lung, fat) conduct electricity, so those
cardiac currents set up an electric potential field throughout the torso. Where
that field reaches the skin, an electrode measures it — that is a "lead" of the
ECG. The **forward problem** asks: *given the heart's electrical sources, what
potentials appear on the body surface?* (The much harder **inverse problem** goes
the other way — from surface measurements back to cardiac sources — and is used
in electrocardiographic imaging; see *Where this sits in the real world*.)

The forward problem is the workhorse behind:

- simulating what a 12-lead ECG *should* look like for a given heart model,
- building the "lead-field" that the inverse problem needs,
- studying how body shape, electrode placement, and tissue conductivity change the
  recorded signal.

We model the heart's activity with a small number of **equivalent current
dipoles** — the classic simplification that a localized current source-sink pair
behaves, from a distance, like a single dipole. Each dipole has a fixed position
and orientation and a **time-varying strength** that traces out the activation
sequence. This is exactly the "equivalent dipole" / multipole source
representation the catalog names.

---

## The math

### Quasi-static Poisson equation

At ECG frequencies (~1 Hz to ~1 kHz) the electromagnetic wavelength is hundreds of
kilometres, so we drop induction and displacement currents: the problem is
**quasi-static**. Charge conservation in an ohmic medium of conductivity
`σ(**r**)` then gives Poisson's equation for the potential `φ`:

```
∇ · ( σ(r) ∇φ(r) )  =  - ∇ · J_source(r)          (in the torso volume)
      n · ( σ ∇φ )   =  0                            (insulating body surface)
```

where `J_source` is the impressed cardiac current density. This is a linear
elliptic boundary-value problem (BVP): **linear** in the sources, which is the key
fact that makes the whole method a matrix multiply.

### The dipole Green's function (our closed-form kernel)

For a **homogeneous, unbounded** conductor of conductivity `σ`, the potential at a
field point **r** due to a single current dipole of moment **p** (units A·m) at
position **r₀** is the closed-form Green's function

```
              1        p · (r - r₀)
φ(r)  =  ---------- · ----------------
           4 π σ         |r - r₀|³
```

This is the physically exact forward map for an infinite homogeneous medium and is
the didactic heart of every ECG forward model. It is implemented once, in
`ecg::dipole_potential` (`src/ecg_core.h`), and shared by the CPU and GPU.

### From physics to a matrix

Write each dipole `s` as `p_s(t) = x_s(t) · d̂_s` — a fixed **unit direction** `d̂_s`
times a scalar **strength** `x_s(t)`. Superposition (the problem is linear) gives
the potential at electrode `e`:

```
φ_e(t) = Σ_s A[e][s] · x_s(t) ,   A[e][s] = (1/4πσ) · d̂_s · (r_e - r₀_s) / |r_e - r₀_s|³
```

The matrix **A** (`L × S`) is the **lead-field / transfer matrix**: entry `A[e][s]`
is the potential electrode `e` records per unit strength of source `s`. It depends
only on geometry, so it is computed **once**. Stacking all `T` time frames into a
source matrix **X** (`S × T`) turns the whole forward problem into a single dense
matrix product:

```
Φ  =  A · X            (Φ is L × T:  body-surface potentials over time)
```

That is the entire computation: **build A once, then apply Φ = A·X**.

---

## The algorithm

1. **Load** the torso/heart model: `L` electrode positions, `S` dipole positions
   and directions, and the `S × T` strength time series `X`.
2. **Build the lead field** `A` (`L × S`): for every `(e, s)`, evaluate the dipole
   Green's function. Work: `O(L · S)`.
3. **Apply the forward map**: `Φ = A · X` (`L × T`). Work: `O(L · S · T)` — this is
   the dominant cost, and the step that recurs at *every* time step of an EP
   simulation, so it is what we hand to a tuned GEMM.
4. **Summarize**: per-lead peak-to-peak swing, the largest-swing lead, one
   signature value.

### Serial vs. parallel complexity

| Step | Serial work | Parallelism available |
|------|-------------|-----------------------|
| Build `A` | `O(L·S)` | fully independent per entry → `L·S`-way |
| Apply `Φ = A·X` | `O(L·S·T)` | dense GEMM → `L·T` outputs, each a length-`S` dot product |

Both steps are embarrassingly (or at least abundantly) parallel, which is why the
GPU is a natural fit — and why cuBLAS, whose entire job is fast GEMM, is the right
tool for step 3.

---

## The GPU mapping

### Step 2 — building `A` (a custom kernel)

`build_lead_field_kernel` (in `src/kernels.cu`) uses the **independent-jobs**
pattern (PATTERNS.md §1): every entry of `A` is a self-contained evaluation, so we
launch a **2-D grid over the `L × S` output**:

- **block**: `16 × 16 = 256` threads — a multiple of the 32-lane warp, good
  occupancy on `sm_75…sm_89`.
- **grid**: `ceil(S/16) × ceil(L/16)` blocks to cover the matrix; ragged edges are
  guarded with `if (e >= L || s >= S) return;`.
- **thread → data**: thread `(col = s, row = e)` owns `A[e·S + s]`.
- **memory**: each thread reads one electrode `Vec3` and one source
  `Vec3`/direction from global memory and writes one `double`. It is a light,
  bandwidth-friendly kernel; the heavy lifting is the GEMM that follows.

### Step 3 — applying `Φ = A·X` (cuBLAS DGEMM)

We do **not** hand-roll the matrix multiply — that is a solved problem, and
CLAUDE.md §6.1.6 says to use the library but explain it. `cublasDgemm` computes
`C = α·op(A)·op(B) + β·C` in double precision.

The one subtlety is **storage order**. cuBLAS is **column-major**; our matrices are
**row-major**. The no-copy trick: a row-major `[m×n]` buffer is, byte-for-byte, the
column-major `[n×m]` buffer (its transpose). So, as cuBLAS sees them:

```
our A  (row-major L×S)  ==  column-major S×L   =: Ac
our X  (row-major S×T)  ==  column-major T×S   =: Xc
our Φ  (row-major L×T)  ==  column-major T×L   =: Φc
```

Transposing the target identity `Φ = A·X` gives `Φᵀ = Xᵀ·Aᵀ`, i.e. in
column-major storage `Φc[T×L] = Xc[T×S] · Ac[S×L]`. That is a plain DGEMM with no
transposes:

```
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m = T, n = L, k = S,
            α = 1,
            Xc, lda = T,     // our X, row-major S×T
            Ac, ldb = S,     // our A, row-major L×S
            β = 0,
            Φc, ldc = T);    // our Φ, row-major L×T
```

Because `Φc` *is* our row-major `Φ` in memory, the result copies straight back with
no rearrangement. Hand-rolling this would mean a tiled, register-blocked
shared-memory GEMM with conflict-free loads and epilogue handling — cuBLAS already
does all of that, tuned per architecture.

### Memory hierarchy notes

- The lead-field kernel is **global-memory bound** but tiny; there is no reuse to
  exploit with shared memory at this size.
- The GEMM's data reuse (each `A` row and `X` column is read many times) is exactly
  what cuBLAS captures with shared-memory/register tiling internally. On large `S`
  and `T`, that reuse is where the GPU's advantage comes from.

---

## Numerical considerations

- **Precision.** Everything is `double` (FP64). ECG potentials span several orders
  of magnitude (near vs. far sources via the `1/dist³` falloff), so FP64 keeps the
  small far-field contributions meaningful.
- **The softening `EPS`.** `dipole_potential` clamps the distance to `1e-9 m` so a
  source coincident with an electrode cannot divide by zero. With our geometry
  (heart strictly inside the torso) this never triggers, but it keeps the result
  finite and identical on CPU and GPU.
- **Determinism & the summation-order caveat.** The lead field is built by the same
  formula in the same order on both sides, so `A_gpu == A_cpu` to ~machine epsilon.
  The **matrix multiply is different**: cuBLAS sums each length-`S` dot product in a
  different order than the CPU triple loop and uses fused multiply-add. Because
  floating-point addition is **not associative**, `Φ_gpu` and `Φ_cpu` agree only to
  ~`1e-12` relative — a real, teachable effect (PATTERNS.md §4). We keep the printed
  result deterministic by taking all reported numbers from the (verified) GPU path
  and printing a fixed number of digits.
- **No atomics.** Neither step reduces across threads with `atomicAdd`, so there is
  no float-atomic nondeterminism to worry about here.

---

## How we verify correctness

`main.cu` runs the CPU reference and the GPU path and compares:

1. **Lead field:** worst entrywise `|A_gpu − A_cpu|` over the `L×S` matrix, against
   `LEAD_TOL = 1e-12` (both sides use the identical shared formula).
2. **Potentials:** worst entrywise `|Φ_gpu − Φ_cpu|` over the `L×T` matrix, against
   `PHI_TOL = 1e-9` (the GEMM summation-order tolerance above; still far below any
   physical signal, since potentials are `O(1e-2…1e0)`).

The demo also checks a **physical** invariant independent of CPU==GPU agreement:
the electrode nearest the strongest, most-swinging source must record the largest
peak-to-peak deflection. The synthetic sample is engineered so this ground truth is
electrode 0; the program reports `RECOVERED` when the computed answer matches
(PATTERNS.md §6). Passing both the numeric check *and* the physical recovery is the
Definition-of-Done for this project's science.

Edge cases guarded: malformed/short input files throw with a clear message; a
degenerate zero-direction source is handled by `ecg::normalize`; the distance
softening prevents singularities.

---

## Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13). Real ECG
forward solvers differ in three big ways:

1. **Bounded, inhomogeneous torso.** A real torso is finite and made of tissues with
   very different conductivities (blood ~0.7, lung ~0.05, skeletal muscle
   anisotropic, …). The closed-form infinite-homogeneous Green's function is
   replaced by a numerical solve of the Poisson BVP on a torso mesh — the **finite
   element method (FEM)** or **boundary element method (BEM)** the catalog lists.
   Each electrode's column of the lead field then comes from solving one BVP (the
   reciprocity / "one BVP per electrode" idea). That is exactly the step cuSOLVER /
   batched cuSOLVER would accelerate in a full implementation; here it collapses to
   the analytic kernel so the project runs offline in milliseconds.
2. **Realistic cardiac sources.** Instead of a handful of dipoles, production tools
   (openCARP, Cardioid/LLNL) drive the forward map with transmembrane potentials
   from a full electrophysiology (EP) simulation on a heart mesh, using the
   bidomain/monodomain equations.
3. **The inverse problem.** Electrocardiographic imaging *inverts* `Φ = A·X` to
   recover cardiac sources from measured body-surface maps. That is ill-posed and
   needs regularization — **Tikhonov**, **total variation**, etc. (the catalog's
   inverse-problem algorithms). The forward operator `A` we build here is precisely
   the matrix those inverse methods regularize and invert.

Even in the full pipeline, the **`Φ = A·X` apply step is a dense matrix-vector /
matrix-matrix product evaluated at every time step**, so the cuBLAS DGEMM shown
here is not a toy shortcut — it is the actual hot loop, just with a much bigger `A`
that came from an FEM/BEM solve instead of a formula.

---

## References to study (not to copy)

- **SCIRun / ECGSim (Utah)** — interactive ECG forward/inverse toolkits; the EDGAR
  database ships torso geometries and body-surface potential maps.
- **openCARP** — open cardiac EP simulator with ECG lead post-processing.
- **Cardioid (LLNL)** — HPC cardiac simulator including an ECG forward module.
- Plonsey & Barr, *Bioelectricity: A Quantitative Approach* — the volume-conductor
  and dipole-source theory in full.
