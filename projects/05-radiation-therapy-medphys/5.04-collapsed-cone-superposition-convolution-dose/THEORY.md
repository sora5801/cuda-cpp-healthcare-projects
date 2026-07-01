# THEORY — 5.4 Collapsed-Cone / Superposition-Convolution Dose

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

In external-beam radiotherapy a linear accelerator (linac) fires a beam of
high-energy (MV) photons into a patient to kill a tumour while sparing healthy
tissue. Planning that treatment requires knowing, in advance, **how much dose
(energy absorbed per unit mass, in gray, Gy = J/kg) lands in every cubic
millimetre of tissue.** That is the job of a *dose engine*.

Two physical facts make dose calculation hard — and make it worth a whole
algorithm family:

1. **Photons do not deposit dose where they interact.** When a photon interacts
   (Compton scatter, photoelectric, pair production), the energy is handed to fast
   secondary electrons (and scattered photons) that travel *millimetres to
   centimetres* before stopping. So the dose at a point is a **blurred, spread-out**
   version of where energy was released. The blurring kernel is the *dose-spread
   kernel* (or *point-spread / energy-deposition kernel*), precomputed by Monte
   Carlo simulation of a single interaction in water.

2. **The body is heterogeneous.** Lung (ρ ≈ 0.25 g/cm³), soft tissue (≈ 1.0), and
   bone (≈ 1.85) attenuate the beam and stop electrons very differently. A dose
   engine that ignores this — treating everything as water — is clinically wrong
   near lung/bone interfaces, exactly where many tumours sit.

**Superposition-convolution (SC)** dose calculation handles both: it convolves the
Monte-Carlo dose-spread kernel with the energy *released* in the patient, scaling
distances by density so the kernel stretches through lung and shrinks through bone.
**Collapsed-cone convolution (CCC)** is the fast, GPU-friendly discretization of
that convolution. This project builds a 2-D teaching version so you can *see* the
depth-dose curve reshape at a lung and a bone slab.

## 2. The math

**Inputs.** A density map ρ(**r**) on a voxel grid (water-relative, dimensionless),
a beam entering the top surface with primary fluence Ψ₀, and material parameters:
the mass-attenuation coefficient μ/ρ [cm²/g] and the cone kernel decay a [1/(g/cm²)].

**Stage 1 — TERMA.** *Total Energy Released per unit MAss.* As the primary beam
travels a path, its fluence attenuates by Beer-Lambert, but in a heterogeneous
medium distance is measured in **radiological path length** (density-weighted):

$$ d_\text{rad}(\mathbf r) = \int_0^{\mathbf r} \rho(\mathbf l)\, dl \quad [\text{g/cm}^2], \qquad
   \Psi(\mathbf r) = \Psi_0 \, e^{-(\mu/\rho)\, d_\text{rad}(\mathbf r)}. $$

TERMA is the energy those interactions release per unit mass:

$$ T(\mathbf r) = \frac{\mu}{\rho}\, \Psi(\mathbf r). $$

**Stage 2 — Superposition/convolution.** The dose is TERMA convolved with the
density-scaled dose-spread kernel h:

$$ D(\mathbf r) = \int T(\mathbf r')\; h\!\big(d_\text{rad}(\mathbf r' \!\to\! \mathbf r),\, \hat\Omega\big)\, d^3\mathbf r'. $$

Here the kernel argument is the **radiological** distance between release point **r′**
and deposition point **r** (heterogeneity correction), and h depends on direction
Ω̂ (kernels are forward-peaked).

**Collapsed-cone approximation.** Discretize h onto a small set of *cones* {Ω̂_c}.
Along a cone the point-kernel is taken as an exponential

$$ h_c(r) = a\, e^{-a\, r}, \qquad \int_0^\infty h_c(r)\,dr = 1, $$

so each cone conserves the energy it carries. Marching a ray one step of
radiological length `d_rad = Δℓ·ρ`, the fraction of the travelling dose deposited
is `(1 − e^{−a·d_rad})` and the rest transmits — the 1-D recurrence Stage 2
implements. Summing over cones reconstructs the full kernel (the "superposition").

**Fixed-point dose.** For deterministic GPU accumulation (see §5) each deposit is
quantized to an integer number of *dose-units*: `units = round(D_contrib · dose_scale)`.

## 3. The algorithm

```
STAGE 1  TERMA ray-trace           STAGE 2  Collapsed-cone superposition
for each beam column x:            for each source voxel s with TERMA T_s:
  rad = 0                            for each cone c in 0..n_cones:
  for y = 0..ny-1:                     carry = T_s / n_cones
    rad_center = rad + ½·ρ·Δ           walk ray from s along (dx_c,dy_c):
    T[y,x] = (μ/ρ)·Ψ0·exp(−μ/ρ·rad_c)   d_rad   = step_cm · ρ_here
    rad += ρ·Δ                           transmit= exp(−a·d_rad)
                                         deposit = carry·(1−transmit)
                                         carry  -= deposit
                                         dose_units[here] += round(deposit·scale)
                                         stop when carry·scale < ½ or off-grid
```

**Complexity.** Let N = nx·ny voxels, C = n_cones, L = max ray length (≈ nx+ny).
- Stage 1: O(N) — each voxel touched once.
- Stage 2 (serial): O(N · C · L). This dominates; it is the SC engine's cost.
- Parallel **work** is the same O(N·C·L), but the **depth** collapses: Stage 1 has
  depth O(ny) per independent column; Stage 2 has depth O(C·L) per independent
  source voxel, with N source voxels running concurrently.

**Data-access pattern.** Stage 1 is a strided march down columns (coalescing-
friendly if threads = adjacent columns). Stage 2 is a **scatter**: each source
voxel writes into many destinations along diagonal/axis rays — hence atomics.

## 4. The GPU mapping

Two kernels, each embarrassingly parallel (PATTERNS.md §1):

**Stage 1 `terma_kernel` — one thread per beam column.**
- Thread-to-data map: thread `t` owns column `x = beam_x0 + t`; it marches
  `y = 0..ny-1`, keeping a running `rad_above` in a **register**.
- Launch: `block = 256`, `grid = ceil(beam_width/256)`. Columns are disjoint, so
  **no atomics, no shared memory** — writes never collide.

**Stage 2 `ccc_kernel` — one thread per source voxel.**
- Thread-to-data map: thread `s` owns source voxel `(sx,sy) = (s%nx, s/nx)`.
- It loops over cones and marches each ray, doing `atomicAdd` into `dose_units`
  because different source voxels' rays overlap in the same destination.
- Launch: `block = 256`, `grid = ceil(nx·ny/256)`.

```
 grid of blocks (Stage 2)         one thread = one SOURCE voxel s
 +----+----+----+----+            s spreads along n_cones rays:
 | b0 | b1 | b2 | .. |                    ^   ^   ^
 +----+----+----+----+                     \  |  /
 | .. |    |    |    |               <----   [s]   ---->     (atomicAdd at each
 +----+----+----+----+                     /  |  \            voxel the ray hits)
   256 threads / block                    v   v   v
```

**Memory hierarchy.**
- **Registers:** the ray state (`carry`, `x`, `y`) — the hot loop is register-resident.
- **Global:** `rho`, `terma`, `dose_units`. Stage 2 re-reads `rho` along each ray;
  a natural optimization (left as an exercise) is to stage a **shared-memory density
  strip** so a block of source voxels reuses cached density — this is exactly the
  "shared memory for the density strip along the current cone ray" the catalog notes.
- **No CUDA library is used**: the kernels are custom. Where a production engine
  would FFT the convolution (cuFFT) for a *spatially-invariant* kernel, CCC keeps
  the convolution in real space precisely because the density scaling makes the
  kernel spatially *variant* — so we hand-roll the ray march. (If you wanted the
  homogeneous-water special case, that step is a cuFFT-based convolution; writing it
  by hand is the O(N·C·L) march here.)

**Occupancy / bandwidth.** Both kernels are simple and register-light, so occupancy
is high. On a tiny grid the launch overhead dominates (see §7). At clinical scale
Stage 2 is bandwidth-bound on the `rho`/`dose` traffic, which is what the shared-
memory strip optimization targets.

## 5. Numerical considerations

- **Precision.** TERMA and the cone recurrence run in **double precision** (FP64)
  on both host and device so the exponentials match bit-for-bit. The *shared*
  `__host__ __device__` header (ccc_physics.h, PATTERNS.md §2) guarantees the CPU
  and GPU evaluate the identical arithmetic.
- **The determinism problem.** Stage 2 is a scatter: many threads `atomicAdd` into
  the same dose voxel in a **nondeterministic order**. Floating-point addition is
  **not associative**, so a `float` atomicAdd would give a slightly different sum
  every launch and would never exactly equal the serial CPU sum.
- **The fix (fixed-point).** We quantize each deposit to an **integer** dose-unit
  and atomicAdd 64-bit integers. Integer addition **is** associative/commutative, so
  the tally is order-independent → **bit-reproducible run to run and identical to
  the CPU** (PATTERNS.md §3). CUDA's 64-bit `atomicAdd` takes `unsigned long long*`;
  all deposits are non-negative, so we accumulate unsigned and reinterpret into the
  signed host grid (the bit patterns match for non-negative values).
- **Edge cases handled:** the ragged last block (guarded), rays leaving the grid,
  and carry decaying below half a unit (early-out so we never add zeros forever).

## 6. How we verify correctness

`main.cu` runs the **CPU reference** (`reference_cpu.cpp`) and the **GPU** on the
same phantom and checks two things:

1. **Dose grid — exact integer equality.** Because both sides use fixed-point
   accumulation of the identical per-voxel physics, the two integer grids must match
   with **0 mismatches** (tolerance `== 0`, PATTERNS.md §4). This is the strongest
   possible check: not "close", but *identical*.
2. **TERMA — tiny FP tolerance (`1e-9`).** TERMA is double-precision and computed by
   the same code path, so it too is essentially bit-identical; the small tolerance
   only guards against a compiler contracting an FMA differently.

Why is CPU==GPU convincing? The reference is a plain serial loop written to be
*obviously* correct — no parallelism, no atomics, no cleverness. An independent
implementation (massively parallel, atomic scatter) reproducing it **exactly** is
strong evidence the GPU code has no race, no indexing bug, and no lost update. As a
science check, the depth-dose curve also shows the expected qualitative shape
(surface build-up, a lung dip, a bone pile-up).

## 7. Where this sits in the real world

Production photon dose engines share this skeleton but add scale and fidelity:

- **3-D and hundreds of cones.** Real CCC uses ~48–400 cones tessellating the
  sphere on 512³ voxels — where the GPU's ~10 min → <10 s win (catalog) appears.
  Our 2-D, 8-cone version is a teaching reduction.
- **Polyenergetic, depth-hardening kernels.** Kernels are Monte-Carlo dose-spread
  *arrays* (DSAs) that vary with the beam's spectrum and its "hardening" with depth;
  we use one analytic exponential per cone.
- **AAA (Anisotropic Analytical Algorithm)** — Varian Eclipse's pencil-beam-plus-
  scatter-kernel superposition — is a close cousin of what we build.
- **Acuros XB / linear Boltzmann transport** solves the transport equation
  deterministically on the grid — more accurate in heterogeneities than CCC, and a
  different algorithm entirely (a grid PDE, closer in spirit to project 5.6).
- **matRad, Plastimatch, CERR** (Prior art in the README) show the full data model:
  beam geometry, DICOM CT/RT structures, beamlet superposition, and plan optimization
  wrapped around the dose kernel.

The general **Siddon ray tracer** we specialized to vertical columns handles an
*oblique* ray by computing the parametric intersections of the ray with the voxel
grid planes and the intersection length in each voxel; that generalization (plus
beam divergence and penumbra) is what turns this teaching tracer into a real one.

---

## References

- **T. R. Mackie et al. (1985)** — the convolution/superposition dose model
  (TERMA ⊗ kernel). The origin of this whole family.
- **A. Ahnesjö (1989), "Collapsed cone convolution of radiant energy for photon
  dose calculation in heterogeneous media"** — the canonical CCC paper; the cone
  discretization and density scaling used here descend from it.
- **R. L. Siddon (1985), "Fast calculation of the exact radiological path…"** —
  the ray-voxel tracer specialized in Stage 1.
- **AAPM TG-105** — benchmarking guidance for convolution/superposition and
  Monte-Carlo dose engines; the reference test cases to compare against.
- **matRad** (<https://github.com/e0404/matRad>) — read its photon collapsed-cone
  engine to see cone kernels and beamlet superposition organized in real code.
- **Plastimatch** (<https://plastimatch.org/>) — ray-tracing and GPU dose components.
