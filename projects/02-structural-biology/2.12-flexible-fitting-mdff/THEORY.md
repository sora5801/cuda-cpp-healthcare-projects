# THEORY — 2.12 Flexible Fitting / MDFF

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Cryo-electron microscopy (cryo-EM)** flash-freezes a purified molecular complex
— a ribosome, a virus capsid, a membrane channel — in a thin film of vitreous ice
and images thousands of randomly-oriented copies. Averaging those projections
reconstructs a 3-D **density map** `ρ(r)`: a scalar field over space whose value is
roughly "how much electron scattering happened here", i.e. where the atoms are. A
good map shows the backbone and bulky side chains; a 3–5 Å map is fuzzy at the
atomic scale.

But a density map is **not a structure**. To do biology you need the actual atomic
coordinates: which residue is where, which loop moved, which pocket opened. The
density tells you *roughly* where atoms are; you must build a chemically-sensible
all-atom model that *explains* that density.

If you already have a related model — say the same protein in a different
conformational state, or a homolog, or an AlphaFold prediction — you do not start
from scratch. You **flexibly fit** the existing model into the new map: deform it
until its atoms sit on the density ridges, while keeping bonds, angles, and
non-bonded contacts physically valid. That is **MDFF (Molecular Dynamics Flexible
Fitting)**, introduced by Trabuco, Villa, Schulten et al. (2008) and the workhorse
behind countless ribosome and virus structures.

The key idea: treat the density map as an **external potential** that an MD
simulation feels, on top of the normal molecular-mechanics force field. Atoms are
pulled toward high density; the force field stops the structure from distorting
unphysically. Run MD, and the model relaxes into the map.

## 2. The math

**Inputs**
- A density map `ρ : ℝ³ → ℝ`, sampled on a regular grid of `nx·ny·nz` voxels with
  isotropic spacing `vox` (Å/voxel). A world point `p` maps to grid coordinates
  `g = p / vox`.
- A starting model: atom positions `x₁ … x_N ∈ ℝ³`.

**The MDFF potential.** Each atom `a` feels an external potential that is *low*
where density is *high* (so the gradient points uphill in density). The standard
MDFF form is `U_MDFF(x) = −w · Φ(ρ(x))`, where `Φ` is a monotonic transfer (often
the identity or a clamped/normalised density) and `w > 0` is a coupling weight. The
**density-derived force** on atom `a` is

```
F_dens(a) = −∇ U_MDFF(x_a) = w · Φ'(ρ) · ∇ρ(x_a).
```

With `Φ = identity` this is simply `F_dens(a) = w · ∇ρ(x_a)` — push the atom along
the spatial gradient of the density, toward denser regions. `∇ρ` is the
**cross-correlation gradient**: it is exactly the derivative of the model-to-map
overlap with respect to the atom's position.

**The restraint / force field.** Left alone, the density force would drag every
atom into the single densest voxel. Real MDFF prevents this with the full MM force
field `U_FF` (bonds, angles, dihedrals, Lennard-Jones, electrostatics) plus
secondary-structure restraints. In this teaching version we replace `U_FF` with one
**harmonic restraint** to a reference position `x_ref`:

```
U_rest(x) = ½ k Σ_a |x_a − x_ref,a|²     ⇒     F_rest(a) = −k (x_a − x_ref,a).
```

**The objective** is the **cross-correlation** `CC` between the model's simulated
density and the map; maximising `CC` is what the density force does. We report a
simple proxy `CC ≈ (1/N) Σ_a ρ(x_a)` — the mean density at the atoms (higher =
better fit). RMSD-to-target (when a ground truth exists) is the other yardstick.

**Symbol table**

| Symbol | Meaning | Units |
|---|---|---|
| `ρ(r)` | density map (scalar field) | arbitrary density |
| `∇ρ` | spatial gradient of density | density / length |
| `w` (`w_dens`) | density-force coupling weight | force / (density/length) |
| `k` (`k_rest`) | restraint stiffness | force / length |
| `x_a`, `x_ref,a` | atom / restraint-anchor position | length (Å, "world units") |
| `step` | overdamped SD step size | length·time / force |
| `vox` | voxel spacing | length / voxel |

## 3. The algorithm

We integrate the atoms under `F = F_dens + F_rest` by **overdamped steepest
descent** (a first-order, friction-dominated proxy for MD): `x ← x + step · F`.
One fitting iteration:

```
for each atom a (in parallel):
    g       = grad_rho(x_a)                  # trilinear interpolation + finite diff
    F_dens  = w_dens * g
    F_rest  = -k_rest * (x_a - x_ref,a)
    x_new_a = x_a + step * (F_dens + F_rest)
swap(x, x_new)                               # Jacobi double-buffer
```

repeated `iters` times.

**Trilinear interpolation** (`sample_density`). To read `ρ` at an arbitrary point
`p` we find the enclosing voxel cube and blend its 8 corners by the fractional
offsets `(fx,fy,fz) = frac(p/vox)`:

```
ρ(p) = Σ_{i,j,k∈{0,1}}  c_{ijk} · (fx if i else 1−fx)(fy if j else 1−fy)(fz if k else 1−fz)
```

This is the same gather-and-blend used by CT backprojection (4.01) and by GPU
texture units. Nearest-voxel sampling would give a piecewise-constant field with
zero gradient almost everywhere — useless for a force — so interpolation is
essential.

**Gradient** (`sample_gradient`). A symmetric finite difference through the
interpolated field, with a half-voxel probe `h = vox/2`:
`∂ρ/∂x ≈ [ρ(p+h·x̂) − ρ(p−h·x̂)] / (2h)`, and similarly for `y`, `z`. (An analytic
trilinear gradient is exact and faster — see Exercise 1 — but the finite difference
shares one code path and is the most transparent.)

**Complexity.** Let `N` = atoms, `T` = iterations.
- Per atom per iteration: trilinear sampling is `O(1)` (8 voxel reads); the
  gradient does 6 samples = `O(1)`. So one iteration is `O(N)` and the whole fit is
  `O(N·T)`. The density rasterisation done once up front is `O(nx·ny·nz·N)` in this
  naive teaching version (real tools use a per-atom cutoff box → `O(N)`).
- **Serial:** the CPU loops over `N` atoms, `T` times: `O(N·T)`, fully sequential.
- **Parallel:** each iteration's `N` atom updates are independent → **work `O(N·T)`,
  depth `O(T)`** (the iterations are inherently sequential; the atoms within an
  iteration are not). Arithmetic intensity is low (a handful of FLOPs per voxel
  read), so the force evaluation is **memory-bound** on the density map.

## 4. The GPU mapping

**Thread-to-data.** One thread owns one atom for the whole run:

```
atom index  i = blockIdx.x * blockDim.x + threadIdx.x
```

Each iteration, thread `i` reads `x_old[i]` and `x_ref[i]`, samples the (shared,
read-only) density and its gradient, and writes `x_new[i]`. Because every thread
reads only from `x_old` and writes a distinct `x_new[i]`, there are **no data
races and no atomics** — this is a **Jacobi** update with **ping-pong** buffers,
the same structure as soft-tissue PBD (10.02) and SEIR ensembles (9.02).

**Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp;
8 warps to hide the density-map memory latency), `grid = ceil(N / 256)` blocks. The
last block is ragged, guarded by `if (i >= natoms) return;`.

**Memory hierarchy.**
- **Global memory** holds the density map `rho[nx·ny·nz]` and the position arrays.
  The density is uploaded **once** and read by every thread on every iteration —
  ideal because it is large and never changes. Its 8-voxel trilinear gathers hit
  the L2 cache well when neighbouring atoms are near each other in space.
- **Registers** hold `MdffParams` (passed by value), the atom's position, and the
  accumulating force — all per-thread scalars, no spilling.
- **No shared memory** is needed here: atoms do not share neighbours the way a
  stencil does. (A production engine that also computes bonded forces *would* tile
  bonded neighbour lists into shared memory.)
- **Texture memory** is the natural next step (Exercise 3): binding `rho` to a 3-D
  CUDA texture lets the hardware do trilinear filtering in one instruction — this is
  precisely how NAMD's GridForce reads the map.

```
                density map (global, read-only, uploaded once)
                          rho[nx*ny*nz]
                               │  trilinear gather (8 voxels) + ∇
                               ▼
 atoms:   x_old[0] x_old[1] x_old[2] ...        x_old[N-1]
            │        │        │                    │
          thread0  thread1  thread2     ...     thread N-1      (one atom each)
            │        │        │                    │
            ▼        ▼        ▼                    ▼
          x_new[0] x_new[1] x_new[2] ...        x_new[N-1]
                               │
                          swap(x_old, x_new)  ── repeat `iters` times
```

**Which CUDA library does what.** This reduced-scope version uses **only the CUDA
runtime** (`cudart`) — the trilinear sampler and integrator are hand-written so
nothing is a black box. Production MDFF additionally uses **cuFFT**: the map–model
cross-correlation and its gradient can be computed in **reciprocal space** as a
multiplication of Fourier transforms (the convolution theorem), which is faster than
real-space correlation for large maps. Writing that by hand means implementing a 3-D
FFT — exactly why one reaches for cuFFT (see project 8.03 for the cuFFT idiom).

## 5. Numerical considerations

- **Precision: FP64 (double).** The fit is a 200-iteration descent; we want the
  CPU and GPU to agree to many digits, and double keeps the trilinear interpolation
  and the finite-difference gradient stable (a half-voxel probe in FP32 loses
  significance near flat regions of the map). Real MDFF runs in FP32 mixed
  precision for speed; we trade that for didactic clarity and tight verification.
- **No atomics, deterministic.** Each `x_new[i]` is written by exactly one thread,
  so there is no floating-point summation order to depend on. Within numerical
  noise the GPU result is reproducible run-to-run, which is why **stdout is
  byte-stable** (the timing, which is not, goes to stderr).
- **FMA divergence.** Even in double precision, the GPU contracts `a*b + c` into a
  single fused-multiply-add with one rounding, while the host compiler may use two
  roundings. Over 200 iterations this makes the GPU and CPU trajectories drift by
  ~`1e-6` — small, real, and worth understanding (the same lesson as projects 10.02
  and 14.02).
- **Stability.** `step` must be small enough that an atom does not overshoot the
  blob (`step · w_dens · max|∇ρ| ≲ vox`); too large a step oscillates or diverges.
  The restraint `k_rest` damps drift toward the global maximum.

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp::fit_cpu`) runs the **identical math** as
the kernel — both call `mdff_step_atom()` from `mdff.h`, a `__host__ __device__`
function — with the same Jacobi double-buffering. Two independent code paths (one
plain serial C++ compiled by cl.exe, one CUDA compiled by nvcc) producing the same
fitted coordinates is strong evidence the GPU implementation is correct.

- **Tolerance: `1e-4`** on final atom positions of magnitude ~10 (so ~5–6 significant
  figures). This is *not* bit-exact: it accounts for the FMA drift in §5. We verify
  to a physically-negligible distance and **say so** rather than pretending the
  results are bit-identical (PATTERNS.md §4, the long-iterative-solver case).
- **A stronger, scientific check:** the synthetic problem embeds a **ground-truth
  target** (the atoms that generated the density). We report **RMSD-to-target**
  before and after — it must *drop* — and the **cross-correlation** before and
  after — it must *rise*. That validates that the fit recovers the right structure,
  not merely that CPU == GPU.
- **Edge cases:** the trilinear sampler clamps indices to the map edge (boundary
  atoms read edge voxels instead of reading out of bounds); the ragged last thread
  block is guarded; a length mismatch makes RMSD/`worst_atom_diff` return `+∞`.

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version**. Production MDFF differs
in several big ways:

- **Real MD, not steepest descent.** NAMD/OpenMM integrate Newtonian dynamics with
  a Langevin thermostat and the full CHARMM/AMBER force field (bonds, angles,
  dihedrals, impropers, Lennard-Jones, PME electrostatics). Our single harmonic
  restraint stands in for *all* of that. The density force is added as an external
  "GridForce" potential on top.
- **Secondary-structure & symmetry restraints** keep helices/sheets and chirality
  intact during large deformations — without them, fitting tears the model.
- **cuFFT cross-correlation.** Large-map correlation and its gradient are computed
  in reciprocal space (convolution theorem), not by per-atom real-space sampling.
- **Resolution-aware density simulation.** The model's density is simulated at the
  map's resolution (Gaussian blur ~ `sigma(resolution)`), and the objective is the
  normalised map–map cross-correlation, not the mean density at atoms.
- **Scale.** The whole point of the GPU is ribosomes and capsids: 10⁵–10⁶ atoms,
  thousands of steps — where one-thread-per-atom force evaluation is the difference
  between minutes and days. `phenix.real_space_refine` and **cryo-EM-aware** Rosetta
  attack the same objective with different optimisers; Coot is the interactive,
  human-in-the-loop counterpart.

The piece you *do* learn here transfers directly: the **per-atom trilinear gather of
the density gradient** is, line for line, what NAMD's CUDA GridForce kernel computes.

---

## References

- Trabuco, Villa, Mitra, Frank, Schulten, *"Flexible fitting of atomic structures
  into electron microscopy maps using molecular dynamics"*, **Structure** 16 (2008)
  673–683 — the original MDFF method and the density-derived force.
- Trabuco, Villa, Schreiner, Harrison, Schulten, *"Molecular dynamics flexible
  fitting: a practical guide…"*, **Methods** 49 (2009) — the VMD/NAMD workflow.
- **NAMD MDFF** docs (<https://www.ks.uiuc.edu/Research/namd/>) — the production CUDA
  GridForce implementation; study how the map becomes an external potential.
- **VMD MDFF plugin** (<https://www.ks.uiuc.edu/Research/vmd/>) — "mdff sim" is the
  Gaussian-blob density simulation we mimic; "mdff setup" prepares the restraints.
- **phenix.real_space_refine** (<https://phenix-online.org>) — an alternative,
  gradient-based optimiser for the same real-space density-fit objective.
- For the cuFFT-based reciprocal-space correlation, see project **8.03** in this
  repo for the cuFFT R2C idiom.
