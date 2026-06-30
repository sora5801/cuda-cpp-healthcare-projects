# THEORY — 2.5 Coarse-Grained / MARTINI Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A protein or a lipid membrane is made of tens of thousands of atoms. An
**all-atom** molecular-dynamics (MD) simulation tracks every one, including
hydrogen atoms whose bonds vibrate on a ~1 femtosecond (10⁻¹⁵ s) timescale. To
resolve those vibrations the integration timestep must be ~2 fs, so reaching even
1 microsecond of biology takes ~500 million steps. Many interesting processes —
a lipid bilayer self-assembling, a membrane protein inserting, a vesicle budding,
a virus capsid breathing — happen on microseconds to milliseconds and are simply
out of reach.

**Coarse-graining** trades resolution for reach. The **MARTINI** force field (the
field's most widely used CG model) lumps roughly **four heavy atoms into one
"bead"** — for example, a water bead represents ~4 real water molecules, and a
lipid's tail becomes a chain of a few apolar beads. Two things speed up at once:

1. **Fewer particles.** ~4× fewer interaction sites means ~4× fewer (and, because
   pair work scales super-linearly, even fewer) non-bonded interactions.
2. **Smoother energy landscape.** Removing the stiff hydrogen vibrations lets the
   timestep grow to ~20–40 fs, and the effective dynamics are faster still.

Together these give MARTINI its famous **~100× reach** over all-atom MD. The
price is that you can no longer ask atom-level questions (a specific hydrogen
bond), but you *can* watch large-scale organization — and that is exactly what CG
models are for. The single most characteristic MARTINI behaviour is **demixing**:
apolar ("oil-like") beads and polar ("water-like") beads avoid each other, which
is what drives lipids to spontaneously form bilayers. This project reproduces
that demixing in miniature.

## 2. The math

We simulate `N` beads with positions `r_i ∈ ℝ³` and velocities `v_i ∈ ℝ³` in a
cubic box of edge `L` with **periodic boundary conditions**. Each bead has a type
`t_i ∈ {C, P}` (apolar / polar).

**Non-bonded potential.** Each pair `(i, j)` interacts through a **Lennard-Jones
(LJ)** potential:

```
U(r) = 4·ε_ab · [ (σ / r)¹² − (σ / r)⁶ ]      for r < r_cut,   else 0
```

- `r` = distance between the two beads (nm), under the **minimum-image
  convention** (interact with the nearest periodic copy).
- `σ` = contact distance (nm); here one value shared by all pairs.
- `ε_ab` = well depth for the type pair `(a, b)` — **the MARTINI interaction
  matrix**. We use `ε_CC = ε_PP = 4` (like-likes-like) and `ε_CP = 1` (weak
  cross-attraction). Because like pairs attract more strongly than unlike pairs,
  C and P **demix** — the toy model of oil/water immiscibility.
- `r_cut` = cutoff radius (nm); pairs beyond it are ignored.

The `r⁻¹²` term is a steep repulsive wall (excluded volume); the `r⁻⁶` term is the
softer attractive well.

**Force.** The force on bead `i` from bead `j` is `f = −∇U`, which along the
separation vector `r_ij = r_i − r_j` is

```
f_ij = (24·ε_ab / r²) · [ 2·(σ/r)¹² − (σ/r)⁶ ] · r_ij
```

and the **total** force on `i` is `F_i = Σ_{j≠i} f_ij`.

**Equations of motion.** Newton: `m · d²r_i/dt² = F_i`, integrated numerically
(next section). Symbols: `m` = bead mass, `dt` = timestep.

**Inputs:** initial `{r_i, v_i, t_i}`, the parameters `(L, dt, steps, r_cut, m, σ,
ε)`. **Outputs:** the final `{r_i, v_i}`, plus diagnostics — total energy
`E = Σ_i ½m|v_i|² + Σ_{i<j} U(r_ij)` and the C/P centroid separation.

## 3. The algorithm

**Velocity-Verlet** is the standard MD integrator: time-reversible and
*symplectic* (it conserves a shadow energy, so it does not drift like naive
Euler). One step advances `(r, v)` as:

```
(A) half-kick : v_i += ½ · (F_i / m) · dt          # using the OLD force
(B) drift     : r_i += v_i · dt                     # move at the new velocity
(C) forces    : recompute F_i at the new positions  # the O(N²) work
(D) half-kick : v_i += ½ · (F_i / m) · dt          # using the NEW force
```

Splitting the velocity kick into two halves around the position drift is what
makes the scheme second-order and stable.

**Complexity.**

| Quantity | Cost |
|---|---|
| Force evaluation (all-pairs) | `O(N²)` per step |
| Kick + drift | `O(N)` per step |
| Whole run | `O(steps · N²)` |

The `O(N²)` force sum dominates and is **embarrassingly parallel across beads**:
`F_i` depends only on reading all positions, never on another `F_j`. Production
codes cut the `O(N²)` to `O(N)` with a **neighbour/cell list** (only nearby beads
are within `r_cut`); we keep the all-pairs version because it is the clearest
thing to map to the GPU first (Exercise 2 adds the list).

**Data-access pattern.** The force kernel is read-heavy: each thread streams all
`N` positions and types from global memory (`O(N)` reads per thread, `O(N²)`
total) and writes a single force vector. Arithmetic intensity is moderate (a
handful of flops per loaded position), so for large `N` the kernel becomes
bandwidth-bound — which motivates the shared-memory tiling optimization.

## 4. The GPU mapping

**Thread-to-data mapping.** One **thread per bead**:

```
i = blockIdx.x * blockDim.x + threadIdx.x      // bead this thread owns
```

Thread `i` loops over all `j ≠ i`, accumulates `F_i` in registers, and writes
`force[i]`. Because each thread owns a distinct output slot, there are **no
atomics and no races** — the cleanest possible parallel reduction.

**Launch configuration.** Block of **256 threads** (a multiple of the 32-lane
warp; 8 warps to hide the long latency of the global-memory position loads),
grid `= ⌈N / 256⌉` blocks. The kick and drift kernels use the same shape.

**Why three kernels per step, not one.** Step (C) reads *all* beads' positions, so
every bead must finish its drift (B) before *any* force is recomputed. A single
kernel cannot synchronise across the whole grid mid-flight, but **a kernel
boundary is a global barrier**. So the host issues, per step:
`kick_drift_kernel → force_kernel → kick_kernel`. The launches are cheap next to
the `O(N²)` force kernel.

```
   per velocity-Verlet step
   ┌──────────────────────────────────────────────────────────┐
   │ kick_drift_kernel   (one thread/bead: half-kick + drift)  │
   │        │  [kernel boundary = global barrier]              │
   │ force_kernel        (one thread/bead: sum O(N) partners)  │  <-- the hot loop
   │        │  [kernel boundary = global barrier]              │
   │ kick_kernel         (one thread/bead: second half-kick)   │
   └──────────────────────────────────────────────────────────┘
```

**Memory hierarchy.**

- **Global memory** holds `pos[]`, `vel[]`, `type[]`, `force[]`. The force kernel
  reads all of `pos`/`type` (the bandwidth bottleneck for large `N`).
- **Registers** hold each thread's running force accumulator and its own bead's
  position — the inner loop touches no shared memory in this teaching version.
- **Shared memory (optimization, Exercise 3):** the classic *N-body tiling* stages
  a block of `B` partner positions into `__shared__` memory so the `B` threads in
  a block reuse each load `B` times — turning a bandwidth-bound kernel into a
  compute-bound one. Omitted here for clarity; this is exactly what cuda-samples'
  `nbody` does.

**Which CUDA library does what.** This teaching version uses **none** — the LJ
force is hand-written so nothing is a black box. The catalog lists **cuFFT for CG
PME**: real MARTINI adds long-range electrostatics via *Particle-Mesh Ewald*,
which spreads charges onto a grid, takes a 3-D FFT (this is the cuFFT call),
multiplies by a Green's function in reciprocal space, and inverse-FFTs back.
Writing that FFT by hand (a Cooley-Tukey radix-2/mixed-radix 3-D transform with
the right normalization and memory layout) is a project in itself — see flagship
`8.03` for a cuFFT walk-through.

## 5. Numerical considerations

- **Precision:** we use **FP64** (`double`) throughout. MD energy conservation is
  sensitive to round-off accumulated over thousands of force sums; double keeps
  the drift negligible and makes the CPU/GPU comparison meaningful. Production CG
  codes often run **mixed precision** (FP32 forces, FP64 accumulators) for speed —
  Exercise 5 explores the trade-off.
- **Stability:** velocity-Verlet is stable provided `dt` is small relative to the
  fastest oscillation. Too large a `dt` and the energy drifts upward and the
  simulation "blows up" — a hands-on lesson in Exercise 4.
- **Race conditions:** none. Each thread writes one `force[i]`; reads are all from
  read-only arrays. No `atomicAdd`, so no float-summation reordering.
- **Determinism:** this is the subtle part. Floating-point addition is **not
  associative**, so a force computed by summing partners in a different order
  gives a (very slightly) different result. We dodge this by having the CPU
  reference and every GPU thread sum over `j = 0, 1, …, N−1` in the **identical
  index order** (the shared `compute_force_on` in `martini.h`). The only residual
  difference is that the GPU **contracts** `a*b + c` into a single fused-multiply-
  add (FMA) instruction while the host compiler may not — a ~1 ulp effect per
  operation that accumulates to ~`1e-11` over 200 steps. We report that and verify
  to a looser-but-still-tiny tolerance rather than pretend the runs are bit-
  identical.

## 6. How we verify correctness

`src/reference_cpu.cpp` runs the **same** velocity-Verlet loop serially, calling
the **same** `martini.h` functions the GPU kernels call. From an identical initial
state, `src/main.cu` compares the final positions:

```
worst = max_i max_axis | r_i^CPU − r_i^GPU |   ≤   TOLERANCE = 1e-6
```

In practice `worst ≈ 1.5e-11` for the committed sample — far inside `1e-6`. The
tolerance is set to `1e-6` (not `0`) for the honest reason in §5: the GPU's FMA
contraction makes the trajectories agree to ~`1e-11`, not bit-for-bit, even in
double precision. (This matches PATTERNS.md §4: a small physical tolerance for a
many-step iterative solver.)

A **second, physics-level** check backs up the CPU==GPU agreement: the **total
energy is nearly conserved** (it changes by < 0.3% over the run, drifting slightly
*down* as the beads relax into a lower-energy demixed arrangement). A broken
integrator or a sign error in the force would show up as energy blowing up or
collapsing. Edge cases handled: the ragged last thread block (guarded by
`i >= n`), self-interaction (`j == i` skipped), and coincident/near-zero
separations (`r² < 1e-12` skipped to avoid divide-by-zero).

## 7. Where this sits in the real world

Production MARTINI runs in **GROMACS** and differs from this toy in every
direction that matters for accuracy and scale:

- **Full interaction matrix.** MARTINI 3 has ~20 bead types and ~800 calibrated
  interaction levels (we use 2 types and a 2×2 matrix). The levels are fit to
  reproduce experimental partitioning free energies.
- **Bonded terms.** Real beads are wired into molecules with bond, angle, and
  dihedral potentials; a lipid is a specific bonded topology, not loose beads. The
  **Gō-MARTINI / elastic-network** overlay adds harmonic springs between protein
  beads to preserve secondary/tertiary structure.
- **Electrostatics.** Charged beads interact through a (reaction-field or PME)
  Coulomb term; PME uses **cuFFT** on the GPU. We omit charges entirely.
- **Shifted potentials & neighbour lists.** MARTINI uses a smoothly **shifted**
  LJ/Coulomb so energy and force are continuous at `r_cut`, and a Verlet
  **neighbour list** to get `O(N)` scaling. We use a hard cutoff and all-pairs.
- **Thermostat/barostat.** Real runs hold temperature/pressure with a thermostat
  (v-rescale) and barostat (Parrinello-Rahman); ours is plain NVE (constant
  energy).
- **Backmapping.** CG configurations can be **backmapped** to atomistic detail for
  finer analysis — a whole pipeline this project does not attempt.

This project is therefore a **reduced-scope teaching version**: it nails the one
idea that makes CG-MD fast on a GPU — the embarrassingly parallel non-bonded pair
force — and points at the production machinery above for everything else.

---

## References

- **MARTINI 3** — Souza et al., *Nat. Methods* 18, 382 (2021). The current force
  field; defines the bead types and interaction matrix this toy reduces.
- **GROMACS** (<https://github.com/gromacs/gromacs>) — production GPU CG-MD;
  study its Verlet neighbour-list and non-bonded GPU kernels.
- **cgmartini.nl** — the official MARTINI parameter repository and tutorials.
- **`insane.py`** (<https://github.com/Tsjerk/Insane>) / **TS2CG**
  (<https://github.com/weria-pezeshkian/TS2CG>) — how real CG membranes are built.
- **Frenkel & Smit, *Understanding Molecular Simulation*** — the canonical text on
  velocity-Verlet, periodic boundaries, and the minimum-image convention.
- **NVIDIA cuda-samples `nbody`** — the textbook shared-memory tiling for the
  all-pairs force (Exercise 3).
