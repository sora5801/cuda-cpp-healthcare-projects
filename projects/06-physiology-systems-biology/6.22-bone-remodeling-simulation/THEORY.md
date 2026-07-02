# THEORY — 6.22 Bone Remodeling Simulation

> A reduced-scope, didactic model of **mechano-regulated bone remodeling**: how a
> patch of bone senses the mechanical load it carries and adds or removes material
> to adapt. We keep the remodeling *biology* (the mechanostat / SED rule) and
> replace the production **finite-element stress solve** with a cheap,
> physically-motivated **stencil** proxy so the whole thing is one clean GPU
> pattern a learner can follow end to end. See §7 for exactly what we simplified.
>
> _Educational only — not for clinical use._

---

## 1. The science

Bone is a *living* tissue in perpetual turnover. Two cell types run a coupled
feedback loop:

- **Osteoclasts** dissolve ("resorb") old or under-used bone.
- **Osteoblasts** lay down ("form") new bone where it is needed.

Their balance is governed by the mechanical environment. **Wolff's law** (Julius
Wolff, 1892) observed that the internal trabecular architecture of bone aligns
with the principal stress trajectories — the porous lattice inside a femoral head
or a vertebral body is not random, it is an *optimized truss* for the loads that
body actually experiences. Harold Frost's **"mechanostat"** (1987) formalized the
control law: each site senses a mechanical stimulus and compares it to a
homeostatic setpoint, with a **"lazy zone"** (dead band) in between where nothing
happens. Rik Huiskes and colleagues (with Prendergast) cast the stimulus as
**strain energy density (SED)** and built the SED remodeling rule that underlies
most computational bone models. At the molecular level the resorb/form balance is
tuned by **RANKL/OPG** signaling, but the *mechanical* control loop above is what
carves the visible architecture, and it is what we model here.

Why does this matter? Disuse (bed rest, spaceflight) → under-loaded bone →
resorption → osteoporosis. A hip implant that shields the surrounding bone from
load → "stress shielding" → the bone around the implant wastes away. Simulating
remodeling lets engineers predict these outcomes and design implants that keep
bone healthily loaded.

**Not for clinical use.** This is an educational, qualitative model on synthetic
data (CLAUDE.md §1).

---

## 2. The math

We work on a 2-D voxel grid, `nx` columns × `ny` rows, row-major. Each voxel
`(x,y)` has a **relative density** `rho(x,y) ∈ [rho_min, 1]` (rho_min ≈ marrow/void,
1 = fully mineralized). Row `y=0` is the loaded top surface; row `ny-1` is the
supported base.

**1. Mechanical stimulus field `S`.** In reality `S` comes from a strain field,
which comes from solving linear elasticity `K u = f` on the voxel mesh — a big
sparse linear system. Here we use a **transport proxy**: stimulus is injected at
the loaded footprint and *conducts* through the tissue preferentially along stiff
(dense) material, reaching a steady state that solves a density-weighted Poisson-
like equation. Discretely, the steady `S` satisfies, at every voxel, the local
balance

```
S(x,y) = [ q(x,y) + Σ_n c_n · S(neighbour_n) ] / [ 1 + Σ_n c_n ]
```

where

- `q(x,y) = load` if `(x,y)` is under the loaded footprint (`y=0`,
  `load_x0 ≤ x ≤ load_x1`), else `0` — the source term;
- the sum runs over the in-grid **von-Neumann neighbours** (±x, ±y);
- the **conductance** `c_n = ½(rho(x,y) + rho(neighbour_n))` is the mean density of
  the pair, so a void voxel (`rho ≈ rho_min`) barely conducts and a dense strut
  conducts well;
- the `+1` in the denominator is a small self-damping term (a leak to ground) that
  keeps `S` bounded even inside a void and makes the fixed-point iteration
  strictly contractive.

This is exactly one **Jacobi relaxation step** of the linear system
`(1 + Σc)·S = q + Σ c·S_neighbour`. Because each update is a *convex combination*
(a weighted average whose weights sum to `Σc/(1+Σc) < 1`), the iteration is
unconditionally stable and converges to the unique steady field.

**2. The remodeling signal.** For a fixed load, softer (less dense) bone strains
more, so the biologically relevant stimulus is the **SED-per-unit-mass**

```
phi(x,y) = S(x,y) / rho(x,y).
```

**3. The mechanostat (Frost dead band / Huiskes SED rule).** With setpoint `k` and
lazy-zone half-width `w`:

```
        ⎧ rho + rate·(phi − (k+w))     if phi > k+w    (over-loaded → FORM)
rho' =  ⎨ rho − rate·((k−w) − phi)     if phi < k−w    (under-loaded → RESORB)
        ⎩ rho                          otherwise        (lazy zone → homeostasis)
```

followed by the clamp `rho' ← min(1, max(rho_min, rho'))`. The lazy zone is what
gives bone its hysteresis: without it the tissue would chatter around the setpoint
forever. The clamp keeps `rho` physical and keeps `phi = S/rho` finite.

**Inputs:** grid size, iteration counts, load + footprint, `k`, `w`, `rate`,
`rho_min`, `rho_init`. **Output:** the remodeled density field `rho` and the last
settled stimulus field `S`. **Objective:** find the density distribution the
feedback loop settles into (a load-adapted architecture).

The simulation alternates: **settle `S` for the current `rho`**, then **apply the
mechanostat to update `rho`**, and repeat for `remodel_steps` iterations.

---

## 3. The algorithm

```
initialize rho = rho_init everywhere,  S = 0
for step in 1 .. remodel_steps:                     # outer "months"
    for it in 1 .. relax_iters:                     # settle the stimulus field
        for every voxel (x,y):                      #   (Jacobi sweep, ping-pong)
            S_new(x,y) = relax(x, y, load, footprint, S_old, rho)
        swap(S_old, S_new)
    for every voxel (x,y):                           # apply the mechanostat
        rho_new(x,y) = mechanostat(x, y, k, w, rate, rho_min, S, rho)
    swap(rho, rho_new)
report total mass, per-column mass, mechanostat-state histogram
```

**Complexity.** Let `N = nx·ny` voxels. Each Jacobi sweep is `O(N)` (a fixed 5-point
stencil per voxel); each remodeling step does `relax_iters` sweeps plus one `O(N)`
density update, so the whole run is
`O(remodel_steps · relax_iters · N)`. Serially this is one thread doing all of it;
in parallel every voxel is independent within a sweep, so the **span** (critical
path) per sweep is `O(1)` given `N` processors — the parallel-friendly structure
the GPU exploits. **Arithmetic intensity** is low (a handful of FLOPs per voxel per
sweep against 5 global-memory reads), so the kernels are **bandwidth-bound**.

---

## 4. The GPU mapping

This is the classic **stencil + ping-pong** pattern (PATTERNS.md §1, "grid PDE /
nearest-neighbour update"), the same shape as flagships **6.04** (lattice-Boltzmann)
and **14.02** (reaction-diffusion).

- **Thread ↔ voxel.** We launch a 2-D grid of `16×16`-thread blocks; thread
  `(x,y) = (blockIdx.x·blockDim.x+threadIdx.x, blockIdx.y·blockDim.y+threadIdx.y)`
  owns voxel `(x,y)`. A boundary guard `if (x>=nx || y>=ny) return;` handles the
  ragged edge tiles.
- **Two kernels, mirroring the two physics functions** (`src/kernels.cu`):
  `relax_kernel` does one Jacobi sweep of `S`; `remodel_kernel` does one density
  update. Both are thin wrappers whose body is a single call into the shared
  `__host__ __device__` physics in `src/bone_remodel.h`.
- **Ping-pong buffers, no races.** Within a sweep each thread writes only *its own*
  `S_new(x,y)` and reads neighbours from a *separate* read-only `S_old` buffer, so
  there are **no data races and no atomics** — we just swap the two device pointers
  between sweeps. Density uses its own second buffer the same way. The host drives
  the two nested loops and launches one kernel per sweep/step.
- **Memory hierarchy.** Row-major `S`/`rho` live in **global memory**; the `+x`
  neighbour of consecutive threads is the consecutive address, so reads are
  **coalesced**. Each voxel's own values sit in **registers**. This teaching version
  does *not* tile the halo into **shared memory** — a deliberate simplification and
  an exercise below; because the stencil is bandwidth-bound, shared-memory halo
  tiling is the first real optimization.
- **Occupancy.** 256 threads/block (16×16) is a multiple of the 32-lane warp and
  gives the scheduler 8 warps/block to hide global-memory latency, with many blocks
  resident on sm_75..sm_89.
- **No CUDA library needed.** The production pipeline would use **cuSPARSE**
  (assemble the sparse stiffness matrix `K`) and **cuSOLVER/PCG** (solve `K u = f`);
  our proxy replaces that whole solve with the custom stencil kernel, so this
  teaching version links only the CUDA runtime. §7 explains exactly what those
  libraries would do and what hand-rolling them entails.

```
   voxel grid (nx x ny), row-major            2-D block grid of 16x16 threads
   x -->                                      +--------+--------+  ...
 y  [ (0,0) (1,0) (2,0) ... (nx-1,0) ]        | block  | block  |
 |  [ (0,1) (1,1) (2,1) ...          ]  --->   |(0,0)   |(1,0)   |  each thread
 v  [  ...                            ]        | 16x16  | 16x16  |  == one voxel
    [ (0,ny-1) ...      (nx-1,ny-1)  ]        +--------+--------+  ...
     row 0 = loaded top; row ny-1 = base       thread(x,y) owns voxel(x,y)
```

---

## 5. Numerical considerations

- **Precision.** Everything is `double` (FP64). Density and stimulus span a modest
  dynamic range, so FP64 is comfortably accurate; we use it mainly so CPU↔GPU
  agreement is tight and easy to reason about.
- **Stability.** The Jacobi update is a damped convex combination (weights sum to
  `Σc/(1+Σc) < 1`), so `S` can never blow up regardless of `relax_iters`. The
  density clamp bounds `rho ∈ [rho_min, 1]`, and `rho_min > 0` guarantees the
  `phi = S/rho` division is always finite.
- **Determinism.** No atomics and no reductions with nondeterministic order: every
  voxel's new value is a pure function of the previous buffers, so the result is
  **bit-identical run to run** and independent of block scheduling. The report's
  histogram uses **integer** counts. Consequently `stdout` is byte-for-byte stable
  (diffed by the demo); timings go to `stderr` (PATTERNS.md §3).
- **CPU/GPU parity.** The per-voxel math lives in **one** `__host__ __device__`
  header (`bone_remodel.h`), included by both the host reference and the kernels,
  so both run the identical sequence of arithmetic (PATTERNS.md §2). The host
  reference's ping-pong bookkeeping — including the parity-dependent "copy the fresh
  buffer back into `Sa`" after an odd number of sweeps — is mirrored exactly in the
  GPU wrapper so the two paths never diverge in *which* values they combine.

---

## 6. How we verify correctness

1. **GPU vs. CPU.** `main.cu` runs both `bone_cpu()` and `bone_gpu()` on the same
   parameters and takes the max absolute difference over the final density field.
   Observed: `~1.1e-16` (a couple of ULPs), verified against a small **physical**
   tolerance `1e-9`. Why not exactly zero? Even in double precision the GPU may
   contract a multiply-add into a single **FMA** where the host compiler emits
   separate ops; over `60 × 80` sweeps that few-ULP difference is nudged by the
   nonlinear (clamped, dead-band) map. We verify to a physically-negligible
   tolerance and say so, rather than pretend bit-identity (PATTERNS.md §4, the
   "long iterative solver" case). An independent serial implementation agreeing
   with the parallel one to machine precision is strong evidence both encode the
   same math and the parallelization introduced no bug.
2. **A known-answer sanity check.** The synthetic job loads *only* the center of the
   top edge. Wolff's law predicts bone should concentrate under that footprint and
   waste away on the flanks. The reported **per-column mass profile does exactly
   that** — it peaks at the load center (~3.31) and falls to the `rho_min` floor
   (`0.8 = 16·0.05`) on the flanks — so the result recovers the qualitative
   physics, not just CPU==GPU agreement.
3. **Edge cases.** `load_bone` rejects a truncated file, a footprint outside the
   grid, `rho_min ≤ 0`, and `rho_init` outside `[rho_min, 1]`, so the `S/rho`
   division and the grid indexing are always well-defined.

---

## 7. Where this sits in the real world

The honest simplifications (CLAUDE.md §13), and how production tools differ:

- **The stress field.** We replace the linear-elastic **finite-element solve**
  `K u = f` with a density-weighted diffusion of an applied load. Real voxel-FEM
  bone codes (e.g. **FEBio**, ETH's **ParOSol / VoxFEM**) assemble the sparse
  stiffness matrix `K` from per-voxel hexahedral elements (**cuSPARSE** for the
  structured-sparse assembly) and solve for the displacement `u` with a
  preconditioned conjugate gradient (**cuSOLVER / custom PCG**) *every remodeling
  step*. The true SED is then `½ σ:ε` computed from that `u`. To hand-roll it you
  would build the 8-node element stiffness matrix, scatter-add it into a global
  sparse `K` (respecting the shared-node race with atomics or coloring), apply
  boundary conditions, and iterate CG with a Jacobi/multigrid preconditioner. That
  solve is a whole project on its own; our proxy captures the *feedback behavior*
  (load flows through stiff paths; struts thicken, unloaded tissue resorbs) without
  it.
- **Dimensionality & scale.** Production runs are 3-D at µCT resolution (10–50 µm →
  ~10⁸ voxels), which is precisely where GPU per-voxel parallelism is essential
  (see PATTERNS.md §7 on the honest-timing rule — our tiny grid is launch-bound and
  the GPU is *slower* than the CPU here). Our 2-D 24×16 demo is a legibility choice.
- **Biology.** Real models add the **RANKL/OPG** signaling ODEs, cell-population
  dynamics, mineralization lag, and anisotropic fabric tensors; some use **cellular
  automata** for the osteoclast/osteoblast fronts. We fold all of that into a single
  scalar mechanostat rule.
- **Homogenization & topology optimization.** Apparent stiffness of a trabecular
  region is obtained by **homogenization**; and the same SED-driven "add material
  where stressed, remove where idle" logic is **SIMP** topology optimization, used
  to design lightweight implants and 3-D-printed prosthetics — bone remodeling and
  structural optimization are mathematically siblings.

---

## References

- **Wolff, J. (1892).** *Das Gesetz der Transformation der Knochen.* The original
  observation that bone architecture follows stress trajectories.
- **Frost, H.M. (1987).** *Bone "mass" and the "mechanostat".* The dead-band control
  law we implement.
- **Huiskes, R. et al. (1987, 2000).** SED-based remodeling and simulation of
  trabecular self-organization — the model family this project distills.
- **FEBio** — <https://github.com/febiosoftware/FEBio>: production nonlinear FEM for
  bone & cartilage; study its element assembly and solver structure.
- **ParOSol / VoxFEM** (ETH Zürich): parallel/GPU voxel FEM for trabecular bone —
  the real version of the stress solve we proxy.
- **FreeFEM** — <https://freefem.org>: general PDE/FEM solver adaptable to remodeling.
- **OpenFOAM** — <https://github.com/OpenFOAM/OpenFOAM-dev>: poroelastic
  fluid–structure bone modeling.
