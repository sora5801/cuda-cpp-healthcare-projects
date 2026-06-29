# THEORY — 6.04 Lattice-Boltzmann Blood/Airflow Solver

> For a reader who knows C++ but is new to CUDA and to computational fluid
> dynamics. See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

Blood through arteries and air through the bronchial tree are fluid flows
governed by the Navier-Stokes equations. Solving those PDEs directly in complex,
moving geometry is hard. The **Lattice-Boltzmann Method (LBM)** takes a different
route: it models the fluid as fictitious particles hopping on a regular grid,
recovering Navier-Stokes behaviour in the macroscopic limit. Its locality (each
node talks only to neighbours) makes it a natural fit for GPUs and the method of
choice for large-scale hemodynamics (HemeLB) and respiratory airflow.

## 2. The math

At each node we store **populations** `f_i(x, t)`, `i = 0..8` for D2Q9 — the
density of particles moving with discrete velocity `c_i`. The lattice-Boltzmann
equation with the **BGK** (single-relaxation) collision operator is

```
f_i(x + c_i, t+1) = f_i(x, t) - (1/τ)·( f_i(x, t) - f_i^eq(x, t) )
```

The left side is **streaming** (move to the neighbour); the right side is
**collision** (relax toward equilibrium). The equilibrium is a truncated
Maxwellian:

```
f_i^eq = w_i ρ [ 1 + 3(c_i·u) + 4.5(c_i·u)^2 − 1.5 u·u ]
```

with weights `w_i` (4/9, 1/9×4, 1/36×4) and lattice sound speed `c_s^2 = 1/3`.
Macroscopic density and velocity are moments: `ρ = Σ f_i`, `ρu = Σ c_i f_i`. A
Chapman-Enskog expansion shows this recovers Navier-Stokes with kinematic
viscosity `ν = c_s^2 (τ − 1/2) = (τ − 1/2)/3`. No-slip walls are imposed by
**bounce-back**: a population hitting a wall is reflected back the way it came.

## 3. The algorithm

```
initialize f_i = w_i      (rest, rho=1, u=0)
repeat steps:
    for each node:        # PARALLEL
        stream:  pull f_i from neighbour (x - c_i); bounce-back at walls
        moments: rho, u = sum f_i, sum c_i f_i / rho
        force:   shift equilibrium velocity by tau*gx (drives the flow)
        collide: f_i_new = f_i - (f_i - f_i^eq)/tau
    swap buffers
```

**Complexity.** Each step is `Θ(N_nodes · 9)` work — perfectly parallel across
nodes. Total cost is `steps × N_nodes`. Throughput is measured in **MLUPS**
(mega-lattice-updates per second), the standard LBM benchmark.

## 4. The GPU mapping

**Decomposition.** One thread per lattice node on a 2-D grid of 16×16 blocks. The
host runs the time loop, launching `lbm_step_kernel` once per step and
**ping-ponging** two device buffers: read `f_old`, write `f_new`, swap pointers.

**Why two buffers.** Streaming reads neighbours' *old* values. If we wrote in
place, a node might read a neighbour that was already updated this step — a race.
Double buffering makes every read come from the frozen previous state, so all
nodes are independent within a step (no atomics, no `__syncthreads`).

**Memory layout.** Populations are stored **structure-of-arrays** — all of
direction `i` contiguous — so a warp of threads (consecutive `x`) reads
contiguous memory for each direction: **coalesced** access, which matters because
LBM is memory-bandwidth bound. Production codes add **shared-memory tiling**: load
a tile plus a one-node halo into shared memory so the streaming neighbour reads
hit on-chip memory (Exercise 1).

**CPU/GPU parity.** The entire per-node update is one `__host__ __device__`
function (`lbm_collide_stream`), so the GPU and CPU execute the *identical*
double-precision arithmetic. The measured velocity fields here agree to ~`2e-16`
(machine epsilon) — essentially bit-identical.

## 5. Numerical considerations

- **Precision.** We use **double** precision: LBM populations are `O(1)` but the
  flow velocity is tiny (`~10^-3`), so the interesting physics lives in the
  low-order bits — single precision would lose it. (Production GPU LBM often uses
  clever single/mixed precision; that is an optimization, not the default lesson.)
- **Stability.** Requires `τ > 1/2` (positive viscosity) and low Mach number
  (`|u| ≪ c_s`); we keep the body force small so velocities stay `~10^-3`.
- **Determinism.** No reduction across nodes during a step, double-buffered reads
  ⇒ the result is reproducible and matches the CPU.

## 6. How we verify correctness

Two independent checks. (1) **Cross-implementation:** the GPU and CPU velocity
fields are compared node-by-node and agree to machine precision. (2) **Physics:**
the steady profile across the channel is the analytic **Poiseuille parabola** —
zero at the no-slip walls, maximum at the centerline — which is the textbook
solution for force-driven flow between plates. Matching a known analytic solution
is strong evidence the kinetic model and boundary conditions are right, not just
that two codes agree.

## 7. Where this sits in the real world

Production hemodynamics/airflow solvers (HemeLB, PALABOS, USERMESO) replace every
simplification here: **3-D** stencils (D3Q19/D3Q27), **multi-relaxation-time
(MRT)** or entropic collision for stability at physiological Reynolds numbers,
**sparse** representations of segmented vessel/airway geometry, the **immersed
boundary method** to couple deformable red-blood-cell membranes, and **Guo
forcing** for accurate body forces. The collide+stream stencil and the
node-per-thread GPU mapping you learn here are exactly their inner loop.

## References

- Krüger et al., *The Lattice Boltzmann Method: Principles and Practice* (2017) — the standard text.
- Bhatnagar, Gross & Krook (1954) — the BGK collision operator.
- Mazzeo & Coveney, **HemeLB** — LBM for sparse vascular geometries.
- NVIDIA CUDA C++ Programming Guide — 2-D grids, shared-memory tiling, coalescing.
