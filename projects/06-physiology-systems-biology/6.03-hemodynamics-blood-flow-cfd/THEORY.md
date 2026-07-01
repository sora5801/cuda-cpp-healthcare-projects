# THEORY — 6.3 Hemodynamics / Blood-Flow CFD

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope teaching
> version** (CLAUDE.md §13); §7 explains what the full research-grade version
> adds._

---

## 1. The science

Blood is a fluid, and where it flows matters medically. In arteries, the force
the moving blood exerts *tangentially* on the vessel wall — the **wall shear
stress (WSS)** — regulates the health of the endothelial cells lining the vessel.
Regions of **low or oscillating WSS** (e.g. the outer walls of bifurcations, the
inner curve of the aortic arch, downstream of a stenosis) are where
**atherosclerotic plaque** preferentially forms. Computing the velocity field and
its near-wall gradient therefore predicts *where a vessel is at risk* — the core
promise of patient-specific hemodynamics.

Two features make real blood flow hard:

1. **Blood is non-Newtonian.** Its viscosity is not constant: red blood cells
   aggregate at low shear (thick blood) and disaggregate at high shear (thin
   blood) — **shear thinning**, captured by the Carreau-Yasuda model.
2. **Vessels are compliant.** Arterial walls flex with each heartbeat, so the
   fluid and the solid wall are coupled (**fluid-structure interaction, FSI**),
   and the fluid domain itself moves (ALE — arbitrary Lagrangian-Eulerian).

This project models the fluid mechanics (incompressible Navier-Stokes) and the
non-Newtonian rheology (Carreau-Yasuda), on a **rigid straight channel** (no FSI),
and computes the velocity profile and WSS — the clinically-relevant output — in a
setting simple enough to verify against an exact solution.

## 2. The math

**Governing equations.** For an incompressible Newtonian/​generalized-Newtonian
fluid of constant density ρ and velocity **u** = (u, v):

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}
   = -\frac{1}{\rho}\nabla p + \nabla\!\cdot(\nu\,\nabla\mathbf{u}) + \mathbf{g},
\qquad \nabla\cdot\mathbf{u} = 0 .
$$

| Symbol | Meaning | Units (SI / demo) |
|---|---|---|
| **u**=(u,v) | velocity | m/s / (dimensionless) |
| p | pressure (÷ρ absorbed in projection) | Pa / — |
| ρ | density (blood ≈ 1060) | kg/m³ / 1 |
| ν | kinematic viscosity (may depend on shear rate) | m²/s / 0.1 |
| **g**=(gₓ,0) | body force = driving pressure gradient | m/s² / 1e-4 |
| ∇·u=0 | incompressibility (no compression) | — |

**Non-Newtonian viscosity (Carreau-Yasuda).** With shear rate
$\dot\gamma$ (the magnitude of the strain-rate tensor):

$$
\nu(\dot\gamma) = \nu_\infty + (\nu_0-\nu_\infty)\,
   \bigl[\,1+(\lambda\dot\gamma)^{a}\,\bigr]^{(n-1)/a}.
$$

Setting ν₀ = ν∞ recovers a **Newtonian** (constant-ν) fluid — the mode the demo
runs, because only then does an exact solution exist to check against.

**The exact check (Poiseuille flow).** A Newtonian fluid driven by a uniform body
force gₓ between two no-slip walls a distance H apart reaches a steady **parabolic**
profile:

$$
u(y) = \frac{g_x}{2\nu}\Bigl[\bigl(\tfrac{H}{2}\bigr)^2-\bigl(y-\tfrac{H}{2}\bigr)^2\Bigr],
\qquad u_{\max}=\frac{g_x H^2}{8\nu}.
$$

The **wall shear stress** is $\tau_w=\mu\,\frac{du}{dy}\big|_{\text{wall}}
=\rho\,g_x\,H/2$ (μ=ρν is the dynamic viscosity). These closed forms are our
science-level ground truth (PATTERNS.md §4).

## 3. The algorithm

We use **Chorin's fractional-step (projection) method** — the classic way to
enforce incompressibility. Per time step Δt:

1. **Predictor** — advance velocity ignoring pressure:
   $\mathbf{u}^{*}=\mathbf{u}^{n}+\Delta t\,[-(\mathbf u\cdot\nabla)\mathbf u
   +\nabla\!\cdot(\nu\nabla\mathbf u)+\mathbf g]$.
   Advection uses **first-order upwind** (stable at the low Reynolds numbers of
   small-vessel flow); diffusion uses the **5-point Laplacian**; ν comes from
   Carreau-Yasuda at the local shear rate.
2. **Pressure Poisson** — u* is not divergence-free. Solve
   $\nabla^2 p=\frac{\rho}{\Delta t}\,\nabla\!\cdot\mathbf u^{*}$ so that the
   correction removes exactly that divergence.
3. **Corrector / projection** —
   $\mathbf u^{n+1}=\mathbf u^{*}-\frac{\Delta t}{\rho}\nabla p$, which is now
   divergence-free.

We solve step 2 with **Jacobi iteration** (a fixed number `p_iters` of sweeps of
$p\leftarrow\frac{1}{4}(p_W+p_E+p_S+p_N-h^2\,\text{rhs})$), the simplest
relaxation for the discrete Poisson equation.

**Complexity.** Let N = nx·ny cells and T = steps.
- **Serial cost:** O(T · p_iters · N) — the Jacobi sweeps dominate.
- **Parallel work/depth:** each sweep is O(N) work at O(1) depth (all cells
  independent), so the *depth* is O(T · p_iters) with unlimited processors.
- **Data-access pattern:** every sub-step reads only the 4 nearest neighbours — a
  **stencil**. Arithmetic intensity is low (a handful of flops per cell per memory
  load), so the solver is **memory-bandwidth bound**, the norm for CFD.

## 4. The GPU mapping

The whole method is a sequence of stencils, so the mapping is the canonical
**"one thread per grid cell"** (PATTERNS.md §1, exemplar 6.04 lattice-Boltzmann):

- **Thread-to-data map:** thread (x,y) = (blockIdx·blockDim + threadIdx) owns cell
  (x,y). Four kernels — `predictor`, `divergence`, `pressure` (one Jacobi sweep),
  `corrector` — each apply one shared per-cell function from `nse_channel.h`.
- **Launch config:** 16×16 = 256 threads per block (a multiple of the 32-lane
  warp, enough warps to hide global-memory latency on sm_75..sm_89); the grid is
  `ceil(nx/16) × ceil(ny/16)` blocks; edge threads guard `if (x>=nx||y>=ny)`.
- **Ping-pong buffers:** velocity swaps u↔u_new and the Jacobi solve swaps the two
  pressure buffers between sweeps, so every kernel reads a *frozen* "old" buffer
  and writes a fresh one — no read/write races, no atomics.
- **Memory hierarchy:** all fields live in **global memory**; the row-major layout
  (`idx(x,y)=y·nx+x`) makes adjacent threads read adjacent addresses
  (**coalesced**). A production kernel would stage each tile's halo in **shared
  memory** to cut redundant global reads — the classic next optimization.

```
   grid of 16x16 blocks over the nx x ny cells        one time step =
   +----+----+----+                                   predictor  -> u*
   | B  | B  | B  |   each block: 256 threads          divergence -> rhs
   +----+----+----+   thread (x,y) updates cell (x,y)  pressure   -> p  (x p_iters)
   | B  | B  | B  |   reads N,S,E,W neighbours          corrector  -> u^{n+1}
   +----+----+----+   (a 5-point stencil)               then swap buffers
```

**Which CUDA library does what.** This teaching version deliberately uses **no**
library for the pressure solve — Jacobi is hand-written so the learner sees the
stencil. Production codes replace it with **AmgX** (GPU algebraic multigrid) or a
**cuSPARSE**-based Krylov solver, because Jacobi converges slowly (its error
decays like the spectral radius per sweep). Writing multigrid by hand means
restriction/prolongation operators and a hierarchy of coarse grids — a large
project in itself, which is exactly why the library exists.

## 5. Numerical considerations

- **Precision: FP64 throughout.** The projection method accumulates small
  divergences; double precision keeps the pressure solve well-conditioned and lets
  the CPU and GPU match closely. FP32 would drift visibly over 40000 steps.
- **Stability.** The explicit diffusion term needs
  $\Delta t < h^2/(4\nu)$ (2-D); `make_synthetic.py` asserts this. Upwind
  advection adds numerical diffusion but is unconditionally stable for our low
  velocities — a deliberate teaching trade-off (accuracy for robustness).
- **Determinism.** Every kernel writes only its own cell; there are **no atomics**
  and no floating-point reductions whose order could vary, so the GPU result is
  bit-reproducible run to run. That is why stdout can be diffed (PATTERNS.md §3).
- **CPU/GPU divergence.** Both paths call the *same* FP64 functions in
  `nse_channel.h`, but the GPU may fuse multiply-adds (FMA) where the host
  compiler does not. Over a long iterative solve this can drift by ~1e-12…1e-6
  (PATTERNS.md §4). On this problem they happen to stay **bit-identical**
  (max diff = 0), so a 1e-9 tolerance is comfortably safe and honest.

## 6. How we verify correctness

Two independent checks:

1. **CPU vs GPU (implementation check).** `reference_cpu.cpp` runs the identical
   fractional-step scheme serially. `main.cu` compares the two velocity fields
   cell-by-cell; the max absolute difference must be ≤ **1e-9** (it is 0 here).
   Agreement between an obviously-correct serial loop and the parallel kernel is
   strong evidence the GPU code is right.
2. **Simulation vs analytic Poiseuille (science check).** The printed centreline
   `u_max` is compared to the exact $g_xH^2/8\nu$. At 40000 steps it is within
   ~4.8%, and the profile is a symmetric parabola — the physics is right, and the
   residual gap is finite-time-to-steady-state plus discretization error (running
   to 80000 steps closes it to <0.3%).

## 7. Where this sits in the real world

| Aspect | This teaching version | Production (SimVascular/svFSI, OpenFOAM, HemeLB) |
|---|---|---|
| Geometry | 2-D rigid straight channel | 3-D patient-specific vessel from CT/MRI angiography |
| Mesh | uniform structured grid | unstructured polyhedral finite-volume mesh |
| Walls | rigid no-slip | **compliant** (FSI + ALE mesh motion, RBF morphing) |
| Rheology | Carreau-Yasuda (Newtonian in demo) | Carreau-Yasuda / Casson, full non-Newtonian |
| Pressure solve | hand-written Jacobi | **algebraic multigrid** (AmgX), cuSPARSE SpMV |
| Time | fixed body force → steady flow | pulsatile inlet over the cardiac cycle (~1000 steps), OSI |
| Scale | one small GPU | domain decomposition, MPI+CUDA, NCCL halo exchange |

The clinically-used outputs are **time-averaged WSS (TAWSS)** and the
**oscillatory shear index (OSI)** over a full heartbeat — we compute steady WSS
only. HemeLB takes a different route entirely: a sparse lattice-Boltzmann method
over the vessel voxels (the same stencil idea as flagship 6.04), avoiding an
explicit pressure solve.

---

## References

- **SimVascular / svFSI** — <https://github.com/SimVascular/svFSI>. The
  image-to-simulation pipeline; study how it turns a segmented geometry into a
  finite-element FSI solve.
- **OpenFOAM** — <https://github.com/OpenFOAM/OpenFOAM-dev>. Canonical
  finite-volume CFD; read `icoFoam`/`pimpleFoam` for the PISO/SIMPLE pressure-
  velocity coupling this project's projection step approximates.
- **HemeLB** — <https://github.com/hemelb-codes/hemelb>. Sparse lattice-Boltzmann
  hemodynamics; the alternative to a pressure-Poisson solver.
- **AmgX** — NVIDIA's GPU algebraic-multigrid library; what replaces the Jacobi
  loop at scale.
- Chorin, A. J. (1968), *Numerical solution of the Navier-Stokes equations* — the
  original fractional-step method implemented here.
