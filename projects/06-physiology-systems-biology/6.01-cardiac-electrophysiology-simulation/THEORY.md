# THEORY — 6.1 Cardiac Electrophysiology Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Every heartbeat is triggered by **electricity**. Cardiac muscle cells
(cardiomyocytes) are *excitable*: at rest their membrane sits at a negative
voltage, but if a neighbouring cell depolarises them past a **threshold**, they
fire a stereotyped spike called an **action potential** — a fast upstroke, a long
plateau, then a slow recovery back to rest. During recovery the cell is
**refractory**: it cannot fire again for a while. Because adjacent cells are
electrically coupled (through gap junctions), a firing cell drags its neighbours
past threshold, and the action potential **propagates** as a travelling wave.
That wave is what an ECG measures, and what makes the muscle contract in a
coordinated squeeze.

The clinically important questions are about *propagation*: How fast does the wave
travel (**conduction velocity**)? Can it get stuck circulating around scar tissue
(**re-entry**, the mechanism of tachycardia and fibrillation)? Where should a
pacemaker or ablation lesion go? Answering these requires simulating the wave over
realistic tissue — the subject of computational cardiac electrophysiology (EP).

This project simulates the wave on a 2-D sheet using the simplest model that still
shows all the essential physics.

## 2. The math

**Monodomain reaction-diffusion PDE.** Let `V(x,y,t)` be the transmembrane voltage
over a tissue domain. The monodomain equation is

```
  ∂V/∂t = D ∇²V  −  I_ion(V, w)
  ∂w/∂t = f(V, w)
```

- `∇²V = ∂²V/∂x² + ∂²V/∂y²` is the **Laplacian** — the *diffusion* term. It models
  electrotonic coupling: current flows from depolarised cells to their resting
  neighbours, so `V` spreads out. `D` is the diffusion coefficient (units
  length²/time); it sets the conduction velocity.
- `I_ion(V, w)` is the local **ionic current** — the *reaction* term. It is what
  makes each cell excitable. `w` is a **recovery** (or "gating") variable that
  turns the cell off and enforces refractoriness.

**The FitzHugh-Nagumo (FHN) cell model.** Real ionic models (ten Tusscher-Panfilov,
O'Hara-Rudy) express `I_ion` through 50–200 coupled ODEs for individual ion-channel
gates. FHN is the canonical *2-variable* reduction that keeps the qualitative
behaviour:

```
  I_ion(V, w) = V (V − a)(V − 1) + w          (cubic, N-shaped)
  f(V, w)     = ε (V − b w)                    (slow linear recovery)
```

Symbols (all dimensionless in FHN):

| symbol | meaning | role |
|---|---|---|
| `V` | transmembrane voltage | 0 ≈ rest, 1 ≈ fully depolarised |
| `w` | recovery variable | rises during the AP, forces return to rest |
| `a` | excitation threshold, `0<a<1` | cell fires only if `V>a` |
| `ε` | recovery time-scale (small) | small ε ⇒ slow recovery ⇒ long AP |
| `b` | recovery coupling | tunes refractoriness / return to rest |
| `D` | diffusion coefficient | sets conduction velocity |

The cubic `V(V−a)(V−1)` is the heart of excitability: for `V<a` it pushes `V` back
to 0 (rest is stable), but once `V>a` it pushes `V` toward 1 (the upstroke). `w`
then grows and drags the cell back down — a self-terminating spike.

**Inputs:** grid size, `dt`, `dx`, `D`, `(a, ε, b)`, and an S1 stimulus patch that
sets `V=stim_v` at `t=0`. **Output:** the voltage field `V(x,y)` after `steps`
iterations — a snapshot of where the wave is.

## 3. The algorithm

**Operator splitting (Godunov).** The PDE has two very different pieces: a stiff
*pointwise* reaction ODE and a *spatial* diffusion term. Splitting advances them
**separately** each timestep, which is both simpler and lets each piece use the
integrator that suits it:

```
for each timestep:
    (A) REACTION : for every cell,  (V,w) ← integrate the ODE for dt   (no neighbours)
    (B) DIFFUSION: for every cell,  V     ← V + dt·D·∇²V                (5-point stencil)
```

**(A) Reaction** — forward Euler on FHN, per cell:

```
  I_ion = V(V−a)(V−1) + w
  V ← V + dt·(−I_ion)
  w ← w + dt·ε(V − b w)
```

**(B) Diffusion** — explicit forward Euler with the standard **5-point Laplacian**:

```
  ∇²V ≈ (V_left + V_right + V_up + V_down − 4·V_center) / dx²
```

with **no-flux (Neumann)** boundaries: an off-grid neighbour is mirrored to the
centre value, so no current leaves the tissue edge (an insulated heart).

**Complexity.** Grid of `N = nx·ny` cells, `S` timesteps. Each step is `O(N)` work
(both half-steps are `O(1)` per cell), so the serial cost is `O(S·N)` time,
`O(N)` space. The *work* is `O(S·N)` and the *depth* (critical path) is `O(S)` —
within a step all `N` cells are independent, so a parallel machine collapses the
`N` factor. Arithmetic intensity is low (a handful of flops per cell per step
against several memory accesses), so the computation is **memory-bandwidth bound** —
exactly the regime GPUs excel at.

## 4. The GPU mapping

**Thread-to-data.** One thread per grid cell, laid out on a 2-D grid of `16×16`
blocks. Thread `(x,y) = (blockIdx*blockDim + threadIdx)` owns cell `(x,y)` and its
row-major index `i = y·nx + x`. Two kernels run per timestep:

- `react_kernel` — thread `(x,y)` updates *its own* `V[i], w[i]` in place. No
  neighbour access ⇒ no shared memory, no atomics, no races.
- `diffuse_kernel` — thread `(x,y)` reads its 4 neighbours from a **read-only**
  input buffer `V_in` and writes to a **separate** output buffer `V_out`.

**Ping-pong buffers.** Because diffusion reads neighbours, it must not overwrite a
value another thread still needs. We keep two voltage buffers and swap them each
step (`src → dst`, then `swap`). A thread thus always reads the *previous* step's
field and writes the *next* one — deterministic and race-free. This is the same
stencil + ping-pong idiom as the `6.04` lattice-Boltzmann and `14.02`
reaction-diffusion flagships (docs/PATTERNS.md §1).

```
   grid of cells (nx × ny)              blocks tile the grid (16×16 threads each)
   +----+----+----+----+                +---------+---------+
   | c00| c10| c20| .. |                | block   | block   |   thread (tx,ty)
   +----+----+----+----+                | (0,0)   | (1,0)   |     owns one cell
   | c01| c11| c21| .. |    ------>      +---------+---------+     i = y*nx + x
   +----+----+----+----+                | block   | block   |
   | .. | .. | .. | .. |                | (0,1)   | (1,1)   |
   +----+----+----+----+                +---------+---------+

   per step:   react_kernel(V,w)  then  diffuse_kernel(V_in=src → V_out=dst); swap
```

**Memory hierarchy.** Buffers live in **global memory**. The row-major layout makes
a warp of threads with consecutive `x` read consecutive addresses ⇒ **coalesced**
loads. The reaction kernel keeps `V,w` in **registers** during the update. This
teaching version does *not* tile the stencil into **shared memory**; a production
kernel would stage each block's tile (plus a halo) into shared memory so the 4
neighbour reads hit on-chip memory instead of global — a standard optimization
left as an exercise. Occupancy: `16×16 = 256` threads/block is a multiple of the
32-lane warp and gives the scheduler enough warps to hide global-memory latency.

**No CUDA library here (and why).** The explicit stencil needs no library — it is a
handful of global-memory reads per cell. The catalog's `cuSPARSE`/`cuSOLVER`
targets belong to the *implicit* diffusion route (§7): there, one timestep is a
sparse linear solve `(I − dt·D·L) Vⁿ⁺¹ = Vⁿ`, where `L` is the Laplacian matrix.
That is `SpMV` (cuSPARSE) inside a conjugate-gradient loop (hand-rolled or
cuSOLVER). We keep the explicit version because it is the clearest first encounter
with the same physics, and it makes the GPU↔CPU verification exact.

## 5. Numerical considerations

**Precision.** We use **FP64 (double)** throughout. Cardiac reaction-diffusion is
sensitive: the cubic nonlinearity amplifies small errors, and thousands of steps
accumulate them. FP64 keeps the CPU and GPU close and the wave well-behaved.

**Stability (CFL).** Explicit forward-Euler diffusion in 2-D is *conditionally*
stable — it only converges if

```
  dt ≤ dx² / (4·D)              (the CFL / von Neumann stability bound)
```

The loader (`load_monodomain`) *refuses* a sample that violates this, rather than
emit exploding numbers. This constraint is precisely why production solvers prefer
*implicit* diffusion, which is unconditionally stable (§7). `cfl_limit()` in
`cardiac_cell.h` computes the bound; the demo prints it.

**Determinism & races.** There are **no atomics and no reductions** here. Each
output element is written by exactly one thread, reading only read-only inputs, so
the result is independent of thread scheduling — the GPU output is **bit-identical
run to run**. That is what lets `demo/run_demo` diff stdout against a captured
`expected_output.txt` (docs/PATTERNS.md §3). Timings, which vary, go to stderr.

**Why CPU and GPU still differ slightly.** Both call the *same* `react_step` and
`diffuse_cell` from the shared header, but the GPU may contract `a*b+c` into a
single fused multiply-add (FMA) with one rounding where the host does two. Over
`400` steps this drifts the fields by `~1e-9`. That is far below any physical
meaning, so we verify to `1e-6` (§6) and say so honestly.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct serial implementation:
plain nested loops over the grid, the same operator-split structure, the same
shared physics. `main.cu` runs **both** the CPU reference and the GPU solver from
the **same** initial state (`init_state`) and computes `max |V_cpu − V_gpu|` over
every cell.

**Tolerance = `1e-6`.** The two paths run identical double-precision operations
(shared `__host__ __device__` core), so they *would* be bit-identical but for
FMA-contraction differences that accumulate to `~1e-9` over the run (§5). `1e-6` is
comfortably above that residual and comfortably below any physically meaningful
voltage difference (docs/PATTERNS.md §4). Agreement between two independently
written implementations — one trivially serial, one massively parallel — is strong
evidence the GPU kernels are correct.

**Edge cases exercised by the sample:** the no-flux boundary (the S1 patch spans
the full left edge, so the wavefront touches the top/bottom walls), the ragged
last block (a `32×32` grid tiles evenly, but `diffuse_kernel`'s bounds guard is
still there for non-multiples), and the CFL check (the loader rejects unstable
`dt`).

## 7. Where this sits in the real world

Production cardiac EP codes — **openCARP**, **MonoAlg3D_C**, **Cardioid** (LLNL,
a Gordon Bell finalist), **Chaste** — keep this project's *reaction* kernel almost
verbatim (one thread per cell, per-cell ODE) but differ in four big ways:

1. **Real ionic models.** They integrate ten Tusscher-Panfilov or O'Hara-Rudy
   (50–200 state variables per cell), using the **Rush-Larsen** method: gating
   variables of the form `dm/dt = (m∞ − m)/τ` are advanced with the *exact
   exponential* `m ← m∞ + (m − m∞)·exp(−dt/τ)` instead of forward Euler, which is
   stable at much larger `dt`. FHN's `w` is the 2-variable cartoon of this.
2. **Implicit diffusion.** Instead of the CFL-limited explicit stencil, they solve
   `(I − dt·D·L)Vⁿ⁺¹ = Vⁿ` each step (backward Euler / Crank-Nicolson) with
   **conjugate gradient + ILU(0) preconditioning**, using **cuSPARSE** for the
   sparse mat-vec and **cuSOLVER**/custom CG for the solve. Unconditional
   stability lets them take physiological `dt` (~10–50 µs) without blowing up.
3. **Real anatomy.** A 3-D voxel or finite-element mesh of an actual heart
   (segmented from **UK Biobank**/**ACDC** cardiac MRI), with **anisotropic**
   diffusion following muscle-fiber orientation, and a **Purkinje** conduction
   network. ~10⁸ nodes → MPI across many GPUs, with **streams** overlapping halo
   exchange and compute.
4. **Bidomain & outputs.** The *bidomain* model adds a separate extracellular
   potential (needed for defibrillation and body-surface ECG), and couples to
   mechanics (contraction) and a forward ECG model.

Everything above is *engineering scale-up* of the same reaction-diffusion physics
this project makes visible on a 2-D sheet.

---

## References

- **FitzHugh, R. (1961)** & **Nagumo et al. (1962)** — the FHN excitable-medium
  model; the origin of the 2-variable reduction used here.
- **ten Tusscher & Panfilov (2006)** — a widely-used human ventricular ionic model
  (the kind that plugs into the reaction slot).
- **O'Hara, Virág, Varró, Rudy (2011)** — the O'Hara-Rudy human ventricular model.
- **Rush & Larsen (1978)** — the exponential-integrator method for stiff gating
  ODEs that production solvers use instead of forward Euler.
- **openCARP** (<https://git.opencarp.org/openCARP/openCARP>) — study its
  operator-split monodomain solver and its ionic-model library.
- **MonoAlg3D_C** (<https://github.com/rsachetto/MonoAlg3D_C>) — a finite-volume
  GPU monodomain solver; read how it structures the per-cell ODE kernel and the
  sparse diffusion solve.
- **Cardioid / LLNL** (<https://github.com/llnl/cardioid>) — a multiscale suite
  (EP + mechanics + ECG); an example of scaling this physics to whole hearts.
- **Chaste** (<https://github.com/Chaste/Chaste>) — Oxford's bidomain solver with a
  cardiac mechanics module; a reference for the bidomain extension in §7.
