# THEORY — 14.02 Spatial Reaction-Diffusion (Gray-Scott)

> For a reader who knows C++ but is new to CUDA and to pattern formation. See
> [README.md](README.md) for the tour and build. _Educational only; this is a
> reduced-scope teaching version of a frontier problem._

## 1. The science

How does a uniform ball of cells become a patterned organism — stripes, spots,
segments? Alan **Turing** (1952) showed that two diffusing, reacting chemicals can
spontaneously break symmetry and form stationary patterns: a slowly-diffusing
"activator" and a fast-diffusing "inhibitor". The same mechanism is invoked for
skin/coat patterns, coral, and intracellular organization of signaling molecules.
The **Gray-Scott** model is a clean two-chemical reaction-diffusion system that
exhibits an astonishing variety of these patterns.

## 2. The math

Two fields `U(x,y,t)` and `V(x,y,t)` on a periodic grid:

```
dU/dt = Du ∇²U − U V² + F (1 − U)        # U is fed in at rate F, consumed by reaction
dV/dt = Dv ∇²V + U V² − (F + k) V        # V is produced by reaction, removed at rate F+k
```

`U + 2V` would be conserved without feed/kill; the autocatalytic term `U V²`
(two V plus one U make three V) is the nonlinearity that drives pattern growth.
The uniform state `(U=1, V=0)` is stable, so patterns only appear where a
perturbation can grow — a region of the `(F, k)` plane mapped famously by Pearson:
spots, stripes, mazes, self-replicating "mitosis", and spatiotemporal chaos.

## 3. The algorithm

```
init: U=1, V=0; seed a central square with U=0.5, V=0.25
each step:
    for every cell (x,y):
        lapU = U[N]+U[S]+U[E]+U[W] - 4 U      # 5-point Laplacian (periodic)
        U' = U + dt (Du lapU - U V^2 + F (1-U))
        V' = V + dt (Dv lapV + U V^2 - (F+k) V)
    swap (U,V) <- (U',V')
```

**Complexity.** Each step is `Θ(nx·ny)` cell updates; total `steps · nx · ny`.
Perfectly parallel across cells.

## 4. The GPU mapping

**Decomposition.** One thread per grid cell, on a 2-D grid of 16×16 blocks. The
host runs the time loop, launching the stencil kernel once per step and
**ping-ponging** two (U,V) buffer pairs.

**Why two buffers.** The update reads a cell's neighbours' *current* values and
writes its *next* value. Writing in place would let a cell read a neighbour already
updated this step (a race). Double buffering freezes the read state, so all cells
are independent within a step — no atomics, no `__syncthreads`. (Identical reasoning
to the lattice-Boltzmann project `6.04`.)

**Memory.** U and V live in global memory; each cell reads its 4 neighbours
(periodic indexing via modulo). The access is local and regular; the kernel is
memory-bandwidth bound, and shared-memory **tiling** of a block + halo is the
standard optimization (Exercise 2). One kernel launch per step makes the GPU
launch-bound on small grids; the advantage grows with grid size.

**CPU/GPU parity.** The per-cell update is one `__host__ __device__` function in
double precision, so the GPU reproduces the CPU field. The labyrinth regime is a
stable attractor, so the two stay within ~`1e-7` over 8000 steps (unlike the
chaotic regimes, where they would diverge — a reproducibility caveat shared with
project 10.02).

## 5. Numerical considerations

- **Stability.** Explicit Euler diffusion is conditionally stable: roughly
  `dt < dx² / (4·max(Du,Dv))`. With `dx=1`, `Du=0.16`, that is `dt < ~1.56`, so
  `dt=1` is safe. Larger `dt` blows up.
- **Determinism.** No reductions/atomics, double-buffered reads → reproducible and
  CPU-matching for stable regimes.
- **Sensitivity.** Some `(F,k)` give chaotic dynamics where tiny FP differences
  amplify; the committed sample uses a stable pattern-forming regime so the demo
  is reproducible.

## 6. How we verify correctness

`main.cu` runs the simulation on CPU and GPU and compares the final U and V fields
cell-by-cell (`worst diff ≈ 1e-7`). Beyond CPU/GPU parity, the result is the
*right kind of thing*: from a tiny seed the system self-organizes into a connected
Turing labyrinth covering about half the grid — the hallmark behaviour of the
model, not a flat or blown-up field.

## 7. Where this sits in the real world

This continuum PDE is the textbook entry point. The catalog's actual project is
**particle-based reaction-diffusion (PBRD)** at *molecular resolution*: track each
molecule as it diffuses (Brownian motion) and reacts on encounter, resolving
sub-micron gradients and receptor clustering that a smooth concentration field
cannot. GPU PBRD (ReaDDy, Smoldyn) parallelizes over *molecules* with cell-list
neighbour search; **eGFRD** is more accurate still. A whole minimal cell (~500k
molecules) at millisecond timescales is ~10¹² particle-timestep updates — a
multi-GPU grand challenge. The stencil you parallelize here is the mean-field limit
of that molecular picture.

## References

- Turing (1952), *The Chemical Basis of Morphogenesis*.
- Gray & Scott (1984); Pearson (1993), *Complex Patterns in a Simple System* (the (F,k) phase diagram).
- ReaDDy / Smoldyn / MCell / STEPS — particle-based reaction-diffusion engines.
- NVIDIA CUDA C++ Programming Guide — 2-D stencils, shared-memory tiling.
