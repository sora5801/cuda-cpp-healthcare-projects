# THEORY — 10.02 Real-Time Soft-Tissue Deformation (PBD)

> For a reader who knows C++ but is new to CUDA and to physics simulation.
> See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

A surgical trainer must let a tool push, stretch, and cut virtual tissue while a
**haptic** device renders the reaction force — all within ~10 ms per frame so it
feels real. Classical finite-element elasticity is accurate but often too slow
for that budget on large meshes. **Position-Based Dynamics (PBD)** trades a
little physical rigour for speed and rock-solid stability by working directly with
*positions* and *geometric constraints* instead of forces and stiff ODEs.

## 2. The math

Represent the tissue as `N` particles with positions `x_i`, velocities `v_i`, and
inverse masses `w_i` (`w_i = 0` pins a particle). Springs become **distance
constraints**: for a pair `(i, j)` with rest length `L`,

```
C(x_i, x_j) = |x_i - x_j| - L = 0
```

PBD satisfies each constraint by moving the particles along the constraint
gradient, weighted by inverse mass:

```
Δx_i = -(w_i / (w_i + w_j)) · C · n,   Δx_j = +(w_j / (w_i + w_j)) · C · n
```

where `n = (x_i - x_j)/|x_i - x_j|` is the unit direction. A pinned particle
(`w = 0`) does not move and its partner absorbs the full correction. A `stiffness`
factor in `[0,1]` and several iterations control how rigidly the constraints are
enforced.

## 3. The algorithm

```
each timestep:
    predict:  p_i = x_i + v_i·dt + g·dt^2        # gravity, explicit
    for it in 1..iters:                          # constraint projection
        for each particle i:  p_i += correction_i(p)   # Jacobi (read-only p)
    finalize: v_i = (p_i - x_i)/dt ; x_i = p_i
```

**Gauss-Seidel vs Jacobi.** The textbook PBD projects constraints *sequentially*
(Gauss-Seidel): each correction sees the previous one. That is inherently serial.
The GPU-friendly variant is **Jacobi**: every particle computes its correction
from the *same* read-only snapshot, so all particles update independently; we
average a particle's incident corrections and relax to stay stable. It needs more
iterations than Gauss-Seidel but parallelizes perfectly.

## 4. The GPU mapping

**Decomposition.** One thread per particle. Three kernels run each step:
`predict_kernel`, then `constraint_kernel` `iters` times, then `finalize_kernel`.

**Double buffering (the key to Jacobi).** The projection reads neighbours'
positions and writes its own — if it wrote in place, a particle could read a
neighbour already updated this iteration (a race). We **ping-pong** two position
buffers: read `src`, write `dst`, swap. Every read in an iteration comes from the
frozen previous iteration, so all particles are independent — no atomics, no
`__syncthreads`.

**Memory.** Positions/velocities/inverse-masses live in global memory; each
particle reads its 8 neighbours (structural + shear). The access is local and
regular. The state is small, so the kernels are launch- and latency-bound on tiny
meshes — the GPU's advantage appears as the mesh grows to the 10⁵+ particles real
simulators use.

## 5. Numerical considerations (a real reproducibility lesson)

We integrate in **double** precision, and the per-particle math is shared between
CPU and GPU (`pbd.h`) — yet the final meshes differ at the **~1e-5** level, not
round-off. Why? The GPU contracts `a*b + c` into a **fused multiply-add (FMA)**
by default; the host compiler does not. Each op differs by ~1e-16, but PBD runs
the projection **thousands of times** (steps × iters), and the draping sheet is a
mildly chaotic dynamical system, so those differences **amplify** to ~1e-5. This
is a genuine and important lesson: *bit-identical CPU/GPU results are not
guaranteed for long iterative solvers*, even in double precision. The fix if you
need it is `nvcc --fmad=false` (matches the host, slightly slower); here we simply
verify to a physically-negligible tolerance (`1e-3` on positions of size ~10), and
the agreement to ~6 significant figures confirms the kernel is correct.

## 6. How we verify correctness

`main.cu` simulates the same mesh on CPU (`simulate_cpu`) and GPU
(`simulate_gpu`) and compares every particle's final position. Beyond CPU/GPU
agreement, the result is physically sensible: the pinned edge holds, the sheet
drapes symmetrically under gravity to a stable depth, and constraints keep
neighbouring particles near their rest spacing (the sheet does not blow up — PBD's
hallmark stability).

## 7. Where this sits in the real world

Production surgical simulators (SOFA, iMSTK, FleX) extend this with: **XPBD**
(compliant constraints whose behaviour is iteration-count-independent),
**tetrahedral volume** constraints and **bending** constraints for true 3-D tissue,
**collision/self-collision** with the surgical tool, **graph-coloured Gauss-Seidel**
for faster convergence, **topological cuts** (dissection/suturing), and the
**Material Point Method (MPM)** for tearing. The particle-per-thread,
double-buffered Jacobi projection you learn here is the parallel core of all of it.

## References

- Müller, Heidelberger, Hennix & Ratcliff (2007), *Position Based Dynamics*.
- Macklin, Müller & Chentanez (2016), *XPBD: Position-Based Simulation of Compliant Constrained Dynamics*.
- NVIDIA, *FleX* technical talks — GPU PBD at scale.
- Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic* — FMA/reproducibility.
