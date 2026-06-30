# THEORY — 2.9 Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics

> The deep dive. Read `README.md` first for orientation, then this for the
> *why*. Code references point at `src/` (`pbe.h` is the shared per-cell core;
> `kernels.cu` is the GPU red-black sweep; `reference_cpu.cpp` is the trusted
> serial baseline).

---

## 1. The science — proteins are charged objects in salty water

A protein is studded with charged and polar atoms (carboxylates, amines,
backbone dipoles). What those charges *do* — how strongly two molecules bind,
which titratable residues are protonated (their pKa), how a colloidal drug
carrier's surface repels its neighbours (zeta potential) — is governed by
**electrostatics in a medium**, not in vacuum.

The medium matters enormously. Inside the tightly packed protein, electronic
polarizability is low: the relative permittivity (dielectric constant) is
**ε ≈ 2–4**. The surrounding water is a sea of rotating dipoles and is extremely
polarizable: **ε ≈ 80**. Dissolved salt (Na⁺, Cl⁻) adds **mobile ions** that
rearrange to *screen* charges, weakening long-range interactions.

Rather than simulate every water molecule and ion (expensive, and most of the
detail is irrelevant to the mean field), **continuum electrostatics** replaces
solvent + ions with a structureless dielectric continuum plus a mobile-ion
density that responds to the local potential. This is the model behind **APBS**
and **DelPhi**. The governing equation is the **Poisson–Boltzmann equation
(PBE)**.

The "S" in the title — the **solvent-accessible surface** — is the geometric
companion: the surface a water-sized probe (radius 1.4 Å) can touch. It defines
where the low-dielectric interior ends and, in production codes, where the
dielectric boundary sits. We compute its **area** (SASA) as a deterministic
scalar via Shrake–Rupley sampling (`compute_sasa` in `reference_cpu.cpp`).

---

## 2. The math — from Poisson to linearized Poisson–Boltzmann

**Poisson's equation** relates potential φ to charge density ρ in a medium of
permittivity ε:

```
  ∇·( ε(r) ∇φ(r) )  =  − ρ_fixed(r) / ε0
```

In ionic solvent the *mobile* ions add their own charge density. At thermal
equilibrium each ion species follows a Boltzmann distribution in the potential,
which (for a symmetric 1:1 salt) gives the **nonlinear Poisson–Boltzmann
equation**:

```
  ∇·( ε ∇φ )  −  ε_w κ² · (kT/e) · sinh( eφ / kT )  =  − ρ_fixed / ε0
```

where **κ** is the inverse **Debye length** — κ² is proportional to the ionic
strength, and 1/κ is the distance over which the solvent screens a charge.

When the potential is small (eφ ≪ kT), `sinh(x) ≈ x`, and the equation
**linearizes** (LPBE) — the form we solve:

```
  ∇·( ε(r) ∇φ(r) )  −  ε_w κ²(r) φ(r)  =  − ρ_fixed(r) / ε0          (LPBE)
```

with the screening term switched on only in solvent (κ²=0 inside the protein,
where there are no mobile ions). Setting κ=0 recovers pure Poisson.

**Reduced units.** To keep the teaching arithmetic clean we fold the physical
constants into one factor, `charge_to_phi = 4π/h` (`build_problem`), so the
potential comes out O(1) in units of kT/e. The *absolute* scale is illustrative,
not calibrated to a real titration — but it is **identical on CPU and GPU**,
which is exactly what the verification needs. §7 lists what a quantitative
solver fixes.

---

## 3. The algorithm — finite differences + red-black Gauss–Seidel

**Discretize.** Lay an `n × n × n` grid of spacing `h` over the molecule
(`build_problem` centres it on the atoms' bounding box). Approximate the
operator with the standard **7-point stencil**: for an interior cell *c* with
six axis-neighbours,

```
  ∇·(ε∇φ) ≈ (ε/h²) · ( Σ_neighbours φ  −  6 φ_c )
```

Plugging into the LPBE and isolating φ_c gives the **pointwise update**
(`pbe_relax_cell` in `pbe.h`):

```
  φ_c  ←  ( (ε/h²)·Σ_neighbours φ  +  ρ_c )  /  ( 6ε/h² + ε_w κ²_c )
```

The denominator is the stencil **diagonal** (`pbe_diag`). Sweeping this update
over all cells repeatedly is **Gauss–Seidel relaxation**; it converges to the
solution of the linear system because the discrete LPBE operator is symmetric
positive-definite (diagonally dominant).

**Complexity.** One sweep touches every interior cell once: **O(n³)** work. We
run a fixed `iters` sweeps (600 for the sample), so total work is
**O(iters · n³)**. Serial Gauss–Seidel is `O(iters · n³)` *sequential* — that is
the bottleneck the GPU attacks. (Multigrid, §7, cuts the iteration count from
O(n) to O(1) and is the production win; we keep flat GS for clarity.)

**The parallelization problem.** Plain Gauss–Seidel uses *already-updated*
neighbours within a sweep — cell *c* depends on cells visited before it. That
chain is inherently serial. The fix is **red-black ordering**:

```
  colour(x,y,z) = (x + y + z) mod 2          # 0 = red, 1 = black
```

On the 7-point stencil every neighbour of a red cell is black and vice-versa, so
within one colour **no cell depends on another of the same colour**. A sweep
becomes two *independent* half-sweeps:

```
  for each iteration:
     update all RED cells   (each reads only black neighbours)   # parallel
     update all BLACK cells (each reads the fresh red neighbours) # parallel
```

`solve_cpu` does exactly this colour order serially; `solve_gpu` does it in
parallel. Same arithmetic, same order → same answer (§6).

---

## 4. GPU mapping — one thread per cell, two launches per sweep

```
  grid  (3-D) : ceil(n/8) × ceil(n/8) × ceil(n/8) blocks      → covers n³ cells
  block (3-D) : 8 × 8 × 8 = 512 threads                       (BX,BY,BZ in kernels.cu)
  thread (gx,gy,gz) ─ owns grid cell (gx,gy,gz)
```

`relax_color_kernel` (in `kernels.cu`) is launched **twice per iteration** — once
with `color=0`, once with `color=1`. Each thread:

1. returns immediately if it is on the outer boundary shell (held at φ=0) or is
   the wrong colour this pass (the parity test) — the two `return`s that make the
   in-place update race-free;
2. otherwise calls the shared `pbe_relax_cell` and writes its one cell.

**Memory hierarchy.** `phi`, `rho`, `eps`, `kappa2` live in **global memory** for
the whole solve; we pay the host↔device copy once in and once out, never per
sweep. The kernel is a low-arithmetic-intensity stencil — it is **bandwidth
bound**: each cell reads 6 neighbour φ + its own ρ/ε/κ². The classic next
optimization is **shared-memory tiling** (stage a block's cells + a halo into
shared memory so neighbour reads hit on-chip memory instead of global), exactly
as the lattice-Boltzmann project 6.04 discusses. We keep the simple global-memory
version because it is the clearest first reading and the demo grid is small; the
tiled version is left as an exercise (README).

**Why no ping-pong buffer?** Reaction-diffusion (14.02) double-buffers because
it is *Jacobi*-style (all cells read the frozen previous state). Gauss–Seidel
updates **in place** and *wants* fresh neighbour values — the red-black colouring
is precisely what makes the in-place write safe in parallel, so a single `phi`
buffer suffices.

```
        RED pass                     BLACK pass
   ┌───┬───┬───┬───┐            ┌───┬───┬───┬───┐
   │ R │ b │ R │ b │            │ r │ B │ r │ B │
   ├───┼───┼───┼───┤            ├───┼───┼───┼───┤   R/B = being written
   │ b │ R │ b │ R │            │ B │ r │ B │ r │   lowercase = read-only
   └───┴───┴───┴───┘            └───┴───┴───┴───┘   (frozen this pass)
```

---

## 5. Numerical considerations — precision, determinism, stability

- **Double precision (FP64) throughout.** The relaxation accumulates over
  hundreds of sweeps; FP32 would lose significance and the CPU/GPU drift would
  swamp the verification. FP64 keeps both sides to ~16 digits.
- **Determinism.** Each cell's update is a fixed, short sequence of FP64 adds and
  one divide (`pbe_relax_cell`). Within a colour the cells are independent, so
  the *order* the GPU processes a colour in does not change any cell's inputs →
  the result is independent of thread scheduling. stdout is byte-identical every
  run (verified). This is why we did **not** need the integer/fixed-point trick
  that the atomic-reduction projects (5.01, 11.09) use: there is no cross-thread
  floating accumulation here.
- **Stability / convergence.** Gauss–Seidel on this SPD system is unconditionally
  convergent (no CFL-like timestep limit); more sweeps → tighter convergence. The
  diagonal `6ε/h² + ε_w κ²` is always strictly positive (ε>0), so the divide is
  always safe.
- **Boundary condition.** We hold the outer shell at **φ = 0** (a grounded box).
  This is acceptable only because the grid has a solvent margin around the
  molecule where the screened potential has already decayed. The production
  choice is a **Debye–Hückel** boundary (each boundary cell set to the analytic
  screened-Coulomb sum of the charges) — more accurate on tight grids (§7).

---

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **CPU vs GPU agreement.** `solve_cpu` and `solve_gpu` run the *identical*
   red-black update (`pbe_relax_cell` from `pbe.h`) in the *identical* colour
   order. `main.cu` takes the worst `|φ_cpu − φ_gpu|` over all cells. Measured:
   **5.55e-17** — machine precision — against a documented tolerance of
   **1.0e-9**. The tolerance is loose on purpose: it leaves head-room for the
   GPU's fused-multiply-add to differ from the host compiler in the last bits on
   other cards, while still meaning "the same field to ~9 significant digits". We
   do **not** claim bit-identity (even though this grid happens to achieve it).

2. **Physical sanity (validates the science, not just CPU==GPU).** The synthetic
   molecule is a **symmetric dipole**: net +1 e on one lobe, −1 e on the other.
   The solved field must therefore be **antisymmetric** — and it is:
   `min = −0.367998`, `max = +0.367998` (equal and opposite), and the potential
   at the geometric centre (the dipole midpoint) is ≈ 0. A sign error in the
   stencil, a charge-deposition bug, or a broken boundary would all break this
   symmetry.

Edge cases handled: charges that would land on the boundary are skipped
(`build_problem`); a non-positive radius or malformed header aborts with a clear
message (`load_atoms`); the divide is guarded by ε>0.

---

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production continuum-electrostatics
solvers differ in ways worth knowing:

| Aspect | This project (teaching) | Production (APBS / DelPhi) |
|---|---|---|
| Equation | **Linearized** PBE | full **nonlinear** PBE (Newton/inexact-Newton outer loop) |
| Dielectric boundary | sharp van-der-Waals (cell in/out of an atom) | smoothed molecular surface, **harmonic-mean** face dielectrics |
| Solver | flat **red-black Gauss–Seidel**, fixed sweeps | **multigrid** / preconditioned CG: O(N) work, O(1) iterations |
| Boundary condition | grounded box (φ=0) | **Debye–Hückel** screened-Coulomb boundary |
| Charge mapping | nearest grid point | trilinear / cubic B-spline (smoother, less grid artifact) |
| Units | reduced (illustrative) | calibrated kT/e, energies in kcal/mol, pKa shifts |
| Precision | FP64 CPU+GPU | mixed; GPU multigrid (DelPhi-GPU) |

The catalog's **CUDA pattern** notes `cuSPARSE` for the sparse Laplacian and
texture memory for the dielectric. We deliberately **hand-roll** the relaxation
instead of calling a library: the whole didactic point is to *see* the stencil
and the red-black colouring. A production multigrid PB solve would assemble the
sparse operator (where `cuSPARSE` SpMV helps the smoother/residual) and recurse
across grid levels — described here, not implemented, per CLAUDE.md §13.

**Real data path.** Swap the synthetic molecule for a real one: fetch a PDB
structure (RCSB), assign charges + radii with **PDB2PQR**, and reformat into this
project's one-line-header `.pqr`-style file (`scripts/download_data.*`).

### Further reading

- **APBS** — <https://github.com/Electrostatics/apbs> — the reference open-source
  PB solver; study its multigrid (PMG/geoflow) backends and test cases.
- **DelPhi** — <http://compbio.clemson.edu/delphi> — classic finite-difference PB;
  its GPU branch parallelizes exactly the relaxation we wrote.
- **PDB2PQR** — <https://github.com/Electrostatics/pdb2pqr> — turns a PDB into the
  charged/radii input PB solvers need.
- **OpenMM GBSA** — <https://github.com/openmm/openmm> — the *analytic*
  Generalized-Born alternative to a grid PB solve (a good contrast: no grid).
