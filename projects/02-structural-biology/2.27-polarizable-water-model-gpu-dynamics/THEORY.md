# THEORY — 2.27 Polarizable Water Model GPU Dynamics

> The deep dive. Read `README.md` first for the overview, then this for the
> science → math → algorithm → GPU mapping → numerics → verification → real world.
> The implementing files are `src/polar.h` (the shared physics), `src/kernels.cu`
> (GPU), and `src/reference_cpu.cpp` (CPU baseline).
>
> _Educational only — not for clinical use._

---

## 1. The science: why water needs to be *polarizable*

Water is the solvent of biology, and almost every biomolecular simulation spends
most of its atoms (and its compute) on water. The cheapest water models —
**TIP3P**, **SPC/E**, **TIP4P** — put *fixed* point charges on each molecule.
Fixed charges cannot respond to their environment: a real water molecule's
electron cloud distorts when a neighbour, an ion, or a protein side-chain comes
close, changing its dipole moment from ~1.85 D (gas) to ~2.6–3.0 D (liquid). That
**electronic polarization** is missing from fixed-charge models, which is why they
struggle with dielectric constants, ion solvation, and interfaces.

**Polarizable** water models add a degree of freedom that *does* respond:

- **Inducible point dipoles** (AMOEBA, this project) — each polarizable site
  carries a dipole `µ = αE` proportional to the local field.
- **Drude oscillators** (SWM4-NDP, CHARMM-Drude) — a charged particle on a spring,
  which is mathematically equivalent to an inducible dipole.
- **Many-body potentials** (**MB-pol**) — the gold standard: 1-body + explicit
  2-body + 3-body terms fit to CCSD(T) quantum data, with induction on top. MB-pol
  reproduces water's properties across phases almost quantitatively, but the
  2-body and 3-body terms make it **orders of magnitude more expensive** — exactly
  the cost that GPU acceleration (the **MBX** library) exists to tame.

The single computational kernel shared by *all* of these — and the one this
project implements — is the **self-consistent induced-dipole solve**. It is the
expensive, iterative heart of polarizable MD. (We implement the inducible-dipole
form; MB-pol's many-body dispersion/repulsion is described in §8 as the
research-grade extension.)

---

## 2. The math

### 2.1 Inducible dipoles

Each polarizable site *i* has an isotropic polarizability `α_i` and feels a local
electric field `E_i`. The induced dipole is linear in that field:

```
  µ_i = α_i · E_i                                               (1)
```

The field at *i* has two sources:

```
  E_i = E_i^ext                              (a uniform applied field, optional)
      + Σ_{j≠i} q_j · r_ij / r_ij³           (permanent charges — the "direct" field)
      + Σ_{j≠i} T_ij · µ_j                   (the OTHER induced dipoles)         (2)
```

where `r_ij = r_i − r_j` and `T_ij` is the **dipole field tensor**

```
  T_ij = (3 r̂ r̂ᵀ − I) / r_ij³                                  (3)
```

so that `E_i^dip = Σ_{j≠i} [ 3(r̂·µ_j) r̂ − µ_j ] / r_ij³`. In our unit system
Coulomb's constant is 1 (lengths in Å, charge in e, α in Å³); the energy then
comes out in e²/Å, which we also report in kcal/mol via the factor 332.0637.

The catch: equation (1) is **implicit** — `µ_i` depends on `E_i`, which depends on
every other `µ_j`, which depends on `µ_i`. Substituting (2) into (1) gives a
linear system

```
  A µ = E^perm ,   with  A = α⁻¹ − T                            (4)
```

a `3N × 3N` system whose solution is the set of self-consistent dipoles.

### 2.2 Thole damping (avoiding the polarization catastrophe)

At short range the bare tensor (3) diverges like `1/r³` and the SCF can blow up
(two dipoles reinforcing each other without bound — the **polarization
catastrophe**). The fix (Thole) is to smear each dipole over a short distance,
multiplying the two terms of (3) by damping factors `λ₃, λ₅` that → 1 at long
range. With the exponential ("Thole-exp") form and screening parameter `a`:

```
  u  = r / (α_i α_j)^{1/6}              (a polarizability-scaled distance)
  λ₃ = 1 − exp(−a u³)
  λ₅ = 1 − (1 + a u³) exp(−a u³)
```

so the damped field tensor is `E_i^dip = Σ [ 3 λ₅ (r̂·µ_j) r̂ − λ₃ µ_j ] / r³`.
This is exactly the AMOEBA damping, implemented in `thole_lambdas()` in
`src/polar.h`.

### 2.3 The induction energy

For linearly induced dipoles the polarization (induction) energy is

```
  U_pol = −½ Σ_i µ_i · E_i^perm                                 (5)
```

— minus one half of each dipole's dot with the **permanent** field that created
it. The ½ is the work done charging the dipole against its own polarizability.
This single scalar is the project's headline result (`polarization_energy_site()`).

---

## 3. The algorithm: Jacobi self-consistent field (SCF)

We could solve the linear system (4) directly (it is `3N×3N`), but that is `O(N³)`
and obscures the physics. Instead we use **fixed-point (Picard) iteration**, the
transparent method that production codes also start from:

```
  1.  E^perm_i  ←  E^ext + Σ_{j≠i} q_j r_ij / r_ij³        (once; O(N²))
  2.  µ_i       ←  α_i · E^perm_i                          (direct/zeroth guess)
  3.  repeat (sweep k = 1, 2, …):
        E^dip_i ←  Σ_{j≠i} T_ij(λ) · µ_j^(k)              (field of OTHER dipoles)
        µ_i^(k+1) ←  α_i · (E^perm_i + E^dip_i)            (re-induce every site)
        δ        ←  max_i,c |µ_i^(k+1) − µ_i^(k)|          (residual)
        µ        ←  µ^(k+1)                                (commit)
      until δ ≤ tol  or  k = max_iters
  4.  U_pol     ←  −½ Σ_i µ_i · E^perm_i
```

This is **Jacobi** iteration: every site is updated from the *previous* sweep's
dipoles (we read `µ`, write `µ_next`, then swap). That is the crucial property for
parallelism — within a sweep, the N updates are **independent**, so they map to N
GPU threads with no synchronization inside the sweep.

### Complexity

| | per sweep | total |
|---|---|---|
| Serial (CPU) | `O(N²)` field eval | `O(N² · n_sweeps)` |
| Parallel (GPU) | `O(N)` work per thread, N threads | same FLOPs, wall-time `O(N · n_sweeps / P)` for P cores |

`n_sweeps` is small (≈10 here) and depends on how strongly the dipoles couple; it
is roughly **independent of N** at fixed density, so the method is effectively
`O(N²)` and the GPU's win grows with N. (Real codes add a neighbour-list cutoff to
make the per-sweep cost `O(N)` — see §8.)

### Why not Gauss–Seidel?

Gauss–Seidel (update `µ_i` using already-updated `µ_{<i}` in the same sweep)
converges in fewer sweeps, but it is **inherently sequential** — site *i* must
wait for site *i−1*. Jacobi trades a few more sweeps for full parallelism, which
is the right trade on a GPU. This is the same Jacobi-vs-Gauss-Seidel choice as the
PBD soft-tissue flagship (10.02) and the stencil solvers.

---

## 4. The GPU mapping

The pattern (PATTERNS.md: *iterative relaxation + N-body field evaluation*):

```
  one GPU thread     ⟷  one site i
  one kernel launch  ⟷  one Jacobi sweep
  host loop          ⟷  the SCF iteration + convergence test
  two device buffers ⟷  ping-pong (mu_in read, mu_out written)
```

### Kernels (`src/kernels.cu`)

| Kernel | Role | Thread *i* does |
|---|---|---|
| `permanent_field_kernel` | E^perm, once | loops over all *j*, sums `q_j r_ij / r³` |
| `init_dipoles_kernel` | direct guess | `µ_i = α_i E^perm_i` |
| `dipole_sweep_kernel` | one Jacobi sweep | loops over all *j*, sums damped dipole field, re-induces `µ_i`, writes to the OTHER buffer, folds its residual into a global max |
| `energy_kernel` | U_pol | `−½ µ_i · E^perm_i`, accumulated |

The host (`solve_dipoles_gpu`) drives the sweep loop: launch
`dipole_sweep_kernel`, copy back the residual, swap the two dipole buffers, and
stop when the residual ≤ tol. Geometry is uploaded **once** as Structure-of-Arrays
(`px, py, pz, q, alpha`) so consecutive threads read consecutive addresses
(coalesced loads).

```
  host                          device
  ----                          ------
  upload px,py,pz,q,alpha  -->  global memory (read-only during the solve)
  permanent_field_kernel   -->  E^perm[i]
  init_dipoles_kernel      -->  d_muA  (= "current")
  for k in 1..max:
    dipole_sweep_kernel(cur=d_muA, nxt=d_muB) --> d_muB, atomicMax residual
    copy residual D2H; if <= tol break
    swap(d_muA, d_muB)         (ping-pong)
  energy_kernel(cur)       -->  U (fixed-point atomicAdd)
```

### Memory hierarchy

- **Global memory** holds positions, charges, polarizabilities, `E^perm`, and the
  two dipole buffers. Each sweep re-reads all `µ_j` from global memory — for these
  small clusters that is fine; §8 notes the shared-memory tiling a large run would
  use (the same idea as a tiled N-body force kernel).
- **Registers** hold thread *i*'s accumulators (`Edip`, its position).
- The **ping-pong buffers** (`d_muA`, `d_muB`) are the double-buffering that makes
  the update a true Jacobi step — no thread ever reads a dipole another thread is
  simultaneously writing.

### Occupancy

128 threads/block: each thread's inner loop carries several double-precision
accumulators, so a smaller block keeps register pressure in check while still
giving the scheduler enough warps to hide the global-memory latency of the `µ_j`
reads on sm_75–sm_89.

---

## 5. Numerical considerations

- **Precision: FP64 throughout.** Electrostatic sums and the SCF residual need the
  dynamic range of double precision; FP32 would lose the small dipole changes near
  convergence. Turing/Ampere/Ada run FP64 slowly, but this is a teaching code, not
  a throughput benchmark.
- **Determinism (PATTERNS.md §3).** Two things could make the parallel result
  depend on thread order:
  1. The per-site field sum over *j*. We loop *j = 0…N−1* in the **same order** as
     the CPU, so the floating-point accumulation is bit-identical → dipoles agree
     to round-off.
  2. The two scalar reductions (max residual, total energy). Floating-point
     `atomicAdd`/`atomicMax` are **not** order-independent, so we reduce in
     **fixed point**: scale by `1e12`, round to `int64`, and use integer
     `atomicMax`/`atomicAdd` (integer ops commute). The host divides back. This
     makes `stdout` byte-identical every run — the same idiom as flagships 5.01
     and 11.09. (The energy can be negative; we add the two's-complement-encoded
     integers, which is exact as long as the true sum fits in `int64`.)
- **Convergence test.** We stop when the largest per-component dipole change drops
  below `tol = 1e-9 e·Å`. Jacobi converges geometrically here because the cluster
  is dilute and Thole-damped (the spectral radius of `αT` is well below 1).
- **Polarization catastrophe.** Without Thole damping, two close dipoles can make
  `αT` have an eigenvalue ≥ 1 and the iteration diverges. The damping (§2.2) keeps
  the iteration contractive — try setting `a_thole` to 0 in the sample to see it
  misbehave (an exercise).

---

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **CPU == GPU.** `reference_cpu.cpp` runs the identical Jacobi iteration on the
   identical shared physics (`polar.h`). `main.cu` compares every dipole component
   and the energy; the worst difference is ~2×10⁻¹⁶ (dipoles) and ~2×10⁻¹³
   (energy), far inside the `1e-9` tolerance. This proves the GPU port is faithful.
2. **Physics check against an analytic answer.** Site 0 is an isolated polarizable
   probe in a known uniform field `E^ext`, 50 Å from the cluster. Its converged
   dipole must equal the closed-form `µ = α E^ext = 0.0722 e·Å`. The demo prints
   `|µ0|` next to the analytic value; they agree to ~2×10⁻⁶ (the residual is the
   cluster's field at 50 Å, ∝ 1/r³ — real physics, quantified in stderr). This
   validates the *science*, not just internal consistency.

The two waters' oxygens get measurably different induced dipoles (0.810 vs 1.031
e·Å) because the cluster geometry is asymmetric — a sanity check that mutual
polarization is actually happening (a symmetric cluster would give equal dipoles).

---

## 7. Edge cases

- **Pure fixed charges** (`alpha = 0`, the hydrogens) never carry a dipole and are
  skipped in the dipole-field loop; they still contribute to `E^perm`.
- **Ragged last block:** `if (i >= N) return;` guards threads past the site count.
- **Self-interaction:** the `j == i` skip avoids a site polarizing itself.
- **Divide-by-zero:** sites never coincide in the sample; a real loader would add
  an exclusion list for bonded 1-2/1-3 pairs (omitted here for clarity).

---

## 8. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). A production
polarizable-water engine differs in several ways, each a natural next step:

- **Conjugate gradient, not Jacobi.** The catalog calls for a *GPU conjugate-
  gradient inducible-dipole solver*. Preconditioned CG solves the same system (4)
  in ~5–10 matvecs instead of dozens of Jacobi sweeps; each matvec is the same
  `T·µ` field evaluation we already have. Jacobi is the transparent stepping stone.
- **Neighbour lists + PME.** Real boxes have 10³–10⁶ sites; the `O(N²)` all-pairs
  sum becomes a cutoff neighbour list for short range plus **Particle-Mesh Ewald**
  (an FFT, cf. flagship 8.03) for the long-range tail. That is what makes a sweep
  `O(N)`.
- **Shared-memory tiling.** A large dipole-field kernel tiles the *j* loop through
  shared memory (load a block of `µ_j` cooperatively, reuse across the block's
  threads) — the classic tiled N-body optimization.
- **Many-body terms (MB-pol).** Beyond induction, MB-pol adds explicit 2-body and
  3-body energies fit to coupled-cluster data; the **MBX** library evaluates these
  on the GPU. cuBLAS handles the many-body expansion tensor contractions the
  catalog mentions. That is the "orders of magnitude more expensive" frontier GPU
  acceleration targets.
- **Integrating MD.** A real run wraps this dipole solve inside an MD time step:
  solve the dipoles, compute forces (including the dipole-gradient terms), advance
  positions, repeat for millions of steps. The dielectric constant and density
  anomaly that validate a water model emerge only after nanoseconds of such MD.

What this project keeps faithful: the self-consistent field equation (1)–(4),
Thole damping, the induction energy (5), FP64 determinism, and the exact
parallel structure (one thread per site, ping-pong Jacobi) that the big codes use
inside their CG solvers.

---

## References (study, don't copy wholesale)

- Thole, *Chem. Phys.* **59**, 341 (1981) — the damping model.
- Ren & Ponder, *J. Phys. Chem. B* **107**, 5933 (2003) — AMOEBA polarizable water.
- Lamoureux et al., *Chem. Phys. Lett.* **418**, 245 (2006) — SWM4-NDP Drude water.
- Babin, Medders, Paesani — MB-pol (the many-body water potential).
- **MBX** <https://github.com/paesanilab/MBX>, **OpenMM**
  <https://github.com/openmm/openmm>, **Tinker-HP**
  <https://github.com/TinkerTools/tinker-hp>, **i-PI** <https://github.com/i-pi/i-pi>.
