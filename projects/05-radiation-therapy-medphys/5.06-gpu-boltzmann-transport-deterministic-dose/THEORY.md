# THEORY — 5.6 GPU Boltzmann Transport (Deterministic Dose)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._ This is a **reduced-scope teaching
> version** (CLAUDE.md §13): a 1-D, mono-energetic, isotropic-scattering slab
> that exercises the exact algorithm the catalog names (discrete ordinates,
> source iteration, the transport sweep) without the intractable 6-DoF clinical
> phase space. §7 describes the full production approach.

---

## 1. The science

When a radiation beam (photons, electrons) enters tissue, particles stream,
scatter, and are absorbed. The clinically relevant quantity is the **absorbed
dose**: energy deposited per unit mass, which drives whether a tumor is
sterilized and whether healthy tissue is spared. Dose is proportional to the
**particle fluence** (how many particles pass through a point, in all
directions) times the local absorption.

There are two ways to compute fluence:

- **Stochastic (Monte Carlo):** simulate millions of individual particle
  histories with random numbers (that is project 5.01 in this repo). Accurate,
  but noisy — the statistical uncertainty falls only as 1/√N, so cutting noise in
  half costs 4× the histories. Slow in low-density regions (lung), where
  particles travel far between interactions.
- **Deterministic (this project):** solve the *equation* that the fluence
  distribution obeys — the **linear Boltzmann transport equation (LBTE)** —
  directly on a grid, with **no random numbers and no statistical noise**. The
  answer is smooth by construction.

Deterministic transport shines exactly where Monte Carlo struggles: **tissue /
low-density interfaces** (lung, air cavities) and **bone / tissue interfaces**,
where scattering and range effects make MC either noisy or expensive. The
commercial embodiment is **Acuros XB** (Varian Eclipse), a GPU-accelerated LBTE
dose engine that matches Monte-Carlo accuracy in heterogeneous anatomy at a
fraction of the runtime. This project is a didactic miniature of that idea.

## 2. The math

The steady-state, mono-energetic, 1-D LBTE for the **angular flux** ψ(x, μ) —
the fluence of particles at position `x` traveling in a direction whose cosine
with the x-axis is `μ = cos θ ∈ [−1, 1]` — is

```
  μ ∂ψ/∂x  +  Σ_t(x) ψ(x,μ)  =  (Σ_s(x)/2) φ(x)  +  q(x)/2.        (1)
```

Symbol dictionary (units in brackets):

| Symbol | Meaning | Units |
|---|---|---|
| `ψ(x,μ)` | angular flux (fluence per unit direction) | particles·cm⁻²·s⁻¹ per unit μ |
| `μ` | direction cosine with the x-axis, `∈ [−1,1]` | — |
| `Σ_t(x)` | total macroscopic cross-section (removal rate) | 1/cm |
| `Σ_s(x)` | scattering cross-section (`≤ Σ_t`) | 1/cm |
| `Σ_a = Σ_t − Σ_s` | absorption cross-section | 1/cm |
| `q(x)` | fixed external source (isotropic) | particles·cm⁻³·s⁻¹ |
| `φ(x)` | **scalar flux** = angle-integrated ψ | particles·cm⁻²·s⁻¹ |

The **scalar flux** closes the system:

```
  φ(x) = ∫_{-1}^{1} ψ(x,μ) dμ.                                     (2)
```

The physics of each term in (1): `μ ∂ψ/∂x` is streaming (particles move along
x at speed μ); `Σ_t ψ` is removal (scatter or absorb); the right-hand side is the
**in-scattering + external source**, isotropic here so each direction receives an
equal `1/2` share of the total emission `Σ_s φ + q` spread over the μ-interval of
length 2. The coupling of *all* directions through `φ` is what makes this an
integro-differential equation rather than a simple ODE — and is why we iterate.

**Outputs.** The converged `φ(x)`, and the **absorbed-dose proxy**
`D(x) ∝ Σ_a(x) φ(x)` (energy-deposition-rate density; true dose multiplies by
particle energy over mass density, constant here).

## 3. The algorithm

Three nested ideas turn (1)–(2) into arithmetic.

**(a) Discrete ordinates (Sₙ) — discretize angle.** Replace the integral (2) by
an `N`-point **Gauss-Legendre quadrature** `{μ_n, w_n}` on `[−1,1]`:

```
  φ(x) ≈ Σ_n w_n ψ_n(x),    ψ_n(x) := ψ(x, μ_n),    Σ_n w_n = 2.   (3)
```

Gauss-Legendre is the standard choice: it integrates polynomials up to degree
`2N−1` exactly, and its nodes come in ± pairs so half the directions move +x and
half −x. `S₈` (used in the demo) means 8 ordinates.

**(b) The transport sweep — discretize space, per direction.** For a *fixed*
ordinate `μ_n`, equation (1) is a 1-D first-order ODE in `x`. Discretize the slab
into cells of width `h` and integrate cell by cell in the direction of travel
("**sweep**"). Using the **diamond-difference** closure
`ψ_center = (ψ_in + ψ_out)/2`, the outgoing edge flux of a cell is closed-form:

```
  ψ_out = ( (|μ|/h − Σ_t/2) ψ_in + Q ) / ( |μ|/h + Σ_t/2 ),        (4)
```

with cell source `Q = (Σ_s φ_old + q)/2`. A `μ_n > 0` direction sweeps left→right
(the known/upwind edge is the left one); a `μ_n < 0` direction sweeps
right→left. Equation (4) is the per-cell heart of every deterministic transport
code — in this repo it lives in `src/boltzmann_sn.h::sn_diamond_out`.

**(c) Source iteration (SI) — resolve the φ coupling.** Because `Q` needs `φ`,
which needs all the `ψ_n`, iterate:

```
  φ⁽⁰⁾ = 0
  repeat:
      for each ordinate n:  sweep the slab (eq. 4) using φ⁽ᵏ⁾  ->  ψ_n
      φ⁽ᵏ⁺¹⁾(x) = Σ_n w_n · (cell-average ψ_n)             (eq. 3)
  until  ‖φ⁽ᵏ⁺¹⁾ − φ⁽ᵏ⁾‖_∞ / ‖φ⁽ᵏ⁺¹⁾‖_∞ ≤ tol
```

**Complexity.** Each iteration costs `O(N · ncell)` (every ordinate sweeps every
cell). SI converges geometrically with rate ≈ the **scattering ratio**
`c = max(Σ_s/Σ_t)`: the error shrinks by ~`c` per iteration, so the iteration
count grows like `log(tol)/log(c)`. In the demo `c = 0.8` (tissue), which is why
it takes 66 iterations. Highly scattering media (`c → 1`) converge painfully
slowly — the motivation for **diffusion synthetic acceleration** (§7).

The spatial sweep (b) is a **sequential recurrence** in `x` (cell `i+1` needs
cell `i`); the arithmetic intensity is low (a handful of FLOPs per cell edge).
The parallel opportunity is elsewhere — see §4.

## 4. The GPU mapping

The key design decision: **what to parallelize.** The sweep along `x` is a
recurrence and resists cheap parallelization. But within one source iteration the
`N` ordinates are **independent**. So:

- **`sweep_kernel` — one thread per ordinate.** Thread `n` owns direction `μ_n`,
  performs the entire left→right or right→left spatial sweep for that direction,
  and writes `w_n · ψ_avg` per cell into **its own private row** of a
  `[N × ncell]` scratch buffer `d_contrib`. No two threads touch the same
  memory ⇒ **no races, no atomics**.
  - Launch: `grid = ceil(N / 128)`, `block = 128`. `N` is small (2…32), so this
    grid is tiny; that is fine — the point is correctness and the pattern.
- **`reduce_kernel` — one thread per cell.** Thread `i` sums the `N`
  contributions in its column, walking `n = 0…N−1` **in fixed order**, into
  `φ_new[i]`. This is the scalar-flux formation (eq. 3).
  - Launch: `grid = ceil(ncell / 128)`, `block = 128`.
- The **host** runs the outer SI loop, relaunching the two kernels each iteration,
  copying `φ_new` back to test convergence, and **swapping the two device flux
  buffers** (ping-pong) so the next sweep reads the updated `φ`.

Memory hierarchy: the material arrays, quadrature, and flux live in **global
memory**; each sweep keeps its running edge flux `ψ_in` in a **register** and
walks the cells. The scattering source is recomputed per cell from global `φ`
(the catalog's "scattering source in shared memory" is an optimization for the
multi-D case; at this size registers/global suffice — see §7).

```
  ORDINATE-PARALLEL SWEEP  (N ordinates, ncell cells)

  thread 0  (μ_0<0) : sweep  <----------------------  right->left   ─┐
  thread 1  (μ_1<0) : sweep  <----------------------  right->left    │  each writes
  ...                                                                 ├─ its own row of
  thread N-1(μ_{N-1}>0): sweep  ------------------->  left->right   ─┘  d_contrib[N x ncell]
                        │           │          │
                        v           v          v
  reduce_kernel:   φ[i] = Σ_n w_n · contrib[n][i]   (fixed order, no atomics)
                        │
                        └── host: converged?  swap φ buffers  repeat
```

**Why not parallelize the sweep across cells?** You can (the multi-D "wavefront"
across a mesh diagonal, or cuSPARSE's upwind triangular solve), and production
codes do — but in 1-D the recurrence is so cheap that the ordinate axis is the
honest, race-free source of parallelism. This mirrors the catalog's "spatial +
angular decomposition"; we take the angular axis and keep the spatial axis
serial-per-thread for clarity.

**No CUDA library is used here.** The sweep and reduction are hand-written so the
learner sees the exact arithmetic. The catalog's suggested cuSPARSE would enter
in ≥2-D, where the per-direction sweep becomes a sparse **upwind triangular
solve** `L ψ = b` (lower-triangular in the sweep ordering); cuSPARSE's
`csrsv`/`bsrsv` triangular solvers do exactly that. Writing it by hand means
topologically ordering the cells along each direction and forward-substituting —
which in 1-D is just our `for` loop, so we show that instead.

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** Source iteration accumulates over
  dozens of sweeps; the diamond-difference update (4) subtracts nearly equal
  terms `(|μ|/h − Σ_t/2)` when `Σ_t` is large, so double precision keeps the
  cancellation harmless. FP32 would visibly drift and force a looser tolerance.
- **Determinism.** Two rules make stdout byte-identical every run:
  1. The angular reduction (`reduce_kernel`) sums ordinate contributions in a
     **fixed index order**, not with `atomicAdd`. Floating-point addition is not
     associative, so an atomic (order-nondeterministic) sum would give slightly
     different bits each run and would *not* match the CPU. A fixed-order loop
     commutes with the CPU's identical fixed-order loop (PATTERNS.md §3).
  2. Deterministic results go to **stdout**; timings and the error magnitude go
     to **stderr** (shown, not diffed).
- **No race conditions.** Each `sweep_kernel` thread writes only its private row;
  each `reduce_kernel` thread writes only its cell. There is nothing to
  synchronize within a kernel.
- **Stability.** Diamond differencing is second-order accurate but can produce
  small negative fluxes in optically thick cells (`Σ_t·h ≫ 1`); our demo keeps
  `Σ_t·h ≤ 0.25`, comfortably in the stable regime. Production codes add negative-
  flux fixups or use the linear-discontinuous FEM the catalog mentions.

## 6. How we verify correctness

Two independent checks, both visible in the demo:

1. **GPU vs. CPU (the gate).** `src/reference_cpu.cpp` runs the *same* source
   iteration serially, calling the *same* `boltzmann_sn.h` per-cell functions and
   summing ordinates in the same order. We assert
   `max_i |φ_cpu[i] − φ_gpu[i]| ≤ 1e-11`. Because the two paths share byte-
   identical math and reduction order, they agree to ~**5e-17** (machine epsilon)
   — the tolerance is met with ~6 orders of magnitude of margin. The tolerance is
   `1e-11` rather than exact-zero only to allow the compiler's freedom to fuse
   multiply-adds differently on host vs. device; empirically that freedom is not
   even exercised here.
2. **Analytic sanity (physics, on stderr).** In an *infinite* uniform medium the
   balance "absorption = source" gives `φ_∞ = q / Σ_a`. Our finite slab has vacuum
   boundaries, so particles leak out and `φ < φ_∞` everywhere; the demo prints a
   source cell's flux next to `φ_∞` so you can see the flux sitting below the
   infinite-medium ceiling — the expected signature of boundary leakage. This
   checks the *science*, not just CPU==GPU agreement.

Why agreement is convincing: the CPU and GPU are structurally different programs
(serial nested loops vs. two parallel kernels + a host driver). A shared bug in
the *physics* would be caught by check (2); a bug in the *parallelization* (a race,
a wrong index, a dropped ordinate) would break check (1). Passing both is strong
evidence.

## 7. Where this sits in the real world

A clinical deterministic dose engine (Acuros XB; ORNL's **Denovo**; **Attila**)
adds, on top of this skeleton:

- **6-DoF phase space:** full 3-D space `(x,y,z)`, 2-D angle `(θ,φ)` with `S₁₆`+
  ordinate sets, and **multi-group energy** `E` — `~10⁹–10¹⁰` unknowns.
- **Coupled photon→electron transport:** photons scatter and liberate electrons;
  the electron LBTE (with its sharply forward-peaked scattering) is solved and
  coupled, which is what makes the dose accurate at interfaces.
- **Legendre scattering expansion:** anisotropic scattering `Σ_s(x, μ·μ')` is
  expanded in Legendre polynomials `P_ℓ` (we truncate at `ℓ=0`, isotropic).
- **Linear-discontinuous FEM** in space (vs. our diamond difference) for
  robustness in thick cells.
- **Diffusion synthetic acceleration (DSA):** source iteration's `c→1` slowdown
  is fixed by preconditioning each SI step with a cheap diffusion solve, cutting
  iteration counts by 10–100× in scattering-dominated regions.
- **The GPU sweep as a wavefront** across the spatial mesh (cuSPARSE upwind
  triangular solves), with the angular-flux tensor streamed through global memory
  and the scattering source staged in shared memory — the catalog's pattern, at
  a scale where it pays off.

Our version keeps all the *concepts* (Sₙ, SI, sweep, diamond difference,
Gauss-Legendre) at a size a learner can read end to end in an afternoon.

---

## References

- **AAPM TG-105** — Chetty et al., *Report of AAPM TG-105: issues associated with
  clinical implementation of Monte Carlo–based external beam treatment planning*
  (Med. Phys. 2007). Context for deterministic vs. MC dose calculation.
- **Acuros XB** — Vassiliev et al., *Validation of a new grid-based Boltzmann
  equation solver for dose calculation in radiotherapy with photon beams* (Phys.
  Med. Biol. 2010). The clinical LBTE engine this project miniaturizes.
- **Lewis & Miller, *Computational Methods of Neutron Transport*** — the standard
  text for discrete ordinates, source iteration, diamond differencing, and DSA.
- **Denovo / Exnihilo (ORNL)** — `https://github.com/ORNL-CEES/Exnihilo` —
  production 3-D deterministic transport; study its sweep and DSA design.
- **OpenMC** — `https://github.com/openmc-dev/openmc` — primarily Monte Carlo,
  useful to contrast the stochastic and deterministic philosophies.
- **"GPU Sn transport CUDA"** — nuclear-engineering literature on GPU-accelerated
  discrete-ordinates sweeps (wavefront parallelism); the source of the catalog's
  CUDA pattern.
