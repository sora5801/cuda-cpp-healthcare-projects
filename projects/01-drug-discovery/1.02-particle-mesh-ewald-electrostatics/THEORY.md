# THEORY — 1.2 Particle-Mesh Ewald Electrostatics

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Every molecular-dynamics (MD) simulation of a drug binding to a protein, an ion
channel gating, or DNA in solution must compute the **electrostatic energy and
forces** between thousands to millions of partial atomic charges. Electrostatics
is the longest-ranged force in biology: the Coulomb potential between two charges
falls off only as `1/r`, so distant atoms still matter.

To model bulk solvent without an astronomically large box, MD uses **periodic
boundary conditions**: the simulation cell is tiled infinitely in all directions.
Now each charge interacts not just with the `N−1` others in the box but with
**all of their infinite periodic images**. The naive lattice sum

```
E = 1/2 * sum_i sum_j sum_n  q_i q_j / |r_ij + nL|
```

(where `n` ranges over all integer image vectors) is **conditionally convergent**:
its value depends on the order of summation, and simply truncating it at a cutoff
produces severe, well-documented artifacts — distorted protein structures,
spurious ion pairing, wrong free energies. You **cannot** just ignore far-away
charges.

**Ewald summation** (1921) is the rigorous fix, and **Particle-Mesh Ewald**
(Darden, York, Pedersen 1993; *smooth* PME, Essmann et al. 1995) is the fast,
FFT-based version that every modern MD code (GROMACS, NAMD, OpenMM, AMBER) uses.
Computing it is typically the **single largest chunk of MD wall-time**, which is
exactly why it has been GPU-accelerated so heavily.

This project computes the PME electrostatic energy of a small periodic charge
system and verifies it three independent ways. The committed sample is a
**synthetic NaCl-like ionic crystal** whose total energy is a stable, recognizable
number (the lattice's Madelung energy) — a clean target to validate against.

## 2. The math

### 2.1 The Ewald split

The trick is to add and subtract a screening Gaussian around each point charge.
Writing `1 = erf(βr) + erfc(βr)` and splitting the `1/r` accordingly turns one
conditionally-convergent sum into **three absolutely-convergent** ones:

```
E_total = E_real + E_recip − E_self
```

with (in reduced units where Coulomb's constant `1/(4πε₀) = 1`):

- **Real space** (short-ranged, screened by the complementary error function):

  ```
  E_real = 1/2 * sum_{i≠j, images}  q_i q_j * erfc(β r_ij) / r_ij
  ```

  `erfc` decays like a Gaussian, so this converges within a cutoff `rcut` and is
  evaluated by direct pairwise summation (here with the **minimum-image
  convention**: each pair uses its nearest periodic copy).

- **Reciprocal space** (smooth, long-ranged, summed over wavevectors
  `k = 2π m / L` for integer vectors `m ≠ 0`):

  ```
  E_recip = (2π / V) * sum_{m≠0}  exp(−|k|² / 4β²) / |k|²  *  |S(m)|²
  ```

  where `V = L³` is the box volume and `S(m) = Σ_j q_j exp(i k·r_j)` is the
  **structure factor**. The Gaussian factor `exp(−|k|²/4β²)` kills high-`k` terms,
  so this also converges quickly.

- **Self energy** (a constant correction removing each Gaussian's interaction
  with itself):

  ```
  E_self = (β / √π) * sum_i q_i²
  ```

### 2.2 Symbols

| Symbol | Meaning | Units |
|---|---|---|
| `q_i` | charge of atom `i` | elementary charge `e` |
| `r_ij` | distance between atoms (minimum image) | length |
| `L`, `V=L³` | cubic box side, volume | length, length³ |
| `β` | Ewald **splitting parameter** | 1/length |
| `m=(mx,my,mz)` | integer reciprocal-lattice vector | — |
| `k = 2π m / L` | physical wavevector | 1/length |
| `K` | FFT grid points per axis (grid is `K³`) | — |
| `p` | B-spline interpolation order (here 4) | — |
| `rcut` | real-space cutoff (= `L/2`) | length |

### 2.3 The key identity: β is a free knob

`β` decides **how much work goes to real vs. reciprocal space** — large `β` makes
`erfc` decay fast (cheap real space) but spreads the Gaussians, needing more
`k`-vectors (expensive reciprocal space), and vice-versa. Crucially, **`E_total`
is independent of `β`.** This is not just trivia — it is the single most useful
*physics* check on an Ewald implementation, and the demo uses it (§6).

### 2.4 From Ewald to *smooth PME*

The reciprocal sum's bottleneck is the structure factor `S(m)`: computing it
directly for all `m` costs `O(N · K³)`. **Smooth PME** approximates it cheaply:

1. **Spread** each point charge onto a regular `K³` grid using **cardinal
   B-splines** of order `p` (each charge touches a `p×p×p` block of grid points).
2. **FFT** the grid: `F[m] = FFT(ρ_grid)` approximates `S(m)`.
3. **Convolve** in reciprocal space: multiply `|F[m]|²` by the Ewald weight
   `exp(−|k|²/4β²)/|k|²` **times a B-spline correction factor** `B(m)` that undoes
   the smoothing bias the spline interpolation introduced.

The FFT makes this `O(K³ log K)`, and with `K ∝ N^{1/3}` the whole method is
`O(N log N)` — the headline result.

The B-spline correction is `B(m) = |b_x(mx)|² |b_y(my)|² |b_z(mz)|²` with

```
|b(m)|² = 1 / | Σ_{j=0}^{p−2} M_p(j+1) exp(2πi m j / K) |²
```

where `M_p(·)` are the cardinal B-spline values at integer nodes (for `p=4`:
`M_4(1)=1/6, M_4(2)=2/3, M_4(3)=1/6`). All of this lives in
[`src/pme.h`](src/pme.h) as shared host+device code.

## 3. The algorithm

The reciprocal-energy pipeline (`pme_recip_*` in code):

```
for each atom a:                                   # SPREAD  — O(N · p³)
    g = (r_a / L) * K                              # scaled grid coordinate
    w = bspline_weights(frac(g))                   # p weights per axis
    scatter q_a * w_x ⊗ w_y ⊗ w_z onto grid[g0 .. g0+p-1]  (wrapped)
F = FFT3D(grid)                                    # FFT    — O(K³ log K)
for each reciprocal bin m:                         # CONVOLVE+ENERGY — O(K³)
    e[m] = mult(m) * B(m) * C(m) * |F[m]|²
E_recip = sum_m e[m]                               # REDUCE — O(K³)
```

with `C(m) = (2π/V) exp(−|k|²/4β²)/|k|²`. Serial complexity is dominated by the
FFT, `O(K³ log K)`; the spreading is `O(N · p³)` and the convolution `O(K³)`.

The CPU reference does the **same** pipeline (so the GPU can be checked exactly)
plus an independent **direct Ewald** reciprocal sum over an explicit `(2m_max+1)³`
shell of `k`-vectors, `O(N · m_max³)` — slow but transparently the *definition* of
`E_recip`, used to confirm SPME is a faithful approximation.

## 4. The GPU mapping

Two of the four stages are data-parallel over atoms or grid cells; the FFT is a
library call. This project combines **two flagship patterns**: cuFFT (like 8.03)
and an **atomic fixed-point scatter** (like 11.09).

### Stage 1 — SPREAD (`spread_kernel`, one thread per atom)

```
grid  = ceil(N / 256) blocks,  block = 256 threads
thread a  ->  atom a
  computes its p×p×p B-spline stencil and atomicAdds q*w onto the grid
```

- **Memory:** reads the atom arrays from global memory; writes the grid via
  `atomicAdd`. No shared memory in this teaching version (a production kernel
  tiles atoms/grid-patches into shared memory to cut global traffic — see §7).
- **Why atomics:** several atoms' stencils overlap the same grid cell, so threads
  add concurrently → a data race without atomics.
- **Why fixed-point (the crux):** float `atomicAdd` is **not associative**, so a
  float grid would be **nondeterministic** and would not match the CPU. We
  accumulate charge as **scaled integers** (`unsigned long long`, scale `2⁴⁰`);
  integer adds commute, so the grid is **bit-identical every run and equals the
  CPU grid exactly** (PATTERNS.md §3; `pme.h::pme_to_fixed`).

### Stage 2 — FFT (cuFFT, **not a black box**)

`cufftExecR2C` computes the 3D real-to-complex transform

```
F[m] = Σ_r ρ(r) exp(−2πi m·r / K)
```

for a `K×K×K` real grid, in `O(K³ log K)`. The **R2C** layout stores only the
non-redundant half-spectrum (`K×K×(K/2+1)`) because a real grid's spectrum is
Hermitian (`F[−m] = conj(F[m])`). Hand-rolling this means a mixed-radix FFT with
twiddle factors and bit-reversal across three axes — exactly the wheel cuFFT lets
us not reinvent. The host reference does the same transform with a transparent
**separable naive DFT** (`O(K⁴)`), so the two are directly comparable.

### Stage 3 — CONVOLVE + ENERGY (`energy_kernel`, one thread per reciprocal bin)

```
grid  = ceil(K·K·(K/2+1) / 256),  block = 256
thread i  ->  reciprocal bin i
  e[i] = mult[i] * influence[i] * |F[i]|²
```

`influence = B(m)·C(m)` is built once on the host and uploaded (so CPU and GPU
share identical coefficients). `mult ∈ {1,2}` is the **Hermitian multiplicity**:
interior half-spectrum bins stand for two physical modes (`+m` and `−m`), so they
count twice; the `m_z=0` and Nyquist planes count once.

### Stage 4 — REDUCE (on the host, deterministically)

Rather than a float `atomicAdd` reduction on the device (order-dependent → not
reproducible), we copy the per-bin energies back and **sum them on the host in
index order** — the *same* order the CPU reference uses. The reported energy is
therefore deterministic and matches the CPU up to the FP32-FFT difference.

```
 atoms ──spread(atomic,fixed-point)──▶ ρ grid (K³, integer→real)
                                            │ cuFFT R2C  (O(K³ log K))
                                            ▼
                                     F[m]  (K·K·(K/2+1) complex)
                                            │ × B(m)·C(m)·mult  (per-bin)
                                            ▼
                                  e[m] ──host sum (index order)──▶ E_recip
```

## 5. Numerical considerations

- **Precision split.** The charge spreading and all coefficients are computed in
  **FP64** and the grid is accumulated in exact fixed-point, so the *only*
  single-precision step is the cuFFT itself (FP32 R2C). This is the real
  engineering tension the catalog calls out: *"double-precision accuracy at float
  throughput."* The FP32 FFT introduces ~`10⁻⁶` relative error per bin, ~`10⁻⁴`
  on the summed energy — acceptable and clearly documented, not hidden.
- **Determinism.** The grid is bit-identical (integer atomics); the reciprocal
  sum is done in fixed host order. So **stdout is reproducible run-to-run**
  (verified by re-running). Had we used a float `atomicAdd` to spread or to
  reduce, the low digits would jitter — the lesson of PATTERNS.md §3.
- **Overflow bound.** A grid cell receives at most `~Σ|q|` of charge; with
  `Σ|q| = O(N)` and `N` in the thousands, times the `2⁴⁰` scale, the integer stays
  far below `2⁶³`. `pme.h` documents this bound.
- **Parameter sanity.** `β = 3/rcut` makes `erfc(β·rcut) ≈ 2×10⁻⁵`, so the
  truncated real-space sum is converged; `m_max = ⌈5βL/π⌉` makes the direct
  reciprocal Gaussian factor negligible at the shell edge.

## 6. How we verify correctness

Three independent, falsifiable checks (all in `main.cu`):

1. **GPU SPME == CPU SPME** to `rtol = 1e-4`. Because the two share `pme.h` and
   the grid is fixed-point exact, the only discrepancy is FP32 cuFFT vs FP64 host
   DFT. Observed: `~6×10⁻⁸`. This proves the **GPU pipeline** is correct.
2. **CPU SPME == direct Ewald** to `rtol = 5e-3`. The direct k-vector sum is the
   textbook *definition* of `E_recip`. Observed: `~7×10⁻¹²` at `K=16, p=4`. This
   proves the **method** (B-spline spreading + influence function) is right, not
   just that two copies of the same code agree.
3. **Total energy invariant to β** to `rtol = 2e-2`. Recomputing `E_real + E_recip
   − E_self` at a different `β` must give the same total. Observed: `~3×10⁻⁵`.
   This is a **physics** check on the whole decomposition; a sign error or a wrong
   prefactor in any of the three terms breaks it. (In fact the reciprocal
   prefactor `2π/V` — with *no* spurious `1/2` — was pinned down precisely by
   requiring this invariance.)

The committed sample is engineered to make these meaningful: the NaCl-like lattice
has a well-defined Madelung energy (`E_total ≈ −27.96` reduced units here), so a
gross bug would show as a wrong, recognizable number.

## 7. Where this sits in the real world

Production PME (GROMACS, NAMD, OpenMM, AMBER) differs from this teaching version
in several deliberate simplifications we made:

- **Forces, not just energy.** Real MD needs the **force** on every atom (the
  gradient of `E_recip`), computed by a *second* interpolation (mesh→particle) of
  the convolved potential back onto atoms. We compute only the energy; the
  gradient is a natural exercise (the same B-spline weights, differentiated).
- **Order 4 → 6 and tuned β, K.** Higher spline order and tuned grids hit a target
  accuracy at lower cost. We fix `p=4` and auto-pick `K, β` for clarity.
- **Shared-memory tiling & domain decomposition.** Our spreading kernel does plain
  global `atomicAdd`; GROMACS/NAMD tile atoms and grid patches into shared memory
  and decompose the box across GPUs to cut atomic contention and PCIe traffic.
- **Mixed precision done carefully.** OpenMM keeps the grid in fixed-point too (a
  real technique we borrow) and reconciles FP32 throughput with FP64-like
  accuracy — the central challenge this project illustrates in miniature.
- **P3M and other variants.** Particle-Particle Particle-Mesh is a close cousin
  with a different influence function; the catalog lists it alongside SPME.

---

## References

- **Essmann, Perera, Berkowitz, Darden, Lee, Pedersen (1995),** *A smooth particle
  mesh Ewald method*, J. Chem. Phys. 103, 8577 — the B-spline construction and
  influence function this code implements.
- **Darden, York, Pedersen (1993),** *Particle mesh Ewald: An N·log(N) method* —
  the original PME paper.
- **GROMACS CUDA PME** — <https://github.com/gromacs/gromacs> — the reference GPU
  PME; study its tiled spreading and the gather (force) kernel we omit.
- **NAMD GPU PME** — <https://www.ks.uiuc.edu/Research/namd/> — tiled,
  domain-decomposed PME across multiple GPUs.
- **OpenMM PME plugin** — <https://github.com/openmm/openmm> — Python-accessible,
  mixed-precision PME; a good model for the fixed-point accuracy trick.
- **cuFFT** — <https://developer.nvidia.com/cufft> — the FFT library all of the
  above use internally and that Stage 2 here calls.
