# THEORY — 1.7 Quantum Chemistry / DFT (reduced-scope RHF/SCF)

> The deep didactic explanation (the "why"). Written for a sharp student who knows
> C++ but is new to CUDA and new to this domain.
>
> _Educational only — not for clinical use._

> **Scope honesty (CLAUDE.md §13).** A production *Density Functional Theory* code
> is a research-grade artifact (exchange-correlation functionals, grids, density
> fitting, periodic boundary conditions…). To keep this a thing you can actually
> read and verify, we implement the **transparent kernel of the same method**:
> **restricted Hartree–Fock (RHF)** self-consistent field on a **minimal STO-3G
> Gaussian basis**, for closed-shell H/He molecules. RHF and Kohn–Sham DFT share
> the *identical* computational skeleton — build integral matrices, then solve a
> generalized eigenproblem self-consistently — and the same O(N⁴) two-electron
> bottleneck the catalog names. The last section explains exactly what we left out.

---

## 1. The science

An atom or molecule is a swarm of nuclei and electrons governed by the
time-independent **Schrödinger equation** `Ĥ Ψ = E Ψ`. The nuclei are ~1800×
heavier than the electrons, so we freeze them (the **Born–Oppenheimer**
approximation) and ask: given fixed nuclear positions, what is the lowest energy
the electrons can have, and what is their wavefunction? That single number — the
**electronic energy** as a function of geometry — is the *potential energy
surface* on which all of chemistry happens: bond lengths, reaction barriers,
binding affinities, vibrational spectra.

The trouble is that electrons **repel each other**, so the wavefunction of N
electrons does not separate into N independent one-electron problems. The
**Hartree–Fock** idea is the simplest honest approximation: pretend each electron
moves in the *average* field of all the others (a "mean field"), and enforce the
Pauli principle (the wavefunction is a single antisymmetric **Slater determinant**
of one-electron orbitals). DFT replaces the exact exchange of HF with an
approximate *exchange-correlation functional* of the electron density, but the
machinery — orbitals expanded in a basis, a self-consistent field — is the same.
Either way the payoff in drug discovery is concrete: optimized geometries of drug
fragments, electrostatic-potential maps for pharmacophores, and QM-derived force
field parameters.

---

## 2. The math

We expand each molecular orbital (MO) `ψ_a` in a fixed set of **basis functions**
`{φ_μ}` (atom-centered Gaussians): `ψ_a(r) = Σ_μ C_{μa} φ_μ(r)`. Substituting into
the Hartree–Fock variational condition turns the differential equation into a
**matrix equation**, the **Roothaan equations**:

```
F C = S C ε
```

- `S_{μν} = ∫ φ_μ φ_ν`  — the **overlap** matrix (the basis is *non-orthogonal*, so
  `S ≠ I`; this is what makes it a *generalized* eigenproblem).
- `F` — the **Fock matrix**, the effective one-electron Hamiltonian:
  `F = H^core + G(P)`, where
  - `H^core_{μν} = T_{μν} + V_{μν}` is kinetic + nuclear-attraction energy, and
  - `G_{μν}(P) = Σ_{λσ} P_{λσ} [ (μν|λσ) − ½(μλ|νσ) ]` is the electron–electron
    term: **Coulomb** repulsion minus **exchange**.
- `(μν|λσ)` — the **two-electron repulsion integrals (ERIs)**:
  `∫∫ φ_μ(r₁)φ_ν(r₁) (1/r₁₂) φ_λ(r₂)φ_σ(r₂) dr₁ dr₂`. There are **O(N⁴)** of them.
- `P_{μν} = 2 Σ_{a∈occ} C_{μa} C_{νa}` — the **density matrix** (factor 2 = two
  spins per occupied spatial orbital).
- `C` — MO coefficients (columns); `ε` — orbital energies (diagonal).

The catch: `F` depends on `P`, which depends on `C`, which comes from
diagonalizing `F`. It is a fixed-point problem, solved by iteration — the
**self-consistent field (SCF)**.

**Gaussian basis functions** make every integral analytic. A primitive s-Gaussian
is `exp(−α|r−A|²)`. The *Gaussian product theorem* says the product of two
Gaussians is a third Gaussian centered between them, which collapses the overlap,
kinetic, and nuclear integrals to closed forms; the Coulomb operator `1/r` is
handled by the **Boys function** `F₀(t) = ½√(π/t)·erf(√t)`. All of these live in
[`src/gaussian_integrals.h`](src/gaussian_integrals.h), derived and commented.

The total energy is `E = ½ Σ_{μν} P_{μν}(H^core_{μν} + F_{μν}) + E_nuclear`, where
`E_nuclear = Σ_{A<B} Z_A Z_B / R_{AB}` is the classical proton–proton repulsion.

---

## 3. The algorithm

```
build basis {φ_μ} from the molecule (STO-3G: one contracted 1s per atom)
build S, T, V  -> H^core               (cheap, O(N^2) integrals)
build (μν|λσ) for all μ,ν,λ,σ          (THE BOTTLENECK, O(N^4) integrals)
F <- H^core                            (core guess: ignore e-e to start)
repeat:
    solve  F C = S C ε                 (generalized eigenproblem, O(N^3))
    P      <- 2 Σ_occ C C^T            (new density)
    F      <- H^core + G(P)            (new Fock matrix, contracts the ERI tensor)
    E_elec <- ½ Σ P (H^core + F)
until |E_elec - E_elec_prev| < tol
E_total = E_elec + E_nuclear
```

**Complexity.** Building the ERIs is **O(N⁴)** and dominates for any real
molecule; the per-iteration eigensolve is **O(N³)**; the number of SCF iterations
is small and roughly constant (typically 10–20). So the wall-clock is "O(N⁴) to
build the integrals, times a handful of O(N³) solves." Production codes reduce the
prefactor with screening (skip negligible integrals) and the resolution-of-identity
(RI) approximation, and reduce the exponent with linear-scaling methods — all
described in §7.

---

## 4. The GPU mapping

This project matches **PATTERNS.md §1 row "dense linear-algebra solve → use
cuSOLVER/cuBLAS"** for the SCF eigensolve, plus a **custom independent-jobs
kernel** for the integrals. Two GPU jobs:

**(a) The ERI kernel — the headline (one thread per integral).** The N⁴ integrals
`(μν|λσ)` are mutually independent, so we launch **N⁴ threads** (block = 256, grid
= ⌈N⁴/256⌉). Thread `t` decodes its flat id into the quartet `(i,j,k,l)` by
div/mod — using the *same* memory layout `eri[((i*N+j)*N+k)*N+l]` as the CPU — then
sums the (≤81) primitive contributions by calling `eri_primitive()` from
[`gaussian_integrals.h`](src/gaussian_integrals.h):

```
thread t  ->  (i,j,k,l)  ->  eri[t] = Σ_primitives c·c·c·c · (ab|cd)
```

```
   N^4 integrals                 GPU grid
  ┌───────────────┐            ┌──┬──┬──┬──┐
  │ (0000)(0001)… │            │t0│t1│t2│…│   one thread = one (μν|λσ)
  │ (0010)(0011)… │   ──────▶  ├──┼──┼──┼──┤
  │      …        │            │  │  │  │  │
  └───────────────┘            └──┴──┴──┴──┘
```

The flat-basis arrays (centers, exponents, coefficients) are tiny and read-only,
so they stay hot in L2 — no shared memory is needed at this scale. The mapping is
deliberately simple because the whole point is the **independence**: thousands of
integrals fall out at once. (A production code further groups quartets by *shell
pair* and stages them through shared memory; see §7.)

**(b) The SCF eigensolve — cuSOLVER, not a black box.** Each cycle solves
`F C = S C ε`. That is a *generalized symmetric-definite* eigenproblem, a solved
library problem, so we hand it to **`cusolverDnDsygvd`** (divide-and-conquer,
`itype=1`, `jobz=VECTOR`). [`src/kernels.cu`](src/kernels.cu) documents exactly
what it computes and the column-major layout it returns. Hand-rolling it would
mean: Cholesky-factor `S`, reduce to a standard eigenproblem, tridiagonalize, run
divide-and-conquer, back-transform — which is precisely the transparent CPU path
in [`reference_cpu.cpp`](src/reference_cpu.cpp) (`solve_generalized` +
`symmetric_eigen`), kept as the reference so you can see what the library hides.

Memory hierarchy used: **global** (the ERI tensor and flat basis), **registers**
(per-thread accumulators), and the library's internal **shared**/register tiling
inside cuSOLVER. The ERI tensor is the only large allocation: N⁴ doubles.

---

## 5. Numerical considerations

- **Precision: FP64 throughout.** Energies must agree to ~1e-6 Ha to be chemically
  meaningful, and the ERIs span several orders of magnitude, so single precision is
  not enough — we use `double` on both the host and the device.
- **CPU/GPU parity (the key idea).** The per-integral formula is one
  `__host__ __device__` inline (`eri_primitive`) compiled for *both* targets
  (PATTERNS.md §2). Because both sides execute the **identical arithmetic in the
  same order**, the CPU and GPU integral tensors are **bitwise identical** (we see
  worst |Δ| ≈ 5e-17, i.e. one rounding ulp from `cudaMemcpy`/reduction ordering).
- **The Boys function near t = 0.** `½√(π/t)·erf(√t)` is `0/0` at the origin in
  floating point, so we switch to its Taylor series `1 − t/3 + t²/10` for small `t`.
  This keeps both nuclear and two-electron integrals smooth and accurate.
- **Determinism (CLAUDE.md §12, PATTERNS.md §3).** Each ERI is computed by a single
  thread (no cross-thread floating reduction), and the SCF/eigensolve are
  deterministic, so **stdout is byte-identical every run**. Timings and the raw
  verification deltas go to **stderr** (shown, not diffed).
- **SCF convergence.** With the small two-electron systems here the core guess
  converges in ~2 iterations to `|ΔE| < 1e-9 Ha`; no damping/DIIS is needed (DIIS
  acceleration is an exercise and is what production codes use for hard cases).

---

## 6. How we verify correctness

Two layers, plus an external check:

1. **GPU integrals vs CPU integrals.** `main.cu` builds the N⁴ ERI tensor *both*
   ways and takes the worst absolute difference. Tolerance **1e-12** (PATTERNS.md
   §4 "exact" tier: identical operations on both sides → machine precision). We
   observe ≈ 5e-17.
2. **GPU SCF energy vs CPU SCF energy.** The same SCF loop is run with the cuSOLVER
   eigensolver and with the CPU Jacobi eigensolver; the total energies must agree
   to **1e-9 Ha**. We observe ≈ 4e-16.
3. **External / analytic check (validates the *science*, not just CPU==GPU).** For
   H₂ at R = 1.4 Bohr the STO-3G total energy is the **published textbook value
   −1.1167 Hartree** (Szabo & Ostlund, *Modern Quantum Chemistry*, §3.5). The
   program lands on −1.11671432 Ha — so the physics, the integrals, and the SCF are
   all correct, not merely self-consistent.

Edge cases handled: malformed input, unsupported elements (only H/He shipped), odd
electron count (closed-shell only), and divide-by-zero guards in the integrals.

---

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production quantum-chemistry codes
(named in the catalog) differ in ways worth knowing:

- **DFT vs HF.** Real DFT replaces HF's exact exchange with an **exchange-correlation
  functional** (B3LYP, ωB97X-D), evaluated by numerical integration on an atom-
  centered grid. The SCF skeleton is identical; only `G(P)` changes. **TeraChem**
  pioneered GPU-native DFT and reports ~100× speedups over single-CPU codes.
- **Bigger bases and higher angular momentum.** We ship only s-type STO-3G (one 1s
  per atom). Real bases (6-31G\*, cc-pVTZ) add p, d, f functions — the integral
  formulas gain polynomial prefactors (the Obara–Saika / McMurchie–Davidson
  recursions) and need the higher Boys functions `F₁, F₂, …`.
- **Beating O(N⁴).** Integral **screening** (Schwarz inequality) skips negligible
  quartets; the **resolution-of-identity (RI)** approximation factorizes the ERIs to
  O(N³); **linear-scaling** DFT exploits locality for O(N). **PySCF** (with the
  GPU4PySCF extension), **CP2K** (Gaussian + plane-wave), and **NWChem** all
  implement these.
- **GPU ERI engines.** Production kernels group integrals by **shell pair**, stage
  primitives through **shared memory**, and use warp-level parallelism over shell
  pairs (the catalog's "two-electron integrals in shared memory") — far more
  involved than our one-thread-per-integral kernel, but the same idea: exploit the
  embarrassing parallelism of the O(N⁴) integral list.
- **Plane-wave DFT** (for periodic solids) replaces Gaussians with plane waves and
  uses **cuFFT** for the kinetic/Coulomb terms — a different but equally GPU-friendly
  route.

What you can take away here is exactly what those codes scale up: an integral list
that is embarrassingly parallel, a generalized eigenproblem each cycle, and a
self-consistent loop that ties them together.

---

## Further reading

- A. Szabo, N. Ostlund, *Modern Quantum Chemistry* — the canonical derivation of
  everything above (Chapter 3 is RHF; the H₂/STO-3G worked example is §3.5).
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory* —
  the integral recursions for higher angular momenta.
- The STO-3G parameters: Hehre, Stewart, Pople, *J. Chem. Phys.* **51**, 2657 (1969).
- PySCF, NWChem, CP2K, TeraChem — see the README's "Prior art".
