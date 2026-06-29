# THEORY — 1.8 Semi-Empirical & Tight-Binding Quantum Methods

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## The science

Electrons in molecules obey quantum mechanics, but solving the full electronic Schrödinger equation
(*ab initio*, or DFT) is expensive — its cost grows steeply with the number of basis functions. For
**screening** tasks in drug discovery (rank 10⁵ conformers, enumerate tautomers, estimate reactivity for a
whole library) that cost is prohibitive. **Semi-empirical and tight-binding methods** make a deliberate
trade: keep the *quantum* skeleton (a Hamiltonian matrix whose eigenvalues are orbital energies) but
replace the costly integrals with cheap, **empirically parameterised** expressions. The result is 100–
10000× faster than DFT and good enough to *rank* molecules — which is what screening needs.

This project teaches the **conceptual ancestor** of all of them: **Hückel Molecular Orbital (HMO)
theory** (Erich Hückel, 1930s), the simplest tight-binding model in chemistry. It describes the
**π-electron system** of a *planar conjugated hydrocarbon* — molecules like benzene or butadiene whose
carbon `2p_z` orbitals (perpendicular to the molecular plane) overlap side-to-side to form delocalised
π molecular orbitals. The π electrons are the chemically interesting ones: they set aromaticity, colour,
and reactivity. Hückel asks one question — *given which carbons are bonded, what are the π-orbital energies?*
— and answers it with a tiny matrix eigenvalue problem.

The key chemical readouts:

- **Total π-electron energy** `E_π` — sum of occupied orbital energies; its excess over isolated double
  bonds is the **delocalisation (resonance) energy** that makes benzene aromatic.
- **HOMO–LUMO gap** — the energy between the Highest Occupied and Lowest Unoccupied MO. A **small gap**
  means the molecule is easily polarised / excited / reactive (think dyes, radicals, antiaromatic rings);
  a **large gap** means kinetic stability. It correlates with chemical hardness, UV absorption, and
  reactivity — which is why it is a workhorse descriptor in QM-based ADMET.

## The math

**Basis.** One π orbital `|i⟩` per sp² carbon `i` (`i = 0 … N−1`). A molecular orbital is a linear
combination `ψ = Σ_i c_i |i⟩` (LCAO). We seek the coefficients `c` and energies `ε` that solve the
matrix eigenvalue problem

```
H c = ε c                    (we use an orthonormal basis, so the overlap matrix S = I)
```

**The Hückel Hamiltonian** is defined by three rules (the *only* empirical input):

```
H_ii = α                      on-site (Coulomb) integral, same for every carbon
H_ij = β   if i,j bonded      resonance (hopping) integral between bonded neighbours
H_ij = 0   otherwise          no direct interaction between non-bonded atoms
```

`α` and `β` are negative energies (bonding lowers energy). We adopt the textbook **relative scale**
`α = 0`, `β = −1`, so every reported energy is in **units of `|β|`** measured from `α`. Then `H` is just
the (signed) **adjacency matrix** of the molecule's π-bond graph — *the molecular graph IS the
Hamiltonian.* This is the essence of "tight binding": the matrix is built from local connectivity, and it
is **sparse** (only bonded pairs are non-zero).

**Eigenvalues = orbital energies.** Diagonalising the real symmetric `H` gives `N` eigenvalues
`ε_0 ≤ ε_1 ≤ … ≤ ε_{N−1}` (the MO energies) and orthonormal eigenvectors (the MO coefficients).

**Filling electrons (Aufbau + Pauli).** A neutral conjugated hydrocarbon contributes **one π electron per
carbon**, so `N_e = N`. Electrons fill the lowest MOs two at a time (opposite spins):

```
E_π = Σ_k  n_k · ε_k ,   n_k = 2 for the lowest ⌊N_e/2⌋ MOs (1 for a singly-filled MO if N_e is odd)
HOMO = ε at the last occupied level,  LUMO = ε at the first empty level,  gap = ε_LUMO − ε_HOMO
```

**Worked closed forms** (the analytic checks the demo uses):

- **Linear polyene of `N` atoms:** `ε_k = α + 2β cos(kπ/(N+1))`, `k = 1…N`.
  Ethylene (`N=2`): `±|β|`, `E_π = 2|β|`, gap `2|β|`. Butadiene (`N=4`): `E_π = 4.472|β|`.
- **Monocyclic ring of `N` atoms (annulene):** `ε_k = α + 2β cos(2πk/N)`, `k = 0…N−1`.
  Benzene (`N=6`): energies `{−2,−1,−1,1,1,2}|β|`, `E_π = 8|β|`, gap `2|β|` → **aromatic**.
  Cyclobutadiene (`N=4`): a **degenerate non-bonding pair at ε = 0**, so the **HOMO–LUMO gap is 0** →
  **antiaromatic**, the most reactive molecule in our batch.
- **Naphthalene** (fused rings, no single formula): `E_π = 13.683|β|`.

## The algorithm

For one molecule:

1. **Build** `H` from the bond list — `O(N²)` to fill the matrix (`O(E)` non-zeros).
2. **Diagonalise** `H` — a dense symmetric eigenproblem, `O(N³)`. We use **cyclic Jacobi**: repeatedly
   apply 2×2 Givens rotations that zero an off-diagonal pair, accumulating them into the eigenvector
   matrix, until the off-diagonal mass is negligible. Convergence is quadratic; a few sweeps suffice.
3. **Sort** eigenvalues ascending — `O(N log N)`.
4. **Fill** electrons and read `E_π`, HOMO, LUMO, gap — `O(N)`.

For a **batch of `M` molecules**, steps 1–4 are independent per molecule. Serial CPU cost ≈ `M · O(N³)`.
The parallel opportunity is twofold: build all `M` matrices at once, and diagonalise all `M` at once.

### Padding (the batch detail that bites)

A batched eigensolver needs **uniform dimension**. We pad every molecule to `N_max` (the largest atom
count in the batch) by adding **isolated atoms**. Naively those padding atoms sit at `ε = α = 0` — and
they would **interleave with physical MOs** when sorted, indistinguishable from the genuine `ε = 0` MOs of
allyl/cyclobutadiene. The fix (in [`tight_binding.h`](src/tight_binding.h)): give each padding atom a
**huge on-site energy** `TB_PAD_DIAG = 10⁶`. The padded matrix is block-diagonal — *(physical block)* ⊕
*(diagonal 10⁶ block)* — so the physical eigenvalues are unchanged, and after an ascending sort the
**first `N_real` eigenvalues are exactly the physical ones**; the padding eigenvalues pile up at the top
where the analysis never looks.

## GPU mapping

```
            adjacency bytes (small)            cuSOLVER batched Jacobi
  host  ───────────────────────────►  d_adj  ───────────────────────────►  d_W  ──► host
                                          │                                  ▲
                                          ▼                                  │
                            build_hamiltonians_kernel  ──►  d_H  ────────────┘
                            (one thread per matrix elem)    (M × N × N)   (overwritten
                                                                          with eigenvectors)
```

**Stage 1 — batched Hamiltonian build (custom kernel).** Each of the `M·N²` matrix entries is independent,
so we assign **one thread per entry**. The launch is a 2-D block (`16×16 = 256` threads, a good occupancy
default on sm_75…sm_89) tiling the `N×N` matrix, with `blockIdx.z` selecting the molecule. Thread
`(mol, i, j)` writes `d_H[mol·N² + i·N + j] = tb_hamiltonian_entry(i, j, …)`. **Memory:** plain global
reads (adjacency) and writes (matrix) — no shared memory, no atomics, because every output is written by
exactly one thread (no races → deterministic). We upload only the small **byte adjacency** and build the
big `double` matrices on-device, saving a host→device matrix copy. This is the catalog's *"parallelise the
sparse Hamiltonian construction over molecule batches"*.

**Stage 2 — batched symmetric eigensolve (cuSOLVER, not a black box).**
`cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)` solves
`A_k x = λx` for **every** symmetric matrix `A_k` in the batch in a **single launch**, using the **Jacobi**
method (same family as our CPU reference → they agree to machine precision). With `jobz = VECTOR` it also
overwrites each `A_k` with that matrix's orthonormal eigenvectors as columns; `uplo = LOWER` reads the
lower triangle. **Data layout:** a contiguous array of column-major `n×n` matrices, stride `n²`. Our
matrices are **symmetric**, so row-major == column-major and we pass `d_H` directly with `lda = n`, *no
transpose*. Eigenvalues come back as `M` contiguous length-`n` ascending blocks in `d_W`. The `syevjInfo`
object lets us set the Jacobi tolerance (`1e-14`) and sweep cap (`100`) so behaviour is reproducible.

**Why use the library** (CLAUDE.md §6.1.6): a batched symmetric eigensolver is a *solved* problem with a
tuned implementation. Hand-rolling it means a per-matrix shared-memory Jacobi kernel with a parallel
off-diagonal reduction and careful convergence handling — a project in itself. We *use* cuSOLVER and
*document exactly what it computes*. (For very small `n`, mapping one matrix per block to a shared-memory
Jacobi is the classic exercise; see the exercises.)

**Memory hierarchy summary:** adjacency and matrices live in **global memory**; the build kernel's per-
thread indices live in **registers**; cuSOLVER manages its own shared-memory rotations internally.

## Numerical considerations

- **Precision: FP64.** Eigenvalues are computed in **double precision**. The matrices are tiny and
  well-conditioned, so this is cheap and removes precision as a worry — both sides land on the same answer.
- **Determinism.** The build kernel is race-free (one writer per element). The eigenvalues are sorted with
  a **stable, index-tie-broken** comparator. Reductions use no floating-point `atomicAdd`, so there is no
  order-dependent summation (PATTERNS.md §3). Run-to-run **stdout is byte-identical** (verified); timings go
  to **stderr**.
- **The `±0` trap.** Several molecules have eigenvalues **exactly 0** by symmetry (allyl's non-bonding MO,
  cyclobutadiene's degenerate pair). Two solvers may return that as `+1e-16` or `−1e-16`, which `"%.6f"`
  would print as `0.000000` vs `−0.000000` — a spurious stdout diff. `clean_zero()` in `main.cu` snaps tiny
  magnitudes to `+0` before printing (far below any physical energy, so nothing chemical changes).
- **Padding shift size.** `TB_PAD_DIAG = 10⁶` is large vs. physical MOs (`~[−3,3]|β|`) yet tiny vs.
  double's range, so it cleanly separates padding eigenvalues without polluting the physical block.

## How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **GPU vs CPU (cross-implementation).** Per molecule, the GPU's cuSOLVER eigenvalues are compared to the
   CPU Jacobi eigenvalues. Both diagonalise the **identical** `double` matrix (guaranteed by the shared
   `tb_hamiltonian_entry()` core), so they agree to **~machine precision**. We assert a worst-case
   difference below `1.0e-09`; we actually observe `~3e-15`.
2. **Against analytic chemistry (the science).** The reported energies match the **closed-form Hückel
   results** above: ethylene `2|β|`, butadiene `4.472|β|`, benzene `8|β|` with gap `2|β|`, naphthalene
   `13.683|β|`, and cyclobutadiene's **zero gap**. This validates not just that CPU==GPU, but that the
   model is implemented correctly.

Edge cases covered: odd electron counts (allyl, cyclopentadienyl as radicals → singly-occupied HOMO),
degeneracies (benzene, cyclobutadiene), and padding (every molecule smaller than `N_max = 10`).

## Where this sits in the real world

Hückel is the **floor** of a tall family; production methods keep the exact same *pipeline* (build a
tight-binding Hamiltonian → diagonalise → occupy → read observables) and make each piece richer:

| Method | What changes vs. Hückel here |
|---|---|
| **Extended Hückel** | All valence orbitals (s, p), distance-dependent off-diagonals via the Wolfsberg–Helmholz formula and a real overlap matrix `S` (generalised eigenproblem `Hc = εSc`). |
| **MNDO/AM1/PM6/PM7** | Include all valence electrons and **electron–electron repulsion** (two-electron integrals, parameterised), solved **self-consistently** (SCF loop): build → diagonalise → update density → repeat. |
| **DFTB / GFN1-/GFN2-xTB** | Tight binding *derived from DFT*; **self-consistent charges** (SCC), element- and distance-dependent `H`/`S`, dispersion. xTB is the modern workhorse for conformer ranking and QM-ADMET; DFTB+ accelerates the eigensolves on GPU (ELPA). |
| **GFN-FF** | A force field *parameterised from* tight binding — drops the eigensolve entirely for speed. |

Crucially, the GPU strategy we demonstrate — **batch thousands of small molecules and diagonalise them
together** — is precisely how these tools achieve library-scale throughput. The eigensolve is the same
dense-symmetric problem; only the matrix's contents and an outer self-consistency loop get more elaborate.
Real energies are reported in kcal/mol or eV (via the method's empirical parameters), not the relative
`|β|` units used here.

## References

- E. Hückel, *Zeitschrift für Physik* **70**, 204 (1931) — the original π-electron theory.
- Grimme et al., *GFN2-xTB*, J. Chem. Theory Comput. **15**, 1652 (2019); the
  [xtb](https://github.com/grimme-lab/xtb) and [TBLite](https://github.com/tblite/tblite) repos.
- [DFTB+](https://github.com/dftbplus/dftbplus) — GPU-accelerated DFTB.
- NVIDIA cuSOLVER documentation — `cusolverDnDsyevjBatched` (batched symmetric eigensolver).
