# THEORY — 2.06 Normal Mode Analysis / Elastic Network Models

> For a reader who knows C++ but is new to CUDA and to structural biology. See
> [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

Proteins are not rigid — they flex, and their *large-scale* motions (a domain
swinging shut on a substrate, a channel breathing open) are central to function
and allostery. Remarkably, these slow collective motions are well captured by a
crude mechanical caricature: ignore chemistry, treat each Cα atom as a bead, and
connect nearby beads with identical springs. This **Elastic Network Model** (ENM),
analyzed by **Normal Mode Analysis** (NMA), reproduces the experimentally observed
flexible regions and motions surprisingly well.

## 2. The math

For `N` Cα atoms, the elastic energy is a sum of `½γ(d − d₀)²` over spring-connected
pairs. The **Hessian** `H` is the matrix of second derivatives of that energy with
respect to the `3N` coordinates — a `3N×3N` symmetric matrix. In the **Anisotropic
Network Model (ANM)**, the off-diagonal `3×3` block for a connected pair `(i,j)` is

```
H_ij = -(γ / d_ij²) · Δ Δᵀ ,   Δ = r_j − r_i      (a 3-vector)
H_ii = -Σ_{j≠i} H_ij                              (so each row sums to zero)
```

The **normal modes** are the eigenvectors of `H`: `H v_k = λ_k v_k`. `λ_k` is the
mode's squared frequency; small `λ` = soft, slow, large-amplitude motion. Because
the energy is invariant under rigid-body translation and rotation, **exactly 6
eigenvalues are zero** (3 translations + 3 rotations) — a built-in correctness
check. The lowest *non-zero* modes are the functional motions.

## 3. The algorithm

```
build H (3N x 3N) from the Cα coordinates and a cutoff
diagonalize H  ->  eigenvalues λ (ascending), eigenvectors v
discard the 6 zero modes; the next few are the functional modes
mobility_i = Σ_{k: λ_k>0} (1/λ_k) · |v_k restricted to atom i|²   (~ a B-factor)
```

**Complexity.** Building `H` is `O(N²)` (pairs). Diagonalizing it is `O((3N)³)` —
the dominant cost, and the GPU target.

## 4. The GPU mapping

**Use the library.** A dense symmetric eigendecomposition is exactly what
**cuSOLVER** provides, so we call it rather than hand-rolling a QR or Jacobi solver.
As with every library here, we **document what it does** (no black box):

```
cusolverDnDsyevd(handle, jobz=VECTOR, uplo=LOWER, n, A, lda, W, work, lwork, info);
// Solves A x = λ x for symmetric A by divide-and-conquer (O(n^3)).
// On exit: W holds the eigenvalues (ascending); A is overwritten with the
//          orthonormal eigenvectors as COLUMNS (column-major).
```

We first call `cusolverDnDsyevd_bufferSize` to learn the workspace size, allocate
it, then run the solve and check the `info` flag for convergence. Because the
Hessian is **symmetric**, its row-major and column-major layouts are identical, so
we upload it directly. The `O(N³)` solve is where the GPU's dense-linear-algebra
throughput pays off — modest here (`n=180`), decisive for real proteins
(`n = 3N` into the tens of thousands).

## 5. Numerical considerations

- **Precision.** Double throughout: eigenvalues span from ~0 (rigid-body) to
  `O(1)`, and the near-zero modes must be resolved cleanly.
- **Determinism.** cuSOLVER `Dsyevd` is deterministic for a fixed input, so the
  reported modes/mobility are reproducible.
- **Degeneracy.** Repeated eigenvalues have an arbitrary eigenvector *basis*, so we
  verify the **eigenvalues** (basis-independent) rather than eigenvectors; the
  mobility uses `|v|²` summed within a residue, which is basis-robust.

## 6. How we verify correctness

`main.cu` diagonalizes the same Hessian with cuSOLVER (GPU) and a transparent
**cyclic Jacobi** eigensolver (CPU) and compares the sorted eigenvalues
(`worst diff ≈ 1e-12`). Beyond CPU/GPU parity, the result is physically right: the
Hessian produces **exactly 6 zero modes** (the rigid-body invariance — a hard check
the model must pass), the next modes are the soft functional motions, and the
mobility profile flags the flexible loops, as a real ENM should.

## 7. Where this sits in the real world

ProDy, Bio3D, and ElNemo run ANM/GNM on real PDB structures and compare the
predicted mobility to crystallographic **B-factors** (they correlate well). For
large proteins one uses a **sparse** Hessian and a **Lanczos** iteration to extract
only the lowest modes (the functional ones) instead of a full dense solve — there
the GPU accelerates the sparse matrix-vector products. All-atom NMA with a real
force field gives true vibrational frequencies at far greater cost. The dense
symmetric eigensolve you call here via cuSOLVER is the computational core of the
ENM family.

## References

- Bahar, Atilgan & Erman (1997) — Gaussian Network Model (GNM).
- Atilgan et al. (2001) — Anisotropic Network Model (ANM).
- Tirion (1996) — single-parameter elastic network for protein dynamics.
- NVIDIA **cuSOLVER** documentation — dense symmetric eigensolvers (`syevd`).
