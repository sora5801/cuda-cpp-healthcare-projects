// ===========================================================================
// src/tight_binding.h  --  The ONE shared "per-element physics" of the model
// ---------------------------------------------------------------------------
// Project 1.8 : Semi-Empirical & Tight-Binding Quantum Methods
//
// THE SHARED __host__ __device__ CORE (PATTERNS.md §2 -- "CPU/GPU parity")
//   This header holds the *one true formula* for every quantity that both the
//   CPU reference (reference_cpu.cpp, compiled by cl.exe/g++) AND the GPU code
//   (kernels.cu, compiled by nvcc) must compute IDENTICALLY. Putting it in a
//   single header that BOTH compilers include is what makes verification exact
//   instead of "close enough": the Huckel matrix the CPU builds is bit-for-bit
//   the matrix the GPU builds, so any eigenvalue disagreement can only come
//   from the two eigensolvers -- which is exactly the thing we want to study.
//
//   To make the SAME inline function callable from host code and from a CUDA
//   kernel, we decorate it `__host__ __device__` -- but ONLY when nvcc is the
//   compiler (the host compiler has never heard of those keywords). The
//   TB_HD macro below expands to the decorators under nvcc and to nothing
//   otherwise. This is the "HD-macro idiom".
//
//   HARD RULE: keep CUDA-only constructs (no `__global__`, no `<<<>>>`, no
//   device-only types) OUT of this header so the plain host compiler can
//   include it without nvcc.
//
// WHAT THE MODEL IS (the science lives in ../THEORY.md):
//   We model the delocalised pi-electrons of a planar conjugated hydrocarbon
//   with Huckel Molecular Orbital (HMO) theory -- the simplest tight-binding
//   model in chemistry and the conceptual ancestor of GFN-xTB / DFTB. Each
//   sp2 carbon contributes ONE pi atomic orbital (one basis function). The
//   model Hamiltonian H is an N x N real symmetric matrix:
//
//       H[i][i] = alpha                 (Coulomb integral; on-site energy)
//       H[i][j] = beta  if i,j bonded   (resonance integral; hopping term)
//       H[i][j] = 0     otherwise
//
//   We work in "beta units" relative to alpha: energies are reported as
//   (E - alpha)/|beta|, i.e. we set alpha = 0 and beta = -1. This is the
//   textbook convention (Huckel), keeps the numbers clean, and is exactly how
//   tight-binding Hamiltonians are tabulated. The eigenvalues of H are the
//   molecular-orbital (MO) energies; the eigenvectors are the MO coefficients.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu (all build on it).
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// TB_HD : decorate inline functions so the SAME source compiles for host & GPU.
//   * Under nvcc, __CUDACC__ is defined -> expand to __host__ __device__ so the
//     function has both a CPU and a GPU code path.
//   * Under the plain host compiler, expand to nothing.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define TB_HD __host__ __device__
#else
#define TB_HD
#endif

// ---------------------------------------------------------------------------
// Huckel parameter convention (see header comment). Reported energies are in
// units of |beta| relative to alpha, so we fix:
//     ALPHA = 0           (zero of energy = the isolated 2p Coulomb integral)
//     BETA  = -1          (resonance integral; negative => bonding is stabilising)
// A more elaborate semi-empirical method (PM7, GFN2-xTB) replaces these two
// constants with element- and distance-dependent parameter functions, but the
// matrix-build / diagonalise / fill-electrons pipeline is identical -- which is
// the whole point of teaching the simplest member of the family first.
// ---------------------------------------------------------------------------
#define TB_ALPHA  (0.0)
#define TB_BETA   (-1.0)

// ---------------------------------------------------------------------------
// TB_PAD_DIAG : the on-site energy we put on PADDING atoms (index >= n_real).
//   THE PADDING PROBLEM (and its fix):
//     cuSOLVER's batched eigensolver needs every matrix in the batch to be the
//     SAME dimension `ld` (= the largest molecule's atom count). Smaller
//     molecules are padded out to `ld` with isolated atoms. If those padding
//     atoms sat at the physical on-site energy alpha=0, their eigenvalues (also
//     0) would INTERLEAVE with the real molecular-orbital energies when we sort
//     ascending -- and several real systems (allyl, cyclobutadiene) genuinely
//     have an MO at exactly 0, so we could not tell padding from physics.
//
//   THE FIX: give every padding atom a HUGE positive on-site energy. Its
//   eigenvalue is then ~ +TB_PAD_DIAG, far above any physical pi MO (which lie
//   in roughly [-3, +3] |beta|). After sorting ascending, the FIRST n_real
//   eigenvalues are exactly the physical ones and the padding eigenvalues pile
//   up at the top where analyze_molecule() never looks. Because padding atoms
//   are isolated (no bonds), this shift cannot perturb the physical block at
//   all -- the matrix is block-diagonal (physical block) + (diagonal padding).
//
//   1e6 is comfortably larger than any pi MO yet tiny vs. double's range, so it
//   introduces no precision loss in the physical eigenvalues.
// ---------------------------------------------------------------------------
#define TB_PAD_DIAG  (1.0e6)

// ---------------------------------------------------------------------------
// tb_hamiltonian_entry
//   The single source of truth for "what is H[i][j] for this PADDED molecule?".
//
//   Inputs:
//     i, j     : padded pi-orbital indices, 0..ld-1
//     adj      : the molecule's ld x ld padded adjacency matrix, row-major,
//                bytes (1 = bonded, 0 = not). adj[i*ld + j] == adj[j*ld + i].
//                Padding rows/cols (index >= n_real) are all zero (isolated).
//     ld       : the PADDED leading dimension (matrix stride) = batch max atoms
//     n_real   : the molecule's TRUE atom count (<= ld); indices >= n_real are
//                padding and receive the large TB_PAD_DIAG on-site energy.
//   Returns:
//     the padded Huckel matrix element H[i][j] as a double.
//
//   WHY a function (not just a memcpy): both the CPU reference and the GPU
//   matrix-builder kernel call THIS, so the matrix they diagonalise is provably
//   identical. The diagonal carries alpha (physical) or TB_PAD_DIAG (padding);
//   off-diagonals carry beta exactly where the adjacency says there is a bond.
//
//   Complexity: O(1). Called ld*ld times to fill one molecule's padded matrix.
// ---------------------------------------------------------------------------
TB_HD inline double tb_hamiltonian_entry(int i, int j,
                                         const unsigned char* adj,
                                         int ld, int n_real) {
    if (i == j) {
        // Diagonal: physical atoms get alpha; padding atoms get the big shift
        // so their eigenvalues never mix with the physical spectrum.
        return (i < n_real) ? TB_ALPHA : TB_PAD_DIAG;
    }
    // Off-diagonal: a hopping (resonance) integral beta IFF atoms i,j are bonded.
    // Padding rows/cols have all-zero adjacency, so this is 0 for them.
    const unsigned char bonded = adj[(std::size_t)i * ld + j];
    return bonded ? TB_BETA : 0.0;
}

// ---------------------------------------------------------------------------
// tb_num_pi_electrons
//   In a neutral conjugated hydrocarbon each sp2 carbon donates exactly ONE
//   electron to the pi-system, so the pi-electron count equals the atom count.
//   We expose it as a function so a future variant (charged species, hetero-
//   atoms donating a lone pair) has one obvious place to change.
//
//   Returns the number of pi electrons for an n-atom neutral pi-system = n.
// ---------------------------------------------------------------------------
TB_HD inline int tb_num_pi_electrons(int n) {
    return n;   // neutral, one 2p_z electron per sp2 carbon
}
