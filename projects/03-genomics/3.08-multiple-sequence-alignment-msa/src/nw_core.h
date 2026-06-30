// ===========================================================================
// src/nw_core.h  --  The ONE shared Needleman-Wunsch core (CPU == GPU parity)
// ---------------------------------------------------------------------------
// Project 3.8 : Multiple Sequence Alignment (MSA)
//
// WHY THIS FILE EXISTS  (PATTERNS.md §2 -- the single most useful idiom)
//   STAGE 1 of progressive MSA scores every pair of sequences with a global
//   Needleman-Wunsch (NW) alignment. We run that scoring in TWO places:
//     * reference_cpu.cpp -- the trusted serial baseline (host compiler), and
//     * kernels.cu        -- the GPU version we are teaching (nvcc).
//   If those two used even slightly different arithmetic, the verification
//   "GPU == CPU" would be approximate and fragile. Instead we put the entire
//   per-pair recurrence in ONE `__host__ __device__` function, `nw_score_core`,
//   compiled by BOTH toolchains from the SAME source. Integer scores + identical
//   code  =>  the GPU and CPU distance matrices are BIT-IDENTICAL (exact check).
//
//   The trick is the FOO_HD macro: under nvcc (`__CUDACC__` defined) it expands
//   to `__host__ __device__`, so the function compiles for both host and device;
//   under the plain host compiler it expands to nothing, so cl.exe/g++ sees an
//   ordinary inline function. No CUDA-only types (no `__global__`, no `dim3`)
//   appear here, which is what lets the host compiler include it.
//
// THE SCIENCE (kept tiny on purpose; full story in THEORY.md)
//   NW finds the optimal GLOBAL alignment of two sequences: it must use every
//   residue of both, inserting gaps as needed. Scoring is integer and linear:
//   +MATCH for identical residues, MISMATCH for a substitution, GAP per inserted
//   gap. The optimal score is the value in the bottom-right corner of the DP
//   matrix H, where
//
//       H[i][j] = max( H[i-1][j-1] + s(a_i, b_j),   // align a_i with b_j
//                      H[i-1][j]   + GAP,            // gap in b (consume a_i)
//                      H[i][j-1]   + GAP )           // gap in a (consume b_j)
//
//   with the boundary rows/cols seeded to i*GAP / j*GAP (every leading residue
//   must be paired against a gap). Unlike Smith-Waterman (project 3.01) there is
//   NO max-with-0: global alignment cannot "restart".
//
// MEMORY NOTE
//   We only need the SCORE, not the traceback, for the distance matrix. So this
//   core uses the classic TWO-ROW rolling buffer: row j depends only on row j-1,
//   so O(L) memory suffices instead of O(L^2). On the GPU those two rows live in
//   per-thread-block scratch; on the CPU they are two std::vector<int>s. Either
//   way the recurrence below is identical.
//
// READ THIS AFTER: nothing (start here); used by reference_cpu.* and kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>

// ---- The host/device decoration macro (PATTERNS.md §2) --------------------
#ifdef __CUDACC__
#define NW_HD __host__ __device__   // nvcc: build for BOTH host and device
#else
#define NW_HD                        // host compiler: the decorators don't exist
#endif

// ---- Scoring scheme (integers keep CPU and GPU bit-identical) -------------
// DNA alphabet, linear gap penalty. These are deliberately simple; a real tool
// would use an affine gap (open/extend) and, for protein, a BLOSUM/PAM matrix
// (see THEORY.md "real world"). Integer scores commute and associate exactly,
// so summation order never changes the result -- the basis of our exact check.
#define NW_MATCH     2     // reward for aligning two identical nucleotides
#define NW_MISMATCH (-1)   // penalty for aligning two different nucleotides
#define NW_GAP      (-2)   // penalty for one gap position (insertion/deletion)

// Substitution score s(x, y): +MATCH if the residues are equal, else MISMATCH.
// `x`,`y` are 0..3 codes (A,C,G,T). Marked NW_HD so the kernel and the CPU
// reference call the EXACT same function.
NW_HD inline int nw_subst(uint8_t x, uint8_t y) {
    return (x == y) ? NW_MATCH : NW_MISMATCH;
}

// ---------------------------------------------------------------------------
// nw_score_core: the optimal global-alignment SCORE of sequences a and b.
//   a, b      : pointers to the two encoded sequences (residues 0..3)
//   la, lb    : their lengths
//   prev,curr : caller-provided scratch, each at least (lb+1) ints. We roll two
//               DP rows through them (no O(L^2) matrix, no traceback). The caller
//               owns this memory -- on the GPU it is a slice of a shared/global
//               scratch buffer; on the CPU it is two vectors. Passing it in keeps
//               this function allocation-free, which it MUST be to run on device.
//   returns   : the integer NW score H[la][lb] (bottom-right corner).
//
// Complexity: O(la * lb) time, O(lb) space. Determinism: pure integer max/add,
// so it is bit-identical on host and device regardless of thread/lane.
//
// HOW THE ROLLING ROWS WORK
//   `prev` holds row (i-1), `curr` is being filled for row i. After finishing
//   row i we swap their roles. prev[0] / curr[0] are the j=0 boundary cells
//   (i*GAP). prev[j] for the very first row is the i=0 boundary (j*GAP).
// ---------------------------------------------------------------------------
NW_HD inline int nw_score_core(const uint8_t* a, int la,
                               const uint8_t* b, int lb,
                               int* prev, int* curr) {
    // Row i=0 boundary: aligning the first 0 residues of `a` against the first j
    // residues of `b` costs j gaps. prev <- {0, GAP, 2*GAP, ...}.
    for (int j = 0; j <= lb; ++j) prev[j] = j * NW_GAP;

    // Sweep rows i = 1..la. Each cell takes the best of diagonal/up/left.
    for (int i = 1; i <= la; ++i) {
        curr[0] = i * NW_GAP;                       // j=0 boundary for this row
        const uint8_t ai = a[i - 1];                // residue a_i (1-based -> 0-based)
        for (int j = 1; j <= lb; ++j) {
            const int diag = prev[j - 1] + nw_subst(ai, b[j - 1]); // align a_i,b_j
            const int up   = prev[j]     + NW_GAP;                  // gap in b
            const int left = curr[j - 1] + NW_GAP;                  // gap in a
            // Branchless-ish max of the three candidates (integer, deterministic).
            int v = diag;
            if (up   > v) v = up;
            if (left > v) v = left;
            curr[j] = v;
        }
        // Roll: this row becomes "prev" for the next iteration. We swap POINTERS,
        // not data -- O(1), and works identically on host and device.
        int* tmp = prev; prev = curr; curr = tmp;
    }
    // After the final swap, `prev` holds row `la`; its last cell is the score.
    return prev[lb];
}

// Self-alignment score of a sequence against itself: every residue matches, no
// gaps, so the score is simply la*MATCH. Used to NORMALISE the pairwise score
// into a distance in [0,1] (see distance_matrix_cpu / the GPU twin). Marked NW_HD
// for symmetry though only the host needs it.
NW_HD inline int nw_self_score(int la) {
    return la * NW_MATCH;
}
