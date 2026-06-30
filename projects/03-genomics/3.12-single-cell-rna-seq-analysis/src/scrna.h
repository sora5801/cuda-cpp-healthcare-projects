// ===========================================================================
// src/scrna.h  --  Shared (host + device) single-cell RNA-seq primitives
// ---------------------------------------------------------------------------
// Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   A scRNA-seq experiment yields a COUNT MATRIX X of shape [N cells x G genes]:
//   X[c][g] = how many mRNA molecules of gene g were captured in cell c. The
//   standard downstream pipeline is: normalize -> select features -> PCA ->
//   build a k-nearest-neighbour (KNN) CELL GRAPH -> cluster/embed. The single
//   most GPU-impactful step (the deep dive in the catalog says so explicitly) is
//   the KNN GRAPH: comparing every cell to every other cell is O(N^2), which is
//   exactly the wall a CPU hits and a GPU demolishes. THIS teaching project
//   implements the two steps a learner can fully follow and verify exactly:
//
//     (1) NORMALIZE   : library-size normalize each cell to a fixed target
//                       total count, then take log1p. This removes the trivial
//                       "deeper-sequenced cells look different" artifact.       [per-cell, parallel]
//     (2) KNN GRAPH   : for each query cell, find its k nearest neighbours by
//                       Euclidean distance in the normalized space.            [per-cell vs all, parallel]
//
//   We deliberately do EXACT brute-force KNN, not the approximate ANN
//   (Faiss/HNSW) that production tools use, because the brute-force version is
//   (a) the honest O(N^2) baseline that motivates the GPU, and (b) verifiable
//   bit-for-bit against a CPU reference. THEORY.md "Where this sits in the real
//   world" explains how rapids-singlecell / Scanpy replace this with PCA + ANN.
//
// WHY A GPU
//   Real datasets are 10^5 - 10^7 cells. The KNN step compares N query cells
//   against N reference cells -> N^2 distance evaluations, each over G genes.
//   Every query cell is INDEPENDENT, so we give each query cell its own GPU
//   thread (the "score one item vs N, each independent" pattern -- see
//   docs/PATTERNS.md sec 1, exemplar 1.12). Normalization is even simpler: one
//   thread per cell, no communication.
//
// CPU/GPU PARITY (the load-bearing idiom -- docs/PATTERNS.md sec 2)
//   The per-element math (normalize one entry; squared distance between two
//   cells; insert a candidate into a top-k list) lives HERE as __host__
//   __device__ (SC_HD) inline functions. The CPU reference loops them; the GPU
//   kernel calls them from one thread. Same code -> the KNN index lists match
//   EXACTLY (integer indices, deterministic tie-break) and the normalized values
//   match to the last bit. Keep CUDA-only constructs (no __global__) out of this
//   header so the plain host compiler (cl.exe / g++) can include it too.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu, main.cu.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <cmath>     // std::log1p, std::sqrt  (host); device uses ::log1p/::sqrt

// ---------------------------------------------------------------------------
// SC_HD: the host/device decorator macro.
//   When this header is compiled by nvcc (__CUDACC__ defined), SC_HD expands to
//   `__host__ __device__` so the function is emitted for BOTH the CPU and the
//   GPU. When compiled by the plain host compiler for reference_cpu.cpp, the
//   decorators do not exist, so SC_HD expands to nothing. One source, two
//   targets, identical arithmetic.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define SC_HD __host__ __device__
#else
#define SC_HD
#endif

// ---------------------------------------------------------------------------
// Problem-size limits, fixed at compile time so device threads can use small
// stack arrays (no dynamic allocation inside a kernel).
//   SC_MAX_GENES : the widest gene panel this teaching build accepts. Real
//                  panels are ~20k-30k genes; here we keep a tiny synthetic
//                  panel so the whole demo runs offline in milliseconds. The
//                  loader rejects anything wider with a clear error.
//   SC_MAX_K     : the largest neighbour count k. Each thread keeps a length-k
//                  top list in registers/local memory; small k keeps it cheap.
// ---------------------------------------------------------------------------
static const int SC_MAX_GENES = 64;   // max genes G in this teaching build
static const int SC_MAX_K     = 16;   // max neighbours k per cell

// ---------------------------------------------------------------------------
// sc_normalize_entry: normalize ONE count-matrix entry.
//   Inputs:
//     raw_count   : the raw integer UMI/read count X[c][g] (passed as float).
//     cell_total  : the cell's total counts sum_g X[c][g] (its "library size").
//     target_sum  : the fixed total every cell is scaled to (a constant, e.g.
//                   1e4 -- the classic Scanpy "counts per 10k", CP10K).
//   Returns:
//     log1p( raw_count / cell_total * target_sum )
//
//   WHY this formula. Two cells of the same biological type can differ wildly in
//   total counts purely because one was sequenced deeper. Dividing by the cell
//   total puts every cell on a common scale (a fraction of its library), then
//   multiplying by target_sum restores interpretable magnitudes. log1p =
//   log(1+x) compresses the heavy right tail of expression and is defined at
//   x=0 (a zero count -> log1p(0) = 0), which matters because scRNA-seq matrices
//   are ~90% zeros. This is exactly counts-per-10k + log1p, the de-facto default.
//
//   We compute in double then return double so the CPU and GPU agree to the bit;
//   main.cu stores the normalized matrix as float for the distance step (the
//   cast is identical on both sides).
// ---------------------------------------------------------------------------
SC_HD inline double sc_normalize_entry(double raw_count, double cell_total, double target_sum) {
    // Guard an all-zero cell (cell_total == 0): its normalized row is all zeros.
    // Without this guard we would divide by zero; with it, an empty droplet maps
    // to the origin, which is the sensible "no information" position.
    if (cell_total <= 0.0) return 0.0;
    const double scaled = raw_count / cell_total * target_sum;  // counts-per-target
    return ::log1p(scaled);  // log(1 + scaled); ::log1p resolves to the device intrinsic under nvcc
}

// ---------------------------------------------------------------------------
// sc_sqdist: squared Euclidean distance between two normalized cell vectors.
//   a, b : pointers to length-G rows of the normalized matrix (row-major).
//   G    : number of genes (the vector dimension).
//   Returns sum_g (a[g] - b[g])^2.  We return the SQUARED distance because the
//   square root is monotonic -- ranking neighbours by squared distance gives the
//   identical ordering as ranking by distance, and skipping sqrt is cheaper and
//   avoids a rounding step (so CPU and GPU stay bit-identical). main.cu takes the
//   sqrt only for the final human-readable report.
//
//   Accumulate in double so that, regardless of summation order, both sides get
//   the same value -- the loop here runs in the SAME order on CPU and GPU (g = 0
//   .. G-1), so even the floating-point rounding is identical.
// ---------------------------------------------------------------------------
SC_HD inline double sc_sqdist(const float* a, const float* b, int G) {
    double s = 0.0;
    for (int g = 0; g < G; ++g) {
        const double diff = static_cast<double>(a[g]) - static_cast<double>(b[g]);
        s += diff * diff;
    }
    return s;
}

// ---------------------------------------------------------------------------
// sc_knn_insert: maintain a fixed-size, ascending-by-distance top-k list.
//   This is the heart of brute-force KNN. As we scan every candidate neighbour,
//   we keep only the k smallest distances seen so far, in sorted order, together
//   with their cell indices. A length-k insertion sort is O(k) per insert and k
//   is tiny (<= 16), so the whole scan is O(N*k) per query cell -- cheap next to
//   the O(N*G) distance work.
//
//   Parameters (the list is two parallel arrays the caller owns):
//     dist  : [k] current best squared distances, ascending; UNUSED slots hold
//             a huge sentinel (+inf) so any real candidate beats them.
//     idx   : [k] the cell index paired with each dist slot (-1 when unused).
//     k     : list length.
//     cand_dist : the candidate's squared distance.
//     cand_idx  : the candidate's cell index.
//
//   TIE-BREAK (determinism!): we use STRICT `<` when deciding to insert and we
//   scan candidates in increasing index order, so among equal distances the
//   LOWER cell index always wins and lands first. The CPU reference and the GPU
//   kernel both call THIS function in the SAME candidate order, so the resulting
//   neighbour lists are identical down to the tie ordering -> exact verification.
// ---------------------------------------------------------------------------
SC_HD inline void sc_knn_insert(double* dist, int* idx, int k,
                                double cand_dist, int cand_idx) {
    // Fast reject: if the candidate is not better than the current worst (the
    // last, largest slot), it cannot enter the top-k. Strict `<` => ties with the
    // worst do not displace it (keeps the earlier-seen, lower-index neighbour).
    if (!(cand_dist < dist[k - 1])) return;

    // Find the insertion position p and shift larger entries one slot right,
    // classic insertion sort on a length-k array. We walk from the back so each
    // displaced entry moves exactly once.
    int p = k - 1;
    while (p > 0 && cand_dist < dist[p - 1]) {
        dist[p] = dist[p - 1];   // shift the bigger neighbour rightward
        idx[p]  = idx[p - 1];
        --p;
    }
    dist[p] = cand_dist;         // drop the candidate into its sorted slot
    idx[p]  = cand_idx;
}

// ---------------------------------------------------------------------------
// sc_knn_one_cell: compute the k nearest neighbours of ONE query cell.
//   This is the entire per-query computation, shared verbatim by the CPU loop
//   and the GPU kernel (one GPU thread runs this for its query cell).
//
//   Parameters:
//     X      : [N*G] the full normalized matrix, row-major (row c = cell c).
//     N      : number of cells.
//     G      : number of genes.
//     q      : the query cell index whose neighbours we want.
//     k      : how many neighbours to return.
//     out_idx  : [k] OUTPUT neighbour cell indices, nearest first.
//     out_dist : [k] OUTPUT squared distances, ascending (parallel to out_idx).
//
//   We EXCLUDE the query cell itself (a cell is trivially its own nearest
//   neighbour at distance 0) by skipping c == q -- the standard convention for a
//   KNN cell graph, where self-loops carry no information.
//
//   Local scratch arrays `best*` live on the thread's stack (registers/local
//   memory on the GPU). They are sized to the compile-time SC_MAX_K so no
//   dynamic allocation is needed inside a kernel.
// ---------------------------------------------------------------------------
SC_HD inline void sc_knn_one_cell(const float* X, int N, int G, int q, int k,
                                  int* out_idx, double* out_dist) {
    // Initialize the top-k list to "empty": +inf distances, -1 indices.
    double best_d[SC_MAX_K];
    int    best_i[SC_MAX_K];
    for (int j = 0; j < k; ++j) {
        best_d[j] = 1.0e308;   // ~DBL_MAX sentinel: any real distance is smaller
        best_i[j] = -1;
    }

    const float* qrow = X + static_cast<std::size_t>(q) * G;  // the query's gene vector

    // Scan EVERY other cell as a candidate neighbour, in increasing index order
    // (the order is what makes the tie-break deterministic, see sc_knn_insert).
    for (int c = 0; c < N; ++c) {
        if (c == q) continue;                                  // no self-loops
        const float* crow = X + static_cast<std::size_t>(c) * G;
        const double d = sc_sqdist(qrow, crow, G);             // shared distance math
        sc_knn_insert(best_d, best_i, k, d, c);                // shared top-k update
    }

    // Publish the sorted result.
    for (int j = 0; j < k; ++j) {
        out_idx[j]  = best_i[j];
        out_dist[j] = best_d[j];
    }
}
