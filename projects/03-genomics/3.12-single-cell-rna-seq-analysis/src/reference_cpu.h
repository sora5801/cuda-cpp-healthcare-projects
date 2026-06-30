// ===========================================================================
// src/reference_cpu.h  --  Dataset type + CPU reference prototypes
// ---------------------------------------------------------------------------
// Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
//
// WHY A SEPARATE HEADER
//   reference_cpu.cpp is compiled by the plain C++ compiler and must NOT see any
//   CUDA/__global__ syntax, so its prototypes cannot live in kernels.cuh. Both
//   main.cu (nvcc) and reference_cpu.cpp (cl.exe/g++) include THIS pure-C++
//   header so they agree on the dataset layout and function signatures. The
//   actual per-element math is shared separately in scrna.h (host+device).
//
// THE CONTRACT
//   We load a tiny scRNA-seq COUNT MATRIX, normalize it (counts-per-target +
//   log1p), then build an exact k-nearest-neighbour CELL GRAPH. The CPU
//   reference here does all of that serially; the GPU twin (kernels.cu) does it
//   in parallel; main.cu asserts the two agree (CLAUDE.md section 5).
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu.  READ scrna.h FIRST (the math).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Dataset: a tiny in-memory scRNA-seq experiment.
//   counts : [N*G] RAW integer count matrix, row-major (row c = cell c, stored
//            as float for uniform I/O; values are non-negative whole numbers).
//   labels : [N] OPTIONAL ground-truth cell-type id per cell (synthetic data
//            embeds 3 separable types so we can sanity-check that KNN connects
//            same-type cells). -1 means "unknown". Not used by the algorithm,
//            only by the human-readable report.
//   N, G   : number of cells and genes.
//   k      : neighbours per cell to find (the KNN graph degree).
//   target_sum : the fixed total counts every cell is normalized to (e.g. 1e4).
// ---------------------------------------------------------------------------
struct Dataset {
    std::vector<float> counts;   // [N*G] raw counts, row-major
    std::vector<int>   labels;   // [N] ground-truth type id, or -1
    int    N = 0;                // cells
    int    G = 0;                // genes
    int    k = 0;                // neighbours per cell
    double target_sum = 0.0;     // normalization target total
};

// ---------------------------------------------------------------------------
// KnnGraph: the result of the pipeline.
//   normalized : [N*G] the normalized (counts-per-target + log1p) matrix.
//   nbr_idx    : [N*k] neighbour cell indices, row c = cell c's k neighbours,
//                nearest first.
//   nbr_dist   : [N*k] the EUCLIDEAN distance (sqrt of squared) to each
//                neighbour, ascending, parallel to nbr_idx. (We store the true
//                distance here for the human report; ranking used squared.)
// ---------------------------------------------------------------------------
struct KnnGraph {
    std::vector<float> normalized;  // [N*G]
    std::vector<int>   nbr_idx;     // [N*k]
    std::vector<float> nbr_dist;    // [N*k] Euclidean distance
};

// load_dataset: parse the tiny text sample (format documented in data/README.md
//   and scripts/make_synthetic.py). Throws std::runtime_error on a malformed or
//   missing file so demos fail loudly instead of running on garbage.
Dataset load_dataset(const std::string& path);

// run_cpu: the serial reference pipeline (normalize -> brute-force KNN).
//   d   : the loaded dataset (read-only).
//   out : filled with the normalized matrix + KNN graph.
//   Returns nothing; results land in `out`. This is the baseline the GPU is
//   verified against (main.cu compares neighbour indices exactly and normalized
//   values / distances within a documented tolerance).
void run_cpu(const Dataset& d, KnnGraph& out);
