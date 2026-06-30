// ===========================================================================
// src/reference_cpu.h  --  Hi-C matrix container + shared host helpers + CPU ref
// ---------------------------------------------------------------------------
// Project 3.15 : Hi-C / 3D Genome Contact Analysis
//
// Pure C++ (no CUDA). The per-element math (fixed-point quantization, corrected
// contact) lives in hic.h and is shared with the GPU. This header declares:
//   * HicMatrix   -- the loaded sparse contact matrix (COO upper triangle).
//   * load_matrix -- read the tiny text sample (data/README.md format).
//   * the ICE post-step (apply a fresh row sum to the bias, then renormalise)
//     and the insulation-score / boundary-calling helpers, all reused by BOTH
//     the CPU reference and the GPU wrapper so the two produce identical output.
//   * ice_balance_cpu / insulation_score / call_boundaries -- the trusted serial
//     baselines main.cu verifies the GPU result against.
//
// READ THIS AFTER: hic.h. READ BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "hic.h"   // CooEntry, hic_to_fixed, hic_corrected, HIC_SCALE

// ---------------------------------------------------------------------------
// A loaded Hi-C contact matrix in sparse COO (coordinate) form.
//   We store only the upper triangle (i <= j); symmetry gives the rest. `n`
//   is the number of genomic bins (the matrix is n x n). `entries` holds the
//   nonzeros. This mirrors the .cool / .hic sparse storage that real tools use.
// ---------------------------------------------------------------------------
struct HicMatrix {
    int n = 0;                       // number of bins (matrix is n x n)
    std::vector<CooEntry> entries;   // nonzeros, upper triangle (i <= j)
};

// Load the text sample: first line "n nnz", then nnz lines "i j count".
// Throws std::runtime_error on a bad/missing file so demos fail loudly.
HicMatrix load_matrix(const std::string& path);

// ---------------------------------------------------------------------------
// ICE shared host helpers (used identically by the CPU ref and the GPU wrapper,
// so the per-iteration bias update is line-for-line the same on both paths).
// ---------------------------------------------------------------------------

// Given the per-bin row sums of the CURRENT corrected matrix, update the bias
// vector in place and report convergence.
//   ICE rule: b_k <- b_k * (rowsum_k / mean_rowsum). A bin already at the target
//   mean keeps its bias; an over-represented bin (rowsum > mean) gets a larger
//   bias, which divides its contacts down next iteration. Empty bins (rowsum==0)
//   are masked: bias stays 0 and they are excluded from the mean.
//   Returns the variance of the (nonzero) row sums about their mean -- our
//   convergence metric (it should fall toward 0 as the matrix balances).
double ice_update_bias(const std::vector<double>& rowsum,
                       std::vector<double>& bias);

// Compute, on the host, the per-bin row sums of M corrected by the given bias,
// using the SAME fixed-point quantization the GPU uses (hic_to_fixed), so the
// CPU sums exactly the integers the GPU atomically accumulates -> bit-identical.
void compute_rowsums_cpu(const HicMatrix& m, const std::vector<double>& bias,
                         std::vector<double>& rowsum);

// ---------------------------------------------------------------------------
// Insulation score + TAD boundary calling (shared metric).
// ---------------------------------------------------------------------------

// Insulation score of every bin from the BALANCED matrix (raw counts / biases).
//   For bin k it is the mean balanced contact over the `window` x `window`
//   "diamond" of pairs (a,b) with a in [k-window, k-1], b in [k, k+window-1]
//   (i.e. contacts that cross position k). A low score => a strong insulating
//   boundary. Bins too close to the matrix edge get a sentinel (-1) and are
//   skipped by the boundary caller. Returns score[n].
std::vector<double> insulation_score(const HicMatrix& m,
                                     const std::vector<double>& bias,
                                     int window);

// Call TAD boundaries as local minima of the insulation score: bin k is a
// boundary if score[k] is valid and strictly less than its `radius` neighbours
// on each side. Returns the sorted list of boundary bin indices (deterministic).
std::vector<int> call_boundaries(const std::vector<double>& score, int radius);

// ---------------------------------------------------------------------------
// CPU reference driver (the trusted serial baseline main.cu verifies against).
// ---------------------------------------------------------------------------

// Run `iters` ICE iterations on the host; fill `bias` (size n). Returns the
// final convergence variance. `bias` starts at all-ones for occupied bins,
// 0 for empty (masked) bins.
double ice_balance_cpu(const HicMatrix& m, int iters, std::vector<double>& bias);
