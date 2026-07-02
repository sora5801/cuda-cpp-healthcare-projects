// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for GRN inference
// ---------------------------------------------------------------------------
// Project 6.13 : Gene Regulatory Network Inference (ARACNE: MI + DPI)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the expression matrix container, the
//   file loader, the discretization pass) and the CPU reference prototypes live
//   here. kernels.cuh also includes this header to reuse the GrnData type and
//   the layout constants -- nothing CUDA-specific leaks in either direction.
//   The per-pair MI math itself lives in grn.h (the __host__ __device__ core).
//
// THE PIPELINE (see ../THEORY.md for the full derivation)
//   1. Load an expression matrix: G genes x S samples (cells), row-major.
//   2. Discretize each gene into N_BINS equal-width bins (grn.h::discretize).
//   3. For every unordered gene pair (i<j): build the B x B joint histogram and
//      score it with mi_from_joint() -> a dense symmetric G x G MI matrix.
//   4. DPI prune: for every triangle (i,j,k), if I(i,j) is the smallest of the
//      three edges by more than a tolerance, mark edge (i,j) as INDIRECT.
//   The GPU (kernels.cu) does steps 2-4 in parallel; this file is the trusted
//   serial twin used to VERIFY it.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.  READ grn.h FIRST.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "grn.h"   // N_BINS, JOINT_CELLS, discretize_value(), mi_from_joint()

// ---------------------------------------------------------------------------
// GrnData : a loaded gene-expression dataset plus its discretized copy.
//   n_genes   : G, the number of genes (rows).
//   n_samples : S, the number of samples / single cells (columns).
//   expr      : [G * S] raw expression, ROW-MAJOR (gene g, sample s at g*S + s).
//   disc      : [G * S] the same matrix discretized to bin indices in [0,B);
//               filled by discretize_matrix() and consumed by the MI step.
//   gene_names: optional G labels (for a readable report); may be empty.
// ---------------------------------------------------------------------------
struct GrnData {
    int n_genes   = 0;
    int n_samples = 0;
    std::vector<double>  expr;         // [G*S] raw, row-major
    std::vector<uint8_t> disc;         // [G*S] discretized bins, row-major
    std::vector<std::string> gene_names;
};

// Load a dataset from the text format documented in data/README.md:
//   line 1:  "<n_genes> <n_samples>"
//   next G:  each line = "<gene_name> v0 v1 ... v(S-1)"  (S expression values)
// Throws std::runtime_error on a missing file or a shape mismatch.
GrnData load_expression(const std::string& path);

// discretize_matrix : fill data.disc from data.expr using grn.h's equal-width
// binning. Each gene is binned against ITS OWN [min,max] range (per-gene
// dynamic range), exactly as the GPU does, so the two `disc` matrices match.
// This is a pure host helper (no CUDA), shared by both code paths' CPU setup;
// the GPU recomputes the identical binning on-device in a kernel.
void discretize_matrix(GrnData& data);

// mi_matrix_cpu : the trusted serial MI computation. Fills `mi` (resized to
// G*G, symmetric, zero diagonal) with I(i;j) in nats for every gene pair, using
// the discretized matrix in `data`. This is the O(G^2 * S) baseline the GPU
// kernel is verified against and timed against.
void mi_matrix_cpu(const GrnData& data, std::vector<double>& mi);

// dpi_prune_cpu : apply the Data Processing Inequality. Given the symmetric MI
// matrix, produce a G*G byte mask `keep` where keep[i*G+j]==1 means edge (i,j)
// survived pruning (a DIRECT edge) and 0 means it was removed as INDIRECT or is
// below `mi_threshold`. `tolerance` is the DPI slack (an edge is pruned only if
// it is smaller than BOTH other triangle edges by more than this, avoiding the
// removal of near-ties). Deterministic: threshold comparisons only.
void dpi_prune_cpu(const std::vector<double>& mi, int n_genes,
                   double mi_threshold, double tolerance,
                   std::vector<uint8_t>& keep);
