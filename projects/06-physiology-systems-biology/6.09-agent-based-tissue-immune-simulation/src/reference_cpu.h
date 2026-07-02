// ===========================================================================
// src/reference_cpu.h  --  Loader, spatial binning, CPU reference + result type
// ---------------------------------------------------------------------------
// Project 6.9 : Agent-Based Tissue / Immune Simulation
//
// Pure C++ (no CUDA). kernels.cu reuses AbmParams/Cells and the spatial-binning
// helpers declared here (the binning is a host-side sort, identical for CPU and
// GPU, so the neighbour order matches exactly). The per-cell / per-grid physics
// itself lives in the shared abm_core.h so CPU and GPU compute byte-for-byte
// identical results.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "abm_core.h"   // AbmParams, Cells, and the shared __host__ __device__ math

// ---------------------------------------------------------------------------
// AbmResult: the DETERMINISTIC summary of a run that main.cu prints to stdout
// and that CPU and GPU must agree on. Chosen to be reproducible integers/rounded
// doubles so the demo's stdout is byte-identical every run.
// ---------------------------------------------------------------------------
struct AbmResult {
    unsigned long long total_quanta = 0; // total chemokine quanta in the field (exact int)
    double total_chemokine = 0.0;        // = total_quanta / QUANTA_PER_UNIT
    double peak_chemokine = 0.0;         // maximum concentration in any grid cell
    int    peak_col = 0, peak_row = 0;   // where that peak is
    double mean_immune_tumor_dist = 0.0; // mean distance from each immune cell to
                                         // the tumor centroid (shrinks as immune
                                         // cells chemotax inward) -- the science check
    int    n_tumor = 0, n_immune = 0;    // agent counts by type
    // The final cell positions (for the exact CPU-vs-GPU verification in main).
    std::vector<double> x, y;
};

// ---------------------------------------------------------------------------
// SpatialBins: the O(N) neighbour-search acceleration structure. We hash each
// cell into a square bin of side `bin_size` (>= the interaction diameter, so all
// possible overlaps are within the 3x3 neighbouring bins), then produce a
// bin-sorted list of cell indices. Built identically for CPU and GPU so the
// neighbour scan order -- and therefore the summed force -- matches exactly.
//   bin_start[b] .. bin_start[b]+bin_count[b]  index into `sorted`
//   sorted[k]                                  is a cell index
// ---------------------------------------------------------------------------
struct SpatialBins {
    int bins_x = 0, bins_y = 0;      // bin grid dimensions
    double bin_size = 0.0;           // bin side length (domain units)
    std::vector<int> bin_start;      // [bins_x*bins_y] first slot of each bin in `sorted`
    std::vector<int> bin_count;      // [bins_x*bins_y] number of cells in each bin
    std::vector<int> sorted;         // [n] cell indices grouped by bin, ascending
};

// Build the spatial bins for the current cell positions. Deterministic: cells
// are placed into `sorted` in ascending cell-index order within each bin (a
// counting sort), which fixes the neighbour scan order for both CPU and GPU.
void build_bins(const AbmParams& p, const Cells& c, SpatialBins& bins);

// Load AbmParams from the one-line sample file (format documented in data/README):
//   gx gy dx steps dt D decay secretion radius k_rep chemotaxis seed n_tumor n_immune
// The cell layout is then generated deterministically from `seed` (see .cpp) so
// the whole simulation is reproducible from the sample alone.
AbmParams load_abm(const std::string& path, Cells& cells);

// CPU reference: run `steps` of secrete -> diffuse -> move on the host and return
// the deterministic summary. This is the trusted baseline the GPU is checked
// against. `field_out` returns the final chemokine field for field-level checks.
AbmResult abm_cpu(const AbmParams& p, const Cells& cells0,
                  std::vector<double>& field_out);

// Compute the deterministic AbmResult summary from a final cell/field state.
// Shared by both paths so the comparison is apples-to-apples.
AbmResult summarize(const AbmParams& p, const std::vector<double>& x,
                    const std::vector<double>& y, const std::vector<int>& type,
                    const std::vector<double>& field);
