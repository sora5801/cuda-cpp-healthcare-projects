// ===========================================================================
// src/reference_cpu.h  --  Volume + shared feature helpers + CPU reference
// ---------------------------------------------------------------------------
// Project 4.27 : Radiomics Feature Extraction
//
// Pure C++ (no CUDA). The per-voxel math is in radiomics.h. The GLCM->features
// reduction, the first-order statistics, and the loader live here and are reused
// by BOTH the CPU reference (reference_cpu.cpp) and the GPU wrapper (kernels.cu),
// so the two produce identical feature vectors. kernels.cu reuses Volume + these
// helpers; only the GLCM *counting* step differs (serial loop vs. atomic scatter).
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "radiomics.h"   // rad_quantize, rad_fill_directions, indexing, RAD_NUM_DIRECTIONS

// ---------------------------------------------------------------------------
// A loaded ROI volume. The image is a dense nx*ny*nz grid of intensities; a
// parallel MASK marks which voxels belong to the segmented region of interest.
// Only masked voxels contribute to features -- this mirrors real radiomics,
// where the tumour is a small blob inside a larger scan.
// ---------------------------------------------------------------------------
struct Volume {
    int nx = 0, ny = 0, nz = 0;        // grid dimensions (x fastest)
    int Ng = 8;                        // number of gray levels after quantization
    std::vector<float>   intensity;    // [nx*ny*nz] raw intensities (e.g. Hounsfield units)
    std::vector<uint8_t> mask;         // [nx*ny*nz] 1 = inside ROI, 0 = outside
    float vmin = 0.0f, vmax = 0.0f;    // intensity range over the ROI (for quantization)
    int   nroi = 0;                    // number of masked (ROI) voxels

    std::size_t voxels() const {
        return static_cast<std::size_t>(nx) * ny * nz;
    }
};

// ---------------------------------------------------------------------------
// The feature bundle we compute. Two families:
//   * first-order: from the ROI gray-level histogram (order-independent).
//   * Haralick texture: from the direction-summed, symmetrized, normalized GLCM.
// All are doubles: the counts are integers (exact), but the derived features
// involve logs/divisions, so we carry full double precision.
// ---------------------------------------------------------------------------
struct Features {
    // First-order (on quantized gray levels 0..Ng-1 inside the ROI):
    double mean = 0.0;        // average gray level
    double variance = 0.0;    // spread of gray levels
    double energy = 0.0;      // sum of squared levels (uniformity of magnitude)
    double entropy = 0.0;     // Shannon entropy of the histogram (bits); randomness

    // Haralick texture (on the normalized GLCM, summed over the 13 directions):
    double glcm_contrast = 0.0;      // sum P(i,j)*(i-j)^2 ; local intensity variation
    double glcm_energy = 0.0;        // sum P(i,j)^2 (angular second moment); orderliness
    double glcm_homogeneity = 0.0;   // sum P(i,j)/(1+(i-j)^2); closeness to the diagonal
    double glcm_correlation = 0.0;   // linear dependency of i,j gray levels [-1,1]
    double glcm_entropy = 0.0;       // Shannon entropy of the GLCM (bits)

    // Bookkeeping (also printed, and checked):
    long long glcm_total = 0;        // total co-occurrence pairs counted (integer, exact)
};

// ---- Loader --------------------------------------------------------------
// Load from the text format (see data/README.md): a header line
//   "nx ny nz Ng", then nx*ny*nz intensities, then nx*ny*nz mask flags (0/1).
// Also computes vmin/vmax and nroi over the masked voxels.
Volume load_volume(const std::string& path);

// ---- Shared reductions (identical on CPU and GPU paths) -------------------

// Quantize every ROI voxel and build the ROI gray-level HISTOGRAM (length Ng).
// Shared so CPU and GPU report identical first-order features.
void build_histogram(const Volume& v, std::vector<long long>& hist);

// First-order features from a gray-level histogram.
void first_order_from_histogram(const Volume& v, const std::vector<long long>& hist,
                                Features& f);

// SYMMETRIZE + NORMALIZE a raw direction-summed GLCM (counts) into a probability
// matrix P, then read off the Haralick features. `glcm` is the summed Ng x Ng
// COUNT matrix (already added over all 13 directions AND both scan orders on the
// counting side). Shared by CPU and GPU so identical counts -> identical features.
void haralick_from_glcm(const Volume& v, const std::vector<long long>& glcm, Features& f);

// ---- CPU reference (the trusted baseline) --------------------------------

// Build the direction-summed GLCM by a plain serial triple loop over voxels and
// the 13 directions. Returns the summed Ng x Ng count matrix. This is the
// reference the GPU's atomic scatter must reproduce EXACTLY.
void build_glcm_cpu(const Volume& v, std::vector<long long>& glcm);

// Full CPU feature extraction: histogram -> first-order, GLCM -> Haralick.
Features extract_features_cpu(const Volume& v);
