// ===========================================================================
// src/radiomics.h  --  Shared (host + device) radiomics primitives
// ---------------------------------------------------------------------------
// Project 4.27 : Radiomics Feature Extraction
//
// WHAT THIS PROJECT COMPUTES
//   "Radiomics" turns a medical image region-of-interest (ROI) -- e.g. a tumour
//   segmented on a CT scan -- into a vector of quantitative NUMBERS ("features")
//   that a downstream model can correlate with outcomes (survival, response,
//   genotype). This teaching project computes the two workhorse feature families:
//
//     (1) FIRST-ORDER statistics: the histogram of gray levels inside the ROI
//         (mean, variance, energy, entropy, ...). These describe the DISTRIBUTION
//         of intensities but ignore where voxels sit relative to each other.
//
//     (2) TEXTURE features from the GRAY-LEVEL CO-OCCURRENCE MATRIX (GLCM). The
//         GLCM P[i][j] counts how often a voxel of gray level i sits next to a
//         voxel of gray level j, for a given neighbour DIRECTION. Summed over the
//         13 symmetric 3-D directions it captures spatial texture: is the tumour
//         smooth or speckled? Haralick's classic scalars (contrast, energy/ASM,
//         homogeneity, correlation, entropy) are read off this matrix.
//
// WHY A GPU (the catalog's "Deep dive")
//   A single ROI can hold ~10^6 voxels, and a radiomics cohort has thousands of
//   scans. CPU PyRadiomics takes minutes per patient; GPU builds run ~100x faster
//   because the GLCM is a HISTOGRAM SCATTER: every voxel independently looks at
//   its neighbours and increments matrix cells. That is one thread per voxel doing
//   atomicAdd -- exactly the parallel-histogram / atomic-reduction pattern
//   (see docs/PATTERNS.md section 1, exemplar 11.09 k-means).
//
// WHY A SHARED HEADER
//   The per-voxel math -- quantizing an intensity to a gray level, the list of
//   3-D direction offsets, the flat GLCM index -- is IDENTICAL on CPU and GPU.
//   Putting it in ONE __host__ __device__ (RAD_HD) header means the CPU reference
//   and the CUDA kernel run byte-for-byte the same arithmetic, so the GLCM counts
//   match EXACTLY (they are integers) and verification is exact, not approximate
//   (docs/PATTERNS.md section 2). Keep CUDA-only constructs (__global__) OUT of
//   this file so the plain C++ host compiler can include it too.
//
// READ THIS AFTER: nothing -- start here, then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <cstdint>   // fixed-width integers

// RAD_HD expands to __host__ __device__ under nvcc (so the function is compiled
// for BOTH the CPU and the GPU) and to nothing under the plain host compiler
// (which does not know those keywords). This is the "HD-macro idiom".
#ifdef __CUDACC__
#define RAD_HD __host__ __device__
#else
#define RAD_HD
#endif

// ---------------------------------------------------------------------------
// GRAY-LEVEL QUANTIZATION
//   Raw CT intensities are Hounsfield units spanning hundreds of values; a GLCM
//   over that many levels would be enormous and sparse. IBSI-style radiomics
//   therefore DISCRETIZES intensities into a small fixed number of gray levels
//   Ng (here 8) using a fixed-bin-count scheme:
//
//       level = floor( (value - vmin) / (vmax - vmin) * Ng )   clamped to [0,Ng-1]
//
//   value == vmax maps to Ng (out of range) so we clamp the top bin back to Ng-1.
//   Every voxel INSIDE the ROI gets a level in [0, Ng-1]; voxels outside the ROI
//   are excluded entirely (they never contribute to histogram or GLCM).
//
//   NOTE: We pass vmin/vmax (the ROI intensity range) so the SAME edges are used
//   on CPU and GPU -> identical levels -> identical matrices.
// ---------------------------------------------------------------------------
RAD_HD inline int rad_quantize(float value, float vmin, float vmax, int Ng) {
    // Degenerate ROI (all one intensity): everything falls in the first bin.
    const float span = vmax - vmin;
    if (span <= 0.0f) return 0;
    // Fractional position of `value` within [vmin, vmax], in [0, 1].
    const float t = (value - vmin) / span;
    int level = static_cast<int>(t * static_cast<float>(Ng));  // floor via trunc (t>=0)
    if (level < 0)      level = 0;         // guard tiny negative rounding
    if (level >= Ng)    level = Ng - 1;    // fold value==vmax back into the top bin
    return level;
}

// ---------------------------------------------------------------------------
// 3-D NEIGHBOUR DIRECTIONS
//   A voxel in a 3-D grid has 26 neighbours (a 3x3x3 cube minus itself). Each
//   direction and its opposite (e.g. +x and -x) give the SAME co-occurrence
//   information once the GLCM is symmetrized, so we only need the 13 DISTINCT
//   direction pairs. Enumerating all (dz,dy,dx) in {-1,0,+1}^3 and keeping the
//   lexicographically-positive half yields exactly these 13 offsets.
//
//   We hard-code them so CPU and GPU iterate the directions in the identical
//   order. Each entry is (dx, dy, dz) applied to a voxel to find its neighbour.
// ---------------------------------------------------------------------------
#define RAD_NUM_DIRECTIONS 13

// A tiny POD for one direction offset. Kept trivially copyable so it can live in
// __constant__ device memory (see kernels.cu) with no constructor fuss.
struct RadDir {
    int dx;   // step along x (columns), voxels
    int dy;   // step along y (rows),    voxels
    int dz;   // step along z (slices),  voxels
};

// The canonical 13 directions. This helper fills a caller-provided array; both
// the host loader and the device setup call it so the tables are guaranteed
// identical. (A function, not a global array, keeps the header header-only and
// avoids one-definition-rule surprises across translation units.)
RAD_HD inline void rad_fill_directions(RadDir* out) {
    // The 13 half-space offsets (the other 13 are their negations). Order fixed.
    const int table[RAD_NUM_DIRECTIONS][3] = {
        { 1,  0,  0}, { 0,  1,  0}, { 1,  1,  0}, {-1,  1,  0},  // in-plane (dz=0)
        { 0,  0,  1}, { 1,  0,  1}, {-1,  0,  1}, { 0,  1,  1},  // out-of-plane (dz=1)
        { 0, -1,  1}, { 1,  1,  1}, {-1,  1,  1}, { 1, -1,  1},
        {-1, -1,  1}
    };
    for (int k = 0; k < RAD_NUM_DIRECTIONS; ++k) {
        out[k].dx = table[k][0];
        out[k].dy = table[k][1];
        out[k].dz = table[k][2];
    }
}

// ---------------------------------------------------------------------------
// FLAT INDEXING
//   The image is stored row-major as x fastest, then y, then z:
//       voxel (x,y,z) -> linear index  x + nx*(y + ny*z)
//   The GLCM for one direction is an Ng x Ng matrix stored row-major:
//       cell (i,j) -> i*Ng + j
//   Keeping these in one place guarantees the CPU and GPU agree on layout.
// ---------------------------------------------------------------------------
RAD_HD inline std::size_t rad_vox_index(int x, int y, int z, int nx, int ny) {
    return static_cast<std::size_t>(x)
         + static_cast<std::size_t>(nx) * (static_cast<std::size_t>(y)
         + static_cast<std::size_t>(ny) * static_cast<std::size_t>(z));
}

RAD_HD inline int rad_glcm_index(int i, int j, int Ng) {
    return i * Ng + j;
}
