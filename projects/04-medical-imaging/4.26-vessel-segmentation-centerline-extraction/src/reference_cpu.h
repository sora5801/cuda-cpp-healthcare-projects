// ===========================================================================
// src/reference_cpu.h  --  Volume container + CPU reference vesselness pipeline
// ---------------------------------------------------------------------------
// Project 4.26 : Vessel Segmentation & Centerline Extraction
//
// Pure C++ (no CUDA). kernels.cu reuses these structs. The actual per-voxel math
// (Hessian eigenvalues, Frangi score) lives in the shared frangi.h so the CPU
// reference and the GPU kernel compute identical results.
//
// PIPELINE (Frangi vesselness, single scale):
//   1. Gaussian smooth the raw volume at scale sigma (separable 1-D passes).
//   2. Per voxel: build the 3x3 Hessian by central finite differences.
//   3. Eigen-decompose the Hessian (closed form) -> Frangi vesselness in [0,1].
//   4. (Teaching add-on) threshold vesselness to a binary vessel mask and read
//      off the peak-response voxel -- enough to show "segmentation + centerline
//      seed" end to end.
//
// READ THIS AFTER: frangi.h.  READ BEFORE: reference_cpu.cpp, kernels.cu, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "frangi.h"   // FrangiParams, vox_idx, eig_sym3, frangi_response

// ---------------------------------------------------------------------------
// Volume: a 3-D scalar image stored as a flat row-major vector<float>.
//   dims: nx (x fastest) * ny * nz. `data` has size nx*ny*nz.
//   Intensities are in raw synthetic units (see data/README.md).
// ---------------------------------------------------------------------------
struct Volume {
    int nx = 0, ny = 0, nz = 0;      // grid dimensions
    std::vector<float> data;         // nx*ny*nz voxels, row-major (x fastest)

    std::size_t size() const {
        return static_cast<std::size_t>(nx) * ny * nz;
    }
};

// A full job = the input volume + the Frangi knobs + the mask threshold.
struct VesselJob {
    Volume vol;                      // raw input intensities
    FrangiParams fp;                 // filter parameters (see frangi.h)
    double mask_threshold = 0.5;     // vesselness >= this -> counted as "vessel"
};

// ---------------------------------------------------------------------------
// load_volume: read the tiny text volume format written by make_synthetic.py.
//   Format (whitespace-separated):
//     line 1:  nx ny nz sigma alpha beta c bright mask_threshold
//     then:    nx*ny*nz float intensities in row-major (x fastest, then y, z)
//   Throws std::runtime_error on any I/O or shape error so demos fail loudly.
// ---------------------------------------------------------------------------
VesselJob load_volume(const std::string& path);

// ---------------------------------------------------------------------------
// gaussian_smooth: separable 3-D Gaussian blur of `in` at scale sigma, into
//   `out` (same size). Separable = three cheap 1-D passes (x, then y, then z)
//   instead of one expensive 3-D convolution. Border handling: clamp-to-edge.
//   Run on the host for BOTH paths (the GPU path smooths on the host too, then
//   uploads) so the two paths start from identical data.
// ---------------------------------------------------------------------------
void gaussian_smooth(const Volume& in, double sigma, Volume& out);

// ---------------------------------------------------------------------------
// vesselness_cpu: the serial reference. Given the SMOOTHED volume and params,
//   fill `vness` (size nx*ny*nz) with the Frangi score per voxel. O(N) voxels,
//   each doing a constant-size Hessian + eigen + Frangi (see frangi.h).
// ---------------------------------------------------------------------------
void vesselness_cpu(const Volume& smoothed, const FrangiParams& fp,
                    std::vector<float>& vness);

// ---------------------------------------------------------------------------
// summarize: reduce a vesselness field to DETERMINISTIC scalars for the report:
//   * n_vessel : count of voxels with vness >= threshold (segmented volume),
//   * vsum     : sum of vesselness (a stable aggregate for the checksum line),
//   * (px,py,pz),pmax : the argmax voxel (peak vesselness) and its value.
//   Deterministic: fixed row-major scan order, first-wins tie-break on the max.
// ---------------------------------------------------------------------------
void summarize(const Volume& dims, const std::vector<float>& vness,
               double threshold,
               long long& n_vessel, double& vsum,
               int& px, int& py, int& pz, double& pmax);
