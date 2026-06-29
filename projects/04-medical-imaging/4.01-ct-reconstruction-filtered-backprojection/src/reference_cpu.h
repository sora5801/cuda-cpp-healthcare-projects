// ===========================================================================
// src/reference_cpu.h  --  CT geometry, ramp filter, CPU backprojection
// ---------------------------------------------------------------------------
// Project 4.01 : CT Reconstruction (Filtered Backprojection)
//
// WHAT THIS PROJECT COMPUTES
//   Reconstruct a 2-D image from its X-ray PROJECTIONS (a "sinogram"). For
//   parallel-beam geometry the reconstruction is Filtered BackProjection (FBP):
//     1. RAMP-FILTER each projection row (Ram-Lak filter).
//     2. BACKPROJECT: for every output pixel, sum the filtered projection value
//        along the ray through that pixel, over all angles.
//   (The Feldkamp-Davis-Kress / FDK algorithm is the 3-D cone-beam extension of
//   exactly this idea -- see THEORY.md.)
//
// WHY A GPU
//   Backprojection is a per-pixel GATHER: each output pixel reads one
//   interpolated sample from every projection. For a 512^3 volume x 1000
//   projections that is ~10^11 voxel-projection pairs -- hopeless on a CPU,
//   bandwidth-bound and fast on a GPU (and texture units do the interpolation
//   for free). Here we do the 2-D parallel-beam case so the geometry is clear.
//
//   Shared, pure-C++ header (no CUDA). kernels.cu reuses CTProblem.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// One reconstruction problem: the sinogram plus all geometry needed to invert it.
//   sino[k*n_det + j] = line integral at projection angle k, detector bin j.
//   Angles are uniform over [0, pi): theta_k = k * pi / n_angles.
//   Detector bin j sits at offset s_j = (j - (n_det-1)/2) * ds.
//   The image is `img` x `img` pixels covering world [-world_half, world_half]^2.
struct CTProblem {
    int n_angles = 0;     // number of projection angles
    int n_det    = 0;     // detector bins per projection
    int img      = 0;     // output image side length (pixels)
    float ds     = 0.0f;  // detector bin spacing (world units)
    float world_half = 0.0f;  // image spans [-world_half, world_half] in x and y
    std::vector<float> sino;  // [n_angles * n_det] raw projections
};

// Load a CTProblem from the text format in data/README.md:
//   header: "<n_angles> <n_det> <ds> <img> <world_half>"
//   then n_angles rows of n_det floats (the sinogram).
CTProblem load_ct(const std::string& path);

// Precompute cos/sin of every projection angle ONCE on the host, so the CPU and
// GPU backprojections use bit-identical trig (avoids cos vs cosf disagreement).
void compute_trig(int n_angles, std::vector<float>& cosv, std::vector<float>& sinv);

// Ram-Lak ramp filter: convolve each projection row with the discrete ramp
// kernel. `filtered` is sized to the sinogram. This is the "Filtered" in FBP;
// without it backprojection alone gives a blurred (1/r) image.
void ramp_filter(const CTProblem& ct, std::vector<float>& filtered);

// CPU reference backprojection: image[py*img+px] = (pi/n_angles) * sum_k
// interp(filtered_row_k, s = x*cos + y*sin). The trusted baseline the GPU
// kernel is checked against. image is sized to img*img.
void backproject_cpu(const CTProblem& ct, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& image);
