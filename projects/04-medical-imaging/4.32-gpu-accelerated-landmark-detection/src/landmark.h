// ===========================================================================
// src/landmark.h  --  The shared, CPU/GPU-identical "physics" of landmark decode
// ---------------------------------------------------------------------------
// Project 4.32 : GPU-Accelerated Landmark Detection
//
// WHY THIS HEADER EXISTS  (the single most important idiom in the repo, see
// docs/PATTERNS.md section 2)
//   The per-voxel math that turns a heatmap into a landmark coordinate must be
//   computed BYTE-FOR-BYTE IDENTICALLY on the CPU reference and inside the GPU
//   kernel. If we wrote the formula twice (once in reference_cpu.cpp, once in
//   kernels.cu) they could drift, and "GPU == CPU" verification would become a
//   guess. Instead we write each formula ONCE here, in a `__host__ __device__`
//   inline function, and include this header from BOTH sides:
//     * reference_cpu.cpp  (compiled by the plain C++ host compiler), and
//     * kernels.cu / main.cu  (compiled by nvcc for host + device).
//   The LM_HD macro below expands to `__host__ __device__` under nvcc and to
//   nothing under the host compiler, so the same source compiles in both worlds.
//
// KEEP THIS HEADER CUDA-TYPE-FREE
//   No `__global__`, no `dim3`, no `<cuda_runtime.h>` types. Only plain C++ and
//   the LM_HD-decorated inline helpers, so cl.exe / g++ can include it happily.
//   (Kernel launches and device pointers live in kernels.cuh / kernels.cu.)
//
// WHAT LANDMARK DECODE IS  (the science, in one paragraph)
//   A heatmap-regression network (stacked hourglass, 3D U-Net) does NOT output
//   coordinates directly. For each anatomical landmark l it outputs a whole 3D
//   volume H_l[z,y,x] whose value is high near the landmark and low elsewhere --
//   a "probability blob", trained against a Gaussian target centred on the true
//   point. To read a coordinate back out we DECODE the heatmap: find the blob's
//   peak. Two classic decoders, both implemented here:
//     (1) ARGMAX      -- the integer voxel with the largest value (coarse, exact).
//     (2) SOFT-ARGMAX -- the intensity-weighted centroid of a small window around
//                        that peak (sub-voxel: recovers fractional positions the
//                        integer grid cannot represent). This is what production
//                        toolkits (MONAI, nnDetection) use for final accuracy.
//   This file holds the small, exact helpers those decoders share.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>   // std::uint32_t, std::int64_t

// ---------------------------------------------------------------------------
// LM_HD: "landmark host+device". Under nvcc (__CUDACC__ defined) a function
// tagged LM_HD is compiled for BOTH the CPU and the GPU; under the plain host
// compiler the decorators do not exist, so LM_HD expands to nothing. This is
// the mechanism that guarantees identical math on both sides.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define LM_HD __host__ __device__
#else
#define LM_HD
#endif

// ---------------------------------------------------------------------------
// Volume geometry. A heatmap is a dense 3D grid stored row-major (x fastest,
// then y, then z) -- the same layout NIfTI/medical volumes and PyTorch tensors
// use for a [Z,Y,X] array. We keep the dimensions in one small struct so every
// function agrees on the indexing convention.
//
//   nx, ny, nz : grid extents along x, y, z (in voxels)
//   The flat index of voxel (x,y,z) is  ((z*ny) + y)*nx + x.
// One network prediction is L such volumes concatenated: landmark l occupies
// the slab [l * (nx*ny*nz) , (l+1) * (nx*ny*nz)).
// ---------------------------------------------------------------------------
struct VolumeDims {
    int nx;   // extent along x (fastest-varying axis)
    int ny;   // extent along y
    int nz;   // extent along z (slowest-varying axis)
};

// Voxels per heatmap volume = nx*ny*nz. Returned as 64-bit because a real
// 512^3 volume is 134 million voxels, which overflows 32-bit when multiplied
// by a landmark index. (Our sample is tiny, but we teach the correct type.)
LM_HD inline std::int64_t volume_voxels(const VolumeDims& d) {
    return static_cast<std::int64_t>(d.nx) * d.ny * d.nz;
}

// Flatten a 3D voxel coordinate to its row-major offset within ONE volume.
//   x in [0,nx), y in [0,ny), z in [0,nz). No bounds check here (callers guard).
LM_HD inline std::int64_t flat_index(int x, int y, int z, const VolumeDims& d) {
    return (static_cast<std::int64_t>(z) * d.ny + y) * d.nx + x;
}

// ---------------------------------------------------------------------------
// Fixed-point weighting for the soft-argmax centroid.
// ---------------------------------------------------------------------------
// WHY FIXED POINT (docs/PATTERNS.md section 3, rule 2)
//   Soft-argmax is a weighted average: coord = sum(w_i * pos_i) / sum(w_i).
//   On the GPU many threads accumulate into the SAME running sums with
//   atomicAdd. Floating-point addition is NOT associative, so a *float* atomic
//   sum depends on the (nondeterministic) order the threads finish -> the last
//   bits wiggle run-to-run and never exactly match the CPU's left-to-right sum.
//   The fix: convert each weight to a 64-bit INTEGER before summing. Integer
//   addition commutes, so the total is identical regardless of order AND equal
//   to the CPU's integer total -- giving us EXACT (== 0) verification, not
//   "close enough". We divide the two integer sums only at the very end.
//
//   WEIGHT_SCALE sets the fixed-point resolution: a heatmap value in [0,1] maps
//   to an integer weight in [0, WEIGHT_SCALE]. 1e6 keeps ~6 decimal digits,
//   comfortably inside 64 bits even summed over a whole 512^3 volume
//   (1e6 * 1.3e8 ~ 1.3e14 << 9.2e18 = INT64_MAX).
static constexpr std::uint32_t WEIGHT_SCALE = 1000000u;   // 1e6 fixed-point units

// Quantise a (clamped, non-negative) heatmap value to an integer weight.
//   v : a heatmap intensity, expected in [0,1] after the decoder clamps it.
//   Returns floor(v * WEIGHT_SCALE) as an unsigned integer. Using floor (not
//   round) on both CPU and GPU keeps the quantisation deterministic and equal.
LM_HD inline std::uint32_t quantize_weight(float v) {
    if (v <= 0.0f) return 0u;                     // negative/zero -> no weight
    float scaled = v * static_cast<float>(WEIGHT_SCALE);
    if (scaled > static_cast<float>(WEIGHT_SCALE)) // clamp v>1 to weight 1.0
        scaled = static_cast<float>(WEIGHT_SCALE);
    return static_cast<std::uint32_t>(scaled);     // truncation == floor for >=0
}

// ---------------------------------------------------------------------------
// SOFT-ARGMAX WINDOW.
//   We do not average over the whole volume (far-away noise would bias the
//   centroid). Instead we take a cube of half-width RADIUS voxels around the
//   argmax peak. RADIUS=2 -> a 5x5x5 = 125-voxel window, the common choice.
//   Kept here so the CPU and GPU decoders use the identical neighbourhood.
// ---------------------------------------------------------------------------
static constexpr int SOFTARGMAX_RADIUS = 2;

// A decoded landmark: the sub-voxel coordinate plus the peak intensity that
// produced it. Doubles so the final division carries full precision; the
// INPUTS to that division are exact integers, so both sides agree.
struct Landmark {
    double x;      // sub-voxel x coordinate (soft-argmax centroid)
    double y;      // sub-voxel y coordinate
    double z;      // sub-voxel z coordinate
    float  peak;   // heatmap value at the integer argmax voxel (confidence)
    int    px;     // integer argmax voxel x (the coarse peak, before refinement)
    int    py;     // integer argmax voxel y
    int    pz;     // integer argmax voxel z
};

// ---------------------------------------------------------------------------
// finalize_softargmax: divide the accumulated integer sums into a coordinate.
//   sum_w        : total integer weight in the window (denominator)
//   sum_wx/wy/wz : integer weight-times-position sums (numerators)
//   Returns the centroid (num/den) as a double. If the window is empty
//   (sum_w == 0, e.g. a flat/zero heatmap) we fall back to the integer peak
//   voxel so the coordinate is still well-defined. Shared by CPU and GPU so the
//   division is performed identically on both.
// ---------------------------------------------------------------------------
LM_HD inline void finalize_softargmax(std::uint64_t sum_w,
                                      std::uint64_t sum_wx,
                                      std::uint64_t sum_wy,
                                      std::uint64_t sum_wz,
                                      int px, int py, int pz,
                                      double& out_x, double& out_y, double& out_z) {
    if (sum_w == 0ull) {                     // degenerate: no positive weight
        out_x = static_cast<double>(px);
        out_y = static_cast<double>(py);
        out_z = static_cast<double>(pz);
        return;
    }
    double denom = static_cast<double>(sum_w);
    out_x = static_cast<double>(sum_wx) / denom;
    out_y = static_cast<double>(sum_wy) / denom;
    out_z = static_cast<double>(sum_wz) / denom;
}
