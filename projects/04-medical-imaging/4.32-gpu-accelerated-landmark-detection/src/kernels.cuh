// ===========================================================================
// src/kernels.cuh  --  GPU decode interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.32 : GPU-Accelerated Landmark Detection
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls decode_gpu(); kernels.cu
//   implements the host wrapper and the device kernel. Included only by .cu
//   translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- that is why the CPU reference lives behind the
//   CUDA-free reference_cpu.h instead).
//
// THE BIG IDEA  (docs/PATTERNS.md: "score one query vs N items, each independent"
// + "clustering / atomic reduce" -- here it is one INDEPENDENT REDUCTION PER
// LANDMARK)
//   A network predicts L heatmaps; decoding landmark l depends only on volume l.
//   So the landmarks are perfectly independent -> we give each landmark its own
//   THREAD BLOCK. Within a block, the many threads cooperate on that one volume:
//
//     grid  : L blocks               (blockIdx.x = which landmark)
//     block : 256 threads            (threadIdx.x = which voxel-stride lane)
//
//   Each block runs a two-phase, fully on-device decode of its heatmap:
//     PHASE 1 -- PARALLEL ARGMAX. Threads stride over the V voxels; each keeps
//        the best (value, flat-index) it sees. A shared-memory tree reduction
//        then collapses the 256 partial winners to ONE. Ties break by LOWEST
//        flat index, matching the CPU's "first in row-major order" so the two
//        agree exactly even when several voxels share the max value.
//     PHASE 2 -- PARALLEL SOFT-ARGMAX. Threads cooperatively sweep the small
//        window around the winning voxel and atomicAdd their fixed-point
//        (integer) weight contributions into shared 64-bit accumulators. Integer
//        atomics COMMUTE, so the totals are order-independent -> deterministic
//        AND bit-identical to the CPU's serial integer sums. Thread 0 divides
//        (shared finalize_softargmax) and writes the landmark out.
//
//   Why block-scoped shared-memory atomics (not global)? All contributors for a
//   landmark live in one block, so the accumulators can sit in __shared__ memory
//   -- atomicAdd on shared memory is far cheaper than on global, and no other
//   block ever touches them.
//
// READ THIS AFTER: landmark.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "landmark.h"        // VolumeDims, Landmark (shared with the CPU side)
#include "reference_cpu.h"   // HeatmapSet (the input container)

// ---- Device kernel ---------------------------------------------------------
// decode_kernel: decode ALL L heatmaps, one landmark per block.
//   data : device pointer to the L*V float intensities (landmark-major).
//   dims : the grid geometry (nx,ny,nz), passed by value (small POD -> registers).
//   L    : number of landmarks (== number of blocks launched).
//   out  : device pointer to L GpuLandmark records (one per block).
// The launch config lives in decode_gpu(); the thread-to-data mapping is
// documented in kernels.cu at the kernel body.
struct GpuLandmark {   // POD mirror of Landmark for a flat device array
    double x, y, z;    // sub-voxel coordinate
    float  peak;       // intensity at the integer argmax voxel
    int    px, py, pz; // integer argmax voxel
};

__global__ void decode_kernel(const float* __restrict__ data,
                              VolumeDims dims, int L,
                              GpuLandmark* __restrict__ out);

// ---- Host wrapper ----------------------------------------------------------
// decode_gpu: the host-callable "do the whole GPU decode" function.
//   Uploads the heatmap set, launches decode_kernel with L blocks, copies the L
//   decoded landmarks back, and reports the measured KERNEL time (CUDA events)
//   via *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden.
//
//   hs        : the heatmap set (host).
//   out       : host output, resized to hs.num_landmarks (output parameter).
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies).
void decode_gpu(const HeatmapSet& hs, std::vector<Landmark>& out, float* kernel_ms);
