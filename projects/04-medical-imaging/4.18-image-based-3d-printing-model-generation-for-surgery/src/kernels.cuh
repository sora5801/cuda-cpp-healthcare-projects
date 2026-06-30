// ===========================================================================
// src/kernels.cuh  --  GPU marching-cubes interface (the "what the GPU offers")
// ---------------------------------------------------------------------------
// Project 4.18 : Image-Based 3D Printing / Model Generation for Surgery
//
// ROLE IN THE PROJECT
//   Declares the host-callable entry point main.cu uses to run marching cubes
//   on the GPU, plus the two device kernels that do the work. Included only by
//   .cu translation units (it pulls in CUDA types), so the pure-C++ reference
//   never sees it. The CPU/GPU SHARED math is in mc_core.h (read that first).
//
// THE GPU STRATEGY (three steps -- the classic "ragged output" idiom)
//   A cube can emit 0..5 triangles, so we don't know up front where each cube's
//   triangles go. We solve it the standard data-parallel way:
//
//     1) COUNT   kernel: thread `cell` computes how many triangles cube `cell`
//                 emits (num_tris_for_cube) and stores it in d_counts[cell].
//                 Pure, independent, no output races.
//     2) SCAN    (exclusive prefix sum over the INTEGER counts): turns the
//                 per-cell counts into per-cell WRITE OFFSETS. Exclusive scan of
//                 [2,0,3,1] -> [0,2,2,5]; the total (here 6) is the mesh size.
//                 Integer scan is associative and DETERMINISTIC (unlike a float
//                 reduction), so the offsets -- and thus the mesh order -- are
//                 reproducible. See docs/PATTERNS.md §3. We hand-roll this scan
//                 (the classic two-level Blelloch scheme) rather than calling
//                 Thrust/CUB, to keep it a no-black-box teaching artifact -- see
//                 the long comment in kernels.cu and THEORY.md "GPU mapping".
//     3) GENERATE kernel: thread `cell` re-classifies its cube and writes its
//                 triangles starting at d_offsets[cell]. Because offsets are
//                 ascending in cell index, the output mesh is ordered exactly
//                 like the CPU reference's serial sweep -> exact comparison.
//
//   This count -> scan -> write pattern is how production GPU MC (and stream
//   compaction in general) works; it is the single most reusable idea here.
//
// READ THIS AFTER: mc_core.h, util/cuda_check.cuh, util/timer.cuh.  THEN kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "mc_core.h"        // VolDims, Triangle, MC tables (HD-shared with CPU)
#include "reference_cpu.h"  // MCProblem (the volume + iso) -- reused, not redefined

// ---- Device kernels (defined in kernels.cu) ------------------------------

// count_kernel: thread `cell` -> d_counts[cell] = #triangles cube `cell` emits.
//   grid : ceil(num_cells / B) blocks of B threads (1-D over the flat cell list)
//   thread (blockIdx.x, threadIdx.x) owns cell = blockIdx.x*blockDim.x + tx
//   Touches only its own 8 corners (global reads) + writes one int. No atomics.
__global__ void count_kernel(const float* __restrict__ vol, VolDims dims,
                             float iso, int n_cells, int* __restrict__ counts);

// generate_kernel: thread `cell` writes its triangles at out[ d_offsets[cell] ].
//   Same launch shape as count_kernel. Re-does the (cheap) classification so we
//   don't have to stash per-cell state between the two passes -- recompute is
//   cheaper than the extra memory traffic. Writes Triangle structs to global.
__global__ void generate_kernel(const float* __restrict__ vol, VolDims dims,
                                float iso, int n_cells,
                                const int* __restrict__ offsets,
                                Triangle* __restrict__ out_tris);

// ---- Host wrapper (defined in kernels.cu) --------------------------------
// marching_cubes_gpu: run the full count -> scan -> generate pipeline on the GPU
//   and return the extracted mesh on the host.
//
//   prob       : the volume + iso-value (host-side; uploaded internally).
//   out        : host output mesh, resized to the triangle count (out-param).
//   kernel_ms  : out-param, milliseconds spent in the two kernels + the scan
//                (CUDA-event timed). Reported to stderr by main.cu.
//
//   All device allocation, the H2D upload, the Thrust scan, and the D2H copy of
//   the compacted mesh are hidden here so main.cu reads cleanly.
void marching_cubes_gpu(const MCProblem& prob, std::vector<Triangle>& out,
                        float* kernel_ms);
