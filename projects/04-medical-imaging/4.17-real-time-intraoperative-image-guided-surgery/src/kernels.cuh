// ===========================================================================
// src/kernels.cuh  --  GPU ICP interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls icp_gpu(); kernels.cu
//   implements the host driver and the device kernel. Included only by .cu
//   translation units (it pulls in CUDA types), so the plain-C++ CPU reference
//   never sees it. The per-point math is shared with the CPU via icp.h.
//
// THE BIG IDEA -- two parallel patterns, one per ICP step
//   Each ICP iteration does:
//     (A) CORRESPOND + REDUCE  (this is the GPU kernel):
//         one thread per MOVING point p_i -> transform it by the current guess,
//         brute-force its NEAREST fixed point q_{c(i)} (independent search), and
//         atomicAdd its contribution to the shared fixed-point accumulators
//         (sumP, sumQ, sumPQ, count). Independent search + atomic reduction --
//         exactly the pattern of flagship 11.09 (k-means), see docs/PATTERNS.md.
//     (B) ALIGN  (done on the HOST, in icp.h's solve_rigid):
//         a single 3x3 SVD on the reduced accumulators -> the incremental rigid
//         transform. Tiny and serial, so we keep it on the host where it is
//         easy to make bit-identical to the CPU reference.
//   The determinism trick (fixed-point integer atomics) lives in icp.h and is
//   what makes the GPU reduction reproducible AND equal to the CPU reduction.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, icp.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "icp.h"            // Vec3, Rigid, AccumFixed (device-visible via nvcc)

// ---- Host driver ----------------------------------------------------------
// icp_gpu: run `iters` fixed ICP iterations on the GPU, starting from identity.
//   P, Q      : host point clouds (moving, fixed). Uploaded once; Q stays
//               resident, P stays resident (we transform it on the device each
//               iteration inside the kernel using the current transform).
//   iters     : number of ICP iterations (fixed, for determinism).
//   history   : filled with the RMS error (mm) after each iteration (computed on
//               the host from the returned per-iteration transforms -- shares
//               rms_error() with the CPU path for one identical metric).
//   kernel_ms : out-param, total milliseconds spent in the correspondence/reduce
//               kernels across all iterations (CUDA-event timed).
// Returns the final recovered rigid transform (maps P onto Q). main.cu checks it
// equals the CPU reference's transform bit-for-bit (the fixed-point guarantee).
Rigid icp_gpu(const std::vector<Vec3>& P, const std::vector<Vec3>& Q,
              int iters, std::vector<double>& history, float* kernel_ms);
