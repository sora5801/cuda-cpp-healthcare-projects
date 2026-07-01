// ===========================================================================
// src/kernels.cuh  --  GPU SART reconstruction interface
// ---------------------------------------------------------------------------
// Project 5.15 : Proton CT & Ion Imaging Reconstruction
//
// THE BIG IDEA
//   SART sweeps over all protons; within one sweep every proton's contribution
//   is INDEPENDENT, so we assign ONE GPU THREAD PER PROTON. Each thread:
//     1. walks its most-likely path (MLP) sampling the current RSP image
//        (a GATHER, exactly like CT backprojection 4.01),
//     2. forms its WEPL residual, and
//     3. SCATTERS a length-weighted correction into shared per-voxel
//        accumulators via atomicAdd.
//   Because many protons cross the same voxel, the scatter is a many-writer
//   reduction. Float atomicAdd would be non-deterministic, so we accumulate in
//   FIXED-POINT int64 (order-independent, bit-identical to the CPU) -- see
//   docs/PATTERNS.md section 3 and reference_cpu.h.
//
//   A second tiny kernel then applies the SART update rsp += relax*num/den, one
//   thread per voxel. The host wrapper loops these two kernels `iters` times.
//
//   This header is included ONLY by .cu files (it declares __global__ kernels);
//   the CPU reference uses the pure-C++ reference_cpu.h instead.
//   READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, pct_physics.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // PctProblem, PctGeom, Proton (pure C++, safe in .cu)

// ---- Device kernels (declared here, defined in kernels.cu) ---------------

// tally_kernel: thread `i` owns proton i. Forward-projects along the MLP, forms
// the residual, and atomicAdds fixed-point num/den into the shared accumulators.
//   protons      : [n_protons] device array of histories
//   rsp          : [n*n] current RSP image (read-only this sweep)
//   n_protons, geom, path_samples : problem parameters (by value)
//   num_fx, den_fx : [n*n] int64 fixed-point accumulators (zeroed before launch)
__global__ void tally_kernel(const Proton* __restrict__ protons, int n_protons,
                             const float* __restrict__ rsp, PctGeom geom,
                             int path_samples,
                             long long* __restrict__ num_fx,
                             long long* __restrict__ den_fx);

// update_kernel: thread `v` owns voxel v. Applies rsp[v] += relax*num/den when
// the voxel was touched (den>0). One-to-one with the CPU update loop.
__global__ void update_kernel(float* __restrict__ rsp, int cells, float relax,
                              const long long* __restrict__ num_fx,
                              const long long* __restrict__ den_fx);

// ---- Host wrapper --------------------------------------------------------
// reconstruct_gpu: run the full SART reconstruction on the GPU.
//   Uploads protons once; each sweep zeroes the accumulators, launches
//   tally_kernel (one thread/proton) then update_kernel (one thread/voxel);
//   copies the final RSP image back. Reports summed KERNEL time via *kernel_ms.
//   result    : host output, resized to n*n (output parameter)
//   kernel_ms : out-param, total ms in the kernels (not copies)
void reconstruct_gpu(const PctProblem& prob, std::vector<float>& result,
                     float* kernel_ms);
