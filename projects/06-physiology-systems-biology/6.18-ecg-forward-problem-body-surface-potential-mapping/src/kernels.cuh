// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls the two host wrappers below;
//   kernels.cu implements them plus the device kernel. Included only by .cu
//   translation units (it names __global__ / device types), so the plain C++
//   compiler never sees it -- that is why the CPU reference lives in the separate
//   pure-C++ header reference_cpu.h.
//
// THE TWO GPU STEPS (mirroring the catalog's "GPU pattern")
//   1. gpu_build_lead_field : an EMBARRASSINGLY-PARALLEL kernel fills the
//      lead-field (transfer) matrix A [L x S]. Each output entry A[e][s] is one
//      independent evaluation of ecg::dipole_potential, so we give each entry its
//      own thread -- the classic "grid over a 2-D output" mapping.
//   2. gpu_apply_forward : cuBLAS DGEMM applies A to the source time series X
//      [S x T] to get the body-surface potentials Phi [L x T] = A * X. This is
//      the catalog's "cuBLAS DGEMV per time step", done as ONE dense matrix
//      multiply over all T frames at once (a batched DGEMV *is* a DGEMM).
//
//   Both call the SAME per-entry physics as the CPU reference (ecg_core.h), so
//   the GPU and CPU results agree to within a documented, tiny tolerance.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, ecg_core.h. Then
// kernels.cu; compare each wrapper with its serial twin in reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <vector>

#include "ecg_core.h"   // ecg::Vec3 (the geometry the kernel reads)

// ---------------------------------------------------------------------------
// gpu_build_lead_field: build A [L x S] row-major on the GPU.
//   Inputs (host vectors, copied to device inside the wrapper):
//     electrode : [L] torso-surface electrode positions (metres)
//     src_pos   : [S] dipole anchor positions           (metres)
//     src_dir   : [S] dipole UNIT directions            (unitless)
//   Output:
//     A         : resized to L*S, row-major; A[e*S+s] = potential at electrode e
//                 from unit source s  (== the CPU build_lead_field_reference).
//   Timing:
//     kernel_ms : out-param, milliseconds spent in the build kernel (CUDA events).
// ---------------------------------------------------------------------------
void gpu_build_lead_field(const std::vector<ecg::Vec3>& electrode,
                          const std::vector<ecg::Vec3>& src_pos,
                          const std::vector<ecg::Vec3>& src_dir,
                          int L, int S,
                          std::vector<double>& A,
                          float* kernel_ms);

// ---------------------------------------------------------------------------
// gpu_apply_forward: Phi [L x T] = A [L x S] * X [S x T], via cuBLAS DGEMM.
//   Inputs (host vectors, copied to device inside the wrapper):
//     A : [L*S] row-major lead field (from gpu_build_lead_field or the CPU ref)
//     X : [S*T] row-major source strength time series
//   Output:
//     Phi : resized to L*T, row-major; Phi[e*T+t] = electrode e's potential at
//           time frame t  (== the CPU apply_forward_reference within tolerance).
//   Timing:
//     gemm_ms : out-param, milliseconds spent in the cuBLAS DGEMM (CUDA events).
// ---------------------------------------------------------------------------
void gpu_apply_forward(const std::vector<double>& A,
                       const std::vector<double>& X,
                       int L, int S, int T,
                       std::vector<double>& Phi,
                       float* gemm_ms);
