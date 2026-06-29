// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.33 : Interaction Fingerprinting & Binding-Mode Clustering
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls build_ifps_gpu() (STAGE A)
//   and ifp_cluster_gpu() (STAGE B); kernels.cu implements the host wrappers and
//   the device kernels. Included only by .cu translation units (it pulls in CUDA
//   types), which is why the CPU reference lives behind the pure-C++
//   reference_cpu.h instead.
//
// THE TWO GPU PATTERNS (see ../THEORY.md "GPU mapping" and PATTERNS.md sec 1)
//   STAGE A  -- one thread per POSE builds that pose's IFP by scanning all
//               NUM_RESIDUES residues (a small inner loop). Embarrassingly
//               parallel: poses never interact. Mirrors the "independent jobs"
//               pattern of flagship 1.12.
//   STAGE B  -- consensus-bit Tanimoto k-means:
//                 ASSIGN  : one thread per pose -> nearest centroid (popcount).
//                 TALLY   : one thread per pose -> atomicAdd into integer per-bit
//                           counters (integer adds commute -> deterministic,
//                           the lesson of flagship 11.09).
//                 UPDATE  : majority vote on the host (shared with the CPU).
//
// READ THIS AFTER: ifp.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // Dataset + the shared host helpers (init/update/cost)

// ---------------------------------------------------------------------------
// build_ifps_gpu : STAGE A on the device.
//   Uploads residues + poses, launches one thread per pose to build the packed
//   IFP bit-vectors, copies them back. Output `fps` is resized to [P*FP_WORDS]
//   and must equal the CPU build_ifps() bit-for-bit.
//     d         : the loaded dataset (residues + poses).
//     fps       : host output, [P*FP_WORDS] 64-bit words (output parameter).
//     kernel_ms : out-param, measured kernel time in ms (CUDA events; STAGE A).
// ---------------------------------------------------------------------------
void build_ifps_gpu(const Dataset& d, std::vector<uint64_t>& fps, float* kernel_ms);

// ---------------------------------------------------------------------------
// ifp_cluster_gpu : STAGE B on the device.
//   Runs `iters` Lloyd iterations of consensus-bit Tanimoto k-means on the
//   already-built fingerprints `fps` ([P*FP_WORDS]). ASSIGN + per-bit TALLY run
//   on the GPU; the majority-vote UPDATE runs on the host (update_centroids),
//   exactly as the CPU reference does -> bit-identical centroids and labels.
//     fps       : the IFP bit-vectors, [P*FP_WORDS] (typically from STAGE A).
//     P, K      : number of poses, number of clusters.
//     iters     : fixed Lloyd iterations (deterministic; no convergence test).
//     centroids : host output [K*FP_WORDS]; labels [P]; sizes [K] (out-params).
//     kernel_ms : out-param, summed assign+tally kernel time in ms (CUDA events).
//   Returns the final clustering cost (sum of Tanimoto distances).
// ---------------------------------------------------------------------------
double ifp_cluster_gpu(const std::vector<uint64_t>& fps, int P, int K, int iters,
                       std::vector<uint64_t>& centroids, std::vector<int>& labels,
                       std::vector<unsigned int>& sizes, float* kernel_ms);
