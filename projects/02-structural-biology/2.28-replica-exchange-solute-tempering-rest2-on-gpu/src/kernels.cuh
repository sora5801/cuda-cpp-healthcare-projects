// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls gpu_sample_round() once per
//   round; kernels.cu implements the host wrapper and the device kernel. This
//   header is included only by .cu translation units (it declares a __global__
//   kernel), so the plain-C++ CPU reference must never see it -- that is why the
//   config/result types live in the pure-C++ reference_cpu.h, which we include
//   here to reuse them.
//
// THE BIG IDEA (PATTERN: INDEPENDENT JOBS -- one thread per replica)
//   In replica-exchange the M replicas evolve INDEPENDENTLY between swaps: each
//   runs its own Monte-Carlo (here) or molecular dynamics (in production) with no
//   communication until an exchange is attempted. That is a textbook "embarrassing
//   parallelism" map: give replica r to GPU thread r, let it run the full
//   sweeps_per_round MC loop in registers, and write back its updated coordinates
//   and accept count. The ONLY synchronization point is the periodic exchange,
//   which we do on the host (it touches just M energies -- negligible). This is
//   exactly how GPU REST2 engines (GROMACS, NAMD, OpenMM) structure the work:
//   independent GPU integration per replica, a tiny exchange handshake between
//   rounds (THEORY.md "Where this sits in the real world").
//
//   Because each thread calls the SAME mc_sweep() the CPU loop calls (rest2.h),
//   and the RNG is a deterministic counter hash, the GPU reproduces the CPU
//   trajectory bit-for-bit -> we verify with tolerance 0 (PATTERNS.md section 4).
//
// READ THIS AFTER: rest2.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>           // uint64_t (RNG counters)
#include <vector>

#include "reference_cpu.h"   // SimConfig, ReplicaParams (pure C++, safe in .cu)

// ---- Device kernel --------------------------------------------------------
// sample_round_kernel: thread r advances replica r by cfg.sweeps_per_round MC
//   sweeps. One thread owns one replica's entire state (its N_SOLUTE beads live
//   in registers/local memory). No inter-thread communication -> no shared memory,
//   no atomics needed inside the kernel.
//     cfg      : run + physics settings (passed BY VALUE -> a per-thread copy in
//                registers; it is tiny and read-only, ideal for this).
//     reps     : [n_replicas] per-replica (lambda, seed), device pointer.
//     coords   : [n_replicas * N_SOLUTE] bead coordinates, in/out, device pointer.
//     accepted : [n_replicas] running accept counts, in/out, device pointer.
//     rng_ctr  : [n_replicas] per-replica RNG stream cursors, in/out, device ptr.
//   grid/block reasoning is documented at the launch site in kernels.cu.
__global__ void sample_round_kernel(SimConfig cfg,
                                    const ReplicaParams* __restrict__ reps,
                                    double*   __restrict__ coords,
                                    long*     __restrict__ accepted,
                                    uint64_t* __restrict__ rng_ctr);

// ---- Host wrapper ---------------------------------------------------------
// gpu_sample_round: run ONE sampling round of ALL replicas on the GPU.
//   It is the exact GPU twin of cpu_sample_round(): it uploads the current
//   coords/accepted/rng-counters, launches one thread per replica, and copies
//   the updated state back to host. main.cu then performs the (shared) exchange
//   step on the returned energies before calling this again for the next round.
//
//   We re-upload/-download each round because the exchange (which can SWAP whole
//   configurations between replicas) happens on the host in this teaching design;
//   a production engine would keep state resident on the GPU and exchange via
//   NCCL/peer copy. The copies are tiny (M*N_SOLUTE doubles) and make the data
//   flow explicit for the learner (THEORY.md "GPU mapping").
//
//   cfg       : run settings (n_replicas, sweeps_per_round, physics).
//   reps      : host [n_replicas] ladder (lambda, seed).
//   coords    : host [n_replicas*N_SOLUTE], in/out (updated with GPU result).
//   accepted  : host [n_replicas], in/out (accumulated accept counts).
//   rng_ctr   : host [n_replicas], in/out (advanced RNG cursors).
//   kernel_ms : out-param, milliseconds spent in the kernel itself (CUDA events).
void gpu_sample_round(const SimConfig& cfg,
                      const std::vector<ReplicaParams>& reps,
                      std::vector<double>& coords,
                      std::vector<long>& accepted,
                      std::vector<uint64_t>& rng_ctr,
                      float* kernel_ms);
