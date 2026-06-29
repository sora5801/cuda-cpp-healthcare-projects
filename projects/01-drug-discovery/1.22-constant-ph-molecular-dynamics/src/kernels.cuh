// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.22 : Constant-pH Molecular Dynamics (reduced-scope teaching model)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls titrate_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it contains a __global__ declaration), so the plain
//   C++ host compiler never sees it -- which is why the CphProblem/CphResult
//   structs live in the pure-C++ reference_cpu.h that this header reuses.
//
// THE BIG IDEA -- ENSEMBLE MONTE CARLO (PATTERNS.md §1: "same ODE/chain for many
// parameter sets" + "Monte-Carlo histories")
//   The titration runs  n_pH * replicas  INDEPENDENT Metropolis Monte Carlo
//   chains. Independence is the gift: we assign ONE GPU THREAD PER CHAIN. Thread
//   `g = blockIdx.x*blockDim.x + threadIdx.x` decodes its own (pH index k,
//   replica r), seeds its private RNG exactly like the CPU does, runs the shared
//   run_chain() entirely in registers/local memory, and then ATOMICALLY adds its
//   integer per-residue protonation counts into the per-pH tally.
//
//   Two CpHMD-relevant lessons live in this mapping:
//     * PER-THREAD RNG from the chain index -> the CPU reproduces the identical
//       random decisions -> verification is an EXACT integer match, not a noisy
//       statistical one (cph_core.h rng_seed; PATTERNS.md §2).
//     * INTEGER ATOMIC SCORING -> many replica-threads add into the same per-pH,
//       per-residue counter with atomicAdd; because the addends are integers the
//       sum is order-independent and deterministic, so it equals the CPU tally
//       byte-for-byte (a float fraction would NOT have this property -- that is
//       exactly why we tally counts, not fractions; PATTERNS.md §3).
//
// READ THIS AFTER: cph_core.h, util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // CphProblem, CphResult (pure C++, safe inside .cu)

// ---- Device kernel -------------------------------------------------------
// titrate_kernel: each thread runs ONE (pH, replica) chain and atomically adds
// its integer protonation counts into the shared device tally.
//   sys        : the system parameters (by value -> resident in each thread)
//   pH_min/pH_max/n_pH : the pH grid (thread decodes its pH from its chain index)
//   replicas   : chains per pH (thread decodes its replica from its chain index)
//   seed       : base RNG seed (chain (k,r) seeds from (seed, chain_id(k,r)))
//   d_prot     : device [n_pH * n_res] uint64 tally, updated via atomicAdd
// Launch: grid-stride over the n_pH*replicas chains with a fixed grid (see .cu).
__global__ void titrate_kernel(CphSystem sys, double pH_min, double pH_max,
                               int n_pH, int replicas, unsigned long long seed,
                               unsigned long long* __restrict__ d_prot);

// ---- Host wrapper --------------------------------------------------------
// titrate_gpu: the host-callable "do the whole GPU titration" function.
//   Allocates the device tally, launches one thread per chain, copies the
//   integer counts back, and reports the measured KERNEL time (CUDA events) via
//   *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//
//   prob      : the loaded titration problem (defines the ensemble + physics)
//   out       : host output; out.prot_count resized to n_pH*n_res, filled with
//               the GPU integer tally; out.tallied_per_pH set (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void titrate_gpu(const CphProblem& prob, CphResult& out, float* kernel_ms);
