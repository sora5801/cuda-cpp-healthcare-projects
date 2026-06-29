// ===========================================================================
// src/kernels.cuh  --  GPU NNP interface (one thread per atom)
// ---------------------------------------------------------------------------
// Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
//
// THE BIG IDEA (pattern: INDEPENDENT JOBS + CONSTANT-MEMORY MODEL)
//   The total energy is a sum of per-atom energies E_i, and each E_i depends only
//   on neighbors within a cutoff. So the n atoms are independent jobs: give each
//   atom its OWN GPU thread. Every thread reads the same model (descriptor
//   hyperparameters + MLP weights) but never writes it -> the model lives in
//   CONSTANT memory, whose broadcast cache serves an entire warp from one
//   address in a single transaction (exactly like the query fingerprint in the
//   1.12 Tanimoto flagship). This is the closest cookbook row in PATTERNS.md:
//   "score one query vs N items, each independent + constant-memory query".
//
//   The per-atom math (descriptor + MLP) is the SHARED __host__ __device__ core
//   in nnp.h, so the kernel and the CPU reference compute identical values.
//   kernels.cu defines the kernel and the host wrapper.
//
// READ THIS AFTER: nnp.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Structure, AcsfParams, AtomicNet (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// nnp_energy_gpu: the host-callable wrapper around the GPU energy kernel.
//   Inputs:
//     s         : the molecular structure (n atoms, flat coordinates)
//     p         : ACSF hyperparameters (uploaded to constant memory)
//     net       : the per-atom MLP weights (uploaded to constant memory)
//   Outputs:
//     e_atom    : resized to s.n; e_atom[i] = E_i computed on the GPU
//     kernel_ms : the kernel's on-device time in milliseconds (CUDA events)
//   Returns: the total energy E = sum_i E_i (summed on the host in atom order so
//            it is deterministic and matches the CPU reference exactly).
//
//   It performs the canonical CUDA steps: upload coords (global) + model
//   (constant), launch one-thread-per-atom, copy per-atom energies back, sum.
// ---------------------------------------------------------------------------
double nnp_energy_gpu(const Structure& s, const AcsfParams& p, const AtomicNet& net,
                      std::vector<double>& e_atom, float* kernel_ms);
