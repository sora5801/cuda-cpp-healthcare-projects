// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for covalent docking
// ---------------------------------------------------------------------------
// Project 1.28 : Covalent Docking
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls score_all_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it contains a __global__ declaration, so the plain
//   C++ compiler must never see it -- that is why the CPU reference and the
//   shared physics live in separate pure-C++ headers, reference_cpu.h/docking.h).
//
// THE BIG IDEA
//   The covalent search evaluates one energy per torsion-grid conformation, and
//   every conformation is INDEPENDENT (it only reads the shared, read-only
//   DockProblem). So we give each conformation its OWN GPU THREAD: thread
//   global-index `id` scores conformation `id` by calling the very same
//   score_conformation() (docking.h) the CPU reference uses. A grid-stride loop
//   lets a modest grid cover the exponentially many conformations.
//
//   This is the canonical "score N independent candidates" pattern (cf. project
//   1.12 Tanimoto). The GPU pays off because the conformation count grows like
//   GRID_PER_DOF^N_TORSIONS -- the curse of dimensionality that makes flexible
//   docking expensive on a CPU. See ../THEORY.md "GPU mapping".
//
//   We deliberately do the final argmin on the HOST (after copying the energy
//   array back), not with a device atomic. A floating-point atomicMin race is
//   nondeterministic in the tie case; copying the array and reducing on the host
//   keeps stdout byte-identical run to run (docs/PATTERNS.md section 3).
//
// READ THIS AFTER: docking.h, reference_cpu.h, util/cuda_check.cuh,
// util/timer.cuh. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // DockProblem, DockResult, n_conformations (pure C++)

// ---- Device kernel -------------------------------------------------------
// score_kernel: one logical thread per conformation, via a grid-stride loop so
// a fixed-size grid covers any number of conformations.
//   p   : the docking problem, passed BY VALUE (small POD -> copied into each
//         thread's parameter space; read-only, identical for all threads).
//   M   : total number of conformations (guards the grid-stride bound).
//   out : device pointer to M doubles; out[id] = energy of conformation id.
__global__ void score_kernel(DockProblem p, long long M, double* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// score_all_gpu: the host-callable "score every conformation on the GPU".
//   Allocates the device energy buffer, launches score_kernel, times ONLY the
//   kernel (CUDA events, not the copies), copies the energies back, and frees.
//   p         : the docking problem (host-side; copied to the kernel by value)
//   energies  : host output, resized to n_conformations(); one energy per id
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void score_all_gpu(const DockProblem& p, std::vector<double>& energies,
                   float* kernel_ms);
