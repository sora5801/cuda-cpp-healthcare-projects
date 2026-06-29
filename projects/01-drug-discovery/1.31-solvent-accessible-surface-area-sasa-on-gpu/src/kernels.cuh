// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for Shrake-Rupley SASA
// ---------------------------------------------------------------------------
// Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
//
// THE BIG IDEA
//   Computing SASA for n atoms is n INDEPENDENT jobs: each atom's exposed-point
//   count depends only on the (read-only) positions of the other atoms, never on
//   another atom's result. So we give each atom its OWN GPU THREAD. With a block
//   of B threads we launch ceil(n / B) blocks (plus a grid-stride loop so a fixed
//   grid covers any n); thread (blockIdx.x, threadIdx.x) owns atom
//   i = blockIdx.x * blockDim.x + threadIdx.x.
//
//   The teaching points of this project:
//     * the per-atom physics is the SHARED __host__ __device__ code in
//       sasa_core.h -- the kernel literally calls the same count_exposed_points()
//       the CPU reference does, so the GPU result is verifiable to the exact
//       integer; and
//     * the atom array is small and read by EVERY thread, so it is a natural fit
//       for SHARED MEMORY tiling (we stage atoms into a block-local cache and
//       loop over tiles) -- the classic O(n^2) all-pairs pattern (N-body style).
//
//   This header is included only by .cu units (it declares a __global__). main.cu
//   calls sasa_gpu(). The pure-C++ data model is in reference_cpu.h.
//
// READ THIS AFTER: sasa_core.h, util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// Then read kernels.cu. The GPU-mapping rationale is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Molecule, Atom (pure C++, safe inside a .cu)

// Device kernel: exposed_out[i] = number of solvent-accessible test points for
// atom i. One thread per atom (grid-stride). It calls the shared
// count_exposed_points() from sasa_core.h, but with a shared-memory-tiled inner
// loop over neighbors (see kernels.cu) -- same arithmetic, faster memory traffic.
//   d_atoms     : [n] device array of Atom (centers + vdW radii)
//   n           : number of atoms
//   probe       : probe radius (Angstrom), passed by value
//   exposed_out : [n] device array of integer exposed-point counts (output)
__global__ void sasa_kernel(const Atom* __restrict__ d_atoms, int n, double probe,
                            int* __restrict__ exposed_out);

// Host wrapper: uploads the atoms, launches sasa_kernel, times ONLY the kernel
// (CUDA events), copies the integer counts back, and derives per-atom areas on
// the host with the shared atom_sasa() (so the float derivation matches the CPU).
//   mol       : the loaded molecule (n atoms)
//   exposed   : resized to n; filled with per-atom exposed-point counts (integers)
//   sasa      : resized to n; filled with per-atom SASA in Angstrom^2
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds
void sasa_gpu(const Molecule& mol,
              std::vector<int>& exposed,
              std::vector<double>& sasa,
              float* kernel_ms);
