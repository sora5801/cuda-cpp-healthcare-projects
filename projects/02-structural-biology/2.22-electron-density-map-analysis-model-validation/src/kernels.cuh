// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (cuFFT + per-voxel kernels)
// ---------------------------------------------------------------------------
// Project 2.22 : Electron Density Map Analysis & Model Validation
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls validate_gpu(); kernels.cu
//   implements the host wrapper and the device kernels. Included only by .cu
//   translation units (it declares __global__ kernels, so the plain C++ compiler
//   must never see it -- that is why the CPU reference and the shared per-voxel
//   math live in separate pure-C++ headers, reference_cpu.h and map_core.h).
//
// THE BIG IDEA (USING A CUDA LIBRARY WITHOUT IT BEING A BLACK BOX)
//   Validating an electron-density map needs the map's Fourier transform. The
//   FFT is a solved problem with a superb GPU library -- cuFFT. The lesson here
//   (PATTERNS.md §1 cuFFT row; exemplar flagship 8.03) is to USE the library but
//   document EXACTLY what its call computes and the data layout it expects, then
//   add only small custom kernels for the parts cuFFT does not do:
//
//     1. cuFFT  : one batched 3-D real-to-complex FFT of each map (A and B).
//     2. extract_complex_kernel : copy cuFFT's float2 output into our portable
//                 Cplx[] (double) so the host can do the deterministic shell
//                 reduction with the SAME fsc_accumulate() the CPU uses.
//     3. rscc_partials_kernel   : block-wise partial sums for the real-space
//                 correlation (RSCC). We reduce the per-block partials on the
//                 host in a FIXED order so the result is bit-reproducible
//                 (PATTERNS.md §3: float atomics are NOT associative).
//
//   Why the host finishes the reductions: a parallel float reduction sums in a
//   nondeterministic order, so its last bits wander run to run -- and we want a
//   byte-identical stdout (demo diffs it). The GPU does the heavy O(N log N) FFT
//   and the O(N) per-voxel work; the tiny final accumulation is deterministic on
//   the host. This is the same split flagship 8.03 uses (cuFFT on GPU, band
//   integration on host).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, map_core.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // DensityMap (pure C++, safe in a .cu)
#include "map_core.h"        // Cplx (POD complex used for the host-side reduction)

// ---- Device kernels (defined in kernels.cu) ------------------------------

// extract_complex_kernel: copy the cuFFT R2C output (float2 == cufftComplex:
//   .x real, .y imag) of ONE map into a Cplx[] (double). One thread per output
//   bin. `total` is the number of complex bins (n*n*(n/2+1) for a 3-D R2C).
//   This makes the spectra portable to the host's deterministic shell binning.
__global__ void extract_complex_kernel(const float2* __restrict__ X, int total,
                                       Cplx* __restrict__ out);

// rscc_partials_kernel: each block computes a PARTIAL of the five RSCC sums
//   (Σa, Σb, Σa², Σb², Σab) over its slice of voxels, using a shared-memory tree
//   reduction, and writes one partial per sum to global memory (indexed by
//   blockIdx.x). The host then sums the `gridDim.x` partials in a fixed order ->
//   deterministic, and matches the CPU's single-pass sums to rounding.
//     a,b    : device map data (n³ floats each)
//     total  : number of voxels
//     part_* : [gridDim.x] per-block partial sums (one element written per block)
__global__ void rscc_partials_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     long long total,
                                     double* __restrict__ part_Sa,
                                     double* __restrict__ part_Sb,
                                     double* __restrict__ part_Saa,
                                     double* __restrict__ part_Sbb,
                                     double* __restrict__ part_Sab);

// ---- Host wrapper --------------------------------------------------------
// validate_gpu: run the WHOLE GPU validation and return results that main.cu
//   compares against the CPU reference.
//     d            : the two maps to validate (host side)
//     rscc         : out -- the real-space correlation coefficient (GPU path)
//     fsc          : out -- the FSC curve, fsc[s] for shell s = round(|k|)
//     shell_count  : out -- voxels per shell (so main can skip empty shells)
//     kernel_ms    : out -- GPU time for the FFTs + kernels (CUDA events)
//
//   Internally: cuFFT R2C of both maps; extract_complex_kernel to portable Cplx;
//   host-side shell binning with the shared fsc_accumulate(); plus
//   rscc_partials_kernel + a deterministic host sum for RSCC. All CUDA/cuFFT
//   bookkeeping is hidden here.
void validate_gpu(const DensityMap& d, double* rscc,
                  std::vector<double>& fsc, std::vector<long long>& shell_count,
                  float* kernel_ms);
