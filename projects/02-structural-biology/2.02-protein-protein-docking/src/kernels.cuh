// ===========================================================================
// src/kernels.cuh  --  GPU FFT-correlation docking interface (cuFFT + kernels)
// ---------------------------------------------------------------------------
// Project 2.2 : Protein-Protein Docking  (catalog id 2.2)
//
// THE BIG IDEA (pattern: USE A CUDA LIBRARY -- cuFFT -- WITHOUT A BLACK BOX)
//   We must score every rigid translation t of the ligand against the receptor:
//       S(t) = sum_x  R(x) * L(x - t)            (a 3D cross-correlation)
//   Done directly that is O(Ng^2) -- Ng outputs, each an Ng-term sum. The
//   Correlation Theorem replaces it with three FFTs and one pointwise multiply:
//       S = IFFT( FFT(R) .* conj(FFT(L)) )        -- O(Ng log Ng)
//   On the GPU the FFTs are done by cuFFT (R2C forward, C2R inverse), and the
//   tiny "multiply two spectra" and "normalize" steps are our own one-line
//   kernels. kernels.cu documents EXACTLY what each cuFFT call computes and the
//   data layout it expects -- the library is used, not hidden.
//
//   WHY conj(FFT(L)): multiplying spectra computes a CONVOLUTION; conjugating
//   one input flips it into a CORRELATION. Conjugating the LIGAND spectrum gives
//   exactly the brute-force convention S(t)=sum_x R(x)L(x-t). See THEORY section
//   "The math" for the one-line derivation.
//
//   The voxelization (atoms -> shape grids R and L) is done on the host with the
//   shared rule in reference_cpu.cpp, so the CPU and GPU consume byte-identical
//   grids and the ONLY source of CPU/GPU divergence is FFT round-off.
//
//   kernels.cu defines the kernels + the dock_gpu wrapper. main.cu calls it.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

// ---- Device kernels (one teaching concept each) --------------------------

// spectral_correlate_kernel: pointwise spectrum product for CORRELATION.
//   For each complex frequency bin k:  P[k] = Rf[k] * conj(Lf[k]).
//   conj(Lf) turns the convolution that a plain product would give into a
//   cross-correlation with correlate_cpu's sign convention. cufftComplex IS
//   float2 (.x real, .y imag); we read both spectra as float2. One thread per
//   frequency bin (the half-spectrum of R2C).
__global__ void spectral_correlate_kernel(const float2* __restrict__ Rf,
                                          const float2* __restrict__ Lf,
                                          int n_complex,
                                          float2* __restrict__ P);

// scale_kernel: cuFFT's forward+inverse pair multiplies the data by Ng (the FFT
//   is UNNORMALIZED), so the C2R output must be divided by Ng to recover the
//   true correlation values. One thread per real voxel.
__global__ void scale_kernel(float* __restrict__ s, int n_real, float inv_ng);

// ---- Host wrapper --------------------------------------------------------

// dock_gpu: compute the full circular cross-correlation score grid S(t) on the
//   GPU using cuFFT, given the host-side receptor and ligand shape grids.
//     N         : grid edge length (grid is N*N*N).
//     R, L      : host shape grids (N*N*N floats each), built by reference_cpu.
//     score     : host output, resized to N*N*N; score[flat3(t)] = S(t).
//     kernel_ms : out-param, GPU time of the FFT + pointwise steps (CUDA events).
//   All CUDA/cuFFT bookkeeping (plans, device buffers, transfers) is hidden here.
void dock_gpu(int N, const std::vector<float>& R, const std::vector<float>& L,
              std::vector<float>& score, float* kernel_ms);
