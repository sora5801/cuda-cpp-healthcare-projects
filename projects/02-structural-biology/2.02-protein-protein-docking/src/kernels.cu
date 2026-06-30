// ===========================================================================
// src/kernels.cu  --  GPU FFT-correlation docking (cuFFT 3D R2C/C2R + kernels)
// ---------------------------------------------------------------------------
// Project 2.2 : Protein-Protein Docking  (catalog id 2.2)
//
// WHAT THIS FILE DOES  (the GPU twin of correlate_cpu in reference_cpu.cpp)
//   Computes S(t) = sum_x R(x)*L(x-t) for ALL translations t at once, using the
//   Correlation Theorem and cuFFT:
//       (a) Rf = FFT(R)          forward 3D real-to-complex   (cuFFT)
//       (b) Lf = FFT(L)          forward 3D real-to-complex   (cuFFT)
//       (c) P  = Rf .* conj(Lf)  pointwise product            (our kernel)
//       (d) S  = IFFT(P)         inverse 3D complex-to-real   (cuFFT)
//       (e) S /= Ng              undo cuFFT's unnormalized scale (our kernel)
//   main.cu then compares this S grid (and its argmax) to the brute-force CPU S.
//
//   THE LIBRARY IS NOT A BLACK BOX: each cuFFT call below is annotated with what
//   it computes mathematically and the exact array layout it reads/writes. The
//   only custom device code is the trivial spectrum multiply and the rescale.
//
// READ THIS AFTER: kernels.cuh (declarations + the theorem), reference_cpu.h.
// ===========================================================================
#include "kernels.cuh"
#include "reference_cpu.h"        // (only for documentation parity; no use needed)
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer (CUDA-event timing)

#include <cufft.h>
#include <cstdio>
#include <cstdlib>

// Threads per block: 256 is a solid sm_75..sm_89 default (8 warps/block, good
// occupancy, multiple of the 32-lane warp). The pointwise kernels are
// memory-bound, so the exact value matters little here.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CUFFT_CHECK: cuFFT has its OWN status type (cufftResult), distinct from
//   cudaError_t, so it needs its own guard macro that mirrors CUDA_CHECK. Every
//   cuFFT call below is wrapped: a plan can fail (unsupported size, out of
//   memory) and an exec can fail (bad pointer); silently ignoring either gives
//   the classic "FFT returns zeros and I don't know why".
// ---------------------------------------------------------------------------
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cufft error %d\n",     \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// spectral_correlate_kernel: P[k] = Rf[k] * conj(Lf[k]) for each freq bin k.
//   Thread map: one thread per complex bin, i = blockIdx.x*blockDim.x+threadIdx.x.
//   Complex multiply with conjugation of the SECOND operand (the ligand):
//     a * conj(b) = (a.x + i a.y)(b.x - i b.y)
//                 = (a.x*b.x + a.y*b.y) + i (a.y*b.x - a.x*b.y)
//   Conjugating one spectrum turns the spectral product from a CONVOLUTION into
//   a CROSS-CORRELATION. WHICH operand you conjugate fixes the SIGN of the shift:
//   conjugating the LIGAND yields exactly correlate_cpu's convention
//       S(t) = sum_x R(x) * L(x - t)
//   (verified bit-for-bit against the brute-force reference). Conjugating the
//   receptor instead would give S(-t) -- a mirrored grid. Pure register math,
//   no shared memory or atomics.
// ---------------------------------------------------------------------------
__global__ void spectral_correlate_kernel(const float2* __restrict__ Rf,
                                          const float2* __restrict__ Lf,
                                          int n_complex,
                                          float2* __restrict__ P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_complex) return;              // guard the ragged last block
    float2 a = Rf[i];                        // receptor spectrum bin
    float2 b = Lf[i];                        // ligand spectrum bin
    float2 p;
    p.x = a.x * b.x + a.y * b.y;             // Re[ a * conj(b) ]
    p.y = a.y * b.x - a.x * b.y;             // Im[ a * conj(b) ]
    P[i] = p;
}

// ---------------------------------------------------------------------------
// scale_kernel: divide every real output voxel by Ng. cuFFT's forward+inverse
//   round-trip multiplies the signal by Ng = N*N*N (the transforms are
//   UNNORMALIZED), so we must rescale to recover the true correlation. One
//   thread per voxel.
// ---------------------------------------------------------------------------
__global__ void scale_kernel(float* __restrict__ s, int n_real, float inv_ng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_real) s[i] *= inv_ng;          // multiply by 1/Ng (precomputed)
}

// ---------------------------------------------------------------------------
// dock_gpu: orchestrate the 5 steps on the device.
//   3D R2C layout (the layout cuFFT mandates, and that our kernels rely on):
//     * REAL input grid:    N*N*N floats, row-major (x fastest), index
//                           (z*N + y)*N + x -- exactly flat3() in reference_cpu.h.
//     * COMPLEX output:     by Hermitian symmetry of a real signal's FFT, cuFFT
//                           stores only the non-redundant half along the LAST
//                           (fastest, x) dimension: N * N * (N/2+1) cufftComplex.
//   We allocate one real buffer and reuse two complex buffers for Rf, Lf, P.
// ---------------------------------------------------------------------------
void dock_gpu(int N, const std::vector<float>& R, const std::vector<float>& L,
              std::vector<float>& score, float* kernel_ms) {
    const int    nx = N, ny = N, nz = N;
    const int    n_real    = nx * ny * nz;            // real voxels (Ng)
    const int    nx_half   = nx / 2 + 1;              // R2C keeps x in [0, N/2]
    const int    n_complex = nz * ny * nx_half;       // complex bins per grid
    score.assign(static_cast<std::size_t>(n_real), 0.0f);

    // ---- Device buffers --------------------------------------------------
    // d_real  : reused as input to each forward FFT and as the final S output.
    // d_Rf    : receptor spectrum.   d_Lf : ligand spectrum (also holds P).
    cufftReal*    d_real = nullptr;     // [n_real]    real grid
    cufftComplex* d_Rf   = nullptr;     // [n_complex] conj-receptor spectrum
    cufftComplex* d_Lf   = nullptr;     // [n_complex] ligand spectrum / product
    CUDA_CHECK(cudaMalloc(&d_real, static_cast<std::size_t>(n_real)    * sizeof(cufftReal)));
    CUDA_CHECK(cudaMalloc(&d_Rf,   static_cast<std::size_t>(n_complex) * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_Lf,   static_cast<std::size_t>(n_complex) * sizeof(cufftComplex)));

    // ---- cuFFT plans (NOT black boxes) -----------------------------------
    // cufftPlan3d(plan, nz, ny, nx, type) builds a 3D FFT plan for an nz x ny x nx
    // grid. CUFFT_R2C: real input -> complex Hermitian-half output (forward).
    // CUFFT_C2R: complex Hermitian-half input -> real output (inverse). For each
    // bin (kz,ky,kx) the forward computes
    //     X[k] = sum_n x[n] * exp(-2*pi*i * (k . n / N))
    // over the 3D grid -- the same sum correlate_cpu does by hand, in O(Ng logNg).
    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan3d(&plan_fwd, nz, ny, nx, CUFFT_R2C));   // R/L -> spectra
    CUFFT_CHECK(cufftPlan3d(&plan_inv, nz, ny, nx, CUFFT_C2R));   // P    -> S

    const int grid_c = (n_complex + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int grid_r = (n_real    + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const float inv_ng = 1.0f / static_cast<float>(n_real);

    GpuTimer timer;
    timer.start();   // time the FFT pipeline (the part the GPU accelerates)

    // (a) Rf = FFT(R). Copy receptor grid H2D, then forward-transform.
    CUDA_CHECK(cudaMemcpy(d_real, R.data(),
                          static_cast<std::size_t>(n_real) * sizeof(cufftReal),
                          cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecR2C(plan_fwd, d_real, d_Rf));

    // (b) Lf = FFT(L). Overwrite d_real with the ligand grid, forward-transform.
    CUDA_CHECK(cudaMemcpy(d_real, L.data(),
                          static_cast<std::size_t>(n_real) * sizeof(cufftReal),
                          cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecR2C(plan_fwd, d_real, d_Lf));

    // (c) P = Rf .* conj(Lf)  (write the product back into d_Lf to save memory).
    spectral_correlate_kernel<<<grid_c, THREADS_PER_BLOCK>>>(d_Rf, d_Lf, n_complex, d_Lf);
    CUDA_CHECK_LAST("spectral_correlate_kernel");

    // (d) S = IFFT(P). C2R writes the real correlation grid back into d_real.
    CUFFT_CHECK(cufftExecC2R(plan_inv, d_Lf, d_real));

    // (e) S /= Ng  (undo cuFFT's unnormalized forward+inverse scaling).
    scale_kernel<<<grid_r, THREADS_PER_BLOCK>>>(d_real, n_real, inv_ng);
    CUDA_CHECK_LAST("scale_kernel");

    *kernel_ms = timer.stop_ms();   // GPU-measured time of steps (a)-(e)

    // ---- Bring the score grid back to the host --------------------------
    CUDA_CHECK(cudaMemcpy(score.data(), d_real,
                          static_cast<std::size_t>(n_real) * sizeof(cufftReal),
                          cudaMemcpyDeviceToHost));

    // ---- Tear down (no GPU garbage collector) ---------------------------
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    CUDA_CHECK(cudaFree(d_real));
    CUDA_CHECK(cudaFree(d_Rf));
    CUDA_CHECK(cudaFree(d_Lf));
}
