// ===========================================================================
// src/kernels.cu  --  GPU SD-OCT reconstruction: custom kernels + cuFFT
// ---------------------------------------------------------------------------
// Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
//
// WHAT THIS FILE DOES
//   The GPU twin of reconstruct_cpu(). It drives the three-stage pipeline:
//     dc_kernel           -- per-A-scan mean (background/DC level)
//     preprocess_kernel   -- DC removal + Hann window + DISPERSION COMPENSATION,
//                            producing the complex cuFFT input (CUSTOM physics)
//     cufftExecC2C        -- ONE batched call = every A-scan's FFT (LIBRARY)
//     power_norm_kernel   -- |A|^2 and per-A-scan normalise -> the image
//   main.cu runs this and the CPU reference and compares them.
//
//   The per-sample math (window, dispersion phase) comes from the SHARED header
//   oct_core.h, so the GPU preprocessing is byte-for-byte the CPU's -- only the
//   FFT itself differs (cuFFT float vs naive DFT double), which bounds the error.
//
// READ THIS AFTER: kernels.cuh (declarations + the two-pattern idea), oct_core.h.
// ===========================================================================
#include "kernels.cuh"
#include "oct_core.h"            // preprocess_sample, Cplx (SHARED with the CPU)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cufft.h>               // cufftPlan1d, cufftExecC2C, cufftComplex
#include <cstdio>
#include <cstdlib>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide memory latency, many blocks resident for good
// occupancy. Used for the per-sample kernel; the per-A-scan kernels launch one
// thread per A-scan under the same block size.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CUFFT_CHECK: cuFFT has its OWN status enum (cufftResult), distinct from
// cudaError_t, so CUDA_CHECK cannot guard it. This mirror macro guards every
// cuFFT call and prints file:line + the numeric status on failure, then aborts
// (a failed FFT means the reconstruction is meaningless -- see cuda_check.cuh
// for the same reasoning). CLAUDE.md §6.1 rule 7: error checks are always visible.
// ---------------------------------------------------------------------------
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cuFFT error %d\n",     \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// dc_kernel: one thread per A-scan. Serially sum that A-scan's N raw spectral
// samples and store the mean. A-scans are short (N ~ 10^3), so this trivial
// per-thread sum is both the clearest code and fast enough; a shared-memory
// block reduction would be premature here (discussed in THEORY "GPU mapping").
//   Thread a = blockIdx.x*blockDim.x + threadIdx.x owns A-scan a.
// ---------------------------------------------------------------------------
__global__ void dc_kernel(const float* __restrict__ raw, int n_ascan, int n_spec,
                          double* __restrict__ dc) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's A-scan
    if (a >= n_ascan) return;                              // guard ragged block

    const float* spec = raw + static_cast<size_t>(a) * n_spec;  // row start
    double sum = 0.0;                                     // accumulate in double
    for (int i = 0; i < n_spec; ++i) sum += static_cast<double>(spec[i]);
    dc[a] = sum / static_cast<double>(n_spec);            // the background level
}

// ---------------------------------------------------------------------------
// preprocess_kernel: THE CUSTOM PHYSICS KERNEL (what cuFFT cannot do for us).
//   One thread per (A-scan, spectral sample). Thread t owns global sample t:
//       a = t / n_spec   (which A-scan)
//       i = t % n_spec   (which sample within it)
//   It calls the SHARED preprocess_sample() -- DC removal, Hann window, complex
//   dispersion phase -- exactly as the CPU reference does, then writes the result
//   as a float2 (cufftComplex) into the cuFFT input buffer. Because the math is
//   the same source text as the CPU (oct_core.h), the FFT inputs match the CPU's
//   to double precision before we down-cast to float2 for cuFFT.
//
//   Memory: reads raw[t] and dc[a] from global memory, writes out[t]. No shared
//   memory / atomics -- every output sample is independent.
// ---------------------------------------------------------------------------
__global__ void preprocess_kernel(const float* __restrict__ raw,
                                  const double* __restrict__ dc,
                                  int n_ascan, int n_spec,
                                  double a2, double a3,
                                  float2* __restrict__ out) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(n_ascan) * n_spec;
    if (t >= total) return;                               // guard ragged block

    const int a = static_cast<int>(t / n_spec);           // A-scan index
    const int i = static_cast<int>(t % n_spec);           // sample within A-scan

    // SHARED per-sample pipeline (identical to the CPU): DC removal + window +
    // exp(-i*dispersion_phase). Computed in double, stored as float2 for cuFFT.
    const Cplx z = preprocess_sample(static_cast<double>(raw[t]), dc[a], i, n_spec, a2, a3);
    out[t] = make_float2(static_cast<float>(z.re), static_cast<float>(z.im));
}

// ---------------------------------------------------------------------------
// power_norm_kernel: |A|^2 + per-A-scan normalisation, one thread per A-scan.
//   cuFFT wrote the full length-N complex spectrum for each A-scan; we read only
//   the first N/2 depth bins (Hermitian symmetry: the rest mirrors them). The
//   thread computes each bin's linear power |A[z]|^2, tracks the A-scan peak, and
//   writes normalised power (0..1) into the image.
//
//   WHY ONE THREAD PER A-SCAN (not one per pixel)? The normalisation needs the
//   A-scan's MAX, a reduction. Doing it inside a single thread keeps the reduction
//   order fixed and free of cross-thread float atomics, so the result is
//   deterministic and matches the CPU exactly (PATTERNS.md #3). N/2 is small
//   (~10^3), so a serial two-pass per thread is fine.
// ---------------------------------------------------------------------------
__global__ void power_norm_kernel(const float2* __restrict__ fft,
                                  int n_ascan, int n_spec,
                                  double* __restrict__ image) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's A-scan
    if (a >= n_ascan) return;                             // guard ragged block

    const int nd = n_spec / 2;                            // depths kept = N/2
    const float2* row = fft + static_cast<size_t>(a) * n_spec;  // this A-scan's FFT
    double* img = image + static_cast<size_t>(a) * nd;    // this A-scan's image row

    // Pass 1: linear power |A[z]|^2 into the image row, track the peak.
    double peak = 0.0;
    for (int z = 0; z < nd; ++z) {
        const float2 v = row[z];
        const double p = static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y;
        img[z] = p;
        if (p > peak) peak = p;
    }
    // Pass 2: normalise to the A-scan's own peak so values are 0..1 (matches CPU).
    if (peak > 0.0)
        for (int z = 0; z < nd; ++z) img[z] /= peak;
}

// ---------------------------------------------------------------------------
// reconstruct_gpu: host wrapper. Allocate, copy in, run the 3-stage pipeline
// (custom kernel -> cuFFT -> custom kernel), copy out, free. We time the GPU
// COMPUTE (both kernels + the batched FFT) with CUDA events -- not the PCIe
// copies, which THEORY discusses separately as the real-time bottleneck.
// ---------------------------------------------------------------------------
void reconstruct_gpu(const OctBscan& b, std::vector<double>& image, float* kernel_ms) {
    const int A = b.n_ascan, N = b.n_spec;
    const int nd = oct_depth_count(N);                    // N/2 depths
    image.assign(static_cast<std::size_t>(A) * nd, 0.0);

    const std::size_t n_samp = static_cast<std::size_t>(A) * N;   // total spectral samples

    // ---- (1) device buffers ------------------------------------------------
    float*        d_raw   = nullptr;   // [A*N]   raw real spectra
    double*       d_dc    = nullptr;   // [A]     per-A-scan mean
    cufftComplex* d_fft   = nullptr;   // [A*N]   FFT in/out (complex, float2)
    double*       d_image = nullptr;   // [A*nd]  normalised power image
    CUDA_CHECK(cudaMalloc(&d_raw,   n_samp * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc,    static_cast<std::size_t>(A) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fft,   n_samp * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_image, static_cast<std::size_t>(A) * nd * sizeof(double)));

    // ---- (2) copy raw spectra H2D -----------------------------------------
    CUDA_CHECK(cudaMemcpy(d_raw, b.raw.data(), n_samp * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- Plan the batched FFT (NOT a black box) ---------------------------
    // cufftPlan1d(&plan, N, CUFFT_C2C, batch) builds a plan for `batch` == A
    // independent length-N complex-to-complex FFTs laid out contiguously (input
    // stride N, output stride N -- exactly our row-major [A*N] buffer). Then
    // cufftExecC2C(plan, in, out, CUFFT_FORWARD) computes, for each A-scan a and
    // depth bin z:
    //     A_a[z] = sum_{i=0}^{N-1} in_a[i] * exp(-2*pi*i * z*i / N)
    // -- the SAME sum reconstruct_cpu() does by hand, but via the O(N log N) FFT.
    // Hand-rolling this would mean writing a Cooley-Tukey radix kernel with shared
    // memory, twiddle factors, and bit-reversal -- a project in itself; cuFFT is
    // the well-optimised, well-tested library so we USE it and explain it.
    // (FORWARD vs INVERSE only flips the sign of the exponent and an overall
    // scale; we match the CPU's forward-DFT sign convention, and the per-A-scan
    // peak-normalisation cancels the scale, so either direction gives the same
    // NORMALISED image -- see THEORY "Numerical considerations".)
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, A));

    // ---- (3) run the pipeline, timed as one GPU stage ---------------------
    const int samp_blocks  = static_cast<int>((n_samp + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const int ascan_blocks = (A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    // 3a. per-A-scan DC.
    dc_kernel<<<ascan_blocks, THREADS_PER_BLOCK>>>(d_raw, A, N, d_dc);
    // 3b. preprocess + dispersion-compensate -> complex cuFFT input.
    preprocess_kernel<<<samp_blocks, THREADS_PER_BLOCK>>>(d_raw, d_dc, A, N, b.a2, b.a3, d_fft);
    // 3c. the batched FFT, IN PLACE on d_fft (forward transform).
    CUFFT_CHECK(cufftExecC2C(plan, d_fft, d_fft, CUFFT_FORWARD));
    // 3d. magnitude^2 + per-A-scan normalise -> the image.
    power_norm_kernel<<<ascan_blocks, THREADS_PER_BLOCK>>>(d_fft, A, N, d_image);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("OCT reconstruction pipeline");   // catch launch + exec errors

    // ---- (4) copy the image D2H -------------------------------------------
    CUDA_CHECK(cudaMemcpy(image.data(), d_image,
                          static_cast<std::size_t>(A) * nd * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // ---- (5) free everything (no GPU garbage collector) -------------------
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_raw));
    CUDA_CHECK(cudaFree(d_dc));
    CUDA_CHECK(cudaFree(d_fft));
    CUDA_CHECK(cudaFree(d_image));
}
