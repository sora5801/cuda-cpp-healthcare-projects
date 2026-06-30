// ===========================================================================
// src/kernels.cu  --  GPU map validation: cuFFT 3-D FFT + per-voxel kernels
// ---------------------------------------------------------------------------
// Project 2.22 : Electron Density Map Analysis & Model Validation
//
// WHAT THIS FILE DOES
//   The GPU twin of reference_cpu.cpp. It computes the same two validation
//   scores -- RSCC (real-space correlation) and the FSC curve (Fourier shell
//   correlation) -- but uses the cuFFT library for the heavy 3-D Fourier
//   transform. The lesson (PATTERNS.md §1, flagship 8.03) is to USE a library
//   kernel WITHOUT it being a black box: we document exactly what cufftExecC2C
//   computes, the data layout it expects, and what hand-rolling would take.
//
//   Division of labour (and WHY):
//     * cuFFT does the O(N log N) 3-D FFT of each map -- the part worth a library.
//     * extract_complex_kernel copies cuFFT's float2 spectra into a portable
//       Cplx[] (double) so the host can finish the reduction deterministically.
//     * rscc_partials_kernel block-reduces the five RSCC sums; the host adds the
//       per-block partials in a FIXED order.
//   The final accumulations run on the HOST on purpose: a parallel FLOAT sum is
//   not associative, so its low bits wander run-to-run and would break the demo's
//   byte-identical stdout (PATTERNS.md §3). The host reduction uses the SAME
//   fsc_accumulate()/pearson_from_sums() the CPU reference uses, so GPU and CPU
//   agree to rounding.
//
//   We use a FULL complex-to-complex (C2C) FFT (not the half-spectrum R2C) so the
//   GPU produces the identical full n³ cube the CPU's naive DFT produces -- making
//   the shell binning a line-for-line match.
//
// READ THIS AFTER: kernels.cuh (declarations + the design), map_core.h (the
//   shared per-voxel math). Compare against reference_cpu.cpp (the CPU twin).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)
#include "map_core.h"            // Cplx, fft_freq, shell_index, fsc_accumulate, ...

#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, eight warps for the scheduler to hide latency, and many blocks
// resident for occupancy. The shared-memory reduction below assumes a power of 2.
static constexpr int THREADS_PER_BLOCK = 256;

// cuFFT has its own status type, so it needs its own check macro (mirrors
// CUDA_CHECK but for cufftResult). Every cuFFT call is guarded and explained.
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cuFFT error %d\n",      \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// extract_complex_kernel: float2 (cufftComplex) -> Cplx (double), one thread per
//   bin. cufftComplex is exactly float2 (.x real, .y imag); we widen to double
//   so the host's shell sums are computed in the same precision as the CPU's.
//   Thread map: i = blockIdx.x*blockDim.x + threadIdx.x owns output bin i.
// ---------------------------------------------------------------------------
__global__ void extract_complex_kernel(const float2* __restrict__ X, int total,
                                       Cplx* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const float2 v = X[i];                 // one complex bin from cuFFT
        out[i].re = static_cast<double>(v.x);  // real part
        out[i].im = static_cast<double>(v.y);  // imaginary part
    }
}

// ---------------------------------------------------------------------------
// rscc_partials_kernel: block-wise partial sums for the five RSCC accumulators.
//   Each thread strides over the voxel array (grid-stride loop) accumulating its
//   private partials, then the block cooperatively tree-reduces them in shared
//   memory and writes ONE partial per sum to global memory at index blockIdx.x.
//
//   Launch: <<<gridDim, THREADS_PER_BLOCK>>> with 5*THREADS_PER_BLOCK doubles of
//   dynamic shared memory (five accumulators per thread). The host later sums the
//   gridDim partials in a fixed order, so the whole RSCC is deterministic.
//
//   We accumulate in DOUBLE to match the CPU's double sums; the per-thread loop
//   reads each voxel's a,b from global memory exactly once.
// ---------------------------------------------------------------------------
__global__ void rscc_partials_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     long long total,
                                     double* __restrict__ part_Sa,
                                     double* __restrict__ part_Sb,
                                     double* __restrict__ part_Saa,
                                     double* __restrict__ part_Sbb,
                                     double* __restrict__ part_Sab) {
    // Dynamic shared memory carved into five contiguous arrays of blockDim.x.
    extern __shared__ double sdata[];
    double* sSa  = sdata;                         // [blockDim.x]
    double* sSb  = sSa  + blockDim.x;
    double* sSaa = sSb  + blockDim.x;
    double* sSbb = sSaa + blockDim.x;
    double* sSab = sSbb + blockDim.x;

    const int tid = threadIdx.x;
    double la = 0, lb = 0, laa = 0, lbb = 0, lab = 0;   // this thread's partials

    // Grid-stride loop: thread g = blockIdx.x*blockDim.x + tid handles voxels
    // g, g + stride, g + 2*stride, ... so any `total` is covered by any grid.
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + tid;
         i < total; i += stride) {
        const double av = a[i];
        const double bv = b[i];
        la += av; lb += bv;
        laa += av * av; lbb += bv * bv; lab += av * bv;
    }
    sSa[tid] = la; sSb[tid] = lb; sSaa[tid] = laa; sSbb[tid] = lbb; sSab[tid] = lab;
    __syncthreads();

    // In-block tree reduction (halving the active range each step). Standard
    // shared-memory reduction; requires blockDim.x to be a power of two.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sSa[tid]  += sSa[tid + s];
            sSb[tid]  += sSb[tid + s];
            sSaa[tid] += sSaa[tid + s];
            sSbb[tid] += sSbb[tid + s];
            sSab[tid] += sSab[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes this block's five partial sums (no atomics: one slot/block).
    if (tid == 0) {
        part_Sa[blockIdx.x]  = sSa[0];
        part_Sb[blockIdx.x]  = sSb[0];
        part_Saa[blockIdx.x] = sSaa[0];
        part_Sbb[blockIdx.x] = sSbb[0];
        part_Sab[blockIdx.x] = sSab[0];
    }
}

// ---------------------------------------------------------------------------
// fft_map_gpu (file-local helper): full 3-D complex FFT of one real map via
//   cuFFT, returning the spectrum as a host vector<Cplx> (n³ entries).
//
//   THE LIBRARY CALL, NOT A BLACK BOX:
//     cufftPlan3d(&plan, n, n, n, CUFFT_C2C) builds a plan for one length-n×n×n
//     complex-to-complex FFT in C-order (z slowest, x fastest -- matching our
//     map layout). cufftExecC2C(..., CUFFT_FORWARD) then computes, for every
//     output bin (kx,ky,kz):
//        F(kx,ky,kz) = Σ_{x,y,z} f(x,y,z) · exp(-2πi (kx·x+ky·y+kz·z)/n)
//     i.e. exactly the triple sum reference_cpu.cpp's dft3d() does by hand, but
//     in O(n³ log n) instead of O(n⁴). We feed a real map as complex (imag = 0).
//
//   To hand-roll this we would implement a radix FFT along each axis with
//   bit-reversal and twiddle factors, mind shared-memory tiling and bank
//   conflicts, and tune for each n -- exactly the work cuFFT has already done.
//
//   Timing of the FFT + extract kernel is added into *kernel_ms by the caller's
//   GpuTimer (passed in via the events); here we just run on the current stream.
// ---------------------------------------------------------------------------
static void fft_map_gpu(const std::vector<float>& real_map, int n,
                        std::vector<Cplx>& spectrum_host) {
    const int total = n * n * n;

    // Pack the real map into interleaved complex (imag = 0) for the C2C transform.
    std::vector<float2> h_in(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) { h_in[i].x = real_map[i]; h_in[i].y = 0.0f; }

    cufftComplex* d_data = nullptr;   // in-place C2C buffer [n³]
    Cplx*         d_cplx = nullptr;   // widened double output [n³]
    CUDA_CHECK(cudaMalloc(&d_data, static_cast<std::size_t>(total) * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_cplx, static_cast<std::size_t>(total) * sizeof(Cplx)));
    CUDA_CHECK(cudaMemcpy(d_data, h_in.data(),
                          static_cast<std::size_t>(total) * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan3d(&plan, n, n, n, CUFFT_C2C));        // 3-D complex plan
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));  // in-place FFT

    const int block = THREADS_PER_BLOCK;
    const int grid = (total + block - 1) / block;
    extract_complex_kernel<<<grid, block>>>(reinterpret_cast<const float2*>(d_data),
                                            total, d_cplx);
    CUDA_CHECK_LAST("extract_complex_kernel");

    spectrum_host.resize(static_cast<std::size_t>(total));
    CUDA_CHECK(cudaMemcpy(spectrum_host.data(), d_cplx,
                          static_cast<std::size_t>(total) * sizeof(Cplx),
                          cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_cplx));
}

// ---------------------------------------------------------------------------
// validate_gpu: the host-callable wrapper that runs the whole GPU validation.
// ---------------------------------------------------------------------------
void validate_gpu(const DensityMap& d, double* rscc,
                  std::vector<double>& fsc, std::vector<long long>& shell_count,
                  float* kernel_ms) {
    const int n = d.n;
    const long long total = d.voxels();

    GpuTimer timer;
    timer.start();   // time everything GPU: both FFTs, extract, and the RSCC kernel

    // ---- 1. RSCC on the GPU: block partials, then deterministic host sum -----
    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, static_cast<std::size_t>(total) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, static_cast<std::size_t>(total) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, d.a.data(), static_cast<std::size_t>(total) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, d.b.data(), static_cast<std::size_t>(total) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Cap the grid so the per-block partial arrays stay small; the grid-stride
    // loop covers any `total` regardless of grid size.
    int grid = static_cast<int>((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    if (grid > 1024) grid = 1024;   // <=1024 partials -> trivial host sum

    double *d_pa, *d_pb, *d_paa, *d_pbb, *d_pab;
    CUDA_CHECK(cudaMalloc(&d_pa,  grid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pb,  grid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_paa, grid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pbb, grid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pab, grid * sizeof(double)));

    const std::size_t shmem = 5 * THREADS_PER_BLOCK * sizeof(double);
    rscc_partials_kernel<<<grid, THREADS_PER_BLOCK, shmem>>>(
        d_a, d_b, total, d_pa, d_pb, d_paa, d_pbb, d_pab);
    CUDA_CHECK_LAST("rscc_partials_kernel");

    // Copy the partials back and sum them in a fixed order (index 0..grid-1) so
    // the result is bit-reproducible. (For grid<=1024 this host sum is trivial.)
    std::vector<double> pa(grid), pb(grid), paa(grid), pbb(grid), pab(grid);
    CUDA_CHECK(cudaMemcpy(pa.data(),  d_pa,  grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pb.data(),  d_pb,  grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(paa.data(), d_paa, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pbb.data(), d_pbb, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pab.data(), d_pab, grid * sizeof(double), cudaMemcpyDeviceToHost));

    double Sa = 0, Sb = 0, Saa = 0, Sbb = 0, Sab = 0;
    for (int i = 0; i < grid; ++i) {
        Sa += pa[i]; Sb += pb[i]; Saa += paa[i]; Sbb += pbb[i]; Sab += pab[i];
    }
    *rscc = pearson_from_sums(static_cast<double>(total), Sa, Sb, Saa, Sbb, Sab);

    // ---- 2. FSC on the GPU: cuFFT both maps, then host shell binning ---------
    std::vector<Cplx> F1, F2;
    fft_map_gpu(d.a, n, F1);   // cuFFT forward FFT of map A
    fft_map_gpu(d.b, n, F2);   // cuFFT forward FFT of map B

    *kernel_ms = timer.stop_ms();   // stop GPU clock (FFTs + kernels are done)

    // Shell binning on the host with the SHARED fsc_accumulate(), so the sums are
    // accumulated in the same order and precision as reference_cpu.cpp -> the GPU
    // and CPU FSC curves match (the FFT values agree to single-precision rounding).
    const int n_shells = max_shell(n);
    std::vector<double> cross(n_shells, 0.0), p1(n_shells, 0.0), p2(n_shells, 0.0);
    shell_count.assign(n_shells, 0);
    for (int z = 0; z < n; ++z) {
        const int kz = fft_freq(z, n);
        for (int y = 0; y < n; ++y) {
            const int ky = fft_freq(y, n);
            for (int x = 0; x < n; ++x) {
                const int kx = fft_freq(x, n);
                const int s = shell_index(kx, ky, kz);
                const std::size_t idx = (static_cast<std::size_t>(z) * n + y) * n + x;
                fsc_accumulate(F1[idx], F2[idx], &cross[s], &p1[s], &p2[s]);
                ++shell_count[s];
            }
        }
    }
    fsc.assign(n_shells, 0.0);
    for (int s = 0; s < n_shells; ++s)
        fsc[s] = fsc_from_sums(cross[s], p1[s], p2[s]);

    // Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_pa));
    CUDA_CHECK(cudaFree(d_pb));
    CUDA_CHECK(cudaFree(d_paa));
    CUDA_CHECK(cudaFree(d_pbb));
    CUDA_CHECK(cudaFree(d_pab));
}
