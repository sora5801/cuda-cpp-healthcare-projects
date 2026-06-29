// ===========================================================================
// src/kernels.cu  --  SPME reciprocal-energy GPU pipeline (spread/FFT/convolve)
// ---------------------------------------------------------------------------
// Project 1.2 : Particle-Mesh Ewald Electrostatics
//
// WHAT THIS FILE DOES
//   The GPU twin of pme_recip_cpu(): build the B-spline charge grid with an
//   atomic fixed-point SCATTER (spread_kernel), FFT it with cuFFT, multiply by
//   the Ewald influence function and form per-bin energies (energy_kernel), then
//   sum on the host in CPU order. main.cu compares this energy to the CPU value.
//
//   The shared per-atom math (B-spline weights, fixed-point quantization) lives
//   in pme.h and is reused VERBATIM by the CPU reference -> the charge grid is
//   bit-identical on both sides; only the FP32 cuFFT vs FP64 host DFT differ, by
//   ~1e-6 relative (documented tolerance in main.cu / THEORY).
//
// READ THIS AFTER: kernels.cuh (the pipeline overview), pme.h (the shared math).
// ===========================================================================
#include "kernels.cuh"
#include "pme.h"                  // PME_ORDER, pme_bspline_weights, fixed-point
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer (CUDA-event timing)

#include <cufft.h>
#include <cstdio>
#include <cstdlib>

// 256 threads/block: a multiple of the 32-lane warp, 8 warps to hide latency,
// a solid default on sm_75..sm_89. Used for the per-atom and per-cell launches.
static constexpr int THREADS_PER_BLOCK = 256;

// cuFFT has its own status enum, so it needs its own check macro (mirrors
// CUDA_CHECK but for cufftResult). Every cuFFT call below is guarded.
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> error %d\n",           \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// spread_kernel  (STAGE 1: particle -> mesh, atomic fixed-point scatter)
//   Launch: one thread per ATOM. grid = ceil(n / 256), block = 256.
//   Thread i owns atom i; it computes the atom's scaled grid coordinate on each
//   axis, the order B-spline weights (pme.h, identical to the CPU), then scatters
//   q * wx * wy * wz onto the order^3 grid block (wrapped periodically).
//
//   WHY ATOMICS: different atoms overlap on the same grid points, so several
//   threads add to the same cell concurrently -> a data race unless the adds are
//   atomic. WHY FIXED-POINT: float atomicAdd is not associative, so a float grid
//   would be nondeterministic AND would not match the CPU. Integer atomicAdd
//   commutes, so the unsigned-long-long grid is bit-identical every run and
//   equals the CPU grid exactly (PATTERNS.md §3; pme.h explains the encoding).
//
//   Memory: reads atom arrays from global memory; writes via atomicAdd to the
//   global fixed-point grid. No shared memory here (kept simple for teaching; a
//   production kernel would tile atoms into shared memory -- see THEORY).
// ---------------------------------------------------------------------------
__global__ void spread_kernel(const double* __restrict__ x,
                              const double* __restrict__ y,
                              const double* __restrict__ z,
                              const double* __restrict__ q,
                              int n, int K, double box,
                              unsigned long long* __restrict__ grid_fixed) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's atom
    if (a >= n) return;                                     // guard ragged block

    const int order = PME_ORDER;
    // Scaled fractional coordinates in [0, K) -- the grid index space.
    const double gx = (x[a] / box) * K;
    const double gy = (y[a] / box) * K;
    const double gz = (z[a] / box) * K;
    const int g0x = static_cast<int>(floor(gx));
    const int g0y = static_cast<int>(floor(gy));
    const int g0z = static_cast<int>(floor(gz));

    double wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];     // per-axis weights
    pme_bspline_weights(gx - g0x, wx);
    pme_bspline_weights(gy - g0y, wy);
    pme_bspline_weights(gz - g0z, wz);

    const double qa = q[a];
    // Scatter onto the order^3 stencil; w[i] is the weight of grid point g0+i.
    for (int i = 0; i < order; ++i) {
        const int ix = ((g0x + i) % K + K) % K;             // wrap periodically
        for (int j = 0; j < order; ++j) {
            const int iy = ((g0y + j) % K + K) % K;
            const double wxy = qa * wx[i] * wy[j];
            for (int k = 0; k < order; ++k) {
                const int iz = ((g0z + k) % K + K) % K;
                const std::size_t idx = (static_cast<std::size_t>(ix) * K + iy) * K + iz;
                // atomicAdd on unsigned long long: integer adds commute, so the
                // final cell value is independent of thread order. pme_to_fixed
                // quantizes the (possibly negative) contribution to signed bits.
                atomicAdd(&grid_fixed[idx], pme_to_fixed(wxy * wz[k]));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fixed_to_real_kernel  (STAGE 2a: integer grid -> real grid for the FFT)
//   One thread per grid cell. Reinterpret the fixed-point accumulator as a real
//   charge density (pme_fixed_to_double) and store it as cufftReal (float) input.
//   We deliberately spread/accumulate in FP64-equivalent fixed-point, then cast
//   once to float here, so the only single-precision step is the FFT itself.
// ---------------------------------------------------------------------------
__global__ void fixed_to_real_kernel(const unsigned long long* __restrict__ grid_fixed,
                                     int total, float* __restrict__ grid_real) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        grid_real[i] = static_cast<float>(pme_fixed_to_double(grid_fixed[i]));
}

// ---------------------------------------------------------------------------
// energy_kernel  (STAGE 3: convolve in reciprocal space + per-bin energy)
//   One thread per reciprocal-grid bin (the R2C half-spectrum, K*K*(K/2+1) bins).
//   Computes e[i] = mult[i] * influence[i] * |F[i]|^2, where:
//     * F[i]         = cuFFT R2C output (float2: .x real, .y imag) -- the DFT of
//                      the charge grid (the structure factor on the mesh).
//     * influence[i] = B(m)C(m), the SPME Ewald coefficient (built on the host,
//                      uploaded; see build_influence in reference_cpu.cpp).
//     * mult[i]      = 1 or 2 -- the Hermitian multiplicity, because the R2C
//                      half-spectrum's interior bins each stand for two physical
//                      modes (+m and -m). Precomputed on the host to match CPU.
//   We do NOT reduce on the device: the per-bin energies are copied back and
//   summed on the host in the SAME order as pme_recip_cpu, giving a deterministic
//   total that matches the CPU bit-for-bit up to the FP32-FFT difference.
// ---------------------------------------------------------------------------
__global__ void energy_kernel(const float2* __restrict__ F,
                              const double* __restrict__ influence,
                              const double* __restrict__ mult,
                              int total, double* __restrict__ e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const float2 v = F[i];
        const double mag2 = static_cast<double>(v.x) * v.x
                          + static_cast<double>(v.y) * v.y;     // |F|^2
        e[i] = mult[i] * influence[i] * mag2;
    }
}

// ---------------------------------------------------------------------------
// pme_recip_gpu: orchestrate the four stages and return E_recip.
//   STAGES (timed together via CUDA events): spread -> convert -> R2C FFT ->
//   convolve. The host then sums the per-bin energies in index order.
//   `influence` and `mult` come precomputed from the host so GPU and CPU use the
//   exact same reciprocal coefficients and multiplicities.
// ---------------------------------------------------------------------------
double pme_recip_gpu(const System& s, const PmeParams& p,
                     const std::vector<double>& influence,
                     const std::vector<double>& mult,
                     float* kernel_ms) {
    const int n = s.n, K = p.K;
    const int Kh = K / 2 + 1;
    const std::size_t NG = static_cast<std::size_t>(K) * K * K;        // real cells
    const std::size_t NF = static_cast<std::size_t>(K) * K * Kh;       // complex bins

    // Host atom arrays are double; copy straight to device double buffers.
    double *d_x = nullptr, *d_y = nullptr, *d_z = nullptr, *d_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x, s.x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, s.y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, s.z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, s.q.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Grids and coefficient arrays.
    unsigned long long* d_grid_fixed = nullptr;   // [NG] fixed-point charge grid
    cufftReal*          d_grid_real  = nullptr;   // [NG] real charge density
    cufftComplex*       d_F          = nullptr;   // [NF] R2C spectrum
    double*             d_influence  = nullptr;   // [NF] B(m)C(m)
    double*             d_mult       = nullptr;   // [NF] Hermitian multiplicity
    double*             d_e          = nullptr;   // [NF] per-bin energies
    CUDA_CHECK(cudaMalloc(&d_grid_fixed, NG * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_grid_real,  NG * sizeof(cufftReal)));
    CUDA_CHECK(cudaMalloc(&d_F,          NF * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_influence,  NF * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mult,       NF * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_e,          NF * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grid_fixed, 0, NG * sizeof(unsigned long long)));  // zero the grid
    CUDA_CHECK(cudaMemcpy(d_influence, influence.data(), NF * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mult, mult.data(), NF * sizeof(double), cudaMemcpyHostToDevice));

    // ---- The cuFFT plan, NOT a black box ----------------------------------
    // cufftPlan3d(plan, K, K, K, CUFFT_R2C) builds a plan for ONE real-to-complex
    // 3D FFT of a K x K x K real grid. cufftExecR2C then computes, for each
    // integer wavevector m = (mx,my,mz):
    //     F[m] = sum_{r} rho[r] * exp(-2*pi*i * (m . r) / K)
    // i.e. the discrete structure factor of the charge grid -- exactly what
    // dft3d_r2c does on the host, in O(K^3 log K) instead of our teaching O(K^4).
    // R2C stores only the non-redundant half (Hermitian symmetry of a real grid):
    // the fast axis is halved to K/2+1, giving the NF-sized output above. Hand-
    // rolling a 3D FFT (mixed-radix, twiddle factors, bit-reversal) is exactly
    // the wheel cuFFT lets us not reinvent.
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan3d(&plan, K, K, K, CUFFT_R2C));

    const int spread_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int cell_blocks   = (static_cast<int>(NG) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int bin_blocks    = (static_cast<int>(NF) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    // STAGE 1: scatter charges into the fixed-point grid (atomic, deterministic).
    spread_kernel<<<spread_blocks, THREADS_PER_BLOCK>>>(d_x, d_y, d_z, d_q, n, K, s.box, d_grid_fixed);
    // STAGE 2a: convert the integer grid to a real density for the FFT.
    fixed_to_real_kernel<<<cell_blocks, THREADS_PER_BLOCK>>>(d_grid_fixed,
                                                             static_cast<int>(NG), d_grid_real);
    // STAGE 2b: forward 3D real-to-complex FFT (the library call).
    CUFFT_CHECK(cufftExecR2C(plan, d_grid_real, d_F));
    // STAGE 3: convolve with the Ewald influence + per-bin energy.
    energy_kernel<<<bin_blocks, THREADS_PER_BLOCK>>>(reinterpret_cast<const float2*>(d_F),
                                                     d_influence, d_mult,
                                                     static_cast<int>(NF), d_e);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("SPME pipeline (spread/convert/convolve)");

    // STAGE 4: copy per-bin energies back and SUM IN INDEX ORDER (the same order
    // pme_recip_cpu uses) for a deterministic, CPU-matching total.
    std::vector<double> e(NF);
    CUDA_CHECK(cudaMemcpy(e.data(), d_e, NF * sizeof(double), cudaMemcpyDeviceToHost));
    double E = 0.0;
    for (std::size_t i = 0; i < NF; ++i) E += e[i];

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_x));  CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));  CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_grid_fixed)); CUDA_CHECK(cudaFree(d_grid_real));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_influence)); CUDA_CHECK(cudaFree(d_mult));
    CUDA_CHECK(cudaFree(d_e));
    return E;
}
