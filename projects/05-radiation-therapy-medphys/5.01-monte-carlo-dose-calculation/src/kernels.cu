// ===========================================================================
// src/kernels.cu  --  Monte Carlo dose kernel (per-thread histories + atomics)
// ---------------------------------------------------------------------------
// Project 5.01 : Monte Carlo Dose Calculation (simplified slab)
//
// GPU twin of dose_cpu(): identical histories (shared mc_physics.h), but run in
// parallel and scored with atomicAdd. main.cu runs both and asserts the dose
// tallies are identical. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "mc_physics.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// dose_kernel: grid-stride over histories. Each iteration simulates one photon
// with its own reproducible RNG stream and atomically adds its deposits.
//   * No shared memory: the dose tally is small but written by many threads, so
//     it lives in global memory and is updated with atomicAdd.
//   * Integer quanta => the atomics commute => deterministic, CPU-matching sum.
//   * "Divergence": different photons take different numbers of steps / branches,
//     so threads in a warp finish at different times -- the classic MC challenge
//     (production codes sort particles by material to reduce it; see THEORY).
// ---------------------------------------------------------------------------
__global__ void dose_kernel(SimParams sp, unsigned long long n_photons,
                            unsigned long long seed,
                            unsigned long long* __restrict__ dose) {
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    const unsigned long long start =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    int bins[MC_MAX_DEPOSITS];
    unsigned long long amts[MC_MAX_DEPOSITS];

    for (unsigned long long i = start; i < n_photons; i += stride) {
        Rng rng = rng_seed(seed, i);
        const int nd = simulate_photon(sp, rng, bins, amts);
        for (int d = 0; d < nd; ++d)
            atomicAdd(&dose[bins[d]], amts[d]);   // many threads -> same bins
    }
}

void dose_gpu(const DoseProblem& prob, std::vector<unsigned long long>& dose, float* kernel_ms) {
    const int n_bins = prob.sp.n_bins;
    dose.assign(n_bins, 0ULL);

    unsigned long long* d_dose = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dose, n_bins * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_dose, 0, n_bins * sizeof(unsigned long long)));

    // Enough blocks to give the GPU plenty of resident warps; the grid-stride
    // loop covers any number of histories with this fixed grid.
    int blocks = 1024;
    GpuTimer timer;
    timer.start();
    dose_kernel<<<blocks, THREADS_PER_BLOCK>>>(prob.sp, prob.n_photons, prob.seed, d_dose);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("dose_kernel");

    CUDA_CHECK(cudaMemcpy(dose.data(), d_dose, n_bins * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dose));
}
