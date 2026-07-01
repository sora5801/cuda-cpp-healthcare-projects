// ===========================================================================
// src/kernels.cu  --  BNCT Monte Carlo kernel (per-thread histories + atomics)
// ---------------------------------------------------------------------------
// Project 5.13 : BNCT Dose Calculation & Optimization (reduced-scope teaching MC)
//
// GPU twin of dose_cpu(): identical neutron histories (shared bnct_physics.h),
// but run in parallel and scored with atomicAdd into a flattened per-component
// depth-dose tally. main.cu runs both and asserts the tallies are identical
// bin-for-bin, component-for-component. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "bnct_physics.h"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 256 threads/block: a multiple of the 32-lane warp and a solid occupancy
// default across sm_75..sm_89. Not tuned -- this is a teaching kernel.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// dose_kernel: one neutron history per grid-stride iteration.
//   Launch config (set in dose_gpu):
//     grid  = 1024 blocks (fixed; the grid-stride loop covers any n_histories)
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: history index = blockIdx.x*blockDim.x + threadIdx.x,
//     then advanced by `stride = blockDim.x*gridDim.x` each loop iteration.
//   Memory: the tally lives in GLOBAL memory (small -- DC_COUNT*n_bins cells --
//     but written by many threads), so updates use atomicAdd. `dep` is a
//     per-thread scratch array in local memory, buffering one history's deposits.
//   Determinism: energy is INTEGER keV quanta, so the atomicAdds COMMUTE ->
//     the summed tally is independent of thread order and equals the CPU's.
//   Divergence: histories branch differently (leak / scatter / capture-by
//     B/N/H) and take different step counts -- the classic MC challenge;
//     production codes sort particles by material to shrink it (THEORY.md).
// ---------------------------------------------------------------------------
__global__ void dose_kernel(SimParams sp, unsigned long long n_histories,
                            unsigned long long seed, int n_bins,
                            unsigned long long* __restrict__ tally) {
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    const unsigned long long start =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    Deposit dep[BNCT_MAX_DEPOSITS];   // per-thread scratch for one history

    // Grid-stride loop: this thread processes histories start, start+stride, ...
    for (unsigned long long i = start; i < n_histories; i += stride) {
        Rng rng = rng_seed(seed, i);                        // history i's stream
        const int nd = simulate_neutron(sp, rng, dep);      // shared transport
        for (int d = 0; d < nd; ++d) {
            // Flatten (component, bin) -> one tally index (row-major, n_bins
            // columns per component). Many threads may hit the same cell -> the
            // atomicAdd serializes them; integer adds => order-independent sum.
            const unsigned long long idx =
                static_cast<unsigned long long>(dep[d].component) * n_bins + dep[d].bin;
            atomicAdd(&tally[idx], static_cast<unsigned long long>(dep[d].keV));
        }
    }
}

// ---------------------------------------------------------------------------
// dose_gpu: host wrapper -- allocate + zero the flat tally, launch all
// histories, copy back, unpack into the DoseTally structure. All CUDA
// bookkeeping (checked via CUDA_CHECK) is hidden from main.cu here. We time
// ONLY the kernel (CUDA events), not the copies -- that is the fair MC figure.
// ---------------------------------------------------------------------------
void dose_gpu(const BnctProblem& prob, DoseTally& t, float* kernel_ms) {
    const int n_bins = prob.sp.n_bins;
    const size_t n_cells = static_cast<size_t>(DC_COUNT) * n_bins;   // flat length

    // Device tally: DC_COUNT rows x n_bins cols, row-major, zero-initialized.
    unsigned long long* d_tally = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tally, n_cells * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_tally, 0, n_cells * sizeof(unsigned long long)));

    // A fixed, generous grid: enough resident warps to hide the long-latency,
    // divergent histories. The grid-stride loop covers ANY number of histories
    // with this one launch (no host-side chunking needed).
    const int blocks = 1024;
    GpuTimer timer;
    timer.start();
    dose_kernel<<<blocks, THREADS_PER_BLOCK>>>(prob.sp, prob.n_histories,
                                               prob.seed, n_bins, d_tally);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("dose_kernel");   // catch launch/exec errors immediately

    // Copy the flat tally back and unpack into the DC_COUNT x n_bins structure.
    std::vector<unsigned long long> flat(n_cells);
    CUDA_CHECK(cudaMemcpy(flat.data(), d_tally,
                          n_cells * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tally));

    t.reset(n_bins);
    for (int c = 0; c < DC_COUNT; ++c)
        for (int b = 0; b < n_bins; ++b)
            t.dose[c][b] = flat[static_cast<size_t>(c) * n_bins + b];
}
