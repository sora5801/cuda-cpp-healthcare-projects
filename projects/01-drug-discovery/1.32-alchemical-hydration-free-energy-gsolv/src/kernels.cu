// ===========================================================================
// src/kernels.cu  --  Ensemble Metropolis kernel (one thread per walker)
// ---------------------------------------------------------------------------
// Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
//
// GPU twin of run_cpu(): each thread runs the SAME Metropolis chain (alchemy.h)
// for one (window, walker) and writes one WalkerResult. main.cu reduces the
// per-walker results into per-window stats and then into delta-G, and verifies
// the GPU per-walker results against the CPU's. See ../THEORY.md section 4.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

// 128 threads/block is a solid occupancy default on sm_75..sm_89. Each thread is
// register-heavy (it holds the solute position, energy, and accumulators), so a
// moderate block size avoids spilling while keeping the SMs busy.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel: thread `gid` owns global walker gid.
//   It decodes its (window w, walker k) from gid, looks up the window's lambda
//   and its neighbours' lambdas (for BAR), then runs the full Metropolis loop and
//   writes one WalkerResult. Pure embarrassing parallelism -- no shared memory, no
//   atomics, no inter-thread communication.
//
//   THREAD -> DATA MAPPING
//     gid = blockIdx.x*blockDim.x + threadIdx.x   in [0, n_windows*n_walkers)
//     w   = gid / n_walkers     (which lambda-window)
//     k   = gid % n_walkers     (which walker in that window)  -- k is unused
//           directly because the walker's RNG is seeded from gid, guaranteeing an
//           independent, reproducible stream that MATCHES the CPU's gid loop.
//
//   The solvent coordinates (d_x/d_y/d_z) are read-only device pointers; we wrap
//   them in a SolventBath on the stack so run_walker() is byte-identical to the
//   host call. n_windows is passed so we can clamp the neighbour windows at 0 / 1.
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(SystemParams sys,
                                const double* __restrict__ d_x,
                                const double* __restrict__ d_y,
                                const double* __restrict__ d_z,
                                int n_solvent, int n_windows, int n_walkers,
                                uint64_t seed, int n_equil, int n_prod,
                                WalkerResult* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_windows * n_walkers;
    if (gid >= total) return;                       // guard the ragged last block

    const int w = gid / n_walkers;                  // this thread's lambda-window

    // Reconstruct this window's lambda and its neighbours' lambdas on a uniform
    // [0,1] grid. We inline the formula (window_lambda lives in a host header that
    // is fine to call here too, but recomputing keeps the kernel self-contained).
    const double inv = (n_windows > 1) ? 1.0 / double(n_windows - 1) : 0.0;
    const double lam      = double(w) * inv;
    const double lam_prev = double((w > 0)             ? w - 1 : w) * inv;
    const double lam_next = double((w < n_windows - 1) ? w + 1 : w) * inv;

    // Wrap the device solvent arrays in the same SolventBath the host uses.
    SolventBath bath{ d_x, d_y, d_z, n_solvent };

    // Run the identical Metropolis walker the CPU runs -> identical result.
    out[gid] = run_walker(bath, sys, lam, lam_prev, lam_next,
                          seed, uint64_t(gid), n_equil, n_prod);
}

// ---------------------------------------------------------------------------
// run_gpu: copy the bath to the device, launch one thread per walker, copy the
// WalkerResults back, and time just the kernel with CUDA events.
//   We deliberately exclude the H2D/D2H copies from the timing: they are tiny
//   here (a few KB of bath + results) and the teaching point is the parallel
//   sampling work, not PCIe traffic. (A throughput study would time the whole
//   pipeline; see THEORY section 5 and the honest-timing note in PATTERNS.md.)
// ---------------------------------------------------------------------------
void run_gpu(const AlchConfig& c, const BathStorage& bath,
             std::vector<WalkerResult>& walkers, float* kernel_ms) {
    const int W = total_walkers(c);
    const int n = c.sys.n_solvent;
    walkers.assign(W, WalkerResult{});

    // --- allocate + upload the read-only solvent bath (SoA doubles) ----------
    double *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    const std::size_t bath_bytes = std::size_t(n) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_x, bath_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bath_bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bath_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, bath.x.data(), bath_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, bath.y.data(), bath_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, bath.z.data(), bath_bytes, cudaMemcpyHostToDevice));

    // --- output buffer: one WalkerResult per walker --------------------------
    WalkerResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, std::size_t(W) * sizeof(WalkerResult)));

    // --- launch one thread per walker, timed with CUDA events ----------------
    const int blocks = (W + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        c.sys, d_x, d_y, d_z, n, c.n_windows, c.n_walkers,
        c.seed, c.n_equil, c.n_prod, d_out);
    *kernel_ms = timer.stop_ms();          // blocks until the kernel finishes
    CUDA_CHECK_LAST("ensemble_kernel");    // catch launch + execution errors

    // --- copy results back and free device memory ----------------------------
    CUDA_CHECK(cudaMemcpy(walkers.data(), d_out,
                          std::size_t(W) * sizeof(WalkerResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
}
