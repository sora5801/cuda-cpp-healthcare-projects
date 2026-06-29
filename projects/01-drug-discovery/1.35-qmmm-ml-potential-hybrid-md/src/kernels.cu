// ===========================================================================
// src/kernels.cu  --  Ensemble hybrid-MD kernel (one thread per trajectory)
// ---------------------------------------------------------------------------
// Project 1.35 : QMMM/ML Potential Hybrid MD   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of integrate_cpu(): each thread runs the SAME velocity-Verlet
//   loop (via run_trajectory() in nnpmm.h) for one ensemble member, then writes
//   one TrajResult. main.cu runs both paths and compares the per-member results.
//
//   Because every per-step computation (the hybrid NNP+LJ force/energy and the
//   integrator) is a shared __host__ __device__ function in nnpmm.h, there is
//   literally one implementation of the physics -- the kernel and the CPU loop
//   call the exact same code. That is what makes the verification meaningful.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea), nnpmm.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide latency, and -- importantly here -- each
// thread is register-heavy (it holds 3*N_ATOMS doubles of MD state), so a
// smaller block keeps register pressure / occupancy reasonable. (Tune per GPU.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel: thread idx owns ensemble member idx.
//   It runs the FULL velocity-Verlet trajectory for that member entirely in
//   registers/local memory (run_trajectory builds the geometry, primes the
//   forces, integrates `steps` steps, and reduces to a TrajResult), then writes
//   that one summary to global memory. There is NO inter-thread communication.
//
//   Divergence note: all members run the same number of steps and the same
//   branch-free force loops, so warps stay coherent; only the member-specific
//   perturbation differs. This is why the ensemble maps so cleanly to the GPU.
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(EnsembleConfig c, TrajResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's member
    if (idx >= ensemble_size(c)) return;                    // guard ragged last block

    // The whole MD trajectory for member idx -- identical call to the CPU path.
    out[idx] = run_trajectory(idx, c.M, c.amp, c.dt, c.steps);
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. The canonical CUDA steps, but the only data that
// crosses the bus is the small output array (the inputs are derived on-device
// from idx + the tiny config), so this is compute-bound, not transfer-bound.
//   (1) allocate the M-element device output buffer
//   (2) launch one thread per member            (timed with CUDA events)
//   (3) copy the results back to the host vector
//   (4) free device memory
// ---------------------------------------------------------------------------
void integrate_gpu(const EnsembleConfig& c, std::vector<TrajResult>& results,
                   float* kernel_ms) {
    const int M = ensemble_size(c);
    results.assign(M, TrajResult{});

    // (1) Device output buffer (one TrajResult per ensemble member). The d_
    //     prefix marks a DEVICE pointer (CLAUDE.md §12) -- never dereference on host.
    TrajResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(TrajResult)));

    // (2) Launch: blocks must cover all M members -> ceiling division.
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("ensemble_kernel");    // catch launch + execution errors

    // (3) Bring the per-member summaries back.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(TrajResult),
                          cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
