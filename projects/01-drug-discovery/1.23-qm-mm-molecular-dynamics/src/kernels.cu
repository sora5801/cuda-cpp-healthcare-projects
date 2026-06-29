// ===========================================================================
// src/kernels.cu  --  Ensemble QM/MM kernel (one thread per trajectory)
// ---------------------------------------------------------------------------
// Project 1.23 : QM/MM Molecular Dynamics   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of integrate_cpu(): each thread runs the SAME velocity-Verlet
//   loop (qmmm::integrate_trajectory in qmmm.h) for one (field, x0) pair from the
//   ensemble sweep, and writes one TrajResult. main.cu compares the per-member
//   results against the CPU reference; because both sides call the identical
//   __host__ __device__ core, agreement is to round-off. See ../THEORY.md.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea), qmmm.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a good default for a REGISTER-HEAVY kernel like this:
// each thread holds the full integrator state (x, v, accel, accumulators) plus
// the Verlet temporaries, so register pressure is the occupancy limiter, not the
// warp count. 128 keeps four warps resident per block while leaving registers for
// the time loop. (256 also works; profile per GPU -- THEORY.md §GPU mapping.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel: thread idx owns ensemble member idx.
//   It reads its (field, x0) from the parameter sweep, then runs the FULL Verlet
//   time loop in registers/local memory and writes one TrajResult. There is NO
//   inter-thread communication -- pure embarrassing parallelism over members.
//   Divergence is mild: every member runs the same `steps` iterations; only the
//   min-gap and product-side branches inside integrate_trajectory differ, and
//   those are cheap predicated updates, not long divergent paths.
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(EnsembleConfig c, qmmm::TrajResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's member
    if (idx >= ensemble_size(c)) return;                     // guard the ragged last block

    double field, x0;
    member_params(c, idx, field, x0);                        // (field, x0) for member idx
    // The whole QM/MM run for this trajectory -- identical math to the CPU path.
    out[idx] = qmmm::integrate_trajectory(x0, c.v0, field, c.dt, c.steps);
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. The canonical CUDA steps, but note there is NO
//   large input buffer to copy up -- the entire problem is the small
//   EnsembleConfig (passed by value into the kernel) plus the output array. So:
//     (1) allocate the device output array (M TrajResults)
//     (2) launch one thread per member (time ONLY the kernel, via CUDA events)
//     (3) copy the results device->host
//     (4) free the device array
//   We time step (2) with CUDA events so the figure is on-device compute, not
//   the PCIe copy of the results (discussed separately in THEORY).
// ---------------------------------------------------------------------------
void integrate_gpu(const EnsembleConfig& c, std::vector<qmmm::TrajResult>& results,
                   float* kernel_ms) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), qmmm::TrajResult{});

    // (1) Device output buffer. d_ marks a DEVICE pointer (CLAUDE.md §12).
    qmmm::TrajResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(qmmm::TrajResult)));

    // (2) Launch: cover all M members with ceil(M / block) blocks ("round up").
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();           // GPU-measured kernel time
    CUDA_CHECK_LAST("ensemble_kernel");     // catch launch + execution errors

    // (3) Bring the per-member results back to the host vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(qmmm::TrajResult),
                          cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
