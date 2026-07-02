// ===========================================================================
// src/kernels.cu  --  Ensemble repressilator kernel (one thread per trajectory)
// ---------------------------------------------------------------------------
// Project 6.10 : Systems-Biology ODE/SDE Network Solver
//
// GPU twin of integrate_cpu(): each thread runs the same RK4 loop (grn.h) for
// one parameter set (alpha, n) and writes one MemberResult. main.cu compares the
// per-member results against the CPU reference. See ../THEORY.md §GPU-mapping.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event stopwatch)

// Block size. 128 threads/block is a solid occupancy default across sm_75..89:
// enough warps to hide latency, small enough that the per-thread register
// footprint of the double-precision RK4 temporaries does not throttle the number
// of resident blocks. THEORY.md §GPU-mapping discusses that register pressure.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel: thread idx owns ensemble member idx.
//   Launch config (set in integrate_gpu):
//     grid  = ceil(M / THREADS_PER_BLOCK) blocks, M = ensemble_size(c)
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x (a flat
//   member index into the alpha x n sweep grid).
//   Pure embarrassing parallelism -- no shared memory, no atomics, no
//   inter-thread communication. Divergence is mild: every member runs the SAME
//   number of steps; only the oscillation-bookkeeping branch differs, and even
//   that is cheap. Because integrate_member() is the SAME __host__ __device__
//   code the CPU calls (grn.h), this thread computes what the reference does for
//   the same inputs, up to floating-point round-off.
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(EnsembleConfig c, MemberResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's member
    if (idx >= ensemble_size(c)) return;                     // guard the ragged last block

    GrnParams pr;
    member_params(c, idx, pr);                               // (alpha,n) + fixed knobs for this member
    out[idx] = integrate_member(c.s0, pr, c.dt, c.steps);    // full RK4 trajectory -> summary
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. Allocates the [M] result buffer, launches one
//   thread per member, times ONLY the kernel (CUDA events), and copies back.
//   No input buffer is needed on the device: the entire problem is described by
//   the small EnsembleConfig, passed BY VALUE into the kernel (it rides in the
//   kernel's constant parameter bank, broadcast to every thread).
// ---------------------------------------------------------------------------
void integrate_gpu(const EnsembleConfig& c, std::vector<MemberResult>& results,
                   float* kernel_ms) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), MemberResult{});

    // Device output: M MemberResults. This is the only device allocation.
    MemberResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(MemberResult)));

    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;   // cover all M members
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time (includes the sync in stop_ms)
    CUDA_CHECK_LAST("ensemble_kernel");    // catch launch-config + in-kernel errors

    // Copy the M summaries back to the host result vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(MemberResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}
