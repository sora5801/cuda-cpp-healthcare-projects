// ===========================================================================
// src/kernels.cu  --  Ensemble whole-heart kernel (one thread per virtual heart)
// ---------------------------------------------------------------------------
// Project 6.2 : Whole-Heart Digital Twin   (REDUCED-SCOPE TEACHING VERSION)
//
// WHAT THIS FILE DOES
//   GPU twin of integrate_cpu(): each thread runs the SAME multi-beat RK4 loop
//   (simulate_heart in heart.h) for one ensemble member (one contractility
//   value) and writes one TwinResult. main.cu compares the per-member results
//   against the CPU baseline. Because the physics is the shared HD-core header,
//   the two agree to round-off. See ../THEORY.md (GPU mapping).
//
//   Memory pattern: pure "independent jobs" -- no shared memory, no atomics, no
//   inter-thread communication. The only global-memory traffic is the tiny
//   TwinResult each thread writes at the end; all the heavy ODE state lives in
//   registers/local memory for the whole time loop.
//
// READ THIS AFTER: kernels.cuh, heart.h, reference_cpu.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a good default for a REGISTER-HEAVY per-thread ODE
// integrator like this one: RK4 over a 4-state system needs many live registers,
// so a smaller block keeps per-thread register pressure from capping occupancy,
// while still giving the scheduler 4 warps per block to hide latency.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel: thread idx owns ensemble member idx (one virtual heart).
//   Launch config (set in integrate_gpu):
//     grid  = ceil(n / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x -> member idx.
//
//   The thread reads its own E_max via member_params(), then runs the full
//   forward heart simulation and writes one TwinResult. Divergence is mild: all
//   members take the SAME number of time steps (steps = beats*bcl/dt); only the
//   data-dependent valve/peak branches differ slightly between threads, which
//   costs a little warp efficiency but keeps every member's math identical to
//   the CPU's.
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(EnsembleConfig c, TwinResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's member
    if (idx >= ensemble_size(c)) return;                     // guard ragged last block

    // Build THIS member's heart (baseline physiology + its own contractility) and
    // run the shared forward model. member_params() and simulate_heart() are the
    // very same __host__ __device__ functions the CPU reference calls.
    const HeartParams p = member_params(c, idx);
    out[idx] = simulate_heart(p, c.dt_ms, c.beats);
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. Because each thread carries its whole ODE state
//   in registers, the ONLY device memory we need is the output array of n
//   TwinResults -- there is no input array to copy up (the config is passed by
//   value into the kernel, and each thread derives its parameters on the fly).
//   The five canonical CUDA steps collapse to: allocate out, launch, copy back,
//   free. We time only the launch with CUDA events (kernel cost, not transfers).
// ---------------------------------------------------------------------------
void integrate_gpu(const EnsembleConfig& c, std::vector<TwinResult>& results, float* kernel_ms) {
    const int n = ensemble_size(c);
    results.assign(static_cast<std::size_t>(n), TwinResult{});
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(TwinResult);

    // (1) Device output buffer. d_ prefix marks a DEVICE pointer (CLAUDE.md 12).
    TwinResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));    // can fail: out of device memory

    // (2) Launch one thread per member. Ceiling division rounds the block count up.
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();             // GPU-measured kernel time (ms)
    CUDA_CHECK_LAST("ensemble_kernel");       // catch launch + execution errors

    // (3) Copy the n results back to the host vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (4) Free device memory (there is no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_out));
}
