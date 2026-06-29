// ===========================================================================
// src/kernels.cu  --  GPU ensemble kernel: one thread per lambda-window
// ---------------------------------------------------------------------------
// Project 1.5 : Free Energy Perturbation / Thermodynamic Integration
//
// GPU twin of integrate_cpu(): each thread runs the SAME Metropolis MC chain
// (run_chain() in alchemy.h) for one lambda-window and writes its estimate of
// < dU/dlambda >_lambda. main.cu trapezoid-integrates those over lambda to get
// DeltaG_TI and compares both to the CPU reference and to the analytic answer.
// See ../THEORY.md "GPU mapping". (No shared memory / no atomics: windows are
// fully independent, so this is the cleanest embarrassing-parallel pattern.)
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a good default here: the work per thread is a long
// serial MC loop (compute-bound, register-heavy), so we do not need a huge block
// to hide memory latency; 128 keeps register pressure modest while giving the
// scheduler several warps. (Window counts are tiny, so occupancy is not the
// bottleneck -- correctness and determinism are what matter.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ti_kernel: thread `w` owns lambda-window w.
//   It runs the entire Metropolis chain for that window in registers/local
//   memory and writes one < dU/dlambda > plus its accepted-move count. No two
//   threads touch the same memory, and the counter-based RNG means the result
//   does not depend on thread scheduling -> fully deterministic and matching the
//   CPU. Divergence is mild: every chain runs the same number of steps; only the
//   accept/reject branch differs per step.
// ---------------------------------------------------------------------------
__global__ void ti_kernel(AlchemyConfig c,
                          double* __restrict__ dvals,
                          long long* __restrict__ accepted) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's window
    if (w >= n_windows(c)) return;                         // guard ragged last block

    long long acc = 0;
    dvals[w]    = run_chain(c, w, &acc);   // the one true sampler (alchemy.h)
    accepted[w] = acc;
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. The canonical CUDA steps:
//   (1) allocate device output buffers (W doubles + W counts)
//   (2) launch ti_kernel with one thread per window
//   (3) copy the per-window results back to the host
//   (4) free device memory
// There are NO input buffers to copy: the whole problem is described by the
// small AlchemyConfig, which we pass by value into the kernel (it lands in
// constant/parameter memory). We time ONLY the kernel (CUDA events).
// ---------------------------------------------------------------------------
void integrate_gpu(const AlchemyConfig& c,
                   std::vector<double>& dvals,
                   std::vector<long long>& accepted,
                   float* kernel_ms) {
    const int W = n_windows(c);
    dvals.assign(W, 0.0);
    accepted.assign(W, 0);

    // (1) Device output buffers. d_ marks DEVICE pointers (CLAUDE.md §12): a host
    //     dereference would crash, so the naming convention is load-bearing.
    double*    d_dvals = nullptr;   // [W] per-window < dU/dlambda >
    long long* d_acc   = nullptr;   // [W] per-window accepted-move counts
    CUDA_CHECK(cudaMalloc(&d_dvals, static_cast<std::size_t>(W) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_acc,   static_cast<std::size_t>(W) * sizeof(long long)));

    // (2) Launch. Cover all W windows: ceil(W / B) blocks (integer round-up).
    const int blocks = (W + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ti_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_dvals, d_acc);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("ti_kernel");          // catch launch + execution errors

    // (3) Bring the per-window results back to the host vectors.
    CUDA_CHECK(cudaMemcpy(dvals.data(), d_dvals,
                          static_cast<std::size_t>(W) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(accepted.data(), d_acc,
                          static_cast<std::size_t>(W) * sizeof(long long),
                          cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_dvals));
    CUDA_CHECK(cudaFree(d_acc));
}
