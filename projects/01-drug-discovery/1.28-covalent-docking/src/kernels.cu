// ===========================================================================
// src/kernels.cu  --  Covalent-docking scoring kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.28 : Covalent Docking
//
// WHAT THIS FILE DOES
//   Implements the device kernel (score_kernel) and the host glue (score_all_gpu)
//   that allocates the device energy buffer, launches the kernel, times it, and
//   brings the energies back. This is the GPU twin of score_all_cpu() in
//   reference_cpu.cpp -- and crucially BOTH call the SAME score_conformation()
//   from docking.h, so their results are bit-identical. main.cu runs both and
//   compares them element-by-element before reporting the docked pose.
//
//   The pattern is "score N independent candidates" (one thread per conformation),
//   the same shape as project 1.12. There are no atomics and no shared memory:
//   each thread writes its own out[id], fully independent of the others.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea),
// docking.h (the shared scoring physics).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps per block to hide latency, and
// leaves many blocks resident for occupancy. Each thread here is compute-bound
// (trig + a small double-precision energy sum) rather than memory-bound, so the
// exact block size matters little; 256 keeps register pressure comfortable.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// score_kernel: one logical thread per conformation, grid-stride loop.
//   Launch config (set in score_all_gpu):
//     grid  = min(ceil(M / THREADS_PER_BLOCK), CAP) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: a thread starts at id = blockIdx.x*blockDim.x +
//   threadIdx.x and strides by the total thread count until id >= M, so a
//   capped grid still covers all M conformations.
//   Memory: reads the by-value DockProblem `p` from the kernel's parameter space
//   (a per-thread constant), writes one double to global memory out[id]. No
//   shared memory, no atomics -- outputs are fully independent.
//   Divergence: mild. All threads run the same forward-kinematics + energy code;
//   the only data-dependent branches are tiny clamps (acos domain, r floor).
// ---------------------------------------------------------------------------
__global__ void score_kernel(DockProblem p, long long M, double* __restrict__ out) {
    // Total number of threads in the grid -- the stride of the grid-stride loop.
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         id < M; id += stride) {
        // The whole per-conformation computation is in score_conformation(),
        // the shared __host__ __device__ core. Calling it from the device here
        // and from the host in reference_cpu.cpp is exactly what makes the GPU
        // and CPU energies bit-identical (the basis of our exact verification).
        out[id] = score_conformation(p, id);
    }
}

// ---------------------------------------------------------------------------
// score_all_gpu: host wrapper. The canonical CUDA steps, minus any input copy
// (the only input is the small DockProblem, passed by value to the kernel):
//   (1) allocate the device energy buffer
//   (2) launch the kernel (timed with CUDA events)
//   (3) copy the energies device->host
//   (4) free device memory
// We time ONLY the kernel so the reported figure is compute cost, not the tiny
// D2H copy (discussed separately in THEORY "Numerical considerations").
// ---------------------------------------------------------------------------
void score_all_gpu(const DockProblem& p, std::vector<double>& energies,
                   float* kernel_ms) {
    const long long M = n_conformations();
    energies.assign(static_cast<std::size_t>(M), 0.0);
    const std::size_t bytes = static_cast<std::size_t>(M) * sizeof(double);

    // (1) Device output buffer. d_ marks a DEVICE pointer (CLAUDE.md section 12):
    //     dereferencing it on the host would crash, so the naming is load-bearing.
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));   // can fail: out of device memory

    // (2) Launch. Cover all M conformations one-thread-each, but cap the grid so
    //     it stays modest; the grid-stride loop handles any larger M.
    int blocks = static_cast<int>((M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    if (blocks > 1024) blocks = 1024;        // cap; grid-stride covers the rest
    if (blocks < 1) blocks = 1;              // always launch at least one block
    GpuTimer timer;
    timer.start();
    score_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, M, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("score_kernel");         // catch launch + execution errors

    // (3) Bring the energies back to the host vector.
    CUDA_CHECK(cudaMemcpy(energies.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
