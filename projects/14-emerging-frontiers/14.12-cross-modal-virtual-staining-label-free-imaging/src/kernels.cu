// ===========================================================================
// src/kernels.cu  --  The GPU kernel and its host wrapper (placeholder: SAXPY)
// ---------------------------------------------------------------------------
// Project 14.12 -- Cross-Modal "Virtual Staining" & Label-Free Imaging   (template skeleton)
//
// WHAT THIS FILE DOES
//   Implements the device kernel (saxpy_kernel) and the host-side glue
//   (saxpy_gpu) that allocates GPU memory, moves data, launches the kernel,
//   times it, and brings the result back. This is the GPU twin of the CPU
//   reference in reference_cpu.cpp; main.cu runs both and compares them.
//
//   TODO(impl): replace the SAXPY math with this project's real kernel. Keep
//   the comment density high (CLAUDE.md section 6.2 targets >= 1:1 in kernels).
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: it is a multiple
// of the 32-lane warp, gives the scheduler 8 warps to hide memory latency, and
// leaves plenty of blocks resident for occupancy. (Tune per project/GPU.)
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// saxpy_kernel: one thread computes one output element.
//   Launch config (set in saxpy_gpu):
//     grid  = ceil(n / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: i = blockIdx.x * blockDim.x + threadIdx.x.
//   Memory: reads x[i], y[i] from global memory, writes out[i]; no shared
//   memory or atomics needed because elements are fully independent.
// ---------------------------------------------------------------------------
__global__ void saxpy_kernel(int n, float a,
                             const float* __restrict__ x,
                             const float* __restrict__ y,
                             float* __restrict__ out) {
    // Global index this thread is responsible for.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // GUARD THE RAGGED LAST BLOCK: n is rarely an exact multiple of the block
    // size, so the final block has threads with i >= n. They must do nothing,
    // or they would read/write out of bounds (an illegal-address crash).
    if (i < n) {
        // The actual work. On the GPU this single fused multiply-add runs in
        // parallel across all n threads at once -- that parallelism is the
        // entire point of the exercise.
        out[i] = a * x[i] + y[i];
    }
}

// ---------------------------------------------------------------------------
// saxpy_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (those are discussed separately in THEORY).
// ---------------------------------------------------------------------------
void saxpy_gpu(int n, float a, const std::vector<float>& x,
               const std::vector<float>& y, std::vector<float>& out,
               float* kernel_ms) {
    out.assign(static_cast<std::size_t>(n), 0.0f);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md 12):
    //     dereferencing one on the host would crash, so the naming matters.
    float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));     // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // (2) Copy inputs H2D. .data() is the contiguous backing array of vector.
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

    // (3) Launch. Blocks must cover all n elements, hence the ceiling division
    //     (n + B - 1) / B -- integer-arithmetic "round up".
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    saxpy_kernel<<<blocks, THREADS_PER_BLOCK>>>(n, a, d_x, d_y, d_out);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("saxpy_kernel");       // catch launch + execution errors

    // (4) Bring the result back to the host vector.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));
}
