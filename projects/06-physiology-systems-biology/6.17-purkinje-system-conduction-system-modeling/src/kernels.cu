// ===========================================================================
// src/kernels.cu  --  Ensemble-of-cables kernel (one thread per Purkinje cable)
// ---------------------------------------------------------------------------
// Project 6.17 : Purkinje System & Conduction System Modeling
//
// WHAT THIS FILE DOES
//   Implements the device kernel (simulate_kernel) and the host-side glue
//   (simulate_gpu) that uploads the CableParams array, launches one thread per
//   cable, times the kernel with CUDA events, and copies the CableResults back.
//   Each thread runs the SAME pk_simulate_cable() as the CPU reference
//   (purkinje.h, shared __host__ __device__ code), so main.cu can compare the two
//   result sets and verify agreement. See ../THEORY.md "GPU mapping".
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea) and purkinje.h (math).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a good default here: each thread carries three
// PK_MAX_NODES-double scratch arrays in local memory, so we favour a modest
// block size (fewer registers/local pressure per SM) over the 256 we would use
// for a light element-wise kernel. Still a multiple of the 32-lane warp.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// simulate_kernel: one thread integrates ONE cable's monodomain PDE.
//   Launch config (set in simulate_gpu):
//     grid  = ceil(n / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: i = blockIdx.x*blockDim.x + threadIdx.x owns cable i.
//
//   Each thread declares its OWN scratch buffers (Va/Vb ping-pong voltage +
//   w recovery) as local arrays -- per-thread private storage, no sharing, no
//   atomics. That is why the ensemble is embarrassingly parallel: cable i's
//   solve touches nothing cable j touches. Divergence is mild -- all cables run
//   the same n_steps; only the threshold-crossing bookkeeping branches differ.
// ---------------------------------------------------------------------------
__global__ void simulate_kernel(const CableParams* __restrict__ params, int n,
                                CableResult* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's cable
    if (i >= n) return;                                    // guard ragged last block

    // Per-thread scratch in LOCAL memory. Sized to the compile-time maximum so
    // the layout is identical to the CPU reference's stack buffers -> identical
    // arithmetic -> identical results. (These arrays are why we keep the block
    // size modest; see the THREADS_PER_BLOCK note above.)
    double Va[PK_MAX_NODES];   // voltage buffer A (ping)
    double Vb[PK_MAX_NODES];   // voltage buffer B (pong)
    double w [PK_MAX_NODES];   // recovery variable

    // Reuse the SHARED host/device stepper -- the single source of truth for the
    // cable physics. main.cu checks that this matches simulate_cpu()'s output.
    out[i] = pk_simulate_cable(params[i], Va, Vb, w);
}

// ---------------------------------------------------------------------------
// simulate_gpu: host wrapper. The canonical CUDA computation steps:
//   (1) allocate device memory   (2) copy CableParams host->device
//   (3) launch the kernel         (4) copy CableResults device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the (tiny) PCIe transfer cost.
// ---------------------------------------------------------------------------
void simulate_gpu(const PurkinjeTree& t, std::vector<CableResult>& results,
                  float* kernel_ms) {
    const int N = tree_size(t);
    results.assign(static_cast<std::size_t>(N), CableResult{});

    // (1) Device buffers. d_ prefix = DEVICE pointer (dereferencing on the host
    //     would crash). One array of inputs, one of outputs.
    CableParams* d_params = nullptr;   // [N] uploaded cable descriptions
    CableResult* d_out    = nullptr;   // [N] per-cable measured results
    CUDA_CHECK(cudaMalloc(&d_params, static_cast<std::size_t>(N) * sizeof(CableParams)));
    CUDA_CHECK(cudaMalloc(&d_out,    static_cast<std::size_t>(N) * sizeof(CableResult)));

    // (2) Copy the flat cables[] array H2D (contiguous std::vector storage).
    CUDA_CHECK(cudaMemcpy(d_params, t.cables.data(),
                          static_cast<std::size_t>(N) * sizeof(CableParams),
                          cudaMemcpyHostToDevice));

    // (3) Launch: enough blocks to cover all N cables (ceiling division).
    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    simulate_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_params, N, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("simulate_kernel");      // catch launch + execution errors

    // (4) Bring the per-cable results back to the host.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(N) * sizeof(CableResult),
                          cudaMemcpyDeviceToHost));

    // (5) Always free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_out));
}
