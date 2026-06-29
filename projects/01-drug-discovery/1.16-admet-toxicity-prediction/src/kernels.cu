// ===========================================================================
// src/kernels.cu  --  ADMET prediction + flag-count kernels and host wrapper
// ---------------------------------------------------------------------------
// Project 1.16 : ADMET / Toxicity Prediction  (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of the CPU reference in reference_cpu.cpp:
//     * predict_kernel    -> the N x M matrix of toxicity probabilities, one
//                            thread per (molecule, endpoint) cell, reading the
//                            endpoint MODELS from constant memory.
//     * flag_count_kernel -> the deterministic per-endpoint flagged-molecule
//                            counts via INTEGER atomicAdd.
//     * admet_screen_gpu  -> the host glue: upload, launch, time, download, and
//                            the tiny serial "worst molecule" argmax.
//   main.cu runs this AND the CPU path and asserts they agree.
//
//   Because the per-element math comes from admet_core.h (the HD-macro shared
//   header), predict_kernel and admet_predict_cpu run identical arithmetic ->
//   exact verification (PATTERNS.md sec.2).
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea),
//   admet_core.h (the shared math).
// ===========================================================================
#include "kernels.cuh"
#include "admet_core.h"          // admet_predict, admet_flagged (HD-shared math)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// THE ENDPOINT MODELS IN CONSTANT MEMORY.
//   Every thread reads the same M endpoint weight vectors + biases, never writes
//   them during a launch, and they are identical across the whole grid -> this
//   is the textbook case for __constant__ memory: its hardware cache BROADCASTS
//   one address to an entire warp in a single transaction, instead of each
//   thread streaming M*D weights from global memory.
//   Footprint: ADMET_M*ADMET_D doubles (12*64*8 = 6144 B) + ADMET_M doubles
//   (96 B) = ~6.2 KB, far under the 64 KB constant bank. Sizes are compile-time
//   constants (admet_core.h), which is exactly what constant memory requires.
//   Filled by cudaMemcpyToSymbol() in admet_screen_gpu().
// ---------------------------------------------------------------------------
__constant__ double c_weights[ADMET_M * ADMET_D];  // [M][D] row-major endpoint weights
__constant__ double c_bias[ADMET_M];               // [M]     endpoint biases

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89 (see THEORY "GPU mapping" for the reasoning).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// predict_kernel: one logical thread per (molecule, endpoint) cell, via a
// grid-stride loop so a fixed-size grid covers an arbitrarily large N*M.
//   Flattened index `cell` in [0, n*M) decodes to:
//       i = cell / ADMET_M   (the molecule)        t = cell % ADMET_M (the endpoint)
//   Memory: molecule i's descriptor from global memory; endpoint t's model from
//   the constant cache; one double written to probs[cell]. No shared memory or
//   atomics -- every cell is independent (the reduction is a separate kernel).
//
//   Thread (blockIdx.x, threadIdx.x) starts at cell = block*blockDim + thread
//   and strides by the total thread count until cell >= n*M.
// ---------------------------------------------------------------------------
__global__ void predict_kernel(const double* __restrict__ desc, int n,
                               double* __restrict__ probs) {
    const long long total  = static_cast<long long>(n) * ADMET_M;  // # of cells
    const int        stride = blockDim.x * gridDim.x;              // threads in grid
    for (long long cell = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < total; cell += stride) {
        const int i = static_cast<int>(cell / ADMET_M);   // molecule
        const int t = static_cast<int>(cell % ADMET_M);   // endpoint

        const double* x = desc + static_cast<long long>(i) * ADMET_D;  // global
        const double* w = c_weights + t * ADMET_D;                     // constant cache
        const double  b = c_bias[t];                                   // constant cache

        // The ONE shared formula -> identical to the CPU reference bit-for-bit.
        probs[cell] = admet_predict(w, x, b, ADMET_D);
    }
}

// ---------------------------------------------------------------------------
// flag_count_kernel: one logical thread per (molecule, endpoint) cell. Threshold
// the probability into a 0/1 flag (the shared admet_flagged()) and atomically
// add it to the per-endpoint counter.
//
//   WHY INTEGER ATOMICS (and not a float reduction): many threads update the
//   same ADMET_M counters concurrently. Floating-point addition is not
//   associative, so a parallel *float* sum depends on the (nondeterministic)
//   completion order and would not match the CPU exactly. INTEGER adds commute,
//   so atomicAdd on an int counter is BOTH correct under contention AND
//   deterministic -- the count is the same every run and equals the CPU's
//   (PATTERNS.md sec.3, the determinism rule). Only ADMET_M (=12) counters are
//   contended, so atomic traffic is negligible.
// ---------------------------------------------------------------------------
__global__ void flag_count_kernel(const double* __restrict__ probs, int n,
                                  int* __restrict__ d_flagged) {
    const long long total  = static_cast<long long>(n) * ADMET_M;
    const int        stride = blockDim.x * gridDim.x;
    for (long long cell = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < total; cell += stride) {
        const int t = static_cast<int>(cell % ADMET_M);        // endpoint
        const int f = admet_flagged(probs[cell], ADMET_THRESHOLD);  // 0 or 1
        if (f) atomicAdd(&d_flagged[t], 1);  // integer add -> deterministic
    }
}

// ---------------------------------------------------------------------------
// pick_blocks: choose a block count that covers `total` cells one-thread-each,
// but cap the grid so it stays modest; the grid-stride loops handle any larger
// problem. Returns at least 1 block.
// ---------------------------------------------------------------------------
static int pick_blocks(long long total) {
    long long b = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (b < 1)    b = 1;
    if (b > 1024) b = 1024;   // cap: grid-stride covers the remainder
    return static_cast<int>(b);
}

// ---------------------------------------------------------------------------
// admet_screen_gpu: the host wrapper. The canonical CUDA steps, with the models
// going to CONSTANT memory. We time ONLY the two kernels (CUDA events), not the
// H2D/D2H copies (discussed separately in THEORY).
// ---------------------------------------------------------------------------
void admet_screen_gpu(const AdmetData& data,
                      std::vector<double>& probs_out,
                      AdmetResult& result_out,
                      float* kernel_ms) {
    const int       n          = data.n;
    const long long total      = static_cast<long long>(n) * ADMET_M;
    const std::size_t desc_bytes  = static_cast<std::size_t>(n) * ADMET_D * sizeof(double);
    const std::size_t probs_bytes = static_cast<std::size_t>(total) * sizeof(double);

    probs_out.assign(static_cast<std::size_t>(total), 0.0);

    // (a) Upload the endpoint models into the __constant__ symbols. cudaMemcpy-
    //     ToSymbol targets the constant bank (a special copy, not ordinary
    //     global memory). Sizes are fixed at compile time, so this is one-shot.
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, data.weights.data(),
                                  sizeof(double) * ADMET_M * ADMET_D));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bias, data.bias.data(),
                                  sizeof(double) * ADMET_M));

    // (b) Allocate + upload the descriptors; allocate the probability matrix and
    //     the per-endpoint flag counters (zeroed so atomicAdd starts from 0).
    double* d_desc  = nullptr;   // [n*D] descriptors
    double* d_probs = nullptr;   // [n*M] probabilities (output)
    int*    d_flag  = nullptr;   // [M]   flag counters (output)
    CUDA_CHECK(cudaMalloc(&d_desc,  desc_bytes));
    CUDA_CHECK(cudaMalloc(&d_probs, probs_bytes));
    CUDA_CHECK(cudaMalloc(&d_flag,  ADMET_M * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_desc, data.desc.data(), desc_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_flag, 0, ADMET_M * sizeof(int)));   // counters := 0

    // (c) Launch both kernels back-to-back and time the pair with CUDA events.
    const int blocks = pick_blocks(total);
    GpuTimer timer;
    timer.start();
    predict_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_desc, n, d_probs);
    flag_count_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_probs, n, d_flag);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("predict/flag_count kernels");

    // (d) Copy the probability matrix and the per-endpoint counts back.
    CUDA_CHECK(cudaMemcpy(probs_out.data(), d_probs, probs_bytes, cudaMemcpyDeviceToHost));
    std::vector<int> flagged(ADMET_M, 0);
    CUDA_CHECK(cudaMemcpy(flagged.data(), d_flag, ADMET_M * sizeof(int), cudaMemcpyDeviceToHost));

    // (e) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_desc));
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_flag));

    // (f) Assemble the AdmetResult. The per-endpoint counts come from the GPU
    //     reduction; the per-molecule totals + worst-molecule argmax are a tiny
    //     O(n*M) serial scan over the GPU probabilities -- not worth a kernel,
    //     and admet_reduce() is the single obviously-correct version shared with
    //     the CPU path. We overwrite its flag counts with the GPU's so main.cu
    //     verifies the GPU reduction (they must match exactly).
    result_out = admet_reduce(data, probs_out);
    result_out.flagged_per_endpoint = flagged;   // GPU-computed counts
}
