// ===========================================================================
// src/kernels.cu  --  Tanimoto similarity kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.12 : Molecular Fingerprint Similarity Search
//
// This is the GPU twin of tanimoto_cpu() in reference_cpu.cpp. main.cu runs
// both and asserts they agree. See ../THEORY.md sec "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// ---------------------------------------------------------------------------
// The query fingerprint in CONSTANT memory.
//   * Every thread reads all FP_WORDS query words but NONE writes them, and they
//     are identical for the whole launch -> constant memory is the ideal home:
//     its hardware cache broadcasts one address to an entire warp in a single
//     transaction, instead of FP_WORDS global loads per thread.
//   * Size is fixed at compile time (FP_WORDS * 8 = 256 bytes), well within the
//     64 KB constant bank. Filled by cudaMemcpyToSymbol() in tanimoto_gpu().
// ---------------------------------------------------------------------------
__constant__ uint64_t c_query[FP_WORDS];

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89 (see THEORY "GPU mapping" for the occupancy reasoning).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// tanimoto_kernel: one logical thread per library molecule, via a grid-stride
// loop so a fixed-size grid still covers an arbitrarily large library.
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count until i >= n.
//   Memory: c_query from constant cache; lib row i from global memory (coalesced
//   when consecutive threads read consecutive rows... see THEORY for the layout
//   trade-off). No shared memory or atomics: outputs are fully independent.
// ---------------------------------------------------------------------------
__global__ void tanimoto_kernel(const uint64_t* __restrict__ lib, int n,
                                float* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint64_t* b = lib + static_cast<std::size_t>(i) * FP_WORDS;  // row i
        int inter = 0, uni = 0;
        // FP_WORDS is a compile-time constant, so the compiler fully unrolls
        // this loop -> no loop overhead, just a straight line of popcounts.
        #pragma unroll
        for (int w = 0; w < FP_WORDS; ++w) {
            const uint64_t a  = c_query[w];   // broadcast from constant cache
            const uint64_t bb = b[w];         // this molecule's word w
            inter += __popcll(a & bb);        // 64-bit popcount = one instruction
            uni   += __popcll(a | bb);
        }
        // Same exact-integer division as the CPU reference -> bit-identical.
        out[i] = uni ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// tanimoto_gpu: the five canonical CUDA steps, with the query going to constant
// memory instead of a global buffer. We time ONLY the kernel (CUDA events), not
// the H2D/D2H copies (those are discussed separately in THEORY).
// ---------------------------------------------------------------------------
void tanimoto_gpu(const FingerprintSet& fps, std::vector<float>& out, float* kernel_ms) {
    const int n = fps.n;
    out.assign(static_cast<std::size_t>(n), 0.0f);
    const std::size_t lib_bytes = static_cast<std::size_t>(n) * FP_WORDS * sizeof(uint64_t);
    const std::size_t out_bytes = static_cast<std::size_t>(n) * sizeof(float);

    // (a) Upload the query to the __constant__ symbol (a special copy that
    //     targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_query, fps.query.data(), FP_WORDS * sizeof(uint64_t)));

    // (b) Allocate + upload the library, and allocate the output scores.
    uint64_t* d_lib = nullptr;   // [n*FP_WORDS] device, row-major
    float*    d_out = nullptr;   // [n] device scores
    CUDA_CHECK(cudaMalloc(&d_lib, lib_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_lib, fps.lib.data(), lib_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-molecule, but capped
    //     so the grid stays modest; the grid-stride loop handles the remainder.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride covers any larger n
    GpuTimer timer;
    timer.start();
    tanimoto_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_lib, n, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("tanimoto_kernel");

    // (d) Copy scores back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_lib));
    CUDA_CHECK(cudaFree(d_out));
}
