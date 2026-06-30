// ===========================================================================
// src/kernels.cu  --  Batched variant-effect kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.19 : Variant Effect / Pathogenicity Prediction
//
// WHAT THIS FILE DOES
//   Implements the device kernel (score_variants_kernel) and the host glue
//   (score_variants_gpu) that uploads the fixed model to CONSTANT memory and the
//   variant windows to GLOBAL memory, launches one thread per variant, times the
//   kernel, and brings the delta scores back. This is the GPU twin of
//   score_variants_cpu() in reference_cpu.cpp; main.cu runs both and asserts they
//   agree. The actual per-variant arithmetic lives in vep_model.h and is shared
//   verbatim with the CPU side (PATTERNS.md sec 2) -- so the two paths match to
//   ~1e-12 and verification is meaningful, not hand-wavy. See ../THEORY.md.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea), and
//   vep_model.h (the math the kernel calls).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// The fixed model in CONSTANT memory.
//   * Every thread reads the SAME weights and NONE writes them, and they are
//     identical for the whole launch -> constant memory is the ideal home: its
//     hardware cache broadcasts one address to an entire warp in a single
//     transaction, instead of every thread streaming the weights from global
//     memory. This is the direct analogue of the constant-memory query in 1.12.
//   * VepModel is a fixed-size, trivially-copyable struct, so sizeof(VepModel)
//     is a compile-time constant comfortably within the 64 KB constant bank
//     (here a few KB). Filled by cudaMemcpyToSymbol() in score_variants_gpu().
// ---------------------------------------------------------------------------
__constant__ VepModel c_model;

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide latency, and many blocks resident for occupancy.
// Each thread does ~2*680 double FMAs (two forward passes) in registers, so this
// kernel is COMPUTE-bound per thread, not memory-bound -- occupancy mainly hides
// the constant-memory and global reads of the small int8 windows.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// score_variants_kernel: one logical thread per variant, via a grid-stride loop
// so a fixed-size grid still covers an arbitrarily large batch.
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count until i >= n.
//   Memory: c_model from the constant cache (broadcast); the two int8 windows
//   ref[i*L..], alt[i*L..] from global memory (each variant's L=21 bytes are
//   contiguous); the score is written once to out[i]. No shared memory or
//   atomics -- outputs are fully independent, the cleanest possible mapping.
// ---------------------------------------------------------------------------
__global__ void score_variants_kernel(const int8_t* __restrict__ ref,
                                      const int8_t* __restrict__ alt,
                                      int n,
                                      double* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;            // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // This variant's reference and alternate windows (contiguous L bytes).
        const int8_t* ref_win = ref + static_cast<std::size_t>(i) * VEP_WINDOW;
        const int8_t* alt_win = alt + static_cast<std::size_t>(i) * VEP_WINDOW;
        // Call the SHARED core: identical code to the CPU reference, so the GPU
        // delta score equals the CPU one up to floating-point rounding only.
        out[i] = vep_variant_effect(c_model, ref_win, alt_win);
    }
}

// ---------------------------------------------------------------------------
// score_variants_gpu: the canonical CUDA steps, with the model going to constant
// memory instead of a global buffer. We time ONLY the kernel (CUDA events), not
// the H2D/D2H copies (discussed separately in THEORY "GPU mapping").
// ---------------------------------------------------------------------------
void score_variants_gpu(const VepModel& m, const VariantSet& vs,
                        std::vector<double>& out, float* kernel_ms) {
    const int n = vs.n;
    out.assign(static_cast<std::size_t>(n), 0.0);
    const std::size_t win_bytes = static_cast<std::size_t>(n) * VEP_WINDOW * sizeof(int8_t);
    const std::size_t out_bytes = static_cast<std::size_t>(n) * sizeof(double);

    // (a) Upload the model to the __constant__ symbol (a special copy that
    //     targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_model, &m, sizeof(VepModel)));

    // (b) Allocate + upload the two window arrays, and allocate the output.
    int8_t* d_ref = nullptr;   // [n*VEP_WINDOW] reference windows, row-major
    int8_t* d_alt = nullptr;   // [n*VEP_WINDOW] alternate windows, row-major
    double* d_out = nullptr;   // [n] delta scores
    CUDA_CHECK(cudaMalloc(&d_ref, win_bytes));
    CUDA_CHECK(cudaMalloc(&d_alt, win_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_ref, vs.ref.data(), win_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alt, vs.alt.data(), win_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-variant, but capped so
    //     the grid stays modest; the grid-stride loop handles any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;        // cap: grid-stride covers the rest
    GpuTimer timer;
    timer.start();
    score_variants_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_ref, d_alt, n, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("score_variants_kernel");// catch launch + execution errors

    // (d) Copy scores back, then (e) free device memory (no GPU GC exists).
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_alt));
    CUDA_CHECK(cudaFree(d_out));
}
