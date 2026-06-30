// ===========================================================================
// src/kernels.cu  --  The MEME E-step window-scoring kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.29 : Motif Finding in Genomic Sequences
//
// WHAT THIS FILE DOES
//   Implements the device kernel (score_windows_kernel) and the host glue
//   (score_windows_gpu) that uploads the sequence bytes + window starts + the
//   current log-odds table, launches the kernel, times it, and brings the
//   per-window scores back. This is the GPU twin of score_windows_cpu() in
//   reference_cpu.cpp; both call the SAME __host__ __device__ window_score()
//   (motif_core.h), so main.cu can verify them against each other EXACTLY.
//
//   This file only accelerates the EXPENSIVE step (scoring ~L windows of width
//   W). The cheap EM bookkeeping (per-sequence softmax, count accumulation,
//   renormalisation) stays on the host in run_meme_em_cpu() -- a deliberate
//   split that mirrors how mCUDA-MEME offloads MEME's hot loop to the GPU.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cstdio>                // std::fprintf
#include <cstdlib>               // std::exit

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide memory latency, and many blocks resident for
// occupancy. The kernel is memory-light (a tiny constant-table lookup per motif
// column) so occupancy, not arithmetic, sets the throughput.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// The current motif's LOG-ODDS table in CONSTANT memory.
//   logodds[p*4 + b] = log2( PWM[p][b] / bg[b] ), row-major, up to MAX_W rows.
//   Why constant memory: EVERY thread reads the WHOLE table but NONE writes it,
//   and it is identical for the whole launch. The constant cache broadcasts one
//   address to an entire warp in a single transaction -- so the W reads each
//   thread does cost ~free compared with fetching from global memory. The table
//   is small (MAX_W*4 floats = 1 KiB) and fixed-size, a textbook constant-memory
//   use (the same idiom as the query in 1.12 Tanimoto). Filled per EM iteration
//   by cudaMemcpyToSymbol() in score_windows_gpu().
// ---------------------------------------------------------------------------
__constant__ float c_logodds[MAX_W * MOTIF_ALPHABET];

// ---------------------------------------------------------------------------
// score_windows_kernel: one logical thread per window, grid-stride looped.
//   Launch config (set in score_windows_gpu):
//     grid  = up to 1024 blocks of THREADS_PER_BLOCK; a grid-stride loop lets
//             this fixed grid cover an arbitrarily large window count.
//     block = THREADS_PER_BLOCK threads.
//   Thread-to-data map: window index
//       win = blockIdx.x*blockDim.x + threadIdx.x, then += stride.
//   For window `win`, start_of_win[win] is the ABSOLUTE index into `data` of the
//   window's first base, so window_score() reads data[start .. start+w-1] and
//   sums the w constant-memory log-odds terms.
//   Memory: `data` and `start_of_win` from global memory (coalesced -- adjacent
//   threads read adjacent start_of_win entries); the log-odds table from the
//   constant cache. No shared memory or atomics: outputs are fully independent.
// ---------------------------------------------------------------------------
__global__ void score_windows_kernel(const unsigned char* __restrict__ data,
                                     const int* __restrict__ start_of_win,
                                     int num_windows, int w,
                                     float* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int win = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's window
         win < num_windows; win += stride) {               // grid-stride loop
        const int start = start_of_win[win];               // abs. start in `data`
        // EXACTLY the CPU's per-window math -- same function, same fixed loop
        // order p=0..w-1 -> the float sum is bit-identical to score_windows_cpu.
        out[win] = window_score(data, start, w, c_logodds);
    }
}

// ---------------------------------------------------------------------------
// score_windows_gpu: host wrapper. The canonical CUDA steps, with the log-odds
// table going to CONSTANT memory rather than a global buffer.
//   (a) upload the log-odds table to the __constant__ symbol,
//   (b) allocate + upload `data` and `start_of_win`, allocate the output,
//   (c) launch + time ONLY the kernel (CUDA events; copies excluded -- they are
//       discussed separately in THEORY), (d) copy scores back, (e) free.
// ---------------------------------------------------------------------------
void score_windows_gpu(const SequenceSet& set, const MotifModel& model,
                       std::vector<float>& out, float* kernel_ms) {
    const int w  = set.w;
    const int nw = set.total_windows();
    out.assign(static_cast<std::size_t>(nw), 0.0f);

    // Guard the constant-memory capacity: a wider motif than MAX_W would silently
    // overflow the symbol, so fail loudly instead (correctness-you-can-see).
    if (w > MAX_W) {
        std::fprintf(stderr,
            "[score_windows_gpu] motif width %d exceeds MAX_W=%d (raise MAX_W in "
            "kernels.cuh and rebuild)\n", w, MAX_W);
        std::exit(EXIT_FAILURE);
    }

    const std::size_t data_bytes  = set.data.size() * sizeof(unsigned char);
    const std::size_t win_bytes   = static_cast<std::size_t>(nw) * sizeof(int);
    const std::size_t out_bytes   = static_cast<std::size_t>(nw) * sizeof(float);
    const std::size_t table_bytes = static_cast<std::size_t>(w) * MOTIF_ALPHABET * sizeof(float);

    // (a) Upload the W x 4 log-odds table to constant memory (a special copy
    //     that targets the constant bank, not ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_logodds, model.logodds.data(), table_bytes));

    // (b) Allocate + upload the sequence bytes and the window-start index, and
    //     allocate the output score buffer.
    unsigned char* d_data  = nullptr;   // [total bases]
    int*           d_start = nullptr;   // [nw] absolute window starts
    float*         d_out   = nullptr;   // [nw] scores
    CUDA_CHECK(cudaMalloc(&d_data,  data_bytes));
    CUDA_CHECK(cudaMalloc(&d_start, win_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,   out_bytes));
    CUDA_CHECK(cudaMemcpy(d_data,  set.data.data(),         data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start, set.start_of_win.data(), win_bytes,  cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks for one-thread-per-window, capped so the grid
    //     stays modest; the grid-stride loop covers any larger window count.
    int blocks = (nw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1)    blocks = 1;
    GpuTimer timer;
    timer.start();
    score_windows_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_start, nw, w, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("score_windows_kernel");

    // (d) Copy scores back, then (e) free device memory (no GPU GC exists).
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_out));
}
