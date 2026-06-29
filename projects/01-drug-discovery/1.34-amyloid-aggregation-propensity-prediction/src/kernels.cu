// ===========================================================================
// src/kernels.cu  --  GPU aggregation scan: one block per protein, tiled window
// ---------------------------------------------------------------------------
// Project 1.34 : Amyloid / Aggregation Propensity Prediction
//
// WHAT THIS FILE DOES
//   The GPU twin of scan_dataset_cpu(). It uploads the amino-acid propensity
//   scale into CONSTANT memory (read by every thread, never changes -> the
//   constant cache broadcasts a value to a whole warp), then launches ONE BLOCK
//   PER PROTEIN. Each block stages its protein's per-residue propensities into
//   SHARED memory once (the tile + halo), every thread reads its W-wide window
//   from there (calling the SAME windowed_mean() the CPU uses), and a single
//   thread reduces the smoothed profile to the protein's AggResult.
//
//   main.cu runs this and scan_dataset_cpu() and asserts they agree.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea), propensity.h (the
//                  shared physics). Compare against reference_cpu.cpp.
// ===========================================================================
#include "kernels.cuh"
#include "propensity.h"          // AA_PROPENSITY, propensity_of_code, windowed_mean
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cstdio>
#include <cstdlib>

// Threads per block. We scan one protein per block; each thread handles one (or
// more, by block-striding) residues of that protein. 256 is a solid sm_75..89
// default: a multiple of the 32-lane warp, 8 warps to hide latency, high
// occupancy. The shared tile sizes to the protein, NOT to the block.
static constexpr int AGG_THREADS = 256;

// The propensity scale in CONSTANT memory. It is 21 floats, identical for every
// thread of every block and constant for the whole launch -- the textbook case
// for __constant__ (a single value broadcasts to all 32 lanes of a warp in one
// transaction). We copy AA_PROPENSITY (from propensity.h) here once per run.
__constant__ float c_scale[AA_COUNT];

// ---------------------------------------------------------------------------
// scale_propensity_device: device-side lookup that reads the CONSTANT-memory
//   copy of the scale (the host/__host__ propensity_of_code in propensity.h
//   reads the plain array; on the device we want the cached constant copy). The
//   numbers are identical, so CPU and GPU still agree exactly.
// ---------------------------------------------------------------------------
__device__ inline float scale_propensity_device(int code) {
    if (code < 0 || code >= AA_COUNT) return 0.0f;   // PAD_CODE / junk -> 0
    return c_scale[code];
}

// ---------------------------------------------------------------------------
// agg_scan_kernel: scan ONE protein per block.
//   grid  : num blocks  (blockIdx.x = protein index p)
//   block : AGG_THREADS threads (threadIdx.x = residue stride origin)
//   shared: dynamic, `len` floats -- this protein's per-residue propensities.
//           We tile the WHOLE protein (len <= AGG_MAX_LEN), so the window mean's
//           halo reads are simply in-tile reads with clamping; no separate halo
//           load is needed (clamping at [0,len) handles the termini exactly as
//           the CPU does). For very long sequences a per-block tile would be
//           split across multiple blocks -- left as an exercise (THEORY §7).
//
//   Thread-to-data map: thread t processes residues t, t+blockDim, t+2*blockDim,
//   ... so all `len` residues are covered regardless of how len compares to the
//   block size (block-stride loop -- the standard "grid-stride" idiom applied
//   within a block).
//
//   Memory spaces touched: global (flat_codes/lengths read, smoothed/results
//   written), shared (the propensity tile), constant (the scale). No atomics:
//   the final reduction is done by thread 0 over the shared tile's smoothed
//   values recomputed deterministically, so the result is bit-reproducible.
// ---------------------------------------------------------------------------
__global__ void agg_scan_kernel(const int* __restrict__ flat_codes,
                                const int* __restrict__ lengths,
                                int stride, int half, float threshold,
                                float* __restrict__ smoothed,
                                AggResult* __restrict__ results) {
    const int p   = blockIdx.x;            // this block's protein index
    const int len = lengths[p];            // its real residue count
    const std::size_t base = static_cast<std::size_t>(p) * stride;  // row start

    // Dynamic shared tile: the protein's per-residue propensities (len floats).
    extern __shared__ float tile[];

    // 1) LOAD + LOOKUP: each thread fills the tile entries it owns by reading the
    //    residue code from global memory and mapping it through the constant-
    //    memory scale. Block-stride so any len <= AGG_MAX_LEN is covered.
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        tile[i] = scale_propensity_device(flat_codes[base + i]);
    }
    __syncthreads();                       // tile fully populated before any read

    // 2) SMOOTH: each thread computes the centered windowed mean for its
    //    residues, reading the W-wide window straight from shared memory. This
    //    is the same windowed_mean() the CPU calls -> identical arithmetic.
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        smoothed[base + i] = windowed_mean(tile, len, i, half);
    }
    __syncthreads();                       // all smoothed[] written before reduce

    // 3) REDUCE: one thread per block scans the smoothed profile to produce the
    //    AggResult. Doing this serially in thread 0 keeps it deterministic and
    //    dead simple; for short chains it is negligible vs. the parallel smooth.
    //    (A parallel reduction would need care to stay deterministic -- and the
    //    longest-APR run length is inherently sequential -- so we keep it here.)
    if (threadIdx.x == 0) {
        AggResult r;
        r.peak_score = -1.0f;              // first residue always wins initially
        int run = 0;                       // current contiguous prone run
        for (int i = 0; i < len; ++i) {
            const float s = smoothed[base + i];
            if (s > r.peak_score) { r.peak_score = s; r.peak_pos = i; }
            if (s >= threshold) {
                ++r.prone_count;
                ++run;
                if (run > r.longest_apr) r.longest_apr = run;
            } else {
                run = 0;
            }
        }
        if (len == 0) r.peak_score = 0.0f; // empty-protein guard (matches CPU)
        results[p] = r;
    }
}

// ---------------------------------------------------------------------------
// scan_dataset_gpu: host wrapper -- the five canonical CUDA steps.
//   (1) upload the propensity scale to constant memory + the flat batch to
//       global; (2) launch one block per protein; (3) time the kernel; (4) copy
//       the smoothed profiles and per-protein results back; (5) free.
//   We time ONLY the kernel (CUDA events), not the H2D/D2H copies (THEORY §6).
// ---------------------------------------------------------------------------
void scan_dataset_gpu(const Dataset& ds, int window, float threshold,
                      std::vector<AggResult>& results,
                      std::vector<float>& smoothed, float* kernel_ms) {
    const int half = (window - 1) / 2;     // window W = 2*half + 1
    results.assign(ds.num, AggResult{});
    smoothed.assign(static_cast<std::size_t>(ds.num) * ds.stride, 0.0f);

    // Guard: we tile each protein wholly in one block's shared memory, so the
    // longest sequence must fit. (Real tools split long chains across blocks --
    // left as an exercise; here we fail loudly rather than silently truncate.)
    if (ds.max_len > AGG_MAX_LEN) {
        std::fprintf(stderr,
            "[scan_dataset_gpu] longest protein has %d residues > AGG_MAX_LEN=%d; "
            "split long chains across blocks (see THEORY exercises).\n",
            ds.max_len, AGG_MAX_LEN);
        std::exit(EXIT_FAILURE);
    }

    // (1) Upload the scale (host AA_PROPENSITY -> device c_scale) once.
    CUDA_CHECK(cudaMemcpyToSymbol(c_scale, AA_PROPENSITY, sizeof(AA_PROPENSITY)));

    // (1b) Device buffers for the flat batch and the outputs.
    int       *d_codes = nullptr, *d_lengths = nullptr;
    float     *d_smoothed = nullptr;
    AggResult *d_results  = nullptr;
    const std::size_t flat_n = ds.flat_codes.size();
    CUDA_CHECK(cudaMalloc(&d_codes,    flat_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths,  ds.num * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_smoothed, flat_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_results,  ds.num * sizeof(AggResult)));

    CUDA_CHECK(cudaMemcpy(d_codes, ds.flat_codes.data(),
                          flat_n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, ds.lengths.data(),
                          ds.num * sizeof(int), cudaMemcpyHostToDevice));

    // (2)+(3) Launch one block per protein; shared mem = max_len floats (the
    //         tile is sized to the longest protein so every block has room).
    const int grid = ds.num;
    const std::size_t shmem = static_cast<std::size_t>(ds.max_len) * sizeof(float);
    GpuTimer timer;
    timer.start();
    agg_scan_kernel<<<grid, AGG_THREADS, shmem>>>(
        d_codes, d_lengths, ds.stride, half, threshold, d_smoothed, d_results);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("agg_scan_kernel");    // catch launch + execution errors

    // (4) Copy results back to the host vectors.
    CUDA_CHECK(cudaMemcpy(smoothed.data(), d_smoothed,
                          flat_n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results.data(), d_results,
                          ds.num * sizeof(AggResult), cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_smoothed));
    CUDA_CHECK(cudaFree(d_results));
}
