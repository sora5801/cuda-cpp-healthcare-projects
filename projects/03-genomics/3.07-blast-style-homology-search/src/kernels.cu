// ===========================================================================
// src/kernels.cu  --  BLAST seed-extend kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.7 : BLAST-Style Homology Search
//
// This is the GPU twin of blast_cpu() in reference_cpu.cpp. main.cu runs both
// and asserts they agree EXACTLY (all-integer scoring). The per-residue scoring
// (gapless X-drop) is shared verbatim via blast_core.h, so the only thing that
// differs between CPU and GPU is the PARALLELISM, not the math.
//
// See ../THEORY.md sec "GPU mapping" for the occupancy and memory reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "blast_core.h"          // gapless_xdrop, SeqView, N_ALPHA (shared HD core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <algorithm>             // std::sort
#include <vector>

// ---------------------------------------------------------------------------
// BLOSUM62 in CONSTANT memory.
//   * Every thread reads the substitution matrix for every scored residue pair,
//     but NONE writes it and it is identical for the whole launch -> constant
//     memory is the ideal home: its hardware cache broadcasts one address to a
//     whole warp in a single transaction, instead of a global load per access.
//   * Size is fixed at compile time (24*24 = 576 bytes), trivially inside the
//     64 KB constant bank. Filled by cudaMemcpyToSymbol() in blast_gpu().
//   * The device kernel passes a pointer to this symbol into gapless_xdrop(),
//     which expects a flat 24x24 int8 matrix (blast_core.h blosum_at) -- the
//     SAME function the CPU calls with the host matrix, hence identical scores.
// ---------------------------------------------------------------------------
__constant__ int8_t c_blosum[N_ALPHA * N_ALPHA];

// 128 threads/block: a multiple of the 32-lane warp. DB sequences vary in length
// so threads in a warp can diverge in their inner loops; a moderate block size
// keeps occupancy healthy on sm_75..sm_89 without over-subscribing registers
// (each thread holds a few scalars). See THEORY "GPU mapping" for the reasoning.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// device_lower_bound: find the first SeedPair whose code >= `target`.
//   Classic binary search over the SORTED seed array (sorted by code in
//   flatten_query_index). Returns an index in [0, n_seeds]. The kernel uses this
//   to locate a DB k-mer's code; if found, all query positions with that code
//   are the contiguous run starting here (because equal codes are adjacent).
//   O(log n_seeds) per DB k-mer -- this is the GPU stand-in for BLAST's hash
//   lookup, written explicitly so there is no black box (CLAUDE.md sec 6).
// ---------------------------------------------------------------------------
__device__ inline int device_lower_bound(const SeedPair* seeds, int n_seeds, int target) {
    int lo = 0, hi = n_seeds;              // search the half-open range [lo, hi)
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);   // avoid overflow vs (lo+hi)/2
        if (seeds[mid].code < target) lo = mid + 1;   // target is to the right
        else                          hi = mid;       // mid is a candidate; go left
    }
    return lo;                              // first index with code >= target
}

// ---------------------------------------------------------------------------
// blast_kernel: one logical thread per DB sequence, via a grid-stride loop so a
// fixed-size grid covers an arbitrarily large database.
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count until i >= n. Each iteration handles ONE
//   DB sequence end to end -- exactly the body of blast_cpu()'s i-loop.
//   Memory: c_blosum from constant cache; query, DB residues, seed array, and
//   offsets from global memory. No shared memory or atomics: outputs (one int
//   per DB sequence) are fully independent, so there is nothing to coordinate.
// ---------------------------------------------------------------------------
__global__ void blast_kernel(const int8_t* __restrict__ d_query, int query_len,
                             const int8_t* __restrict__ d_db_res,
                             const int*    __restrict__ d_db_off,
                             const int*    __restrict__ d_db_len,
                             const SeedPair* __restrict__ d_seeds, int n_seeds,
                             int n, int* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;          // total threads in grid
    const SeqView q{ d_query, query_len };              // the shared query view

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // This thread's DB sequence: a window into the concatenated buffer.
        const SeqView d{ d_db_res + d_db_off[i], d_db_len[i] };
        int best = 0;                                   // best HSP score so far

        // Slide the length-k window across this DB sequence (== blast_cpu inner loop).
        for (int dpos = 0; dpos + SEED_K <= d.len; ++dpos) {
            int code = pack_kmer(d.data, d.len, dpos, SEED_K);   // base-24 code
            if (code < 0) continue;                              // ambiguous -> skip

            // Binary-search the sorted seed array for this code, then walk the
            // contiguous run of equal codes -- each gives a query position (seed).
            int s = device_lower_bound(d_seeds, n_seeds, code);
            for (; s < n_seeds && d_seeds[s].code == code; ++s) {
                int qpos = d_seeds[s].qpos;
                // The SAME extension the CPU runs (blast_core.h), reading BLOSUM
                // from constant memory -> identical integer HSP score.
                int hsp = gapless_xdrop(q, d, qpos, dpos, SEED_K, c_blosum, X_DROP);
                if (hsp > best) best = hsp;
            }
        }
        out[i] = best;
    }
}

// ---------------------------------------------------------------------------
// flatten_query_index (host): hash map -> sorted SeedPair array for the device.
//   We copy every (code, qpos) into a flat vector and sort it by (code, qpos).
//   Sorting by code makes the device binary search valid; the secondary qpos
//   sort makes the run order deterministic (so the kernel visits seeds in a
//   fixed order -- not that it matters for the MAX, but determinism is a virtue).
// ---------------------------------------------------------------------------
std::vector<SeedPair> flatten_query_index(const QueryIndex& qi) {
    std::vector<SeedPair> flat;
    for (const auto& kv : qi.table)
        for (int qpos : kv.second)
            flat.push_back(SeedPair{ kv.first, qpos });
    std::sort(flat.begin(), flat.end(), [](const SeedPair& a, const SeedPair& b) {
        if (a.code != b.code) return a.code < b.code;   // primary: by code
        return a.qpos < b.qpos;                         // secondary: by position
    });
    return flat;
}

// ---------------------------------------------------------------------------
// blast_gpu: the canonical CUDA steps, with BLOSUM62 going to constant memory.
// We time ONLY the kernel (CUDA events), not the H2D/D2H copies (discussed
// separately in THEORY). The five steps: upload constants, allocate+upload
// inputs, launch, copy result back, free.
// ---------------------------------------------------------------------------
void blast_gpu(const SequenceDB& db, const QueryIndex& query_idx,
               std::vector<int>& out, float* kernel_ms) {
    const int n = db.n;
    out.assign(static_cast<std::size_t>(n), 0);

    // Flatten the query index into the device-friendly sorted seed array.
    std::vector<SeedPair> seeds = flatten_query_index(query_idx);
    const int n_seeds = static_cast<int>(seeds.size());

    // (a) Upload BLOSUM62 to the __constant__ symbol (special copy targeting the
    //     constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_blosum, blosum62(),
                                  N_ALPHA * N_ALPHA * sizeof(int8_t)));

    // (b) Allocate + upload the inputs and allocate the output.
    int8_t*   d_query  = nullptr;   // [query_len] encoded query residues
    int8_t*   d_db_res = nullptr;   // [total residues] concatenated DB
    int*      d_db_off = nullptr;   // [n] per-sequence start offsets
    int*      d_db_len = nullptr;   // [n] per-sequence lengths
    SeedPair* d_seeds  = nullptr;   // [n_seeds] sorted query k-mer index
    int*      d_out    = nullptr;   // [n] best HSP score per DB sequence

    const std::size_t q_bytes   = db.query.size()  * sizeof(int8_t);
    const std::size_t res_bytes = db.db_res.size() * sizeof(int8_t);
    const std::size_t off_bytes = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t seed_bytes= static_cast<std::size_t>(n_seeds) * sizeof(SeedPair);
    const std::size_t out_bytes = static_cast<std::size_t>(n) * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_query,  q_bytes));
    CUDA_CHECK(cudaMalloc(&d_db_res, res_bytes));
    CUDA_CHECK(cudaMalloc(&d_db_off, off_bytes));
    CUDA_CHECK(cudaMalloc(&d_db_len, off_bytes));
    CUDA_CHECK(cudaMalloc(&d_seeds,  seed_bytes ? seed_bytes : 1));   // avoid 0-byte malloc
    CUDA_CHECK(cudaMalloc(&d_out,    out_bytes));

    CUDA_CHECK(cudaMemcpy(d_query,  db.query.data(),  q_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_db_res, db.db_res.data(), res_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_db_off, db.db_off.data(), off_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_db_len, db.db_len.data(), off_bytes, cudaMemcpyHostToDevice));
    if (seed_bytes)
        CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data(), seed_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-sequence, capped so the
    //     grid stays modest; the grid-stride loop handles any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1)    blocks = 1;
    if (blocks > 1024) blocks = 1024;
    GpuTimer timer;
    timer.start();
    blast_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_query, static_cast<int>(db.query.size()),
        d_db_res, d_db_off, d_db_len,
        d_seeds, n_seeds, n, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("blast_kernel");

    // (d) Copy scores back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_db_res));
    CUDA_CHECK(cudaFree(d_db_off));
    CUDA_CHECK(cudaFree(d_db_len));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_out));
}
