// ===========================================================================
// src/kernels.cu  --  Profile-HMM Viterbi search kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 2.13 : MSA Generation Acceleration
//
// WHAT THIS FILE DOES
//   The GPU twin of viterbi_search_cpu(). It scores ONE query profile HMM against
//   every database sequence by running, in parallel, one Viterbi dynamic program
//   per sequence -- ONE THREAD BLOCK PER SEQUENCE. main.cu runs this and the CPU
//   reference and asserts they agree EXACTLY (integer scores -> bit-for-bit).
//
//   The math is NOT duplicated here: the per-row recurrence lives once in
//   hmm_core.h (viterbi_step / best_in_row), included by both sides, so parity is
//   guaranteed by construction (PATTERNS.md §2). What lives HERE is purely the
//   PARALLELISATION: how the row update is spread across a block's threads, what
//   sits in shared vs constant vs global memory, and the synchronisation.
//
// THE ROW DEPENDENCY (why the parallel split is correct)
//   Within viterbi_step, for column k:
//     * M[k] and I[k] read ONLY the previous row (prevM/I/D) -> all columns are
//       independent -> THREADS COMPUTE THEM IN PARALLEL.
//     * D[k] reads curM[k-1] and curD[k-1] (the CURRENT row) -> a left-to-right
//       chain -> one thread resolves the whole D row sequentially AFTER M is done.
//   We therefore do each residue in three phases separated by __syncthreads():
//     (1) parallel M,I  (2) one-thread sequential D  (3) parallel best-in-row max.
//   Every integer op matches the CPU's viterbi_step, so the result is identical.
//
// READ THIS AFTER: kernels.cuh, hmm_core.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include "hmm_core.h"            // viterbi_step pieces are re-derived per-phase here

// Threads per block. The block cooperates on ONE sequence's DP; with L<=256
// columns, 128 threads gives each thread ~2 columns and keeps several warps
// resident to hide the shared-memory latency. A multiple of the 32-lane warp.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// CONSTANT MEMORY: the profile's emission log-odds table.
//   Read by every thread of every block, never written during the launch -> the
//   constant cache broadcasts one address to a whole warp in a single transaction
//   (the same reason 1.12 keeps its query here). Fixed size MAX_PROFILE_L*21 ints
//   (~21 KB) fits the 64 KB constant bank. Filled by cudaMemcpyToSymbol below.
//   Layout matches hmm_core.h: column k (1-based), residue a at [(k-1)*21 + a].
// ---------------------------------------------------------------------------
__constant__ int c_emit[MAX_PROFILE_L * ALPHABET_SIZE];

// ---------------------------------------------------------------------------
// viterbi_seq_kernel : one block scores one database sequence.
//
//   Launch config (set in viterbi_search_gpu):
//     grid  = N blocks (one per database sequence)
//     block = THREADS_PER_BLOCK threads (cooperate on this sequence's DP)
//   Block-to-data map: blockIdx.x = database sequence index i.
//
//   Parameters (all device pointers / scalars):
//     L            : profile length (<= MAX_PROFILE_L)
//     t_*          : the seven scaled-integer transition log-odds
//     d_res        : CSR-concatenated residues of ALL sequences (global memory)
//     d_offset     : [N+1] start offsets into d_res
//     d_length     : [N] sequence lengths
//     d_out        : [N] output best scores (one written per block)
//
//   Shared memory (block-private, fast): three pairs of ping-pong rows
//   (prev/cur for M,I,D), each length L+1, plus a small reduction scratch.
//   Sized dynamically via the launch's third <<<>>> argument.
// ---------------------------------------------------------------------------
__global__ void viterbi_seq_kernel(int L,
                                   int t_mm, int t_mi, int t_im, int t_ii,
                                   int t_md, int t_dm, int t_dd,
                                   const uint8_t* __restrict__ d_res,
                                   const int* __restrict__ d_offset,
                                   const int* __restrict__ d_length,
                                   int* __restrict__ d_out) {
    const int i  = blockIdx.x;          // this block's database sequence
    const int tx = threadIdx.x;         // this thread's lane within the block
    const int W  = L + 1;               // row width (column 0 = begin state)

    // --- Carve the dynamic shared memory into our six rows + reduction scratch.
    //     One contiguous extern array, partitioned by hand (the standard idiom).
    extern __shared__ int smem[];
    int* prevM = smem;                  // [W]
    int* prevI = prevM + W;             // [W]
    int* prevD = prevI + W;             // [W]
    int* curM  = prevD + W;             // [W]
    int* curI  = curM  + W;             // [W]
    int* curD  = curI  + W;             // [W]
    int* redux = curD  + W;             // [THREADS_PER_BLOCK] for the best-score max

    const int T   = d_length[i];        // length of this sequence
    const int base = d_offset[i];       // start of this sequence in d_res

    // --- Row -1 (before any residue): only the begin state is reachable.
    //     Every thread initialises a strided slice of the previous row so the
    //     whole length-W row is set cooperatively.
    for (int k = tx; k < W; k += blockDim.x) {
        prevM[k] = NEG_INF; prevI[k] = NEG_INF; prevD[k] = NEG_INF;
    }
    __syncthreads();
    if (tx == 0) prevM[0] = 0;          // free entry at the profile start
    __syncthreads();

    int best = NEG_INF;                 // this thread's running best match score

    // --- March the sequence one residue at a time (the rows are dependent). ---
    for (int r = 0; r < T; ++r) {
        const int a = static_cast<int>(d_res[base + r]);   // this residue's aa index

        // Begin column 0 of the current row (one thread sets it).
        if (tx == 0) { curM[0] = 0; curI[0] = NEG_INF; curD[0] = NEG_INF; }
        __syncthreads();

        // PHASE 1 (parallel): M[k] and I[k] for k=1..L depend ONLY on the
        // previous row, so every thread handles a strided set of columns with no
        // interdependence. These are exactly the M/I formulas in viterbi_step.
        for (int k = 1 + tx; k <= L; k += blockDim.x) {
            const int e = c_emit[(k - 1) * ALPHABET_SIZE + a];   // emission, constant cache
            const int inM = imax(imax(prevM[k - 1] + t_mm, prevI[k - 1] + t_im),
                                 prevD[k - 1] + t_dm);
            curM[k] = (inM <= NEG_INF / 2) ? NEG_INF : inM + e;
            const int inI = imax(prevM[k] + t_mi, prevI[k] + t_ii);
            curI[k] = (inI <= NEG_INF / 2) ? NEG_INF : inI;
        }
        __syncthreads();                // curM, curI fully written before deletes

        // PHASE 2 (sequential): the delete chain D[k] depends on curM[k-1] and
        // curD[k-1] of THIS row -> a strict left-to-right recurrence. One thread
        // resolves it; it is O(L) but cheap, and doing it serially makes the
        // result provably identical to the CPU (PATTERNS.md §3, determinism).
        if (tx == 0) {
            for (int k = 1; k <= L; ++k) {
                const int inD = imax(curM[k - 1] + t_md, curD[k - 1] + t_dd);
                curD[k] = (inD <= NEG_INF / 2) ? NEG_INF : inD;
            }
        }
        __syncthreads();                // curD ready; safe to read the row & swap

        // PHASE 3 (parallel): fold this row's best match score into each thread's
        // running max. Integer max is associative -> order-independent -> the
        // reduction is deterministic regardless of how threads interleave.
        for (int k = 1 + tx; k <= L; k += blockDim.x) best = imax(best, curM[k]);

        // Ping-pong: this row becomes the previous row for the next residue. The
        // shared-memory partition pointers are fixed, so we move the CONTENTS via
        // a cooperative cur->prev copy (W is small, L<=256, so this is cheap). An
        // optimisation would alternate two buffer sets by parity to skip the copy;
        // we copy for clarity -- the recurrence stays unmistakable (THEORY §GPU).
        for (int k = tx; k < W; k += blockDim.x) {
            prevM[k] = curM[k]; prevI[k] = curI[k]; prevD[k] = curD[k];
        }
        __syncthreads();
    }

    // --- Reduce each thread's `best` to a single per-sequence score. ---
    //     Store to scratch, then a standard tree reduction with integer max.
    redux[tx] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) redux[tx] = imax(redux[tx], redux[tx + stride]);
        __syncthreads();
    }
    if (tx == 0) d_out[i] = redux[0];   // the block writes its sequence's hit score
}

// ---------------------------------------------------------------------------
// viterbi_search_gpu : host wrapper. The canonical CUDA steps, adapted:
//   (a) upload the emission table to CONSTANT memory (a special symbol copy)
//   (b) upload the CSR-packed database (residues + offsets + lengths) to global
//   (c) launch ONE BLOCK PER SEQUENCE with dynamic shared memory sized to 6 rows
//       (+ reduction scratch); time ONLY the kernel with CUDA events
//   (d) copy the N scores back; (e) free device memory.
// ---------------------------------------------------------------------------
void viterbi_search_gpu(const SearchProblem& prob, std::vector<int>& out, float* kernel_ms) {
    const ProfileHMM& h = prob.hmm;
    const SeqDB& db = prob.db;
    const int L = h.L;
    const int N = db.N;
    const int W = L + 1;
    out.assign(static_cast<std::size_t>(N), NEG_INF);

    // Guard the teaching cap: the constant array and shared rows are sized for it.
    if (L > MAX_PROFILE_L) {
        std::fprintf(stderr, "[viterbi_search_gpu] profile L=%d exceeds MAX_PROFILE_L=%d\n",
                     L, MAX_PROFILE_L);
        std::exit(EXIT_FAILURE);
    }

    // (a) Emission table -> constant memory symbol (only L*21 of the array used).
    CUDA_CHECK(cudaMemcpyToSymbol(c_emit, h.emit.data(),
                                  static_cast<std::size_t>(L) * ALPHABET_SIZE * sizeof(int)));

    // (b) Database (CSR) -> global memory.
    uint8_t* d_res    = nullptr;   // [sum T] concatenated residues
    int*     d_offset = nullptr;   // [N+1] start offsets
    int*     d_length = nullptr;   // [N] sequence lengths
    int*     d_out    = nullptr;   // [N] output scores
    const std::size_t res_bytes = db.res.size() * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_res,    res_bytes));
    CUDA_CHECK(cudaMalloc(&d_offset, (static_cast<std::size_t>(N) + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_length, static_cast<std::size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,    static_cast<std::size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_res,    db.res.data(),    res_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, db.offset.data(),
                          (static_cast<std::size_t>(N) + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_length, db.length.data(),
                          static_cast<std::size_t>(N) * sizeof(int), cudaMemcpyHostToDevice));

    // (c) Launch: N blocks, dynamic shared = 6 rows of W ints + reduction scratch.
    const std::size_t shmem_bytes =
        (static_cast<std::size_t>(6) * W + THREADS_PER_BLOCK) * sizeof(int);
    GpuTimer timer;
    timer.start();
    viterbi_seq_kernel<<<N, THREADS_PER_BLOCK, shmem_bytes>>>(
        L, h.t_mm, h.t_mi, h.t_im, h.t_ii, h.t_md, h.t_dm, h.t_dd,
        d_res, d_offset, d_length, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("viterbi_seq_kernel");

    // (d) Scores back to host, then (e) free.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, static_cast<std::size_t>(N) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_length));
    CUDA_CHECK(cudaFree(d_out));
}
