// ===========================================================================
// src/kernels.cu  --  Profile-HMM Viterbi/Forward kernels + host wrapper
// ---------------------------------------------------------------------------
// Project 3.28 : Profile HMM (Viterbi / Forward)
//
// WHAT THIS FILE DOES
//   Implements the GPU twin of reference_cpu.cpp's score_one(): one thread runs
//   the full M/I/D dynamic program for ONE database sequence against the shared
//   profile, in log space, and writes that sequence's score. Two kernels (one
//   per algorithm) are thin wrappers around a single templated device routine so
//   the math is written once. The host wrapper phmm_search_gpu() does the CUDA
//   bookkeeping (upload, launch, time, download).
//
//   The per-cell recurrence is the SAME phmm.h code the CPU loops, so GPU and CPU
//   agree to ~machine precision (main.cu verifies this; THEORY §6).
//
// READ THIS AFTER: phmm.h (the shared recurrence), kernels.cuh (the interface).
//   Compare line-by-line against reference_cpu.cpp's score_one().
// ===========================================================================
#include "kernels.cuh"
#include "phmm.h"                // ProfileHMM, the HD recurrence core, MAX_M, MAX_L
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// THE PROFILE IN CONSTANT MEMORY.
//   The whole model (length, emission tables, transition logs) is read by every
//   thread and never modified during the launch -> constant memory is ideal: its
//   hardware cache broadcasts one address to a whole warp in a single transaction
//   instead of forcing each thread to re-read the model from global memory. The
//   image is a fixed compile-time size (sizeof(ProfileHMM) ~ 65*20*8 + ... bytes,
//   well under the 64 KB constant bank). Filled by cudaMemcpyToSymbol() below.
//   This mirrors the constant-memory query fingerprint in flagship 1.12.
// ---------------------------------------------------------------------------
__constant__ ProfileHMM c_profile;

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide latency, and many blocks resident for
// occupancy. Each thread holds 3 small DP rows (size M+1<=65 doubles each) in
// local memory, so register/local pressure -- not block size -- is the limiter
// here (THEORY §4 discusses the occupancy trade-off).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// combine() / combine3(): the algorithm-selecting reduction, as plain templated
// __device__ functions (NOT lambdas -- see the note in score_one_device). For
// Viterbi (max-sum) they are max; for Forward (sum-product in log space) they are
// the log-sum-exp from phmm.h. Both call the SAME phmm.h primitives the CPU uses.
// ---------------------------------------------------------------------------
template <bool IS_VITERBI>
__device__ inline double combine(double a, double b) {
    return IS_VITERBI ? max2(a, b) : log_sum_exp(a, b);
}
template <bool IS_VITERBI>
__device__ inline double combine3(double a, double b, double c) {
    return IS_VITERBI ? max3(a, b, c) : log_sum_exp(log_sum_exp(a, b), c);
}

// ===========================================================================
// THE DEVICE DP  (the GPU twin of reference_cpu.cpp::score_one)
// ---------------------------------------------------------------------------
// score_one_device(): fill the M/I/D lattice for ONE sequence and return its
// score. Templated on IS_VITERBI so the combine operator (max vs log-sum-exp) is
// chosen at COMPILE TIME -- no per-cell branch, and the two kernels share this
// one body. Uses FIXED-SIZE local arrays (length MAX_M+1) rolled across the
// sequence, exactly like the CPU's rolling std::vector rows but stack-allocated
// (CUDA local memory) so there is no dynamic allocation on the device.
//
// Thread-to-data map (set in the kernel): one thread <-> one sequence. The
// sequence's residues are at &res[off] for `len` residues.
// ===========================================================================
template <bool IS_VITERBI>
__device__ double score_one_device(const ProfileHMM& p,
                                    const std::uint8_t* __restrict__ seq, int L) {
    const int M = p.M;

    // combine() / combine3(): the ONE operator that differs between Viterbi and
    // Forward. We use plain static helper functions (defined just above this
    // template) rather than device lambdas -- device lambdas would need the extra
    // nvcc flag --extended-lambda, and plain functions keep the build portable
    // and the device/host code paths textually identical. IS_VITERBI is a
    // compile-time template arg, so the ternary inside is folded away (no branch).
    //
    // Rolling rows in LOCAL memory (sized to the compile-time max so they fit on
    // the stack). Only indices 0..M are used. Initialize to LOG_ZERO.
    double Mprev[MAX_M + 1], Iprev[MAX_M + 1], Dprev[MAX_M + 1];
    double Mrow[MAX_M + 1],  Irow[MAX_M + 1],  Drow[MAX_M + 1];
    for (int k = 0; k <= M; ++k) {
        Mprev[k] = Iprev[k] = Dprev[k] = LOG_ZERO;
        Mrow[k]  = Irow[k]  = Drow[k]  = LOG_ZERO;
    }

    // ---- Row i = 0 : the BEGIN boundary (no residue emitted yet) -----------
    // Enter match column 1 for free (M[0][1] = log 1 = 0), then walk a silent
    // DELETE chain across row 0 to allow skipping leading columns.
    Mprev[1] = 0.0;
    for (int k = 2; k <= M; ++k) {
        const TransLog tk1 = p.trans[k - 1];
        double from_m = Mprev[k - 1] + tk1.md;   // M_{k-1} -> D_k
        double from_d = Dprev[k - 1] + tk1.dd;   // D_{k-1} -> D_k
        Dprev[k] = combine<IS_VITERBI>(from_m, from_d);
    }

    // ---- Rows i = 1..L : emit residue x_i ----------------------------------
    for (int i = 1; i <= L; ++i) {
        const int x = seq[i - 1];   // residue code emitted at this row

        // Reset the current row to "impossible".
        for (int k = 0; k <= M; ++k) { Mrow[k] = Irow[k] = Drow[k] = LOG_ZERO; }

        for (int k = 1; k <= M; ++k) {
            // MATCH M[i][k]: from M/I/D at (i-1, k-1), plus the match emission.
            const TransLog tk1 = p.trans[k - 1];
            double in_m = Mprev[k - 1] + tk1.mm;
            double in_i = Iprev[k - 1] + tk1.im;
            double in_d = Dprev[k - 1] + tk1.dm;
            Mrow[k] = combine3<IS_VITERBI>(in_m, in_i, in_d) + emit_match(p, k, x);

            // INSERT I[i][k]: from M/I at (i-1, k), plus the insert emission.
            const TransLog tk = p.trans[k];
            double in2_m = Mprev[k] + tk.mi;
            double in2_i = Iprev[k] + tk.ii;
            Irow[k] = combine<IS_VITERBI>(in2_m, in2_i) + emit_insert(p, x);

            // DELETE D[i][k]: silent state, reads the CURRENT row at k-1 (same i),
            // so we must sweep k left-to-right (Drow[k-1] already final).
            double in3_m = Mrow[k - 1] + tk1.md;
            double in3_d = Drow[k - 1] + tk1.dd;
            Drow[k] = combine<IS_VITERBI>(in3_m, in3_d);
        }

        // Roll: copy current -> previous for the next residue. (We copy rather
        // than swap pointers because these are fixed local arrays; the compiler
        // keeps the small loop tight. Same end state as the CPU's std::swap.)
        for (int k = 0; k <= M; ++k) {
            Mprev[k] = Mrow[k];
            Iprev[k] = Irow[k];
            Dprev[k] = Drow[k];
        }
    }

    // ---- END: the alignment ends in the final match column M. --------------
    return Mprev[M];
}

// ---------------------------------------------------------------------------
// phmm_kernel: one thread per database sequence, via a grid-stride loop so a
// fixed grid covers an arbitrarily large database.
//   res        : [total residues] flat concatenated residue codes (global mem)
//   off,len    : [n] per-sequence start offset and length (global mem)
//   n          : number of database sequences
//   out        : [n] output scores
//   The profile is read from the __constant__ symbol c_profile (not a param).
//   Thread index i = blockIdx.x*blockDim.x + threadIdx.x; stride = total threads.
// ---------------------------------------------------------------------------
template <bool IS_VITERBI>
__global__ void phmm_kernel(const std::uint8_t* __restrict__ res,
                            const int* __restrict__ off,
                            const int* __restrict__ len,
                            int n, float* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;          // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const std::uint8_t* seq = res + off[i];          // this sequence's residues
        double s = score_one_device<IS_VITERBI>(c_profile, seq, len[i]);
        out[i] = static_cast<float>(s);                  // store as float (matches CPU)
    }
}

// ---------------------------------------------------------------------------
// phmm_search_gpu: the host wrapper. The five canonical CUDA steps, with the
// profile going to constant memory. We time ONLY the kernel (CUDA events), not
// the H2D/D2H copies (THEORY discusses those separately).
// ---------------------------------------------------------------------------
void phmm_search_gpu(const ProfileHMM& p, const SeqDB& db, bool is_viterbi,
                     std::vector<float>& out, float* kernel_ms) {
    const int n = db.n;
    out.assign(static_cast<std::size_t>(n), 0.0f);

    const std::size_t res_bytes = db.res.size() * sizeof(std::uint8_t);
    const std::size_t idx_bytes = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t out_bytes = static_cast<std::size_t>(n) * sizeof(float);

    // (a) Upload the profile to the __constant__ symbol. cudaMemcpyToSymbol is
    //     the special copy that targets the constant bank (not ordinary global).
    CUDA_CHECK(cudaMemcpyToSymbol(c_profile, &p, sizeof(ProfileHMM)));

    // (b) Allocate + upload the ragged database (flat residues + offsets/lengths)
    //     and allocate the output scores.
    std::uint8_t* d_res = nullptr;   // [total residues]
    int*          d_off = nullptr;   // [n]
    int*          d_len = nullptr;   // [n]
    float*        d_out = nullptr;   // [n]
    CUDA_CHECK(cudaMalloc(&d_res, res_bytes));
    CUDA_CHECK(cudaMalloc(&d_off, idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_len, idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_res, db.res.data(), res_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off, db.off.data(), idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len, db.len.data(), idx_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-sequence, capped so the
    //     grid stays modest; the grid-stride loop handles any remainder. We
    //     dispatch the Viterbi or Forward instantiation by the runtime flag.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;
    GpuTimer timer;
    timer.start();
    if (is_viterbi)
        phmm_kernel<true><<<blocks, THREADS_PER_BLOCK>>>(d_res, d_off, d_len, n, d_out);
    else
        phmm_kernel<false><<<blocks, THREADS_PER_BLOCK>>>(d_res, d_off, d_len, n, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST(is_viterbi ? "phmm_kernel<viterbi>" : "phmm_kernel<forward>");

    // (d) Copy scores back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_len));
    CUDA_CHECK(cudaFree(d_out));
}
