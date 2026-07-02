// ===========================================================================
// src/kernels.cu  --  The GPU side: one transformer self-attention encoder block
// ---------------------------------------------------------------------------
// Project 7.3 : Clinical NLP over Notes & Records
//
// WHAT THIS FILE DOES (the GPU twin of reference_cpu.cpp's attention_reference)
//   gpu_attention runs the four stages of a self-attention block on the device:
//     1. build_projections_kernel + gather_kernel : make the shared Wq/Wk/Wv and
//        gather each note's token embeddings X.
//     2. cuBLAS DGEMM x3            : Q = X Wq, K = X Wk, V = X Wv.
//     3. cublasDgemmStridedBatched : scores = Q_h K_hᵀ / sqrt(dh), all B*H heads.
//     4. softmax_kernel            : mask PADs + stable softmax per query row.
//     5. cublasDgemmStridedBatched : O_h = A_h V_h, all B*H heads.
//   Every per-element formula (projection weights, softmax exp/max) is the SHARED
//   one in attn_core.h, so the GPU and CPU produce (near) identical numbers --
//   the whole point of the verification in main.cu.
//
// LIBRARY, NOT BLACK BOX (CLAUDE.md §6.1.6):
//   cuBLAS DGEMM computes C = alpha·op(A)·op(B) + beta·C for double matrices, and
//   is COLUMN-MAJOR (Fortran order). Our arrays are ROW-MAJOR. The one identity
//   we lean on throughout: a row-major [m x n] buffer, read as column-major, IS
//   the [n x m] transpose. So to get a row-major C[m x n] = A[m x k]·B[k x n] we
//   ask cuBLAS for the column-major product Cᵀ = Bᵀ·Aᵀ, i.e. swap A<->B and
//   m<->n and use op = N for both. Each call site below states the exact mapping.
//   Hand-rolling a competitive DGEMM means shared-memory tiling + register
//   blocking + bank-conflict-free loads; cuBLAS already does all of that, and on
//   real hardware dispatches to tensor cores -- the acceleration the deep-dive
//   names.
//
// READ THIS AFTER: kernels.cuh, attn_core.h, util/cuda_check.cuh, util/timer.cuh.
// Compare every stage with its serial twin in reference_cpu.cpp.
// ===========================================================================
#include "kernels.cuh"
#include "attn_core.h"
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer (CUDA-event timing)

#include <cublas_v2.h>            // cublasDgemm, cublasDgemmStridedBatched
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// cuBLAS has its own status enum; guard + explain every call (no black box).
// A failed BLAS call means the attention output is garbage, so we abort loudly
// rather than silently return wrong numbers.
#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "[CUBLAS_CHECK] %s:%d -> status %d\n",        \
                         __FILE__, __LINE__, static_cast<int>(st__));          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (0)

// Threads per block for the 1D/2D helper kernels. 256 is a solid default on
// sm_75..sm_89: a multiple of the 32-lane warp, 8 warps to hide latency.
static constexpr int THREADS_PER_BLOCK = 256;

// ===========================================================================
// KERNEL 1 -- build_projections_kernel: fill Wq/Wk/Wv from the shared recipe
// ---------------------------------------------------------------------------
// The three [D x D] projection matrices are FABRICATED weights (attn_core.h
// explains why we fake rather than train). One thread per matrix entry evaluates
// attn::proj_entry -- the SAME function the CPU reference calls in
// build_projection, so both sides get bit-identical weights.
//   grid/block : 1D over the D*D*3 entries (3 matrices packed back to back).
//   thread map : linear index -> (kind, i, j) via divide/mod.
//   memory     : pure compute + one global write each; no shared/atomics.
// ===========================================================================
__global__ void build_projections_kernel(int D, double* __restrict__ W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // entry across all 3 mats
    int total = 3 * D * D;
    if (idx >= total) return;
    int kind = idx / (D * D);          // 0=Wq, 1=Wk, 2=Wv
    int rem  = idx % (D * D);
    int i    = rem / D;                // row (input dim)
    int j    = rem % D;                // col (output dim)
    W[idx] = attn::proj_entry(i, j, kind);   // shared recipe -> matches CPU
}

// ===========================================================================
// KERNEL 2 -- gather_kernel: build X [(B*S) x D] from the embedding table
// ---------------------------------------------------------------------------
// For every (note b, position s) we copy that token's D-dim embedding row into
// X. This is the "gather" pattern (PATTERNS.md §1): each output row is an
// indirect load indexed by the token id. PAD positions gather the [PAD] row;
// they are harmless because the softmax mask (kernel 4) zeroes their influence.
//   grid/block : 1D over the B*S*D output entries.
//   thread map : linear index -> (row = b*S+s, d). token id = ids[row].
//   memory     : reads ids[row] + embed[token*D + d]; writes X[row*D + d].
// ===========================================================================
__global__ void gather_kernel(const int* __restrict__ ids,      // [B*S] token ids
                              const double* __restrict__ embed,  // [V*D] table
                              int rows, int D,
                              double* __restrict__ X) {          // [rows*D] out
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * D;
    if (idx >= total) return;
    int row = idx / D;                 // which (note,position)
    int d   = idx % D;                 // which embedding dimension
    int tok = ids[row];                // the token id at that slot
    X[idx] = embed[static_cast<std::size_t>(tok) * D + d];
}

// ===========================================================================
// KERNEL 3 -- scale_and_mask_kernel: scale scores by 1/sqrt(dh) and mask PADs
// ---------------------------------------------------------------------------
// cuBLAS gives us the raw dot products Q_h·K_hᵀ. Before softmax we must (a)
// multiply by 1/sqrt(dh) and (b) set any score whose KEY is a [PAD] token to a
// large negative sentinel so it gets ~0 probability. We fold both into one
// element-wise kernel over the [B*H*S*S] score tensor.
//   score layout : ((b*H + h)*S + qi)*S + kj  (row-major, matches AttnResult)
//   thread map   : one thread per score entry; derive (b, kj) to read the key id.
//   NOTE: we scale FIRST then overwrite masked entries, so the sentinel is exact
//   (-1e30, identical to the CPU) rather than a scaled sentinel.
// ===========================================================================
__global__ void scale_and_mask_kernel(double* __restrict__ scores,  // [B*H*S*S]
                                      const int* __restrict__ ids,   // [B*S] tokens
                                      int B, int H, int S, double scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(B) * H * S * S;
    if (idx >= total) return;
    int kj = idx % S;                          // key position
    int tmp = idx / S;                         // -> (b*H + h)*S + qi
    // int qi = tmp % S;                        (not needed here)
    tmp /= S;                                  // -> b*H + h
    int b  = tmp / H;                          // note index
    // Which token sits at key position kj of note b?
    int ktok = ids[static_cast<std::size_t>(b) * S + kj];
    if (ktok == attn::TOK_PAD) {
        scores[idx] = -1.0e30;                 // mask: exact same sentinel as CPU
    } else {
        scores[idx] *= scale;                  // 1/sqrt(dh) scaling
    }
}

// ===========================================================================
// KERNEL 4 -- softmax_kernel: stable per-row softmax over the S keys
// ---------------------------------------------------------------------------
// This is the ONE stage that is not a GEMM. Each attention "row" is the length-S
// score vector of one (note, head, query) triple; softmax turns it into a
// probability distribution over keys. We give ONE BLOCK to each row and let its
// threads cooperate through shared memory to (1) find the row max, (2) sum the
// stabilized exponentials, then every thread writes its normalized probability.
//
//   grid  : B*H*S blocks (one per row).
//   block : THREADS_PER_BLOCK threads; each handles keys kj = tid, tid+blk, ...
//   shared: two reductions (max, then sum) via a tree in shared memory.
//   numerics: uses attn::softmax_exp with the row max, identical to the CPU.
//
//   S here is tiny (the sample seq len), so a single block per row is simple and
//   plenty fast; a production kernel would fuse this with the score GEMM
//   (that is literally what Flash Attention does -- see THEORY §real-world).
// ===========================================================================
__global__ void softmax_kernel(double* __restrict__ scores, int rows, int S) {
    int row = blockIdx.x;                      // this block owns one row
    if (row >= rows) return;
    double* r = scores + static_cast<std::size_t>(row) * S;   // this row's S scores

    extern __shared__ double sh[];             // blockDim.x doubles of scratch
    int tid = threadIdx.x;

    // ---- (1) row maximum (parallel reduction) -----------------------------
    double local_max = -1.0e300;
    for (int k = tid; k < S; k += blockDim.x)
        if (r[k] > local_max) local_max = r[k];
    sh[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && sh[tid + stride] > sh[tid]) sh[tid] = sh[tid + stride];
        __syncthreads();
    }
    double m = sh[0];                          // the row max, broadcast via shared
    __syncthreads();

    // ---- (2) sum of stabilized exponentials -------------------------------
    // Overwrite each score with exp(score - m) (same buffer, saves memory), and
    // accumulate the denominator. attn::softmax_exp keeps CPU/GPU rounding equal.
    double local_sum = 0.0;
    for (int k = tid; k < S; k += blockDim.x) {
        double e = attn::softmax_exp(r[k], m);
        r[k] = e;
        local_sum += e;
    }
    sh[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    double denom = sh[0];
    __syncthreads();

    // ---- (3) normalize -> probabilities ------------------------------------
    for (int k = tid; k < S; k += blockDim.x)
        r[k] = r[k] / denom;
}

// ---------------------------------------------------------------------------
// dgemm_rowmajor: row-major C[m x n] = A[m x k] · B[k x n] via one cuBLAS DGEMM.
//   Using the transpose identity (see file header): we compute the column-major
//   Cᵀ = Bᵀ·Aᵀ by swapping the operands. Concretely, cublasDgemm(N, N, n, m, k,
//   B, ldb=n, A, lda=k, C, ldc=n). alpha/beta are the usual scalars.
//   Factored out so the three projection matmuls share one clearly-explained
//   call instead of three copies of the tricky argument order.
// ---------------------------------------------------------------------------
static void dgemm_rowmajor(cublasHandle_t h, int m, int n, int k,
                           double alpha, const double* A, const double* B,
                           double beta, double* C) {
    CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,          // cuBLAS m',n',k' = n, m, k
                             &alpha,
                             B, n,             // "A" arg = our B, ld = n
                             A, k,             // "B" arg = our A, ld = k
                             &beta,
                             C, n));           // C, ld = n
}

// ===========================================================================
// HOST WRAPPER -- gpu_attention
// ===========================================================================
void gpu_attention(const NoteBatch& nb, AttnResult& res, GpuAttnTimings* t) {
    const int B = nb.B, S = nb.S, D = nb.D, H = nb.H, dh = nb.dh();
    const int rows = B * S;                    // total tokens in the batch
    res.allocate(B, H, S, D);

    // ---- device buffers ---------------------------------------------------
    int*    d_ids   = nullptr;                 // [B*S] token ids
    double* d_embed = nullptr;                 // [V*D] embedding table
    double* d_W     = nullptr;                 // [3*D*D] Wq|Wk|Wv packed
    double* d_X     = nullptr;                 // [(B*S)*D] gathered embeddings
    double* d_Q     = nullptr;                 // [(B*S)*D] projected queries
    double* d_K     = nullptr;                 // [(B*S)*D] projected keys
    double* d_V     = nullptr;                 // [(B*S)*D] projected values
    double* d_scores= nullptr;                 // [B*H*S*S] attention scores/probs
    double* d_out   = nullptr;                 // [(B*S)*D] context output O
    const std::size_t rowsD = static_cast<std::size_t>(rows) * D;
    const std::size_t scoresN = static_cast<std::size_t>(B) * H * S * S;

    CUDA_CHECK(cudaMalloc(&d_ids,   static_cast<std::size_t>(B) * S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_embed, static_cast<std::size_t>(nb.V) * D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W,     static_cast<std::size_t>(3) * D * D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X,     rowsD * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q,     rowsD * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_K,     rowsD * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V,     rowsD * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scores, scoresN * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out,   rowsD * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_ids, nb.token_ids.data(),
                          static_cast<std::size_t>(B) * S * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_embed, nb.embed.data(),
                          static_cast<std::size_t>(nb.V) * D * sizeof(double),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ---- build Wq/Wk/Wv on device (shared recipe) -------------------------
    {
        int total = 3 * D * D;
        int grid = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        build_projections_kernel<<<grid, THREADS_PER_BLOCK>>>(D, d_W);
        CUDA_CHECK_LAST("build_projections_kernel");
    }
    double* d_Wq = d_W;                         // [D*D] slices of the packed buffer
    double* d_Wk = d_W + static_cast<std::size_t>(D) * D;
    double* d_Wv = d_W + static_cast<std::size_t>(2) * D * D;

    // ---- gather X [(B*S) x D] from the embedding table --------------------
    {
        int total = rows * D;
        int grid = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gather_kernel<<<grid, THREADS_PER_BLOCK>>>(d_ids, d_embed, rows, D, d_X);
        CUDA_CHECK_LAST("gather_kernel");
    }

    // ---- STAGE 2: projections Q = X Wq, K = X Wk, V = X Wv (timed) ---------
    // Each is a row-major [(B*S) x D] = [(B*S) x D] · [D x D] matmul. dgemm_rowmajor
    // hides the column-major transpose trick (see its comment). One DGEMM each.
    {
        GpuTimer tm; tm.start();
        dgemm_rowmajor(handle, rows, D, D, 1.0, d_X, d_Wq, 0.0, d_Q);
        dgemm_rowmajor(handle, rows, D, D, 1.0, d_X, d_Wk, 0.0, d_K);
        dgemm_rowmajor(handle, rows, D, D, 1.0, d_X, d_Wv, 0.0, d_V);
        CUDA_CHECK(cudaDeviceSynchronize());
        t->proj_ms = tm.stop_ms();
    }

    // ---- STAGE 3: batched scores = Q_h K_hᵀ (timed) -----------------------
    // We want, for every (note b, head h): scores_bh[S x S] = Q_h[S x dh] · K_hᵀ.
    // Q and K are [(B*S) x D] row-major; head h occupies columns [h*dh, (h+1)*dh).
    // The batch index is g = b*H + h (0..B*H-1). For batch g:
    //   * Q_h(g) starts at row (b*S) col (h*dh)  -> offset (b*S)*D + h*dh in d_Q,
    //     and successive ROWS are D apart -> as a column-major [dh x S] matrix
    //     (the transpose of row-major [S x dh]) its leading dim is D.
    //   * Likewise K_h(g).
    // Row-major target scores[S x S] = Q_h · K_hᵀ. In cuBLAS column-major terms,
    // scoresᵀ (== scores by symmetry of shape, not value) we get with:
    //   op(A)=T on K_h-view, op(B)=N on Q_h-view  -> see the exact args below.
    // The clean way: treat Q_h,K_h as row-major [S x dh]; then
    //   scores[S x S] = Q_h · K_hᵀ  ==(transpose id)==  compute column-major
    //   scoresᵀ = (K_hᵀ)ᵀ · ... ; concretely cuBLAS(N, T, S, S, dh, ...).
    // We derive it as: Cc[S x S] = op(Kview)·op(Qview) with Kview,Qview being the
    // row-major [S x dh] blocks read column-major as [dh x S].
    {
        const double alpha = 1.0, beta = 0.0;
        // Strides between consecutive BATCHES (b,h). Moving h by 1 advances the
        // column offset by dh (contiguous within a row); moving b by 1 advances
        // by a full note (S rows). We iterate g = b*H + h, so the per-batch
        // stride is not constant across the b/h boundary in general -- BUT here
        // batch g and g+1 differ by exactly dh columns when staying in the same
        // note, and by (S*D - (H-1)*dh) at a note boundary. A single strided
        // call needs a CONSTANT stride, so we choose the layout to make it so:
        // heads are contiguous in columns and notes are S rows apart, giving a
        // constant stride of dh only if H divides evenly with no row wrap. To
        // keep the teaching code simple and provably correct we instead loop
        // over notes and issue ONE strided-batched call PER NOTE over its H
        // heads (constant stride dh), B calls total. Still O(1) Python-free host
        // work and all heads of a note run in one launch.
        GpuTimer tm; tm.start();
        for (int b = 0; b < B; ++b) {
            const double* Qb = d_Q + static_cast<std::size_t>(b) * S * D;  // note b
            const double* Kb = d_K + static_cast<std::size_t>(b) * S * D;
            double* Cb = d_scores + static_cast<std::size_t>(b) * H * S * S;
            // For head h: Q_h/K_h are row-major [S x dh] with leading dim D
            // (rows D apart), column offset h*dh. Read column-major they are
            // [dh x S] with lda = D. We want row-major scores[S x S] = Q_h K_hᵀ.
            // Column-major identity: scores_colmajor[S x S] (= our row-major
            // scoresᵀ, but scores is what we store row-major) is obtained with
            //   C(colmaj) = op(A)·op(B), A=Q_h-view[dh x S], B=K_h-view[dh x S],
            //   opA = T (-> [S x dh]), opB = N (-> [dh x S]) => [S x S] = Q_hᵀᵀ...
            // Verified against the CPU reference numerically; the arg order below
            // yields row-major scores[qi][kj] = Σ_c Q[qi,c] K[kj,c].
            CUBLAS_CHECK(cublasDgemmStridedBatched(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                S, S, dh,
                &alpha,
                Kb, D, static_cast<long long>(dh),   // A = K_h view, lda=D, stride dh
                Qb, D, static_cast<long long>(dh),   // B = Q_h view, ldb=D, stride dh
                &beta,
                Cb, S, static_cast<long long>(S) * S,// C = scores, ldc=S, stride S*S
                H));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t->score_ms = tm.stop_ms();
    }

    // ---- STAGE 4: scale + mask, then stable softmax (timed) ---------------
    {
        GpuTimer tm; tm.start();
        long long total = scoresN;
        int grid = static_cast<int>((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        scale_and_mask_kernel<<<grid, THREADS_PER_BLOCK>>>(
            d_scores, d_ids, B, H, S, attn::attn_scale(dh));
        CUDA_CHECK_LAST("scale_and_mask_kernel");

        int rows_sm = B * H * S;                // one block per softmax row
        std::size_t shmem = static_cast<std::size_t>(THREADS_PER_BLOCK) * sizeof(double);
        softmax_kernel<<<rows_sm, THREADS_PER_BLOCK, shmem>>>(d_scores, rows_sm, S);
        CUDA_CHECK_LAST("softmax_kernel");
        t->soft_ms = tm.stop_ms();
    }

    // ---- STAGE 5: batched context O_h = A_h V_h (timed) -------------------
    // A_h is [S x S] (row-major, per note-head), V_h is [S x dh] (columns
    // [h*dh,(h+1)*dh) of d_V, leading dim D). Output O_h is [S x dh] written into
    // the same head-columns of d_out. We again loop per note, one strided-batched
    // call over the note's H heads (constant column stride dh).
    {
        const double alpha = 1.0, beta = 0.0;
        GpuTimer tm; tm.start();
        for (int b = 0; b < B; ++b) {
            const double* Ab = d_scores + static_cast<std::size_t>(b) * H * S * S;
            const double* Vb = d_V + static_cast<std::size_t>(b) * S * D;
            double* Ob = d_out + static_cast<std::size_t>(b) * S * D;
            // Row-major O_h[S x dh] = A_h[S x S] · V_h[S x dh].
            // Column-major identity: Oc = op(V-view)·op(A-view) with V-view read
            // as [dh x S] (lda=D) and A-view read as [S x S] (lda=S); opV=N,
            // opA=N gives Oc[dh x S] = V_hᵀ·A_hᵀ = (A_h V_h)ᵀ, i.e. exactly our
            // row-major O_h. Strides: A batch = S*S, V batch = dh (head columns),
            // O batch = dh.
            CUBLAS_CHECK(cublasDgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dh, S, S,
                &alpha,
                Vb, D, static_cast<long long>(dh),    // A = V_h view, lda=D, stride dh
                Ab, S, static_cast<long long>(S) * S, // B = A_h,      ldb=S, stride S*S
                &beta,
                Ob, D, static_cast<long long>(dh),    // C = O_h,      ldc=D, stride dh
                H));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t->ctx_ms = tm.stop_ms();
    }
    t->total_ms = t->proj_ms + t->score_ms + t->soft_ms + t->ctx_ms;

    // ---- copy results back ------------------------------------------------
    CUDA_CHECK(cudaMemcpy(res.out.data(), d_out, rowsD * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.weights.data(), d_scores, scoresN * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
    cudaFree(d_ids); cudaFree(d_embed); cudaFree(d_W); cudaFree(d_X);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_scores); cudaFree(d_out);
}
