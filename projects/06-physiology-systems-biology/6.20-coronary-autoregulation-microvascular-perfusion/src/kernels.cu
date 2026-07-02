// ===========================================================================
// src/kernels.cu  --  GPU coronary perfusion solve: CSR-SpMV Conjugate Gradient
// ---------------------------------------------------------------------------
// Project 6.20 : Coronary Autoregulation & Microvascular Perfusion
//
// WHAT THIS FILE DOES
//   Implements solve_gpu() (declared in kernels.cuh): the same autoregulation
//   outer loop as the CPU reference, but each sparse SPD solve L p = b runs on
//   the device via Conjugate Gradient whose inner SpMV is a hand-rolled CSR
//   kernel (one thread per network node/row). This is the "iterative CG + sparse
//   SpMV" GPU pattern from PATTERNS.md §1, and it is exactly what the catalog's
//   cuSPARSE-SpMV suggestion computes -- we hand-roll it so nothing is a black
//   box (THEORY §real-world shows the equivalent cuSPARSE calls).
//
// THE OPERATOR (identical MATH to reference_cpu.cpp's spmv, different STORAGE)
//   We build, on the host, an explicit CSR for the boundary-eliminated
//   graph-Laplacian: row = node, diagonal = sum of incident conductances,
//   off-diagonal = -G to each INTERIOR neighbor (fixed neighbors are eliminated
//   into b, keeping L symmetric SPD). Fixed rows are the identity. Building CSR
//   on the host once per solve is cheap (O(nnz)); the expensive, repeated work
//   (SpMV + dots + axpys, done every CG iteration) is on the GPU.
//
// DETERMINISM (PATTERNS.md §3)
//   Dot-products use a two-stage reduction: each block writes one partial sum,
//   then a single-block kernel sums those partials in a FIXED index order. Same
//   order every run -> byte-identical CG trajectory -> byte-identical stdout.
//   (Cross-CPU/GPU we still allow a small tolerance because the CSR row order
//   sums a node's incident edges differently than the CPU's edge loop; see
//   main.cu's TOLERANCE and THEORY §numerics.)
//
// READ THIS AFTER: reference_cpu.cpp (the serial twin), coronary.h (physics).
// ===========================================================================
#include "kernels.cuh"
#include "coronary.h"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Threads per block for all 1-D kernels. 256 is a solid occupancy default on
// sm_75..sm_89 and a power of two (simplifies the shared-memory reduction).
static constexpr int TPB = 256;

// ===========================================================================
// SECTION 1 -- Elementwise + reduction kernels (the CG building blocks)
// ===========================================================================

// --- 1a. CSR SpMV:  y = L x  ------------------------------------------------
// One THREAD PER ROW (node) i. The thread walks its CSR row [row_ptr[i],
// row_ptr[i+1]) and accumulates the dot of that sparse row with x. This is the
// canonical CSR-SpMV thread mapping (and what cuSPARSE does internally).
//   row_ptr : [N+1] CSR row offsets
//   col_idx : [nnz] column (neighbor node) of each stored nonzero
//   val     : [nnz] the matrix value at that position
//   Memory: global reads of col_idx/val are coalesced within a row's run; x is
//   gathered (indexed by col_idx) -- the classic SpMV access pattern.
__global__ void csr_spmv(int N, const int* __restrict__ row_ptr,
                         const int* __restrict__ col_idx,
                         const double* __restrict__ val,
                         const double* __restrict__ x,
                         double* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's row (node)
    if (i >= N) return;                              // guard the ragged last block
    double acc = 0.0;                                // running row dot-product
    const int beg = row_ptr[i], end = row_ptr[i + 1];
    for (int k = beg; k < end; ++k)                  // walk this row's nonzeros
        acc += val[k] * x[col_idx[k]];               // val * x[neighbor]
    y[i] = acc;
}

// --- 1b. Block dot-product:  partial[b] = sum over this block of a[i]*b[i] ---
// Each block reduces TPB products in shared memory (tree reduction) and writes
// ONE partial sum. A later single-block kernel sums the partials deterministically.
__global__ void dot_partial(int N, const double* __restrict__ a,
                            const double* __restrict__ b,
                            double* __restrict__ partial) {
    __shared__ double s[TPB];                        // one slot per thread
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    s[tid] = (i < N) ? a[i] * b[i] : 0.0;            // load & multiply (0 pad tail)
    __syncthreads();
    // Tree reduction: halve the active range each step. Deterministic pairing.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];        // block's contribution
}

// --- 1c. Final reduction of the per-block partials (single block) -----------
// Sums `n` partials in a FIXED sequential order into out[0]. Using one block
// with a fixed loop makes the total deterministic regardless of how many blocks
// produced partials.
__global__ void reduce_final(int n, const double* __restrict__ partial,
                             double* __restrict__ out) {
    __shared__ double s[TPB];
    int tid = threadIdx.x;
    double acc = 0.0;
    // Within ONE block: thread tid sums partial[tid], partial[tid+TPB], ... in
    // increasing index order, then the block tree-reduces those.
    for (int k = tid; k < n; k += blockDim.x) acc += partial[k];
    s[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = s[0];
}

// --- 1d. AXPY family (the vector updates inside CG) -------------------------
// y = a*x + y (classic AXPY). `a` is passed by POINTER because the scalar lives
// in device memory (produced by a reduction) -- avoids a device->host copy on
// the hot path.
__global__ void axpy(int N, const double* __restrict__ a,
                     const double* __restrict__ x, double* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] += (*a) * x[i];
}

// y = y - a*x  (used for the residual update r -= alpha*Lp).
__global__ void axmy(int N, const double* __restrict__ a,
                     const double* __restrict__ x, double* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] -= (*a) * x[i];
}

// p = r + beta*p  (the CG search-direction update). beta is a device scalar.
__global__ void xpby(int N, const double* __restrict__ beta,
                     const double* __restrict__ r, double* __restrict__ p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) p[i] = r[i] + (*beta) * p[i];
}

// --- 1e. Tiny scalar kernels (compute CG's alpha/beta on-device) ------------
// One thread computes alpha = rr / pLp. Guard pLp==0 (a degenerate direction).
__global__ void calc_alpha(const double* rr, const double* pLp, double* alpha) {
    *alpha = (*pLp != 0.0) ? (*rr) / (*pLp) : 0.0;
}
// beta = rr_new / rr_old.
__global__ void calc_beta(const double* rr_new, const double* rr_old, double* beta) {
    *beta = (*rr_old != 0.0) ? (*rr_new) / (*rr_old) : 0.0;
}

// r = b - L x  (residual initialization before the CG loop).
__global__ void residual_init(int N, const double* __restrict__ b,
                              const double* __restrict__ Lx,
                              double* __restrict__ r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) r[i] = b[i] - Lx[i];
}

// --- 1f. Per-segment flow: q_s = G_s * (p_a - p_b)  (one thread per segment) -
// This is the "one thread per vessel segment" mapping the catalog calls for --
// here specialized to the Poiseuille flow readout.
__global__ void segment_flows(int S, const int* __restrict__ seg_a,
                              const int* __restrict__ seg_b,
                              const double* __restrict__ Gseg,
                              const double* __restrict__ p,
                              double* __restrict__ q) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    q[s] = Gseg[s] * (p[seg_a[s]] - p[seg_b[s]]);
}

// ===========================================================================
// SECTION 2 -- Host helpers: CSR assembly + a device dot-product
// ===========================================================================
namespace {

// Grid size to cover `n` elements with TPB-thread blocks (ceil-divide).
inline int grid_for(int n) { return (n + TPB - 1) / TPB; }

// A device dot-product: launches dot_partial then reduce_final, leaving the
// scalar in d_out (device). d_partial must hold >= grid_for(N) doubles.
void device_dot(int N, const double* d_a, const double* d_b,
                double* d_partial, double* d_out) {
    const int blocks = grid_for(N);
    dot_partial<<<blocks, TPB>>>(N, d_a, d_b, d_partial);
    CUDA_CHECK_LAST("dot_partial");
    reduce_final<<<1, TPB>>>(blocks, d_partial, d_out);
    CUDA_CHECK_LAST("reduce_final");
}

// Assemble the boundary-eliminated Laplacian in CSR on the HOST, for the current
// conductances Gseg. Mirrors reference_cpu.cpp::spmv exactly:
//   * fixed row i  -> single nonzero: (i,i)=1
//   * interior i   -> diagonal (i,i)=sum incident G, plus (i,j)=-G for each
//                     INTERIOR neighbor j (fixed neighbors eliminated to b).
// We first build per-node adjacency lists, then flatten to CSR. Done once per
// autoregulation iteration (O(nnz)); the CG iterations that follow are on-GPU.
void build_csr(const Network& net, const std::vector<double>& Gseg,
               std::vector<int>& row_ptr, std::vector<int>& col_idx,
               std::vector<double>& val) {
    const int N = net.n_nodes;
    std::vector<double> diag(N, 0.0);
    std::vector<std::vector<int>>    nbr(N);   // interior neighbor node indices
    std::vector<std::vector<double>> nbrG(N);  // matching conductance (negated later)

    for (int s = 0; s < net.n_segs; ++s) {
        const int a = net.seg_a[s], b = net.seg_b[s];
        const double g = Gseg[s];
        const bool fa = net.is_fixed[a] != 0, fb = net.is_fixed[b] != 0;
        if (!fa) diag[a] += g;                 // diagonal gets every incident G
        if (!fb) diag[b] += g;
        if (!fa && !fb) {                      // keep off-diagonal only interior-interior
            nbr[a].push_back(b); nbrG[a].push_back(g);
            nbr[b].push_back(a); nbrG[b].push_back(g);
        }
    }

    row_ptr.assign(N + 1, 0);
    col_idx.clear();
    val.clear();
    // Count nonzeros per row to fill row_ptr (prefix sum).
    for (int i = 0; i < N; ++i) {
        if (net.is_fixed[i]) row_ptr[i + 1] = row_ptr[i] + 1;                    // identity
        else                 row_ptr[i + 1] = row_ptr[i] + 1 + (int)nbr[i].size();// diag+nbrs
    }
    col_idx.resize(row_ptr[N]);
    val.resize(row_ptr[N]);
    for (int i = 0; i < N; ++i) {
        int k = row_ptr[i];
        if (net.is_fixed[i]) {
            col_idx[k] = i; val[k] = 1.0;               // identity row
        } else {
            col_idx[k] = i; val[k] = diag[i]; ++k;      // diagonal
            for (size_t t = 0; t < nbr[i].size(); ++t) {
                col_idx[k] = nbr[i][t];
                val[k]     = -nbrG[i][t];               // off-diagonal is -G
                ++k;
            }
        }
    }
}

}  // namespace

// ===========================================================================
// SECTION 3 -- The GPU driver: autoregulation loop with on-device CG solves
// ===========================================================================
void solve_gpu(Network& net, int n_autoreg, double cg_tol, int cg_max_iter,
               Solution& out, float* kernel_ms) {
    const int N = net.n_nodes, S = net.n_segs;
    const int nblk = grid_for(N);

    out.p.assign(N, 0.0);
    out.q.assign(S, 0.0);

    // ---- host-side warm start (matches CPU exactly) -----------------------
    double pmax = 0.0, pmin = 1e300;
    for (int i = 0; i < N; ++i) if (net.is_fixed[i]) {
        if (net.fixed_p[i] > pmax) pmax = net.fixed_p[i];
        if (net.fixed_p[i] < pmin) pmin = net.fixed_p[i];
    }
    const double pmid = 0.5 * (pmax + pmin);
    std::vector<double> h_p(N);
    for (int i = 0; i < N; ++i) h_p[i] = net.is_fixed[i] ? net.fixed_p[i] : pmid;

    // ---- persistent device buffers (allocated once, reused every solve) ----
    double *d_p, *d_r, *d_pv, *d_Lp, *d_b, *d_val, *d_partial;
    int    *d_rowptr, *d_colidx;
    double *d_rr, *d_rrnew, *d_pLp, *d_alpha, *d_beta;   // device scalars
    double *d_Gseg, *d_q; int *d_sega, *d_segb;
    CUDA_CHECK(cudaMalloc(&d_p,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pv, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Lp, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial, nblk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rr,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rrnew,sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pLp,  sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha,sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Gseg, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_q,    S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sega, S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_segb, S * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sega, net.seg_a.data(), S * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segb, net.seg_b.data(), S * sizeof(int), cudaMemcpyHostToDevice));

    // Sparsity (nnz) is fixed by topology -- radii change VALUES not structure --
    // so we size the CSR arrays once from an initial assembly.
    std::vector<double> Gseg(S);
    for (int s = 0; s < S; ++s)
        Gseg[s] = coronary::segment_conductance(net.seg_r[s], net.seg_len[s], net.hct);
    std::vector<int> row_ptr, col_idx; std::vector<double> val;
    build_csr(net, Gseg, row_ptr, col_idx, val);
    const int nnz = (int)val.size();
    CUDA_CHECK(cudaMalloc(&d_rowptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colidx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val,    nnz * sizeof(double)));

    std::vector<double> h_b(N, 0.0);   // host RHS (rebuilt each iteration)

    // Autoregulation constants (identical to CPU).
    const double gain = 0.20, rmin = 4.0, rmax = 40.0;

    GpuTimer gtimer; gtimer.start();   // time ALL solves as one teaching figure
    CUDA_CHECK(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    for (int k = 0; k < n_autoreg; ++k) {
        // (a) conductances from current radii (coronary.h; same as CPU)
        for (int s = 0; s < S; ++s)
            Gseg[s] = coronary::segment_conductance(net.seg_r[s], net.seg_len[s], net.hct);
        CUDA_CHECK(cudaMemcpy(d_Gseg, Gseg.data(), S * sizeof(double), cudaMemcpyHostToDevice));

        // (a') CSR values + eliminated-boundary RHS (host assembly, then upload)
        build_csr(net, Gseg, row_ptr, col_idx, val);
        for (int i = 0; i < N; ++i) h_b[i] = net.is_fixed[i] ? net.fixed_p[i] : 0.0;
        for (int s = 0; s < S; ++s) {
            const int a = net.seg_a[s], b = net.seg_b[s];
            const bool fa = net.is_fixed[a] != 0, fb = net.is_fixed[b] != 0;
            if (!fa && fb) h_b[a] += Gseg[s] * net.fixed_p[b];
            if (fa && !fb) h_b[b] += Gseg[s] * net.fixed_p[a];
        }
        CUDA_CHECK(cudaMemcpy(d_rowptr, row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colidx, col_idx.data(), nnz * sizeof(int),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val,    val.data(),     nnz * sizeof(double),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b,      h_b.data(),     N * sizeof(double),    cudaMemcpyHostToDevice));

        // ---- (b) Conjugate Gradient on the device -------------------------
        // r = b - L p ; pv = r ; rr = r.r
        csr_spmv<<<nblk, TPB>>>(N, d_rowptr, d_colidx, d_val, d_p, d_Lp);
        CUDA_CHECK_LAST("csr_spmv(init)");
        residual_init<<<nblk, TPB>>>(N, d_b, d_Lp, d_r);
        CUDA_CHECK_LAST("residual_init");
        CUDA_CHECK(cudaMemcpy(d_pv, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));
        device_dot(N, d_r, d_r, d_partial, d_rr);

        // Relative stopping threshold uses ||b||^2 (host-side sum, once).
        double h_bnorm2 = 0.0;
        for (int i = 0; i < N; ++i) h_bnorm2 += h_b[i] * h_b[i];
        const double thresh = cg_tol * cg_tol * (h_bnorm2 > 0.0 ? h_bnorm2 : 1.0);

        double h_rr = 0.0;
        CUDA_CHECK(cudaMemcpy(&h_rr, d_rr, sizeof(double), cudaMemcpyDeviceToHost));

        int it = 0;
        for (; it < cg_max_iter; ++it) {
            if (h_rr <= thresh) break;                          // converged
            csr_spmv<<<nblk, TPB>>>(N, d_rowptr, d_colidx, d_val, d_pv, d_Lp);
            CUDA_CHECK_LAST("csr_spmv");
            device_dot(N, d_pv, d_Lp, d_partial, d_pLp);        // pLp = p.(Lp)
            calc_alpha<<<1, 1>>>(d_rr, d_pLp, d_alpha);         // alpha = rr/pLp
            CUDA_CHECK_LAST("calc_alpha");
            axpy<<<nblk, TPB>>>(N, d_alpha, d_pv, d_p);         // p += alpha*pv
            CUDA_CHECK_LAST("axpy");
            axmy<<<nblk, TPB>>>(N, d_alpha, d_Lp, d_r);         // r -= alpha*Lp
            CUDA_CHECK_LAST("axmy");
            device_dot(N, d_r, d_r, d_partial, d_rrnew);        // rr_new = r.r
            calc_beta<<<1, 1>>>(d_rrnew, d_rr, d_beta);         // beta = rr_new/rr
            CUDA_CHECK_LAST("calc_beta");
            xpby<<<nblk, TPB>>>(N, d_beta, d_r, d_pv);          // pv = r + beta*pv
            CUDA_CHECK_LAST("xpby");
            CUDA_CHECK(cudaMemcpy(d_rr, d_rrnew, sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(&h_rr, d_rr, sizeof(double), cudaMemcpyDeviceToHost));
        }
        out.cg_iters = it;
        out.cg_resid = std::sqrt(h_rr);

        // (c) flows implied by the new pressures (one thread per segment)
        segment_flows<<<grid_for(S), TPB>>>(S, d_sega, d_segb, d_Gseg, d_p, d_q);
        CUDA_CHECK_LAST("segment_flows");

        // Record the FIRST (cold-start) solve's cost + pre-autoregulation
        // perfusion, mirroring the CPU reference.
        if (k == 0) {
            out.cg_iters_first = out.cg_iters;
            std::vector<double> h_q0(S);
            CUDA_CHECK(cudaMemcpy(h_q0.data(), d_q, S * sizeof(double), cudaMemcpyDeviceToHost));
            out.perfusion_first = inlet_perfusion(net, h_q0);
        }

        // (d) autoregulate radii (host side, exactly as CPU) unless final pass
        if (k + 1 < n_autoreg) {
            std::vector<double> h_q(S);
            CUDA_CHECK(cudaMemcpy(h_q.data(), d_q, S * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s = 0; s < S; ++s)
                net.seg_r[s] = coronary::autoregulate_radius(
                    net.seg_r[s], h_q[s], net.seg_target[s], gain, rmin, rmax);
        }
    }

    *kernel_ms = gtimer.stop_ms();   // total device time (teaching artifact)

    // ---- copy the final solution back to the host -------------------------
    CUDA_CHECK(cudaMemcpy(out.p.data(), d_p, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.q.data(), d_q, S * sizeof(double), cudaMemcpyDeviceToHost));

    // ---- free everything (order irrelevant; all plain cudaFree) ------------
    cudaFree(d_p); cudaFree(d_r); cudaFree(d_pv); cudaFree(d_Lp); cudaFree(d_b);
    cudaFree(d_partial); cudaFree(d_rr); cudaFree(d_rrnew); cudaFree(d_pLp);
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_Gseg); cudaFree(d_q);
    cudaFree(d_sega); cudaFree(d_segb);
    cudaFree(d_rowptr); cudaFree(d_colidx); cudaFree(d_val);
}
