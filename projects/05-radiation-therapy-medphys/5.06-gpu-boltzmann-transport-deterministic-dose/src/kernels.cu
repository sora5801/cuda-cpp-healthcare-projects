// ===========================================================================
// src/kernels.cu  --  GPU S_N sweep + deterministic reduction + SI driver
// ---------------------------------------------------------------------------
// Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
//
// GPU twin of solve_sn_cpu(). Per-cell physics is the shared boltzmann_sn.h, so
// the two agree to floating-point round-off. Two kernels per source iteration:
//   1) sweep_kernel  -- one thread per ordinate does the full spatial sweep,
//   2) reduce_kernel -- one thread per cell sums ordinate contributions (fixed
//                       order -> deterministic, no atomics).
// The host runs the outer source-iteration loop and the convergence test.
// See ../THEORY.md "GPU mapping". Read kernels.cuh first.
// ===========================================================================
#include "kernels.cuh"
#include "boltzmann_sn.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cmath>     // std::fabs
#include <cstddef>   // std::size_t
#include <vector>

// A modest block size. nord (the S_N order) is small (typically 2..32), so the
// sweep grid is tiny; ncell (cells) drives the reduction grid. 128 threads/block
// is a safe occupancy default on sm_75..sm_89 for both kernels.
static constexpr int BLOCK = 128;

// ---------------------------------------------------------------------------
// sweep_kernel: thread n performs the entire spatial sweep for ordinate n.
//   Launch config (set in solve_sn_gpu):
//     grid  = ceil(nord / BLOCK) blocks
//     block = BLOCK threads
//   Thread-to-data map: n = blockIdx.x*blockDim.x + threadIdx.x -> ordinate n.
//   Memory: reads the material arrays + lagged d_phi from global memory; writes
//   its OWN row d_contrib[n*ncell .. n*ncell+ncell-1]. Because each thread owns a
//   distinct row, there are NO races and NO atomics -- the reduction is a
//   separate, deterministic kernel.
//
//   The heavy lifting is in sn_sweep_one_ordinate() (shared with the CPU): it
//   walks the cells in the travel direction, applying the diamond-difference
//   update, and accumulates w_n*psi_avg into this thread's row.
// ---------------------------------------------------------------------------
__global__ void sweep_kernel(int ncell, int nord, double h,
                             const double* __restrict__ d_mu,
                             const double* __restrict__ d_w,
                             const double* __restrict__ d_sigma_t,
                             const double* __restrict__ d_sigma_s,
                             const double* __restrict__ d_q,
                             const double* __restrict__ d_phi,
                             double psi_left_bc, double psi_right_bc,
                             double* __restrict__ d_contrib) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's ordinate
    if (n >= nord) return;                                // guard the ragged block

    // This thread's private output row (length ncell). We zero it first because
    // sn_sweep_one_ordinate ADDS into it (the += accumulation idiom shared with
    // the CPU, which reuses a cleared buffer each iteration).
    double* row = d_contrib + static_cast<std::size_t>(n) * ncell;
    for (int i = 0; i < ncell; ++i) row[i] = 0.0;

    sn_sweep_one_ordinate(d_mu[n], d_w[n], ncell, h,
                          d_sigma_t, d_sigma_s, d_q, d_phi,
                          psi_left_bc, psi_right_bc, row);
}

// ---------------------------------------------------------------------------
// reduce_kernel: thread i sums the nord ordinate contributions for cell i.
//   Launch config: grid = ceil(ncell / BLOCK), block = BLOCK.
//   Thread-to-data map: i = blockIdx.x*blockDim.x + threadIdx.x -> cell i.
//   The sum walks n = 0..nord-1 in order, so the result is DETERMINISTIC and
//   equals the CPU's ordinate-ordered accumulation exactly (PATTERNS.md §3). We
//   deliberately do NOT use atomicAdd here: float atomics reorder additions and
//   would break bit-reproducibility.
// ---------------------------------------------------------------------------
__global__ void reduce_kernel(int ncell, int nord,
                              const double* __restrict__ d_contrib,
                              double* __restrict__ d_phi_new) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's cell
    if (i >= ncell) return;

    double s = 0.0;                                       // running scalar flux
    for (int n = 0; n < nord; ++n)                        // fixed ordinate order
        s += d_contrib[static_cast<std::size_t>(n) * ncell + i];
    d_phi_new[i] = s;
}

// ---------------------------------------------------------------------------
// solve_sn_gpu: the host driver. Five canonical CUDA steps, then the outer
//   source-iteration loop (mirrors solve_sn_cpu):
//     (1) allocate device buffers,
//     (2) upload materials + quadrature,
//     (3) per iteration: sweep_kernel -> reduce_kernel -> copy phi_new back,
//         test relative L-inf change, swap phi,
//     (4) copy the converged phi to the host,
//     (5) free device memory.
//   We time ONLY the iteration loop (step 3) with CUDA events, so the reported
//   figure is compute cost, not one-time upload cost.
// ---------------------------------------------------------------------------
void solve_sn_gpu(const SlabProblem& p, const SnQuadrature& quad,
                  std::vector<double>& phi, int& iters, float* kernel_ms) {
    const int    n    = p.ncell;
    const int    nord = p.nord;
    const double h    = p.h();
    const std::size_t cell_bytes = static_cast<std::size_t>(n)    * sizeof(double);
    const std::size_t ord_bytes  = static_cast<std::size_t>(nord) * sizeof(double);
    const std::size_t contrib_bytes =
        static_cast<std::size_t>(nord) * n * sizeof(double);

    // (1) device buffers. d_ prefix = DEVICE pointer (CLAUDE §12).
    double *d_mu = nullptr, *d_w = nullptr;
    double *d_sigma_t = nullptr, *d_sigma_s = nullptr, *d_q = nullptr;
    double *d_phi = nullptr, *d_phi_new = nullptr, *d_contrib = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mu,      ord_bytes));        // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_w,       ord_bytes));
    CUDA_CHECK(cudaMalloc(&d_sigma_t, cell_bytes));
    CUDA_CHECK(cudaMalloc(&d_sigma_s, cell_bytes));
    CUDA_CHECK(cudaMalloc(&d_q,       cell_bytes));
    CUDA_CHECK(cudaMalloc(&d_phi,     cell_bytes));
    CUDA_CHECK(cudaMalloc(&d_phi_new, cell_bytes));
    CUDA_CHECK(cudaMalloc(&d_contrib, contrib_bytes));    // [nord*ncell] scratch

    // (2) upload the (iteration-invariant) inputs once.
    CUDA_CHECK(cudaMemcpy(d_mu,      quad.mu.data(),   ord_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,       quad.w.data(),    ord_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma_t, p.sigma_t.data(), cell_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma_s, p.sigma_s.data(), cell_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q,       p.q.data(),       cell_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_phi, 0, cell_bytes));         // initial guess phi = 0

    const dim3 sweep_block(BLOCK);
    const dim3 sweep_grid((nord + BLOCK - 1) / BLOCK);    // ceil-div: cover all ordinates
    const dim3 red_block(BLOCK);
    const dim3 red_grid((n + BLOCK - 1) / BLOCK);         // ceil-div: cover all cells

    std::vector<double> phi_host(n, 0.0);      // current iterate mirrored on host
    std::vector<double> phi_new_host(n, 0.0);  // for the convergence test

    // (3) source iteration, timed as a whole.
    iters = 0;
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < p.max_iter; ++it) {
        // a. sweep every ordinate (fills d_contrib rows from the lagged d_phi).
        sweep_kernel<<<sweep_grid, sweep_block>>>(
            n, nord, h, d_mu, d_w, d_sigma_t, d_sigma_s, d_q, d_phi,
            p.psi_left_bc, p.psi_right_bc, d_contrib);
        CUDA_CHECK_LAST("sweep_kernel");

        // b. reduce ordinate contributions -> new scalar flux (deterministic).
        reduce_kernel<<<red_grid, red_block>>>(n, nord, d_contrib, d_phi_new);
        CUDA_CHECK_LAST("reduce_kernel");

        // c. bring phi_new back to test convergence on the host (the outer loop
        //    control). ncell is small in this teaching problem, so this copy is
        //    negligible; a production code would do the reduction + norm on-device
        //    to avoid the round-trip (noted in THEORY §GPU mapping).
        CUDA_CHECK(cudaMemcpy(phi_new_host.data(), d_phi_new, cell_bytes,
                              cudaMemcpyDeviceToHost));

        double num = 0.0, den = 0.0;                      // relative L-inf change
        for (int i = 0; i < n; ++i) {
            const double d = std::fabs(phi_new_host[i] - phi_host[i]);
            if (d > num) num = d;
            const double a = std::fabs(phi_new_host[i]);
            if (a > den) den = a;
        }
        phi_host = phi_new_host;

        // d. accept the new iterate as the lag for the next sweep: swap the
        //    DEVICE pointers (O(1)) so the next sweep reads d_phi_new's data.
        double* tmp = d_phi; d_phi = d_phi_new; d_phi_new = tmp;
        ++iters;

        const double rel = (den > 0.0) ? (num / den) : num;
        if (rel <= p.tol) break;                          // converged
    }
    *kernel_ms = timer.stop_ms();

    // (4) the converged flux now lives in d_phi (after the final swap).
    phi.assign(n, 0.0);
    CUDA_CHECK(cudaMemcpy(phi.data(), d_phi, cell_bytes, cudaMemcpyDeviceToHost));

    // (5) release device memory.
    CUDA_CHECK(cudaFree(d_mu));       CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_sigma_t));  CUDA_CHECK(cudaFree(d_sigma_s));
    CUDA_CHECK(cudaFree(d_q));        CUDA_CHECK(cudaFree(d_phi));
    CUDA_CHECK(cudaFree(d_phi_new));  CUDA_CHECK(cudaFree(d_contrib));
}
