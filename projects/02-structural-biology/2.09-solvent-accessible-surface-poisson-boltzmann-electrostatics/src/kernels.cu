// ===========================================================================
// src/kernels.cu  --  Red-black Gauss-Seidel PBE kernels + host time loop
// ---------------------------------------------------------------------------
// Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
//
// GPU twin of solve_cpu(): the SAME per-cell relaxation (pbe.h), but the grid
// is coloured red/black so each colour's cells update in parallel without
// races. One thread per interior cell; two kernel launches per sweep. main.cu
// compares the resulting field with the CPU reference. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 3-D thread block: 8x8x8 = 512 threads. A cubic block maps naturally onto the
// cubic grid and gives good occupancy on sm_75..sm_89 (THEORY "GPU mapping").
static constexpr int BX = 8, BY = 8, BZ = 8;

// ---------------------------------------------------------------------------
// relax_color_kernel: update every interior cell of ONE colour.
//   Launch config
//     grid  : covers the n^3 cells (one thread per cell; outer-shell threads
//             and wrong-colour threads return early).
//     block : 8x8x8 = 512 threads.
//     thread (global x,y,z) owns grid cell (x,y,z).
//   Memory: reads phi, rho, eps, kappa2 from GLOBAL memory; writes phi in place.
//     Because only THIS colour's cells are written and they read only the OTHER
//     colour's cells, there is no write-after-read hazard within the launch --
//     that is the whole point of the red-black split.
//   Parameters
//     color : 0 = red (parity 0), 1 = black (parity 1).
//   The per-cell math is the shared pbe_relax_cell() (pbe.h), identical to the
//   CPU, which is what makes GPU and CPU agree to ~machine precision.
// ---------------------------------------------------------------------------
__global__ void relax_color_kernel(GridParams P, int color,
                                   double* __restrict__ phi,
                                   const double* __restrict__ rho,
                                   const double* __restrict__ eps,
                                   const double* __restrict__ kappa2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int n = P.n;

    // Only INTERIOR cells are relaxed; the outer shell stays at the grounded
    // boundary phi = 0 (set once at allocation). Guard the ragged edges too.
    if (x < 1 || x >= n - 1 || y < 1 || y >= n - 1 || z < 1 || z >= n - 1) return;

    // Skip cells of the other colour this pass (the parity test that makes the
    // in-place update race-free -- see kernels.cuh).
    if (((x + y + z) & 1) != color) return;

    const int c = pbe_idx(x, y, z, n);
    phi[c] = pbe_relax_cell(x, y, z, P, phi, rho, eps, kappa2);
}

// ---------------------------------------------------------------------------
// solve_gpu: allocate device grids, run the red-black sweeps, copy phi back.
//   The host owns the sweep loop and launches two kernels per sweep (red then
//   black). All buffers live on the device for the whole solve, so we pay the
//   host<->device copy only once in and once out -- the relaxation never leaves
//   the GPU. Timing wraps the entire sweep loop with CUDA events.
// ---------------------------------------------------------------------------
void solve_gpu(const PbeProblem& prob, std::vector<double>& phi, float* kernel_ms) {
    const GridParams& P = prob.P;
    const int n = P.n;
    const size_t N = static_cast<size_t>(n) * n * n;
    const size_t bytes = N * sizeof(double);

    // Device buffers: the unknown field phi (in/out) and the three read-only
    // coefficient grids built on the host (eps, kappa^2, rho).
    double *d_phi = nullptr, *d_rho = nullptr, *d_eps = nullptr, *d_kappa2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phi, bytes));     // can fail: out-of-memory
    CUDA_CHECK(cudaMalloc(&d_rho, bytes));
    CUDA_CHECK(cudaMalloc(&d_eps, bytes));
    CUDA_CHECK(cudaMalloc(&d_kappa2, bytes));

    // phi starts at the boundary value everywhere (0); that also sets the outer
    // shell to the grounded boundary condition, which the kernel never touches.
    CUDA_CHECK(cudaMemcpy(d_phi, phi.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, prob.rho.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eps, prob.eps.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kappa2, prob.kappa2.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BX, BY, BZ);
    dim3 grid((n + BX - 1) / BX, (n + BY - 1) / BY, (n + BZ - 1) / BZ);

    // ---- the relaxation: P.iters sweeps, each = red launch then black launch.
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < P.iters; ++it) {
        relax_color_kernel<<<grid, block>>>(P, /*color=*/0, d_phi, d_rho, d_eps, d_kappa2);
        relax_color_kernel<<<grid, block>>>(P, /*color=*/1, d_phi, d_rho, d_eps, d_kappa2);
    }
    *kernel_ms = timer.stop_ms();
    // One synchronizing error check after the loop catches launch + execution
    // failures from any sweep (the per-launch sync is folded into this).
    CUDA_CHECK_LAST("relax_color_kernel");

    // Copy the converged field back and release device memory.
    CUDA_CHECK(cudaMemcpy(phi.data(), d_phi, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_phi));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_eps));
    CUDA_CHECK(cudaFree(d_kappa2));
}
