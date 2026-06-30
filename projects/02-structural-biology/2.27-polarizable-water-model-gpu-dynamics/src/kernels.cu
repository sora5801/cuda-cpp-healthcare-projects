// ===========================================================================
// src/kernels.cu  --  GPU Jacobi SCF induced-dipole solver (one thread/site)
// ---------------------------------------------------------------------------
// Project 2.27 : Polarizable Water Model GPU Dynamics
//
// WHAT THIS FILE DOES
//   The GPU twin of solve_dipoles_cpu(). It runs the SAME Jacobi self-consistent
//   field loop (polar.h math) but in parallel: one thread per site per sweep.
//   main.cu runs CPU + GPU and compares the converged dipoles and energy.
//
//   Kernels here:
//     permanent_field_kernel : E^perm_i once (thread i loops over all j charges).
//     dipole_sweep_kernel    : one Jacobi sweep -- thread i gathers the dipole
//                              field from every other site's CURRENT dipole,
//                              re-induces mu_i, writes it to the OTHER buffer,
//                              and contributes its per-component change to a
//                              DETERMINISTIC fixed-point max (the SCF residual).
//     energy_kernel          : U_pol = -1/2 sum_i mu_i . E^perm_i, accumulated in
//                              a fixed-point integer so the sum is reproducible.
//
//   Determinism (PATTERNS.md §3): the per-site field loop visits j in the same
//   order as the CPU, and both scalar reductions use integer atomics, so stdout
//   is byte-identical every run and matches the CPU to round-off.
//
// READ THIS AFTER: kernels.cuh, polar.h, reference_cpu.cpp (the CPU twin).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <vector>

// Threads per block. 128 is a good default here: each thread does an O(N) inner
// loop with many registers (positions, accumulators), so a smaller block keeps
// register pressure / occupancy balanced on sm_75..sm_89. (Tune per GPU.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// FIXED-POINT REDUCTION SCALE
//   Floating-point atomicAdd is NOT associative, so a parallel float sum depends
//   on thread-finish order -> nondeterministic. We instead scale each value by a
//   large constant, round to a 64-bit integer, and atomicAdd the integers
//   (integer addition DOES commute), then divide back on the host. SCALE = 1e12
//   keeps ~12 digits of the energy / residual -- far more than the ~1e-9
//   convergence tolerance needs, and well within int64 range for these small
//   clusters. This is the exact idiom used by flagships 5.01 and 11.09.
// ---------------------------------------------------------------------------
static constexpr double FIXED_SCALE = 1.0e12;

// scale_to_ll: deterministic round of a double to a fixed-point long long.
//   __device__ helper used by both reductions. llrint rounds half-to-even
//   identically on host and device for these magnitudes.
__device__ inline long long scale_to_ll(double v) {
    return llrint(v * FIXED_SCALE);
}

// ---------------------------------------------------------------------------
// init_dipoles_kernel: the "direct" (zeroth) guess mu_i = alpha_i * E^perm_i.
//   This is the SAME starting point the CPU uses before its Jacobi loop, so the
//   two solvers count sweeps identically. Thread i owns site i.
// ---------------------------------------------------------------------------
__global__ void init_dipoles_kernel(int N,
                                    const double* __restrict__ alpha,
                                    const Vec3*  __restrict__ Eperm,
                                    Vec3*        __restrict__ mu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    mu[i] = vscale(Eperm[i], alpha[i]);
}

// ---------------------------------------------------------------------------
// permanent_field_kernel: E^perm_i = Eext + sum_{j!=i} q_j r_ij / r_ij^3.
//   grid  : ceil(N / THREADS_PER_BLOCK) blocks
//   block : THREADS_PER_BLOCK threads
//   thread i (= blockIdx.x*blockDim.x + threadIdx.x) owns output Eperm[i].
//   Reads all positions/charges from global memory (SoA: px,py,pz,q). Computed
//   ONCE; the result is reused by every Jacobi sweep, exactly like the CPU.
// ---------------------------------------------------------------------------
__global__ void permanent_field_kernel(int N,
                                       const double* __restrict__ px,
                                       const double* __restrict__ py,
                                       const double* __restrict__ pz,
                                       const double* __restrict__ q,
                                       Vec3 Eext,
                                       Vec3* __restrict__ Eperm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;                                   // guard ragged last block

    const Vec3 pi{px[i], py[i], pz[i]};
    Vec3 E = Eext;                                        // start from external field
    // Loop over all OTHER sites in the SAME order as the CPU (j = 0..N-1, skip i)
    // so the floating-point accumulation is identical on both sides.
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        const double qj = q[j];
        if (qj == 0.0) continue;                          // neutral site: no field
        const Vec3 pj{px[j], py[j], pz[j]};
        E = vadd(E, field_perm_pair(pi, pj, qj));         // shared polar.h math
    }
    Eperm[i] = E;
}

// ---------------------------------------------------------------------------
// dipole_sweep_kernel: ONE Jacobi sweep.
//   Thread i gathers the dipole field at site i from every OTHER site's CURRENT
//   induced dipole (mu_in), adds the (fixed) permanent field, re-induces
//   mu_i = alpha_i * (E^perm + E^dip), and writes it to mu_out. It also folds its
//   own largest per-component change into a global fixed-point maximum (the SCF
//   residual) via atomicMax on int64-encoded values.
//
//   Ping-pong: reading mu_in and writing mu_out (separate buffers) is what makes
//   this a JACOBI update -- every site sees the same previous-sweep dipoles, so
//   the sweep is order-independent and safe to parallelize. The host swaps the
//   two buffers between sweeps.
//
//   grid/block as above; thread i owns site i.
// ---------------------------------------------------------------------------
__global__ void dipole_sweep_kernel(int N, double a_thole,
                                    const double* __restrict__ px,
                                    const double* __restrict__ py,
                                    const double* __restrict__ pz,
                                    const double* __restrict__ alpha,
                                    const Vec3*  __restrict__ Eperm,
                                    const Vec3*  __restrict__ mu_in,
                                    Vec3*        __restrict__ mu_out,
                                    long long*   __restrict__ dmu_fixed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double ai = alpha[i];
    const Vec3   pi{px[i], py[i], pz[i]};

    // Gather the field from every OTHER induced dipole (this sweep's mu_in).
    Vec3 Edip{0.0, 0.0, 0.0};
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        const double aj = alpha[j];
        if (aj == 0.0) continue;                          // fixed charge: no dipole
        const Vec3 pj{px[j], py[j], pz[j]};
        Edip = vadd(Edip, field_dip_pair(pi, pj, mu_in[j], ai, aj, a_thole));
    }

    // Re-induce: mu_i = alpha_i * (E^perm_i + E^dip_i).
    const Vec3 Etot = vadd(Eperm[i], Edip);
    const Vec3 mnew = vscale(Etot, ai);
    mu_out[i] = mnew;

    // Largest per-component change at this site (the local residual).
    const Vec3 d = vsub(mnew, mu_in[i]);
    double local = fabs(d.x);
    local = fmax(local, fabs(d.y));
    local = fmax(local, fabs(d.z));

    // Fold into the GLOBAL max deterministically: encode as a fixed-point int64
    // (residuals are >= 0, so plain integer atomicMax gives the true maximum).
    atomicMax(dmu_fixed, scale_to_ll(local));
}

// ---------------------------------------------------------------------------
// energy_kernel: U_pol = -1/2 sum_i mu_i . E^perm_i, accumulated in fixed point.
//   Each thread computes its site's contribution (polarization_energy_site from
//   polar.h) and atomicAdds the int64-scaled value into a single accumulator.
//   Integer adds commute -> the total is identical regardless of thread order
//   and matches the CPU's serial sum to the fixed-point resolution.
// ---------------------------------------------------------------------------
__global__ void energy_kernel(int N,
                              const Vec3* __restrict__ mu,
                              const Vec3* __restrict__ Eperm,
                              long long* __restrict__ U_fixed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double e = polarization_energy_site(mu[i], Eperm[i]);
    atomicAdd(reinterpret_cast<unsigned long long*>(U_fixed),
              static_cast<unsigned long long>(scale_to_ll(e)));
}

// ---------------------------------------------------------------------------
// solve_dipoles_gpu: host wrapper -- the canonical CUDA lifecycle around the
// three kernels above, with the Jacobi sweep loop driven from the host.
// ---------------------------------------------------------------------------
SolveResult solve_dipoles_gpu(const PolarSystem& sys, float* kernel_ms) {
    const int N = num_sites(sys);

    // (1) Repack the Array-of-Structs sites into Structure-of-Arrays on the host
    //     so consecutive threads read consecutive (coalesced) addresses on the
    //     device. px[k] is site k's x-coordinate, etc.
    std::vector<double> h_px(N), h_py(N), h_pz(N), h_q(N), h_alpha(N);
    for (int k = 0; k < N; ++k) {
        const Site& s = sys.sites[static_cast<std::size_t>(k)];
        h_px[k] = s.pos.x; h_py[k] = s.pos.y; h_pz[k] = s.pos.z;
        h_q[k]  = s.q;     h_alpha[k] = s.alpha;
    }

    // (2) Allocate device buffers. d_ marks DEVICE pointers (CLAUDE.md §12).
    double *d_px=nullptr,*d_py=nullptr,*d_pz=nullptr,*d_q=nullptr,*d_alpha=nullptr;
    Vec3 *d_Eperm=nullptr, *d_muA=nullptr, *d_muB=nullptr;
    long long *d_dmu=nullptr, *d_U=nullptr;
    const std::size_t db = static_cast<std::size_t>(N) * sizeof(double);
    const std::size_t vb = static_cast<std::size_t>(N) * sizeof(Vec3);
    CUDA_CHECK(cudaMalloc(&d_px, db));   CUDA_CHECK(cudaMalloc(&d_py, db));
    CUDA_CHECK(cudaMalloc(&d_pz, db));   CUDA_CHECK(cudaMalloc(&d_q, db));
    CUDA_CHECK(cudaMalloc(&d_alpha, db));
    CUDA_CHECK(cudaMalloc(&d_Eperm, vb));
    CUDA_CHECK(cudaMalloc(&d_muA, vb)); CUDA_CHECK(cudaMalloc(&d_muB, vb));
    CUDA_CHECK(cudaMalloc(&d_dmu, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_U, sizeof(long long)));

    // (3) Upload geometry H2D.
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q,  h_q.data(),  db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha, h_alpha.data(), db, cudaMemcpyHostToDevice));

    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();   // time the whole SCF (permanent field + every sweep + energy)

    // (4a) Permanent field, once.
    permanent_field_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        N, d_px, d_py, d_pz, d_q, sys.Eext, d_Eperm);
    CUDA_CHECK_LAST("permanent_field_kernel");

    // (4b) Direct (zeroth) dipole guess mu = alpha * Eperm, into d_cur. This is
    //      the EXACT starting point the CPU uses, so the sweep counts align.
    Vec3* d_cur = d_muA;   // current dipoles (input to the next sweep)
    Vec3* d_nxt = d_muB;   // scratch (output of the next sweep)
    init_dipoles_kernel<<<blocks, THREADS_PER_BLOCK>>>(N, d_alpha, d_Eperm, d_cur);
    CUDA_CHECK_LAST("init_dipoles_kernel");

    // (4c) Jacobi sweeps until converged. This loop mirrors reference_cpu.cpp's
    //      `for (iter = 1; iter <= max_iters; ++iter)` ONE-FOR-ONE so the reported
    //      iteration count matches the CPU.
    int iter = 1;
    long long h_dmu = 0;
    double dmu_max = 0.0;
    for (iter = 1; iter <= sys.max_iters; ++iter) {
        CUDA_CHECK(cudaMemset(d_dmu, 0, sizeof(long long)));   // reset residual
        dipole_sweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            N, sys.a_thole, d_px, d_py, d_pz, d_alpha, d_Eperm,
            d_cur, d_nxt, d_dmu);
        CUDA_CHECK_LAST("dipole_sweep_kernel");
        // Bring back the fixed-point residual and decode it.
        CUDA_CHECK(cudaMemcpy(&h_dmu, d_dmu, sizeof(long long), cudaMemcpyDeviceToHost));
        dmu_max = static_cast<double>(h_dmu) / FIXED_SCALE;

        Vec3* tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;          // ping-pong swap
        if (dmu_max <= sys.tol) break;                          // converged
    }
    if (iter > sys.max_iters) iter = sys.max_iters;

    // (4d) Polarization energy of the converged dipoles (d_cur), fixed-point.
    CUDA_CHECK(cudaMemset(d_U, 0, sizeof(long long)));
    energy_kernel<<<blocks, THREADS_PER_BLOCK>>>(N, d_cur, d_Eperm, d_U);
    CUDA_CHECK_LAST("energy_kernel");

    *kernel_ms = timer.stop_ms();

    // (5) Copy results back.
    SolveResult r;
    r.mu.assign(static_cast<std::size_t>(N), Vec3{0.0, 0.0, 0.0});
    CUDA_CHECK(cudaMemcpy(r.mu.data(), d_cur, vb, cudaMemcpyDeviceToHost));
    long long h_U = 0;
    CUDA_CHECK(cudaMemcpy(&h_U, d_U, sizeof(long long), cudaMemcpyDeviceToHost));
    r.U_pol = static_cast<double>(h_U) / FIXED_SCALE;
    r.U_pol_kcal = energy_to_kcal_per_mol(r.U_pol);
    r.iters = iter;
    r.final_dmu = dmu_max;

    // (6) Free everything (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_q));  CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_Eperm));
    CUDA_CHECK(cudaFree(d_muA)); CUDA_CHECK(cudaFree(d_muB));
    CUDA_CHECK(cudaFree(d_dmu)); CUDA_CHECK(cudaFree(d_U));
    return r;
}
