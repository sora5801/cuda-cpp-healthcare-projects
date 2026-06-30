// ===========================================================================
// src/kernels.cu  --  The two GPU kernels + their host wrappers
// ---------------------------------------------------------------------------
// Project 2.3 : Cryo-EM Single-Particle Reconstruction  (reduced-scope, 2D)
//
// WHAT THIS FILE DOES
//   Implements the GPU twins of the CPU reference (reference_cpu.cpp):
//     * match_kernel        + match_gpu        -> the E-step (projection match)
//     * backproject_kernel  + reconstruct_gpu  -> the M-step (back-projection)
//   main.cu runs the CPU and GPU versions of both and asserts they agree.
//
//   Crucially, neither kernel re-derives the physics: both call the SHARED
//   __host__ __device__ functions from reference_cpu.h (ncc_score,
//   backproject_pixel), the exact same code the CPU reference runs. That is the
//   single most important idiom in this repo (docs/PATTERNS.md §2): identical
//   arithmetic on both sides => verification is EXACT, not "close enough".
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide memory latency, and many blocks resident
// for occupancy. (See THEORY "GPU mapping" for the occupancy reasoning.)
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CONSTANT MEMORY: the reference projection bank.
//   Every matching thread reads ALL M*PROJ_LEN reference samples but none
//   writes them, and they are identical for the whole launch -> constant memory
//   is ideal: its hardware cache BROADCASTS one address to a whole warp in a
//   single transaction (vs. a global load per thread). Size = N_ANGLES*PROJ_LEN
//   floats = 60*64*4 = 15,360 bytes, comfortably within the 64 KB constant bank.
//   Filled by cudaMemcpyToSymbol() in match_gpu().
// ---------------------------------------------------------------------------
__constant__ float c_refs[N_ANGLES * PROJ_LEN];

// CONSTANT MEMORY: the per-angle view directions (theta_a) for back-projection.
//   N_ANGLES doubles = 480 bytes. Read by every back-projection thread for every
//   particle -> the same broadcast argument as above. Filled in reconstruct_gpu.
__constant__ double c_ref_thetas[N_ANGLES];

// ---------------------------------------------------------------------------
// match_kernel (E-STEP): one thread per particle.
//   Launch (set in match_gpu): grid = ceil(n / B) blocks (capped), block = B,
//   with a grid-stride loop so a modest grid covers an arbitrarily large N.
//   Thread (blockIdx.x, threadIdx.x) -> particle index i.
//
//   Each thread copies its particle profile into a small LOCAL array once, then
//   loops over the M reference projections in constant memory, scoring each with
//   the shared ncc_score(). It keeps the best angle (strict `>` so an exact tie
//   keeps the lower index -- identical to the CPU's tie-break -> bit-exact match).
//   No shared memory or atomics: each particle's result is independent.
// ---------------------------------------------------------------------------
__global__ void match_kernel(const float* __restrict__ particles, int n,
                             int* __restrict__ assign,
                             float* __restrict__ best_score) {
    const int stride = blockDim.x * gridDim.x;                 // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // Load this particle's profile into registers/local memory. PROJ_LEN is
        // a compile-time constant so the array is a fixed size and the inner
        // loops can unroll. (We copy once instead of re-reading global memory
        // M times inside ncc_score.)
        float prof[PROJ_LEN];
        #pragma unroll
        for (int s = 0; s < PROJ_LEN; ++s)
            prof[s] = particles[static_cast<long long>(i) * PROJ_LEN + s];

        int   best_a = 0;
        float best_s = -2.0f;        // below NCC's -1 floor so any real score wins
        for (int a = 0; a < N_ANGLES; ++a) {
            // c_refs + a*PROJ_LEN is reference template a, broadcast from the
            // constant cache. ncc_score is the SHARED host/device function.
            const float s = ncc_score(prof, c_refs + a * PROJ_LEN, PROJ_LEN);
            if (s > best_s) { best_s = s; best_a = a; }       // lowest-index tie-break
        }
        assign[i]     = best_a;
        best_score[i] = best_s;
    }
}

// ---------------------------------------------------------------------------
// backproject_kernel (M-STEP): one thread per OUTPUT pixel (a gather).
//   Launch (set in reconstruct_gpu): a 2D grid of 2D blocks covering the
//   IMG_SIZE x IMG_SIZE density. Thread (px,py) computes recon[py*IMG_SIZE+px].
//
//   Each thread sums every particle's contribution at its pixel via the shared
//   backproject_pixel(), which walks particles i=0..n-1 in a FIXED order. Since
//   each output pixel is owned by exactly one thread (no cross-thread writes)
//   and the per-pixel sum order is fixed, the result is deterministic and
//   bit-identical to reconstruct_cpu -- NO atomics required (contrast project
//   11.09, where scattered writes force fixed-point atomic accumulation).
// ---------------------------------------------------------------------------
__global__ void backproject_kernel(const float* __restrict__ particles,
                                   const int* __restrict__ assign, int n,
                                   float* __restrict__ recon) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;     // pixel column
    const int py = blockIdx.y * blockDim.y + threadIdx.y;     // pixel row
    if (px >= IMG_SIZE || py >= IMG_SIZE) return;             // guard ragged edges
    // c_ref_thetas is the per-angle direction table in constant memory; the
    // shared backproject_pixel() does the gather (same math as the CPU).
    recon[py * IMG_SIZE + px] =
        backproject_pixel(particles, assign, c_ref_thetas, n, px, py);
}

// ---------------------------------------------------------------------------
// match_gpu: host wrapper for the E-step. The canonical CUDA steps, with the
//   reference bank going to CONSTANT memory rather than a global buffer. We time
//   ONLY the kernel (CUDA events), not the H2D/D2H copies (discussed in THEORY).
// ---------------------------------------------------------------------------
void match_gpu(const Dataset& ds, std::vector<int>& assign,
               std::vector<float>& best_score, float* kernel_ms) {
    const int n = ds.n_particles;
    assign.assign(static_cast<std::size_t>(n), 0);
    best_score.assign(static_cast<std::size_t>(n), -2.0f);
    const std::size_t part_bytes = static_cast<std::size_t>(n) * PROJ_LEN * sizeof(float);

    // (a) Upload the reference bank to the __constant__ symbol (special copy
    //     into the constant bank, not ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_refs, ds.refs.data(),
                                  static_cast<std::size_t>(N_ANGLES) * PROJ_LEN * sizeof(float)));

    // (b) Allocate + upload the particles, and allocate the two outputs.
    float* d_part  = nullptr;   // [n*PROJ_LEN] device, row-major
    int*   d_assign = nullptr;  // [n] device
    float* d_score = nullptr;   // [n] device
    CUDA_CHECK(cudaMalloc(&d_part,  part_bytes));
    CUDA_CHECK(cudaMalloc(&d_assign, static_cast<std::size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_score, static_cast<std::size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_part, ds.particles.data(), part_bytes, cudaMemcpyHostToDevice));

    // (c) Launch: enough blocks to cover n one-thread-per-particle, capped so the
    //     grid stays modest; the grid-stride loop handles any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;
    GpuTimer timer;
    timer.start();
    match_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_part, n, d_assign, d_score);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("match_kernel");

    // (d) Copy results back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(assign.data(),     d_assign, static_cast<std::size_t>(n) * sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(best_score.data(), d_score,  static_cast<std::size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_part));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_score));
}

// ---------------------------------------------------------------------------
// reconstruct_gpu: host wrapper for the M-step. Uploads the assignments + the
//   per-angle theta table (constant memory) + the particles, launches the 2D
//   back-projection grid, times it, returns the density.
// ---------------------------------------------------------------------------
void reconstruct_gpu(const Dataset& ds, const std::vector<int>& assign,
                     std::vector<float>& recon, float* kernel_ms) {
    const int n = ds.n_particles;
    recon.assign(static_cast<std::size_t>(IMG_SIZE) * IMG_SIZE, 0.0f);
    const std::size_t part_bytes  = static_cast<std::size_t>(n) * PROJ_LEN * sizeof(float);
    const std::size_t recon_bytes = static_cast<std::size_t>(IMG_SIZE) * IMG_SIZE * sizeof(float);

    // Precompute the per-angle directions on the host (identical values to the
    // CPU reference) and upload them to constant memory.
    double thetas[N_ANGLES];
    for (int a = 0; a < N_ANGLES; ++a) thetas[a] = ref_angle(a);
    CUDA_CHECK(cudaMemcpyToSymbol(c_ref_thetas, thetas, sizeof(thetas)));

    // Device buffers.
    float* d_part   = nullptr;   // [n*PROJ_LEN]
    int*   d_assign = nullptr;   // [n]
    float* d_recon  = nullptr;   // [IMG_SIZE*IMG_SIZE]
    CUDA_CHECK(cudaMalloc(&d_part,   part_bytes));
    CUDA_CHECK(cudaMalloc(&d_assign, static_cast<std::size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recon,  recon_bytes));
    CUDA_CHECK(cudaMemcpy(d_part,   ds.particles.data(), part_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_assign, assign.data(), static_cast<std::size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));

    // 2D launch: 16x16 = 256-thread blocks tile the image; ceil-divide so the
    // grid covers IMG_SIZE in each dimension (the kernel guards the ragged edge).
    const dim3 block(16, 16);
    const dim3 grid((IMG_SIZE + block.x - 1) / block.x,
                    (IMG_SIZE + block.y - 1) / block.y);
    GpuTimer timer;
    timer.start();
    backproject_kernel<<<grid, block>>>(d_part, d_assign, n, d_recon);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("backproject_kernel");

    CUDA_CHECK(cudaMemcpy(recon.data(), d_recon, recon_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_part));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_recon));
}
