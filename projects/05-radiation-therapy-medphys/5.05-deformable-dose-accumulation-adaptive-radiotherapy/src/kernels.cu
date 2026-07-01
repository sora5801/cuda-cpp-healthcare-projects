// ===========================================================================
// src/kernels.cu  --  GPU Demons DIR + deformable dose warp / accumulate / DVH
// ---------------------------------------------------------------------------
// Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
//               (reduced-scope 2-D teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of reference_cpu.cpp. It implements two host wrappers and their
//   device kernels:
//     register_gpu()        -- Stage A: Thirion's Demons DIR (three kernels in a
//                              loop, ping-pong buffers) -> the DVF (ux,uy).
//     accumulate_dose_gpu() -- Stage B: warp the delivered dose by the DVF, sum
//                              deformed doses into a total, and histogram it.
//
//   Every per-voxel formula (warp, gradient, Thirion force, Gaussian weights,
//   dose gather, DVH bin) comes from demons.h / dose.h and is the SAME code the
//   CPU reference runs, so the two sides agree to floating-point rounding (DIR /
//   warped dose) and EXACTLY (the integer DVH). Tolerances live in ../THEORY.md.
//
// READ THIS AFTER: demons.h, dose.h, kernels.cuh, reference_cpu.cpp (the mirror).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// A 16x16 tile = 256 threads per block over the 2-D image. 256 is a multiple of
// the 32-lane warp and gives the scheduler 8 warps per block to hide the global-
// memory latency of the neighbourhood reads. The 2-D block shape mirrors the 2-D
// image so the thread-to-pixel mapping is trivial (see below).
static constexpr int TILE = 16;

// A flat 256-thread block for the 1-D per-voxel kernels (accumulate, DVH). One
// warp-friendly block size reused so occupancy reasoning is uniform.
static constexpr int FLAT = 256;

// ===========================================================================
// STAGE A -- Demons DIR kernels (identical algorithm to project 4.8)
// ===========================================================================

// ---------------------------------------------------------------------------
// demons_force_kernel  (PASS 1 of a Demons iteration)
//   block = (TILE,TILE); grid = ceil(nx/TILE) x ceil(ny/TILE) -> covers every px.
//   Thread (blockIdx,threadIdx) owns pixel (x,y). Reads F,M and the current ux,uy;
//   writes ux[i],uy[i]. NO atomics / NO shared memory: dm_demons_force reads only
//   u[i] at THIS pixel (plus the images), so each thread updates a distinct output
//   element -> the write is race-free by construction.
// ---------------------------------------------------------------------------
__global__ void demons_force_kernel(const double* __restrict__ F,
                                    const double* __restrict__ M,
                                    double* __restrict__ ux,
                                    double* __restrict__ uy,
                                    DemonsParams P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's row
    if (x >= P.nx || y >= P.ny) return;                    // guard ragged edges

    const int i = y * P.nx + x;
    double dux, duy;
    dm_demons_force(F, M, ux, uy, x, y, P, &dux, &duy);    // the one true update
    ux[i] += dux;   // add the step to this pixel's displacement (in place, safe)
    uy[i] += duy;
}

// ---------------------------------------------------------------------------
// gauss_x_kernel  (PASS 2: horizontal half of the separable Gaussian)
//   Each thread reads a (2*radius+1)-wide row window of `src` around its pixel and
//   writes the blurred value to `dst`. src and dst MUST differ (ping-pong) so no
//   thread reads a value another thread is overwriting -- the classic double-
//   buffered stencil (cf. 6.04 / 14.02).
// ---------------------------------------------------------------------------
__global__ void gauss_x_kernel(const double* __restrict__ src,
                               double* __restrict__ dst,
                               DemonsParams P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.nx || y >= P.ny) return;
    dst[y * P.nx + x] = dm_gauss_x(src, x, y, P.nx, P.ny, P.sigma, P.radius);
}

// ---------------------------------------------------------------------------
// gauss_y_kernel  (PASS 3: vertical half of the separable Gaussian)
//   Mirror of gauss_x_kernel along the other axis. After PASS 2 then PASS 3, the
//   displacement component has been convolved with a full 2-D Gaussian in
//   O(radius) work per pixel instead of O(radius^2).
// ---------------------------------------------------------------------------
__global__ void gauss_y_kernel(const double* __restrict__ src,
                               double* __restrict__ dst,
                               DemonsParams P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.nx || y >= P.ny) return;
    dst[y * P.nx + x] = dm_gauss_y(src, x, y, P.nx, P.ny, P.sigma, P.radius);
}

// ===========================================================================
// STAGE B -- deformable dose warp, accumulation, and DVH histogram (NEW in 5.5)
// ===========================================================================

// ---------------------------------------------------------------------------
// warp_dose_kernel  (deformable dose warp)
//   block = (TILE,TILE); grid covers every voxel. Thread (x,y) GATHERS the
//   delivered dose at its deformed position (x+ux,y+uy) via the shared bilinear
//   sampler (warp_dose_at) and writes out[i]. Pure gather: each thread writes its
//   own out[i] -> no atomics, no races (cf. 4.01 CT backprojection).
// ---------------------------------------------------------------------------
__global__ void warp_dose_kernel(const double* __restrict__ dose,
                                 const double* __restrict__ ux,
                                 const double* __restrict__ uy,
                                 double* __restrict__ out,
                                 int nx, int ny) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    out[y * nx + x] = warp_dose_at(dose, ux, uy, x, y, nx, ny);
}

// ---------------------------------------------------------------------------
// accumulate_kernel  (summation of deformed doses)
//   Flat 1-D grid over the nx*ny voxels. total[i] += add[i]. Each thread owns a
//   distinct index i, so the read-modify-write is race-free WITHOUT atomics.
//   Running this once per fraction sums the deformed doses in the planning frame.
// ---------------------------------------------------------------------------
__global__ void accumulate_kernel(double* __restrict__ total,
                                  const double* __restrict__ add,
                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    total[i] += add[i];
}

// ---------------------------------------------------------------------------
// dvh_kernel  (dose-volume histogram via INTEGER atomics)
//   Flat 1-D grid over the voxels. Each thread computes its dose bin (dvh_bin,
//   shared with the CPU) and does atomicAdd(&hist[bin], 1u). Many threads hit the
//   same bin -> a reduction. We add INTEGERS: integer atomicAdd is associative and
//   commutative, so the counts are identical regardless of thread order, and match
//   the serial CPU histogram exactly. A FLOAT atomic sum here would reorder and
//   make the histogram (and thus stdout) non-reproducible (PATTERNS.md §3).
// ---------------------------------------------------------------------------
__global__ void dvh_kernel(const double* __restrict__ dose,
                           unsigned* __restrict__ hist,
                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int b = dvh_bin(dose[i]);         // same binning as the CPU reference
    atomicAdd(&hist[b], 1u);                // integer atomic -> deterministic
}

// ===========================================================================
// HOST WRAPPERS
// ===========================================================================

// ---------------------------------------------------------------------------
// register_gpu: Stage A -- run the full Demons solver on the GPU (five canonical
//   CUDA steps wrapped around the iteration loop). Structurally identical to the
//   DIR flagship (4.8): the whole solver stays resident on the device so there is
//   NO host<->device traffic inside the loop -- the point of GPU DIR.
//
//   BUFFER BOOKKEEPING (two buffers per component so the separable Gaussian never
//   reads a half-written field):
//     d_ux / d_uy    hold the CURRENT displacement field.
//     d_ux2 / d_uy2  are scratch for the Gaussian.
//   Per iteration: force writes d_ux/d_uy in place; smooth-x reads d_ux->d_ux2
//   (and d_uy->d_uy2); smooth-y reads d_ux2->d_ux (and d_uy2->d_uy), so after the
//   Y pass the smoothed field is back in d_ux/d_uy, ready for the next force pass.
// ---------------------------------------------------------------------------
void register_gpu(const ArtCase& c, const DemonsParams& P,
                  std::vector<double>& ux, std::vector<double>& uy,
                  float* kernel_ms) {
    const std::size_t N     = static_cast<std::size_t>(c.nx) * c.ny;
    const std::size_t bytes = N * sizeof(double);

    // (1) Device buffers. d_ = DEVICE pointer (CLAUDE.md §12).
    double *d_F = nullptr, *d_M = nullptr;
    double *d_ux = nullptr, *d_uy = nullptr;     // current displacement field
    double *d_ux2 = nullptr, *d_uy2 = nullptr;   // Gaussian scratch (ping-pong)
    CUDA_CHECK(cudaMalloc(&d_F,   bytes));       // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_M,   bytes));
    CUDA_CHECK(cudaMalloc(&d_ux,  bytes));
    CUDA_CHECK(cudaMalloc(&d_uy,  bytes));
    CUDA_CHECK(cudaMalloc(&d_ux2, bytes));
    CUDA_CHECK(cudaMalloc(&d_uy2, bytes));

    // (2) Upload the images once (plan_img=FIXED, daily_img=MOVING). The field
    //     starts at ZERO (identity map); an all-zero-bytes IEEE-754 double is
    //     exactly +0.0, so cudaMemset(...,0,...) is a correct way to zero doubles.
    CUDA_CHECK(cudaMemcpy(d_F, c.plan_img.data(),  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, c.daily_img.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_ux, 0, bytes));
    CUDA_CHECK(cudaMemset(d_uy, 0, bytes));

    // Launch geometry: a 2-D grid of 16x16 blocks covering the image (ceiling
    // division so partial edge tiles still get a block; kernels guard the range).
    dim3 block(TILE, TILE);
    dim3 grid((c.nx + TILE - 1) / TILE, (c.ny + TILE - 1) / TILE);

    // (3) The Demons iteration loop, timed as one unit.
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < P.iters; ++it) {
        demons_force_kernel<<<grid, block>>>(d_F, d_M, d_ux, d_uy, P);   // PASS 1
        gauss_x_kernel<<<grid, block>>>(d_ux, d_ux2, P);                 // PASS 2
        gauss_x_kernel<<<grid, block>>>(d_uy, d_uy2, P);
        gauss_y_kernel<<<grid, block>>>(d_ux2, d_ux, P);                 // PASS 3
        gauss_y_kernel<<<grid, block>>>(d_uy2, d_uy, P);
    }
    *kernel_ms = timer.stop_ms();          // GPU-measured loop time
    CUDA_CHECK_LAST("demons iteration");   // catch any launch/execution error

    // (4) Bring the final displacement field back to the host.
    ux.resize(N);
    uy.resize(N);
    CUDA_CHECK(cudaMemcpy(ux.data(), d_ux, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy.data(), d_uy, bytes, cudaMemcpyDeviceToHost));

    // (5) Free everything (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_ux));
    CUDA_CHECK(cudaFree(d_uy));
    CUDA_CHECK(cudaFree(d_ux2));
    CUDA_CHECK(cudaFree(d_uy2));
}

// ---------------------------------------------------------------------------
// accumulate_dose_gpu: Stage B end to end on the GPU.
//   Steps:
//     (1) upload the delivered daily dose + the DVF; allocate total (zeroed),
//         a per-fraction warped-dose scratch, and the DVH histogram (zeroed).
//     (2) warp the delivered dose ONCE (the DVF is fixed across the identical
//         fractions in this demo) -> d_warp.
//     (3) accumulate d_warp into d_total `nfractions` times (summation of
//         deformed doses). Re-running the accumulate kernel is the teaching-clear
//         way to show fraction-by-fraction accumulation; a real course has a
//         DIFFERENT DVF (and dose) per fraction, so it would re-warp each time.
//     (4) histogram d_total with integer atomics -> the DVH.
//     (5) copy total + DVH back; free.
//   We CUDA-event-time steps (2)-(4) -- the compute -- not the one-time copies.
// ---------------------------------------------------------------------------
void accumulate_dose_gpu(const ArtCase& c,
                         const std::vector<double>& ux,
                         const std::vector<double>& uy,
                         int nfractions,
                         std::vector<double>& total_out,
                         std::vector<unsigned>& dvh_out,
                         float* kernel_ms) {
    const std::size_t N     = static_cast<std::size_t>(c.nx) * c.ny;
    const std::size_t bytes = N * sizeof(double);
    const int n             = static_cast<int>(N);

    // (1) Allocate + upload.
    double *d_dose = nullptr, *d_ux = nullptr, *d_uy = nullptr;
    double *d_warp = nullptr, *d_total = nullptr;
    unsigned *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dose,  bytes));
    CUDA_CHECK(cudaMalloc(&d_ux,    bytes));
    CUDA_CHECK(cudaMalloc(&d_uy,    bytes));
    CUDA_CHECK(cudaMalloc(&d_warp,  bytes));
    CUDA_CHECK(cudaMalloc(&d_total, bytes));
    CUDA_CHECK(cudaMalloc(&d_hist,  DVH_BINS * sizeof(unsigned)));

    CUDA_CHECK(cudaMemcpy(d_dose, c.daily_dose.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ux,   ux.data(),           bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uy,   uy.data(),           bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_total, 0, bytes));                        // total = 0
    CUDA_CHECK(cudaMemset(d_hist,  0, DVH_BINS * sizeof(unsigned)));  // hist  = 0

    // Launch geometry: 2-D grid for the warp (image-shaped), 1-D for the flat
    // per-voxel kernels (accumulate, DVH).
    dim3 block2d(TILE, TILE);
    dim3 grid2d((c.nx + TILE - 1) / TILE, (c.ny + TILE - 1) / TILE);
    const int gridFlat = (n + FLAT - 1) / FLAT;

    GpuTimer timer;
    timer.start();
    // (2) Warp the delivered dose into the planning frame (once).
    warp_dose_kernel<<<grid2d, block2d>>>(d_dose, d_ux, d_uy, d_warp, c.nx, c.ny);
    // (3) Accumulate it nfractions times (summation of deformed doses).
    for (int f = 0; f < nfractions; ++f)
        accumulate_kernel<<<gridFlat, FLAT>>>(d_total, d_warp, n);
    // (4) Histogram the accumulated total (integer atomics -> deterministic).
    dvh_kernel<<<gridFlat, FLAT>>>(d_total, d_hist, n);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("dose accumulation");

    // (5) Copy results back + free.
    total_out.resize(N);
    dvh_out.resize(DVH_BINS);
    CUDA_CHECK(cudaMemcpy(total_out.data(), d_total, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvh_out.data(), d_hist,
                          DVH_BINS * sizeof(unsigned), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dose));
    CUDA_CHECK(cudaFree(d_ux));
    CUDA_CHECK(cudaFree(d_uy));
    CUDA_CHECK(cudaFree(d_warp));
    CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaFree(d_hist));
}
