// ===========================================================================
// src/kernels.cu  --  GPU GLCM + histogram kernels and the host wrapper
// ---------------------------------------------------------------------------
// Project 4.27 : Radiomics Feature Extraction
//
// WHAT THIS FILE DOES
//   Implements the two device kernels (histogram_kernel, glcm_kernel) and the
//   host glue (extract_features_gpu) that uploads the volume, launches them,
//   copies the tiny count matrices back, and then reuses the SAME host
//   reductions the CPU uses (first_order_from_histogram, haralick_from_glcm)
//   to turn counts into features. Because the per-voxel math (radiomics.h) and
//   the count->feature reductions are shared, and the counts are INTEGERS built
//   with commutative atomicAdd, the GPU features equal the CPU features exactly.
//
//   This is the GPU twin of build_glcm_cpu()/build_histogram() in
//   reference_cpu.cpp; main.cu runs both paths and compares them.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea), radiomics.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include "radiomics.h"           // rad_quantize, RadDir, rad_fill_directions, indexing

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide memory latency, and many blocks resident
// for occupancy. Here it also bounds the shared-memory GLCM contention.
static constexpr int THREADS_PER_BLOCK = 256;

// Largest gray-level count we support in the shared-memory GLCM. Ng=8 -> 64
// cells; we cap at 16 levels (256 cells = 1 KiB of unsigned ints) so the
// block-private copy always fits comfortably in shared memory on every arch.
static constexpr int MAX_NG = 16;

// The 13 direction offsets live in CONSTANT memory: every thread reads the same
// small table, and constant memory's broadcast cache serves all lanes of a warp
// in one shot -- ideal for a read-only lookup shared by the whole grid.
__constant__ RadDir c_dirs[RAD_NUM_DIRECTIONS];

// ---------------------------------------------------------------------------
// histogram_kernel: one thread per voxel. ROI voxels quantize their intensity
//   and atomic-add 1 into the global gray-level histogram. This is the simplest
//   parallel-histogram: independent voxels, colliding bins -> atomicAdd.
//   Grid  : ceil(nvox / THREADS_PER_BLOCK) ; block : THREADS_PER_BLOCK.
//   Thread (blockIdx.x, threadIdx.x) owns linear voxel v = that global index.
// ---------------------------------------------------------------------------
__global__ void histogram_kernel(const float* __restrict__ intensity,
                                 const uint8_t* __restrict__ mask,
                                 int nx, int ny, int nz, int Ng,
                                 float vmin, float vmax,
                                 unsigned int* __restrict__ hist) {
    const std::size_t nvox = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= nvox) return;                 // guard the ragged last block
    if (!mask[v])  return;                  // only ROI voxels contribute
    // Identical quantization to the CPU (shared radiomics.h) -> identical bins.
    const int g = rad_quantize(intensity[v], vmin, vmax, Ng);
    atomicAdd(&hist[g], 1u);               // integer add -> deterministic
}

// ---------------------------------------------------------------------------
// glcm_kernel: one thread per voxel; PRIVATIZED shared-memory co-occurrence
//   histogram. The core lesson of this project.
//
//   WHY SHARED MEMORY FIRST. If every thread atomic-added straight into the
//   global Ng x Ng matrix, thousands of threads would serialize on the same few
//   cells (a textured ROI still has only 64 cells). Instead each BLOCK keeps its
//   own GLCM copy in fast on-chip shared memory; threads atomic-add there
//   (cheap, low-contention), and at the end ONE flush per cell merges the block
//   copy into global memory. This is the classic "privatization" optimization
//   for histograms -- fewer, cheaper global atomics.
//
//   Thread-to-data map: linear voxel index v = blockIdx.x*blockDim.x+threadIdx.x.
//   For each of the 13 directions, if the neighbour is in-grid AND in the ROI,
//   the thread adds the symmetric pair (gi,gj) and (gj,gi) to the shared GLCM.
//   Counts are integers -> the whole reduction is order-independent and matches
//   the serial CPU build exactly.
// ---------------------------------------------------------------------------
__global__ void glcm_kernel(const float* __restrict__ intensity,
                            const uint8_t* __restrict__ mask,
                            int nx, int ny, int nz, int Ng,
                            float vmin, float vmax,
                            unsigned int* __restrict__ glcm) {
    // Block-private GLCM in shared memory (fixed MAX_NG*MAX_NG, we use Ng*Ng).
    __shared__ unsigned int s_glcm[MAX_NG * MAX_NG];

    // Cooperatively zero the shared matrix: threads stride over its cells.
    const int ncells = Ng * Ng;
    for (int c = threadIdx.x; c < ncells; c += blockDim.x)
        s_glcm[c] = 0u;
    __syncthreads();                        // all cells zeroed before any add

    const std::size_t nvox = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // A thread with an in-range ROI centre scatters its 13-direction pairs.
    // (We do NOT `return` early on out-of-range/off-ROI threads, because they
    //  must still reach the __syncthreads() and the flush below with the block.)
    if (v < nvox && mask[v]) {
        // Recover 3-D coordinates from the linear index (x fastest).
        const int x = static_cast<int>(v % nx);
        const int y = static_cast<int>((v / nx) % ny);
        const int z = static_cast<int>(v / (static_cast<std::size_t>(nx) * ny));
        const int gi = rad_quantize(intensity[v], vmin, vmax, Ng);

        #pragma unroll
        for (int k = 0; k < RAD_NUM_DIRECTIONS; ++k) {
            const int xn = x + c_dirs[k].dx;
            const int yn = y + c_dirs[k].dy;
            const int zn = z + c_dirs[k].dz;
            if (xn < 0 || xn >= nx || yn < 0 || yn >= ny || zn < 0 || zn >= nz)
                continue;                    // neighbour off the grid -> no pair
            const std::size_t nidx = rad_vox_index(xn, yn, zn, nx, ny);
            if (!mask[nidx]) continue;        // neighbour outside ROI -> no pair
            const int gj = rad_quantize(intensity[nidx], vmin, vmax, Ng);

            // Symmetric increment into the block-private matrix.
            atomicAdd(&s_glcm[rad_glcm_index(gi, gj, Ng)], 1u);
            atomicAdd(&s_glcm[rad_glcm_index(gj, gi, Ng)], 1u);
        }
    }

    __syncthreads();                         // all block adds finished

    // FLUSH: merge this block's private GLCM into the global matrix. Threads
    // stride over the cells so the writes are coalesced and balanced.
    for (int c = threadIdx.x; c < ncells; c += blockDim.x) {
        const unsigned int val = s_glcm[c];
        if (val != 0u) atomicAdd(&glcm[c], val);   // only touch non-empty cells
    }
}

// ---------------------------------------------------------------------------
// extract_features_gpu: host wrapper. Uploads the volume, launches the two
//   kernels (timed together with CUDA events), copies the small count matrices
//   back, converts unsigned-int counts to the long-long buffers the shared
//   reductions expect, and reuses first_order_from_histogram + haralick_from_glcm
//   so the GPU features match the CPU exactly.
// ---------------------------------------------------------------------------
Features extract_features_gpu(const Volume& v, float* kernel_ms) {
    const std::size_t nvox = v.voxels();
    const int Ng = v.Ng;
    const int ncells = Ng * Ng;

    // (1) Device buffers for the volume, histogram, and GLCM count matrix.
    float*    d_intensity = nullptr;
    uint8_t*  d_mask      = nullptr;
    unsigned int* d_hist  = nullptr;
    unsigned int* d_glcm  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_intensity, nvox * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask,      nvox * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_hist,      static_cast<std::size_t>(Ng) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_glcm,      static_cast<std::size_t>(ncells) * sizeof(unsigned int)));

    // (2) Upload the volume and zero the accumulators.
    CUDA_CHECK(cudaMemcpy(d_intensity, v.intensity.data(), nvox * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, v.mask.data(), nvox * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, static_cast<std::size_t>(Ng) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_glcm, 0, static_cast<std::size_t>(ncells) * sizeof(unsigned int)));

    // Upload the 13 direction offsets into constant memory (same table as CPU).
    RadDir dirs[RAD_NUM_DIRECTIONS];
    rad_fill_directions(dirs);
    CUDA_CHECK(cudaMemcpyToSymbol(c_dirs, dirs, sizeof(dirs)));

    // (3) Launch both kernels (one thread per voxel), timed together.
    const int blocks = static_cast<int>((nvox + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    GpuTimer timer;
    timer.start();
    histogram_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_intensity, d_mask, v.nx, v.ny, v.nz, Ng, v.vmin, v.vmax, d_hist);
    glcm_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_intensity, d_mask, v.nx, v.ny, v.nz, Ng, v.vmin, v.vmax, d_glcm);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("histogram+glcm kernels");

    // (4) Copy the tiny count matrices back.
    std::vector<unsigned int> hist_u(Ng), glcm_u(ncells);
    CUDA_CHECK(cudaMemcpy(hist_u.data(), d_hist,
                          static_cast<std::size_t>(Ng) * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(glcm_u.data(), d_glcm,
                          static_cast<std::size_t>(ncells) * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_intensity));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_glcm));

    // Widen counts to long long and reuse the SHARED host reductions so the GPU
    // feature vector is computed by the identical code the CPU used.
    std::vector<long long> hist(Ng), glcm(ncells);
    for (int g = 0; g < Ng; ++g) hist[g] = static_cast<long long>(hist_u[g]);
    for (int c = 0; c < ncells; ++c) glcm[c] = static_cast<long long>(glcm_u[c]);

    Features f;
    first_order_from_histogram(v, hist, f);
    haralick_from_glcm(v, glcm, f);
    return f;
}
