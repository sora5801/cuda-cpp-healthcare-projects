// ===========================================================================
// src/kernels.cu  --  Frangi vesselness kernel (map pattern) + host wrapper
// ---------------------------------------------------------------------------
// Project 4.26 : Vessel Segmentation & Centerline Extraction
//
// WHAT THIS FILE DOES
//   GPU twin of vesselness_cpu(): one thread per voxel computes the local
//   Hessian (finite differences over the 3x3x3 neighbourhood), its three
//   eigenvalues (closed-form symmetric 3x3, from frangi.h), and the Frangi
//   vesselness score. Identical per-voxel math to the CPU path -> exact-ish
//   verification (~1e-9). main.cu runs both and compares.
//
//   Why one thread per voxel and not shared-memory tiling? The 3x3x3 stencil
//   re-reads each neighbour up to 27 times; a tiled kernel that stages a block's
//   halo into shared memory would cut global traffic. We keep the simple version
//   for teaching and describe the tiled optimization in THEORY (GPU mapping).
//
// READ THIS AFTER: kernels.cuh, frangi.h.  Compare against reference_cpu.cpp.
// ===========================================================================
#include "kernels.cuh"
#include "frangi.h"              // vox_idx, eig_sym3, sort_abs3, frangi_response
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 8x8x4 = 256 threads/block: a multiple of the 32-lane warp, 3-D so the block
// maps onto the 3-D volume, and small in z (volumes are usually thin in z, and
// the x-fastest layout keeps a warp's global reads coalesced along x).
static constexpr int TX = 8, TY = 8, TZ = 4;

// ---------------------------------------------------------------------------
// clampi_dev: device clamp-to-edge index helper (mirrors the host clampi()).
//   Kept tiny and branch-light; used for border voxels so the finite-difference
//   stencil never reads out of bounds.
// ---------------------------------------------------------------------------
__device__ inline int clampi_dev(int v, int hi) {
    return v < 0 ? 0 : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// vesselness_kernel: thread (x,y,z) owns voxel (x,y,z).
//   Grid : ceil(nx/TX) x ceil(ny/TY) x ceil(nz/TZ) blocks
//   Block: TX x TY x TZ threads
//   Memory: reads the 3x3x3 neighbourhood of `s` from global memory; writes one
//   `vness` element. No shared memory / atomics -> voxels are independent.
// ---------------------------------------------------------------------------
__global__ void vesselness_kernel(int nx, int ny, int nz,
                                  const float* __restrict__ s,
                                  FrangiParams fp,
                                  float* __restrict__ vness) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;   // guard ragged edge blocks

    // A local read helper with clamp-to-edge borders (identical to the host's
    // hessian_at V()). Lambda so the finite-difference formulae read cleanly.
    auto V = [&](int xi, int yi, int zi) -> double {
        return s[vox_idx(clampi_dev(xi, nx - 1), clampi_dev(yi, ny - 1),
                         clampi_dev(zi, nz - 1), nx, ny)];
    };
    const double c = V(x, y, z);

    // --- Hessian by central finite differences (voxel spacing = 1) ----------
    const double h00 = V(x + 1, y, z) - 2.0 * c + V(x - 1, y, z);  // d^2/dx^2
    const double h11 = V(x, y + 1, z) - 2.0 * c + V(x, y - 1, z);  // d^2/dy^2
    const double h22 = V(x, y, z + 1) - 2.0 * c + V(x, y, z - 1);  // d^2/dz^2
    const double h01 = (V(x + 1, y + 1, z) - V(x - 1, y + 1, z)
                      - V(x + 1, y - 1, z) + V(x - 1, y - 1, z)) * 0.25;
    const double h02 = (V(x + 1, y, z + 1) - V(x - 1, y, z + 1)
                      - V(x + 1, y, z - 1) + V(x - 1, y, z - 1)) * 0.25;
    const double h12 = (V(x, y + 1, z + 1) - V(x, y - 1, z + 1)
                      - V(x, y + 1, z - 1) + V(x, y - 1, z - 1)) * 0.25;

    // --- Eigenvalues -> sort by magnitude -> Frangi score -------------------
    double e0, e1, e2;
    eig_sym3(h00, h01, h02, h11, h12, h22, e0, e1, e2);
    double l1 = e0, l2 = e1, l3 = e2;
    sort_abs3(l1, l2, l3);
    vness[vox_idx(x, y, z, nx, ny)] = (float)frangi_response(l1, l2, l3, fp);
}

// ---------------------------------------------------------------------------
// vesselness_gpu: host wrapper. Canonical CUDA steps:
//   (1) allocate device buffers  (2) upload the smoothed volume
//   (3) launch the kernel over a 3-D grid (timed with CUDA events)
//   (4) download the score field (5) free device memory
// We time ONLY the kernel so the figure is compute cost, not PCIe transfer.
// ---------------------------------------------------------------------------
void vesselness_gpu(const Volume& s, const FrangiParams& fp,
                    std::vector<float>& vness, float* kernel_ms) {
    const std::size_t n = s.size();
    const std::size_t bytes = n * sizeof(float);
    vness.assign(n, 0.0f);

    // (1) Device buffers. d_ prefix marks DEVICE pointers (CLAUDE.md section 12).
    float *d_s = nullptr, *d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_s, bytes));    // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_v, bytes));

    // (2) Upload the already-smoothed volume.
    CUDA_CHECK(cudaMemcpy(d_s, s.data.data(), bytes, cudaMemcpyHostToDevice));

    // (3) Launch: a 3-D grid that tiles the volume; round each axis UP so every
    //     voxel is covered even when a dimension is not a multiple of the tile.
    dim3 block(TX, TY, TZ);
    dim3 grid((s.nx + TX - 1) / TX, (s.ny + TY - 1) / TY, (s.nz + TZ - 1) / TZ);
    GpuTimer timer;
    timer.start();
    vesselness_kernel<<<grid, block>>>(s.nx, s.ny, s.nz, d_s, fp, d_v);
    *kernel_ms = timer.stop_ms();           // GPU-measured kernel time
    CUDA_CHECK_LAST("vesselness_kernel");   // catch launch + execution errors

    // (4) Download the score field.
    CUDA_CHECK(cudaMemcpy(vness.data(), d_v, bytes, cudaMemcpyDeviceToHost));

    // (5) Free (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_v));
}
