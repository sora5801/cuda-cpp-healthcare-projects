// ===========================================================================
// src/kernels.cu  --  GPU marching cubes: count -> scan -> generate
// ---------------------------------------------------------------------------
// Project 4.18 : Image-Based 3D Printing / Model Generation for Surgery
//
// GPU twin of marching_cubes_cpu(): same per-cube math (from mc_core.h), one
// thread per cube, with a DETERMINISTIC parallel prefix-sum in the middle to lay
// out the ragged triangle output. main.cu runs both and checks they agree. See
// ../THEORY.md "GPU mapping" for the full derivation of why this is the right
// structure.
//
// WHY WE HAND-ROLL THE SCAN (instead of calling Thrust/CUB)
//   docs/PATTERNS.md §5 says "use the library for solved problems" -- and in
//   production you absolutely would call thrust::exclusive_scan or
//   cub::DeviceScan. But CLAUDE.md §6 ("no black boxes") asks us to show what
//   that library call computes and what hand-rolling takes. The exclusive scan
//   is THE foundational data-parallel primitive, so here we implement it
//   ourselves with the classic two-level (Blelloch) approach -- it is short,
//   exact on integers (hence deterministic), and the single most transferable
//   thing a learner takes from this project. THEORY.md §"GPU mapping" explains
//   the algorithm; the production library equivalent is noted there too.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cstdio>    // std::fprintf (the "too large" guard message)
#include <cstdlib>   // std::exit, EXIT_FAILURE

// 256 threads/block: a standard 1-D occupancy default on sm_75..sm_89. The cell
// list is 1-D (flattened), so a 1-D block over it is the natural mapping. The
// scan below also uses this block size; SCAN_BLOCK shared elements per block.
static constexpr int BLOCK = 256;

// ---------------------------------------------------------------------------
// cell_to_ijk: turn a flat cell index into its (ci,cj,ck) base corner.
//   The flattening MUST match reference_cpu.cpp's loop order exactly
//       cell = (ck*CY + cj)*CX + ci          (cz outer, cx inner)
//   or the GPU mesh would be a permutation of the CPU one and the vertex-wise
//   comparison would fail. A __device__ helper keeps both kernels consistent.
// ---------------------------------------------------------------------------
__device__ inline void cell_to_ijk(int cell, int CX, int CY,
                                   int& ci, int& cj, int& ck) {
    ci = cell % CX;
    const int rest = cell / CX;
    cj = rest % CY;
    ck = rest / CY;
}

// ---------------------------------------------------------------------------
// count_kernel: PASS 1 -- how many triangles does each cube emit?
//   Each thread owns one cube, loads its 8 corners via the shared
//   gather_corners(), classifies it, and looks up the triangle count. No two
//   threads write the same location (counts[cell] is private), so there are no
//   races and no atomics -- the cleanest possible parallel pass.
// ---------------------------------------------------------------------------
__global__ void count_kernel(const float* __restrict__ vol, VolDims dims,
                             float iso, int n_cells, int* __restrict__ counts) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;                 // guard the ragged last block

    int ci, cj, ck;
    cell_to_ijk(cell, cells_x(dims), cells_y(dims), ci, cj, ck);

    float val[8];
    Vec3  pos[8];
    gather_corners(vol, dims, ci, cj, ck, val, pos);   // shared with CPU
    const int cube_index = classify_cube(val, iso);    // shared with CPU
    counts[cell] = num_tris_for_cube(cube_index);      // 0..5
}

// ---------------------------------------------------------------------------
// THE EXCLUSIVE PREFIX-SUM (the count -> offsets step), hand-rolled.
//
//   Goal: turn counts[0..n) into offsets[0..n) where
//       offsets[i] = counts[0] + counts[1] + ... + counts[i-1]
//   so each cell knows where its first triangle goes. We do it in three kernels,
//   the standard work-efficient two-level scheme:
//
//     (A) block_scan_kernel : each block scans its own chunk of up to BLOCK
//          elements in SHARED memory (a Hillis-Steele inclusive scan, then shift
//          to exclusive), writes the per-element offsets, AND records that
//          block's TOTAL sum into block_sums[blockIdx].
//     (B) scan_block_sums_kernel : a SINGLE block exclusively scans the
//          block_sums array (there are gridDim of them -- few enough that one
//          block handles them serially-in-shared). block_offsets[b] is then the
//          number of triangles in all blocks BEFORE block b.
//     (C) add_block_offsets_kernel : add block_offsets[blockIdx] back into every
//          element of block b, stitching the per-block scans into one global
//          exclusive scan.
//
//   All adds are on INTEGERS, which are associative, so the result is exactly
//   the serial prefix sum -- bit-for-bit, every run (docs/PATTERNS.md §3). This
//   is the same algorithm thrust::exclusive_scan / cub::DeviceScan run for you.
//
//   NOTE: this assumes the block-sums array fits in a single block (gridDim <=
//   BLOCK), i.e. up to BLOCK*BLOCK = 65536 cells. Our samples are far smaller;
//   THEORY.md notes the fully-recursive generalization for clinical volumes.
// ---------------------------------------------------------------------------

// (A) per-block exclusive scan in shared memory.
__global__ void block_scan_kernel(const int* __restrict__ counts, int n,
                                  int* __restrict__ offsets,
                                  int* __restrict__ block_sums) {
    // Shared scratch: one int per thread in the block. Lives only for this
    // block's lifetime; threads cooperate through it.
    __shared__ int s[BLOCK];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    // Load this element (0 past the end so out-of-range lanes contribute nothing).
    s[tid] = (gid < n) ? counts[gid] : 0;
    __syncthreads();

    // Hillis-Steele INCLUSIVE scan: after this loop s[tid] = sum of s[0..tid].
    // Each step adds the element `offset` positions to the left; doubling offset
    // means log2(BLOCK) steps. We double-buffer via a register `v` to avoid a
    // read-after-write hazard within a step.
    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        int v = 0;
        if (tid >= offset) v = s[tid - offset];
        __syncthreads();                 // all reads done before any write
        if (tid >= offset) s[tid] += v;
        __syncthreads();                 // all writes done before next step
    }

    // The block's total = inclusive scan of the last lane.
    if (tid == blockDim.x - 1) block_sums[blockIdx.x] = s[tid];

    // Convert inclusive -> EXCLUSIVE by subtracting this element's own count:
    //   exclusive[tid] = inclusive[tid] - counts[tid].
    if (gid < n) offsets[gid] = s[tid] - ((gid < n) ? counts[gid] : 0);
}

// (B) exclusive scan of the (small) block_sums array in a single block.
//   n_blocks <= BLOCK by the assumption above, so one block scans them all.
__global__ void scan_block_sums_kernel(int* __restrict__ block_sums, int n_blocks) {
    __shared__ int s[BLOCK];
    const int tid = threadIdx.x;
    s[tid] = (tid < n_blocks) ? block_sums[tid] : 0;
    __syncthreads();

    // Same Hillis-Steele inclusive scan as above.
    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        int v = 0;
        if (tid >= offset) v = s[tid - offset];
        __syncthreads();
        if (tid >= offset) s[tid] += v;
        __syncthreads();
    }
    // Write back the EXCLUSIVE result (inclusive minus own value).
    if (tid < n_blocks) {
        int own = block_sums[tid];
        block_sums[tid] = s[tid] - own;   // = sum of all blocks strictly before tid
    }
}

// (C) add each block's offset back into its elements -> global exclusive scan.
__global__ void add_block_offsets_kernel(int* __restrict__ offsets, int n,
                                        const int* __restrict__ block_offsets) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) offsets[gid] += block_offsets[blockIdx.x];
}

// ---------------------------------------------------------------------------
// generate_kernel: PASS 3 -- write each cube's triangles at its scanned offset.
//   offsets[cell] is the exclusive prefix sum of counts, i.e. the index of this
//   cube's FIRST triangle in the compacted output. We re-classify the cube
//   (cheap, avoids stashing state) and emit its triangles sequentially from
//   there. Because offsets are monotonically increasing in `cell`, the global
//   write order matches the CPU's serial order exactly.
// ---------------------------------------------------------------------------
__global__ void generate_kernel(const float* __restrict__ vol, VolDims dims,
                                float iso, int n_cells,
                                const int* __restrict__ offsets,
                                Triangle* __restrict__ out_tris) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    int ci, cj, ck;
    cell_to_ijk(cell, cells_x(dims), cells_y(dims), ci, cj, ck);

    float val[8];
    Vec3  pos[8];
    gather_corners(vol, dims, ci, cj, ck, val, pos);
    const int cube_index = classify_cube(val, iso);

    int w = offsets[cell];                  // first output slot for this cube
    const int* row = TRI_TABLE[cube_index]; // its triangle list (edge triples)
    for (int t = 0; row[t * 3] != -1; ++t) {
        Triangle tri;
        // Three vertices, each interpolated along its edge -- identical math to
        // the CPU reference (interp_edge lives in mc_core.h).
        tri.v[0] = interp_edge(row[t * 3 + 0], pos, val, iso);
        tri.v[1] = interp_edge(row[t * 3 + 1], pos, val, iso);
        tri.v[2] = interp_edge(row[t * 3 + 2], pos, val, iso);
        out_tris[w++] = tri;
    }
}

// ---------------------------------------------------------------------------
// marching_cubes_gpu: orchestrate count -> scan -> generate and return the mesh.
//   Memory plan (all device-side):
//     d_vol      : the uploaded scalar volume (nx*ny*nz floats)
//     d_counts   : per-cell triangle count            (n_cells ints)
//     d_offsets  : exclusive prefix sum of counts      (n_cells ints)
//     d_blocksum : one running sum per block           (n_blocks ints)
//     d_tris     : the compacted output mesh           (total_tris Triangles)
//   We time the two MC kernels + the three scan kernels with CUDA events (the
//   H2D upload and the final D2H copy are excluded so the figure reflects
//   compute, like flagship 4.01).
// ---------------------------------------------------------------------------
void marching_cubes_gpu(const MCProblem& prob, std::vector<Triangle>& out,
                        float* kernel_ms) {
    const VolDims& d = prob.dims;
    const int n_cells = num_cells(d);
    const std::size_t n_samples = static_cast<std::size_t>(d.nx) * d.ny * d.nz;
    const int n_blocks = (n_cells + BLOCK - 1) / BLOCK;

    // Safety: the single-block block-sum scan assumes n_blocks <= BLOCK. Our
    // committed sample is tiny; this guards a learner who points it at a huge
    // volume (THEORY.md gives the recursive fix). Fail loud, not silently wrong.
    if (n_blocks > BLOCK) {
        std::fprintf(stderr, "[error] volume too large for the single-level scan "
                     "(%d blocks > %d); see THEORY.md for the recursive version.\n",
                     n_blocks, BLOCK);
        std::exit(EXIT_FAILURE);
    }

    // ---- upload the volume + allocate the per-cell scratch ----------------
    float* d_vol = nullptr;
    int*   d_counts = nullptr;
    int*   d_offsets = nullptr;
    int*   d_blocksum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vol, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, (size_t)n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blocksum, (size_t)n_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_vol, prob.vol.data(), n_samples * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block(BLOCK);
    const dim3 grid(n_blocks);

    GpuTimer timer;
    timer.start();

    // ---- PASS 1: count triangles per cell --------------------------------
    count_kernel<<<grid, block>>>(d_vol, d, prob.iso, n_cells, d_counts);
    CUDA_CHECK_LAST("count_kernel");

    // ---- PASS 2: exclusive scan counts -> offsets (3 sub-kernels) ---------
    block_scan_kernel<<<grid, block>>>(d_counts, n_cells, d_offsets, d_blocksum);
    CUDA_CHECK_LAST("block_scan_kernel");
    scan_block_sums_kernel<<<1, block>>>(d_blocksum, n_blocks);
    CUDA_CHECK_LAST("scan_block_sums_kernel");
    add_block_offsets_kernel<<<grid, block>>>(d_offsets, n_cells, d_blocksum);
    CUDA_CHECK_LAST("add_block_offsets_kernel");

    // Total triangle count = offset of last cell + its own count. We read just
    // those two ints back (cheap) to size the output buffer.
    int last_offset = 0, last_count = 0;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_offsets + (n_cells - 1), sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_count, d_counts + (n_cells - 1), sizeof(int),
                          cudaMemcpyDeviceToHost));
    const int total_tris = last_offset + last_count;

    // ---- PASS 3: generate triangles into the compacted buffer -------------
    Triangle* d_tris = nullptr;
    if (total_tris > 0) {
        CUDA_CHECK(cudaMalloc(&d_tris, (size_t)total_tris * sizeof(Triangle)));
        generate_kernel<<<grid, block>>>(d_vol, d, prob.iso, n_cells,
                                         d_offsets, d_tris);
        CUDA_CHECK_LAST("generate_kernel");
    }

    *kernel_ms = timer.stop_ms();   // counts both MC kernels + the 3 scan kernels

    // ---- copy the mesh back to the host ----------------------------------
    out.assign(total_tris, Triangle{});
    if (total_tris > 0) {
        CUDA_CHECK(cudaMemcpy(out.data(), d_tris,
                              (size_t)total_tris * sizeof(Triangle),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_tris));
    }

    CUDA_CHECK(cudaFree(d_vol));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_blocksum));
}
