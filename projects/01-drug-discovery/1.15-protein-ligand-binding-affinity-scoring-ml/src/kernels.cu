// ===========================================================================
// src/kernels.cu  --  Batched 3D-CNN affinity scoring kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
//
// WHAT THIS FILE DOES
//   Implements the GPU twin of score_cpu() in reference_cpu.cpp. main.cu runs
//   both and asserts they agree. The network forward pass for ONE complex has
//   four stages (voxelize -> conv+ReLU -> global-average pool -> dense readout);
//   we assign ONE THREAD BLOCK to each complex and let its threads cooperate
//   over the GRID^3 voxels (the stencil pattern). See ../THEORY.md "GPU mapping".
//
//   The per-element math (atom_contrib, channel_of, grid_index, relu,
//   lcg_weight) is the SAME code the CPU calls, pulled from scoring_core.h via
//   the HD-macro idiom -- that is what makes the two results agree (PATTERNS.md
//   sec.2). Only the LOOP STRUCTURE and the parallel reduction differ here.
//
// READ THIS AFTER: kernels.cuh, scoring_core.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cmath>                 // exp (host side of scoring_core's helpers)

// Threads per block. 128 is a good fit here: each block processes ONE complex
// and its threads stride over GRID^3 = 4096 voxels (so ~32 voxels/thread), a
// multiple of the 32-lane warp, with enough warps to hide global-memory latency
// while keeping the shared-memory pooling reduction small. (THEORY "GPU mapping"
// discusses the occupancy trade-off vs. 256.)
static constexpr int BLOCK = 128;

// ---------------------------------------------------------------------------
// voxelize_pass: deposit every atom of THIS block's complex into its density
//   grid, as a GATHER (each thread owns a stride of voxels and pulls in the atom
//   contributions) rather than a scatter (atoms writing voxels). Why gather?
//   A scatter needs atomics to avoid write races, and atomic double-adds sum in a
//   nondeterministic order -> irreproducible results (PATTERNS.md sec.3). A gather
//   has each voxel written by exactly one thread, so it is race-free AND
//   deterministic AND matches the CPU's per-voxel sum order term-for-term.
//
//   grid : [GRID_SIZE] this complex's grid (in global memory, already zeroed)
//   atoms: the block's atoms (a0..a1 in the global flat array)
// ---------------------------------------------------------------------------
__device__ void voxelize_pass(double* __restrict__ grid,
                              const Atom* __restrict__ atoms, int a0, int a1) {
    // Each thread walks a strided subset of the CIN*GRID^3 grid elements.
    for (int idx = threadIdx.x; idx < GRID_SIZE; idx += blockDim.x) {
        // Decode the flat index back into (channel c, voxel x,y,z). This is the
        // exact inverse of grid_index() so reads and writes stay consistent.
        const int x = idx % GRID;
        const int y = (idx / GRID) % GRID;
        const int z = (idx / (GRID * GRID)) % GRID;
        const int c = idx / VOX_PER_CH;
        // This voxel center's physical coordinates (angstroms).
        const double vx = (x + 0.5) * VOX_A;
        const double vy = (y + 0.5) * VOX_A;
        const double vz = (z + 0.5) * VOX_A;
        double acc = 0.0;
        // Sum the Gaussian contribution of every atom that lands in THIS channel.
        // We loop atoms in increasing index -> identical summation order to the
        // CPU's per-voxel accumulation (it also sums atoms in array order).
        for (int a = a0; a < a1; ++a) {
            const Atom at = atoms[a];
            if (channel_of(at.type, at.is_ligand) != c) continue;  // wrong channel
            const double ddx = vx - at.x, ddy = vy - at.y, ddz = vz - at.z;
            acc += atom_contrib(ddx * ddx + ddy * ddy + ddz * ddz);
        }
        grid[idx] = acc;   // one writer per voxel -> race-free, deterministic
    }
}

// ---------------------------------------------------------------------------
// score_kernel: one BLOCK scores one complex (blockIdx.x). Stages:
//   (1) voxelize_pass fills the block's grid (gather).
//   (2) __syncthreads(): the conv reads neighbor voxels, so the WHOLE grid must
//       be written before any thread starts the convolution.
//   (3) conv + ReLU + partial pool: each thread sweeps its strided voxels; for
//       each it computes all COUT conv responses, ReLUs them, and adds into its
//       OWN per-output-channel partial sums (registers/local). No cross-thread
//       sharing here, so still deterministic per thread.
//   (4) tree reduction in shared memory: combine the per-thread partials into the
//       COUT pooled sums. A binary-tree reduction has a FIXED combination order,
//       so the block's pooled value is reproducible run-to-run (it differs from
//       the CPU's flat left-to-right sum only by ~1e-12 rounding -> we verify to a
//       documented 1e-6 tolerance, see main.cu / THEORY "How we verify").
//   (5) thread 0 finishes the dense readout and the pKd squash, writes out[blk].
//
//   grids : [n * GRID_SIZE] scratch, one grid slice per block (in global memory).
//   out   : [n] predicted pKd per complex.
// ---------------------------------------------------------------------------
__global__ void score_kernel(const Atom* __restrict__ atoms,
                            const int* __restrict__ offset,
                            double* __restrict__ grids,
                            double* __restrict__ out, int n) {
    const int blk = blockIdx.x;
    if (blk >= n) return;                              // ragged guard (n blocks)

    // This block's private grid slice and atom range.
    double* grid = grids + static_cast<std::size_t>(blk) * GRID_SIZE;
    const int a0 = offset[blk], a1 = offset[blk + 1];

    // --- (1) voxelize ------------------------------------------------------
    voxelize_pass(grid, atoms, a0, a1);
    __syncthreads();                                   // (2) grid fully written

    // --- (3) conv + ReLU, accumulate per-thread pooled partials ------------
    // partial[oc] = sum over THIS thread's voxels of ReLU(conv response oc).
    double partial[COUT];
    for (int oc = 0; oc < COUT; ++oc) partial[oc] = 0.0;

    for (int v = threadIdx.x; v < VOX_PER_CH; v += blockDim.x) {
        // Decode the output voxel coordinate (output channels share a voxel grid).
        const int x = v % GRID;
        const int y = (v / GRID) % GRID;
        const int z = (v / (GRID * GRID)) % GRID;
        // For each output map, convolve the KSZ^3 neighborhood over all input ch.
        for (int oc = 0; oc < COUT; ++oc) {
            double s = 0.0;
            for (int ic = 0; ic < CIN; ++ic) {
                for (int dz = 0; dz < KSZ; ++dz) {
                    const int zz = z + dz - KSZ / 2;       // SAME padding
                    if (zz < 0 || zz >= GRID) continue;
                    for (int dy = 0; dy < KSZ; ++dy) {
                        const int yy = y + dy - KSZ / 2;
                        if (yy < 0 || yy >= GRID) continue;
                        for (int dx = 0; dx < KSZ; ++dx) {
                            const int xx = x + dx - KSZ / 2;
                            if (xx < 0 || xx >= GRID) continue;
                            // Weight index identical to reference_cpu.cpp.
                            const uint64_t widx =
                                (((static_cast<uint64_t>(oc) * CIN + ic) * KSZ + dz) * KSZ + dy) * KSZ + dx;
                            s += lcg_weight(widx) * grid[grid_index(ic, xx, yy, zz)];
                        }
                    }
                }
            }
            partial[oc] += relu(s);                        // ReLU then pool-accumulate
        }
    }

    // --- (4) tree-reduce the per-thread partials into pooled[oc] -----------
    // Shared memory: BLOCK lanes x COUT channels of double. We reduce one output
    // channel at a time to keep the shared footprint to BLOCK doubles.
    __shared__ double red[BLOCK];
    __shared__ double pooled[COUT];
    for (int oc = 0; oc < COUT; ++oc) {
        red[threadIdx.x] = partial[oc];
        __syncthreads();
        // Standard halving tree reduction: a fixed, deterministic addition order.
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) pooled[oc] = red[0] / static_cast<double>(VOX_PER_CH);
        __syncthreads();   // pooled[oc] visible before the next oc overwrites red
    }

    // --- (5) dense readout + pKd squash (thread 0 only) --------------------
    if (threadIdx.x == 0) {
        double raw = lcg_weight(static_cast<uint64_t>(WCONV_SIZE) + COUT);  // bias
        for (int oc = 0; oc < COUT; ++oc)
            raw += lcg_weight(static_cast<uint64_t>(WCONV_SIZE) + oc) * pooled[oc];
        const double squashed = 1.0 / (1.0 + exp(-raw));   // logistic -> (0,1)
        out[blk] = 2.0 + 9.0 * squashed;                   // map to pKd in [2,11]
    }
}

// ---------------------------------------------------------------------------
// score_gpu: host wrapper. The canonical CUDA steps, batched over complexes:
//   (1) upload the ragged atom array + CSR offsets,
//   (2) allocate one grid slice PER COMPLEX (scratch) + the output,
//   (3) launch <<<n, BLOCK>>> (one block per complex), timed with CUDA events,
//   (4) copy the per-complex pKd back, (5) free.
//
//   We time ONLY the kernel (not the H2D/D2H copies) so the figure is the compute
//   cost; THEORY discusses why on a tiny batch the copies + launch dominate.
//
//   NOTE on the grid scratch: GRID_SIZE doubles per complex = 8*4096*8 = 256 KB.
//   For a real million-pose batch you would NOT allocate one grid per pose; you
//   would stream poses through a fixed pool of grids (one per resident block).
//   Here n is tiny, so the simple one-slice-per-complex layout is clearest
//   (THEORY "Where this sits in the real world").
// ---------------------------------------------------------------------------
void score_gpu(const ComplexSet& cs, std::vector<double>& out, float* kernel_ms) {
    const int n = cs.n;
    out.assign(static_cast<std::size_t>(n), 0.0);

    const std::size_t atom_bytes  = cs.atoms.size() * sizeof(Atom);
    const std::size_t off_bytes   = cs.offset.size() * sizeof(int);
    const std::size_t out_bytes   = static_cast<std::size_t>(n) * sizeof(double);
    const std::size_t grids_bytes = static_cast<std::size_t>(n) * GRID_SIZE * sizeof(double);

    // (1)+(2) Allocate device buffers (d_ = device pointer; CLAUDE.md sec.12).
    Atom*   d_atoms = nullptr;
    int*    d_off   = nullptr;
    double* d_grids = nullptr;
    double* d_out   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_atoms, atom_bytes));
    CUDA_CHECK(cudaMalloc(&d_off,   off_bytes));
    CUDA_CHECK(cudaMalloc(&d_grids, grids_bytes));     // scratch grids (zeroed below)
    CUDA_CHECK(cudaMalloc(&d_out,   out_bytes));
    CUDA_CHECK(cudaMemcpy(d_atoms, cs.atoms.data(), atom_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off,   cs.offset.data(), off_bytes, cudaMemcpyHostToDevice));
    // The voxelizer OVERWRITES every grid voxel it owns, but the gather only
    // visits voxels near atoms... actually it visits ALL voxels (one per thread),
    // writing acc (0 where no atom is near). So no pre-zeroing is required; we do
    // it anyway as a belt-and-suspenders guard against any future partial-write.
    CUDA_CHECK(cudaMemset(d_grids, 0, grids_bytes));

    // (3) Launch: one block per complex. The grid-stride loops inside the kernel
    //     let BLOCK threads cover GRID^3 voxels regardless of BLOCK.
    GpuTimer timer;
    timer.start();
    score_kernel<<<n, BLOCK>>>(d_atoms, d_off, d_grids, d_out, n);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("score_kernel");

    // (4) Result D2H, then (5) free.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_atoms));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_grids));
    CUDA_CHECK(cudaFree(d_out));
}
