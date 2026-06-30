// ===========================================================================
// src/kernels.cu  --  GPU kernels + host wrapper for inverse-folding design
// ---------------------------------------------------------------------------
// Project 2.10 : Protein Design / Inverse Folding Inference
//
// This is the GPU twin of design_cpu() in reference_cpu.cpp. main.cu runs both
// and asserts they agree EXACTLY (integer math, shared scoring core). The two
// kernels mirror the two serial passes:
//     neighbor_kernel  <-> step 1 (all-pairs burial, O(L^2))
//     design_kernel    <-> step 2 (per-residue argmax over 20 amino acids)
// See ../THEORY.md sec "GPU mapping" for the full reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "inverse_folding.h"     // BackboneResidue, NUM_AA, score_aa_at_residue
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89 (see THEORY "GPU mapping" for the occupancy reasoning). It is
// ALSO the shared-memory tile width in neighbor_kernel below.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// neighbor_kernel: one thread per residue i counts its Calpha contacts.
//   Launch config (set in design_gpu):
//     grid  = ceil(L / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: i = blockIdx.x * blockDim.x + threadIdx.x owns residue i
//   and computes neighbors[i].
//
//   WHY SHARED MEMORY (the teaching point):
//     The naive version has every thread re-read all L residue coordinates from
//     global memory -> O(L^2) global loads, each ~hundreds of cycles. Instead we
//     TILE: the block cooperatively loads a tile of THREADS_PER_BLOCK residues
//     into shared memory (fast, on-chip), every thread compares its residue
//     against that whole tile from shared memory, then we advance to the next
//     tile. Each global coordinate is thus read once per block, not once per
//     thread -- the same trick as a tiled matrix multiply / N-body force kernel.
//
//   Memory: reads res[] (global, staged through shared `tile`); writes
//   neighbors[i] (global). No atomics: each thread owns one independent output.
// ---------------------------------------------------------------------------
__global__ void neighbor_kernel(const BackboneResidue* __restrict__ res, int L,
                                int* __restrict__ neighbors) {
    // Shared tile of residue coordinates, refilled tile-by-tile by the block.
    // Size = block size, so each thread loads exactly one residue per tile.
    __shared__ BackboneResidue tile[THREADS_PER_BLOCK];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's residue

    // Load THIS thread's own residue once into a register. Threads with i >= L
    // (ragged last block) still participate in the cooperative tile loads below
    // -- they just must not WRITE an output. We read a safe dummy for them.
    BackboneResidue me = (i < L) ? res[i] : BackboneResidue{0.f, 0.f, 0.f};

    int count = 0;   // running neighbor count for residue i (integer -> exact)

    // Sweep the residue list one shared-memory tile at a time.
    for (int base = 0; base < L; base += THREADS_PER_BLOCK) {
        // (a) Cooperative load: thread t brings residue (base + t) into shared
        //     memory (guarded so the last partial tile does not read past L).
        const int load_idx = base + threadIdx.x;
        if (load_idx < L) tile[threadIdx.x] = res[load_idx];
        // Barrier: every thread must finish loading before anyone reads the tile.
        __syncthreads();

        // (b) Compare my residue against every residue in this tile.
        const int tile_count = min(THREADS_PER_BLOCK, L - base);  // valid entries
        if (i < L) {
            for (int t = 0; t < tile_count; ++t) {
                const int j = base + t;            // global index of tile entry t
                if (j == i) continue;              // not a neighbor of itself
                const float dx = me.x - tile[t].x;
                const float dy = me.y - tile[t].y;
                const float dz = me.z - tile[t].z;
                const float d2 = dx * dx + dy * dy + dz * dz;   // squared dist (A^2)
                if (d2 <= CONTACT_RADIUS_SQ) ++count;           // within contact
            }
        }
        // Barrier before refilling the tile next iteration (so no thread reads a
        // half-overwritten tile). Both __syncthreads() are required for safety.
        __syncthreads();
    }

    if (i < L) neighbors[i] = count;   // guard the ragged last block on the WRITE
}

// ---------------------------------------------------------------------------
// design_kernel: one thread per residue i picks the best amino acid.
//   Launch config: same grid/block as neighbor_kernel (one thread per residue).
//   Thread-to-data map: i = blockIdx.x * blockDim.x + threadIdx.x owns residue i.
//   Memory: reads neighbors[i] (global), writes designed[i], score[i] (global).
//   No shared memory or atomics: the 20-way argmax is local to each thread, and
//   each output element is independent.
//
//   The argmax loop is IDENTICAL in structure to design_cpu()'s step 2, and
//   calls the SAME score_aa_at_residue() -> the chosen amino acid and its score
//   match the CPU bit-for-bit (the whole reason the score core is a shared
//   __host__ __device__ function in inverse_folding.h).
// ---------------------------------------------------------------------------
__global__ void design_kernel(const int* __restrict__ neighbors, int L,
                              int* __restrict__ designed, int* __restrict__ score) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's residue
    if (i >= L) return;                                    // ragged-block guard

    const int nbr = neighbors[i];                          // this residue's burial

    // Initialize "best" with amino acid 0 so the tie-break (lowest index) is
    // well-defined and matches the CPU exactly.
    int best_aa    = 0;
    int best_score = score_aa_at_residue(0, nbr);
    #pragma unroll
    for (int aa = 1; aa < NUM_AA; ++aa) {
        const int s = score_aa_at_residue(aa, nbr);
        // STRICT '>' keeps the FIRST (lowest-index) amino acid on a tie -> the
        // same deterministic choice the CPU makes.
        if (s > best_score) { best_score = s; best_aa = aa; }
    }
    designed[i] = best_aa;
    score[i]    = best_score;
}

// ---------------------------------------------------------------------------
// design_gpu: host wrapper. The canonical CUDA steps for a two-kernel pipeline:
//   (1) allocate device memory   (2) copy the backbone host->device
//   (3) launch neighbor_kernel    (4) launch design_kernel (uses its output)
//   (5) copy results device->host (6) free device memory
// We time ONLY the two kernels (CUDA events spanning both launches), not the
// H2D/D2H copies -- those are discussed separately in THEORY.
// ---------------------------------------------------------------------------
void design_gpu(const Backbone& bb, DesignResult& out, float* kernel_ms) {
    const int L = bb.size();
    out.neighbors.assign(L, 0);
    out.designed.assign(L, 0);
    out.score.assign(L, 0);

    const std::size_t res_bytes = static_cast<std::size_t>(L) * sizeof(BackboneResidue);
    const std::size_t int_bytes = static_cast<std::size_t>(L) * sizeof(int);

    // (1) Device buffers. d_ marks DEVICE pointers (CLAUDE.md sec 12).
    BackboneResidue* d_res       = nullptr;   // [L] Calpha coordinates
    int*             d_neighbors = nullptr;   // [L] neighbor counts (kernel-1 out / kernel-2 in)
    int*             d_designed  = nullptr;   // [L] designed amino-acid indices
    int*             d_score     = nullptr;   // [L] best per-residue scores
    CUDA_CHECK(cudaMalloc(&d_res,       res_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_neighbors, int_bytes));
    CUDA_CHECK(cudaMalloc(&d_designed,  int_bytes));
    CUDA_CHECK(cudaMalloc(&d_score,     int_bytes));

    // (2) Upload the backbone coordinates (the only large input). bb.res is a
    //     contiguous vector<BackboneResidue>, so one memcpy suffices.
    CUDA_CHECK(cudaMemcpy(d_res, bb.res.data(), res_bytes, cudaMemcpyHostToDevice));

    // (3)+(4) Launch both kernels. Enough blocks to give one thread per residue;
    //     ceiling division (L + B - 1) / B rounds up so no residue is dropped.
    const int blocks = (L + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    // Pass 1: burial. Pass 2: argmax design. design_kernel depends on
    // neighbor_kernel's output, but successive launches on the default stream
    // run in order, so no explicit sync is needed between them.
    neighbor_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_res, L, d_neighbors);
    design_kernel  <<<blocks, THREADS_PER_BLOCK>>>(d_neighbors, L, d_designed, d_score);
    *kernel_ms = timer.stop_ms();          // combined GPU time for both passes
    CUDA_CHECK_LAST("inverse-folding kernels");   // catch launch/exec errors

    // (5) Bring the three result arrays back to the host.
    CUDA_CHECK(cudaMemcpy(out.neighbors.data(), d_neighbors, int_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.designed.data(),  d_designed,  int_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.score.data(),     d_score,     int_bytes, cudaMemcpyDeviceToHost));

    // (6) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_designed));
    CUDA_CHECK(cudaFree(d_score));
}
