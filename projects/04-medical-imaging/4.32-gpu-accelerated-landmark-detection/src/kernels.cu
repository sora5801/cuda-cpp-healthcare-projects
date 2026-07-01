// ===========================================================================
// src/kernels.cu  --  GPU landmark decode: per-block argmax + soft-argmax
// ---------------------------------------------------------------------------
// Project 4.32 : GPU-Accelerated Landmark Detection
//
// GPU twin of decode_cpu() (reference_cpu.cpp). It performs the SAME two-phase
// decode -- integer argmax, then fixed-point soft-argmax -- but in parallel:
// one thread BLOCK per landmark, the block's threads cooperating on that one
// heatmap. Because the per-voxel math (argmax tie-break, quantize_weight, the
// final division) is the shared landmark.h code, the GPU result is bit-identical
// to the CPU for the integer parts, matching within ~1e-9 after the one double
// division. main.cu runs both and verifies. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <cstdint>               // std::uint64_t

// One block decodes one landmark; 256 threads is a solid occupancy default on
// sm_75..sm_89 and a power of two, which the tree reduction below relies on.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// argmax_reduce: collapse each thread's best (value, index) to a single winner
// held by lane 0, using a shared-memory tree reduction.
//
//   s_val[t] : the best heatmap value thread t has seen so far.
//   s_idx[t] : the flat voxel index (within this volume) of that value.
//   The reduction halves the active range each step: lane t merges lane t+stride
//   into itself. The MERGE RULE reproduces the CPU tie-break exactly:
//     - take the strictly larger value; on an EXACT tie, take the LOWER index.
//   Row-major flat index increases in the CPU's scan order (x, then y, then z),
//   so "lowest index wins ties" == "first seen in the CPU scan wins ties".
//   __syncthreads() between steps makes the shared writes of one step visible to
//   the next (all threads must arrive before anyone reads the merged values).
// ---------------------------------------------------------------------------
__device__ inline void argmax_reduce(float* s_val, std::int64_t* s_idx, int t) {
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            float          ov = s_val[t + stride];       // "other" value
            std::int64_t   oi = s_idx[t + stride];       // "other" index
            // Adopt the other lane's candidate if it is strictly better, or an
            // exact tie with a smaller index (deterministic tie-break).
            if (ov > s_val[t] || (ov == s_val[t] && oi < s_idx[t])) {
                s_val[t] = ov;
                s_idx[t] = oi;
            }
        }
        __syncthreads();   // barrier: this step's writes visible before next step
    }
}

// ---------------------------------------------------------------------------
// decode_kernel: decode all L heatmaps, one landmark per block.
//
// THREAD-TO-DATA MAPPING
//   blockIdx.x  = l  -> this block owns landmark l's volume (slab l*V .. (l+1)*V).
//   threadIdx.x = t  -> a lane that strides over voxels: t, t+B, t+2B, ...
//                       (B = blockDim.x) so all V voxels are covered with
//                       coalesced-friendly, balanced work.
//
// SHARED MEMORY LAYOUT (all block-private, freed when the block ends)
//   s_val[B], s_idx[B]         : per-lane argmax partials (phase 1).
//   s_peak, s_px/s_py/s_pz     : the block's winning value + voxel (phase 1 out).
//   s_w, s_wx, s_wy, s_wz      : 64-bit fixed-point soft-argmax accumulators
//                                (phase 2), atomically summed by all lanes.
// ---------------------------------------------------------------------------
__global__ void decode_kernel(const float* __restrict__ data,
                              VolumeDims dims, int L,
                              GpuLandmark* __restrict__ out) {
    const int l = blockIdx.x;           // which landmark this block decodes
    if (l >= L) return;                 // guard a ragged grid (we launch exactly L)
    const int t = threadIdx.x;          // this lane's id within the block
    const int B = blockDim.x;           // threads per block (== THREADS_PER_BLOCK)

    const std::int64_t V = volume_voxels(dims);           // voxels in one volume
    const float* vol = data + static_cast<std::int64_t>(l) * V;  // this heatmap

    // ---- Shared-memory scratch (sized to the block) -----------------------
    __shared__ float          s_val[THREADS_PER_BLOCK];   // argmax partial values
    __shared__ std::int64_t   s_idx[THREADS_PER_BLOCK];   // argmax partial indices
    __shared__ int            s_px, s_py, s_pz;           // winning voxel coords
    __shared__ float          s_peak;                     // winning value
    // Soft-argmax accumulators. unsigned long long is the type atomicAdd supports
    // for 64-bit integer atomics; we alias it to std::uint64_t at read-out.
    __shared__ unsigned long long s_w, s_wx, s_wy, s_wz;

    // ================= PHASE 1: parallel argmax ============================
    // Each lane scans its strided share of voxels, tracking its own best. We
    // keep the flat index so ties resolve identically to the CPU.
    float        best = -3.4e38f;       // ~ -FLT_MAX: below any real heatmap value
    std::int64_t best_i = 0;
    for (std::int64_t i = t; i < V; i += B) {
        float v = vol[i];
        if (v > best || (v == best && i < best_i)) {   // strict-> or tie by lower i
            best = v; best_i = i;
        }
    }
    s_val[t] = best;                    // publish this lane's partial to shared mem
    s_idx[t] = best_i;
    __syncthreads();                    // all partials written before we reduce

    argmax_reduce(s_val, s_idx, t);     // tree-reduce to the single block winner

    // Lane 0 now holds the volume-wide argmax. Decode its flat index back to
    // (x,y,z) via the row-major inverse and publish for phase 2.
    if (t == 0) {
        std::int64_t idx = s_idx[0];
        int z = static_cast<int>(idx / (static_cast<std::int64_t>(dims.nx) * dims.ny));
        std::int64_t rem = idx - static_cast<std::int64_t>(z) * dims.nx * dims.ny;
        int y = static_cast<int>(rem / dims.nx);
        int x = static_cast<int>(rem - static_cast<std::int64_t>(y) * dims.nx);
        s_px = x; s_py = y; s_pz = z;
        s_peak = s_val[0];
        s_w = s_wx = s_wy = s_wz = 0ull;   // zero the phase-2 accumulators
    }
    __syncthreads();                    // winner + zeroed accumulators visible to all

    // ================= PHASE 2: parallel soft-argmax =======================
    // The window is a (2R+1)^3 cube around the peak. We flatten it to a 1D range
    // [0, WIN) and stride the lanes over it, so a block of 256 threads handles a
    // 125-voxel window in one pass with idle lanes simply skipping.
    const int R  = SOFTARGMAX_RADIUS;
    const int WD = 2 * R + 1;           // window side length (e.g. 5)
    const int WIN = WD * WD * WD;       // window voxel count (e.g. 125)
    for (int k = t; k < WIN; k += B) {
        // Unflatten k -> (dx,dy,dz) offsets in [-R, R].
        int dx = (k % WD) - R;
        int dy = ((k / WD) % WD) - R;
        int dz = (k / (WD * WD)) - R;
        int x = s_px + dx, y = s_py + dy, z = s_pz + dz;
        // Clip to the volume (same guard as the CPU's window loop).
        if (x < 0 || x >= dims.nx || y < 0 || y >= dims.ny || z < 0 || z >= dims.nz)
            continue;
        unsigned int w = quantize_weight(vol[flat_index(x, y, z, dims)]);
        if (w == 0u) continue;          // zero weight adds nothing; skip the atomics
        // Integer atomicAdd into shared accumulators. Integer adds commute, so
        // the totals are identical regardless of the order lanes finish -> the
        // result is deterministic AND equal to the CPU's serial integer sums.
        atomicAdd(&s_w,  static_cast<unsigned long long>(w));
        atomicAdd(&s_wx, static_cast<unsigned long long>(w) * x);
        atomicAdd(&s_wy, static_cast<unsigned long long>(w) * y);
        atomicAdd(&s_wz, static_cast<unsigned long long>(w) * z);
    }
    __syncthreads();                    // all contributions in before we finalize

    // Lane 0 finalizes: divide the integer sums (shared finalize_softargmax, the
    // SAME division the CPU does) and write this landmark's record out.
    if (t == 0) {
        GpuLandmark g;
        finalize_softargmax(static_cast<std::uint64_t>(s_w),
                            static_cast<std::uint64_t>(s_wx),
                            static_cast<std::uint64_t>(s_wy),
                            static_cast<std::uint64_t>(s_wz),
                            s_px, s_py, s_pz, g.x, g.y, g.z);
        g.peak = s_peak;
        g.px = s_px; g.py = s_py; g.pz = s_pz;
        out[l] = g;
    }
}

// ---------------------------------------------------------------------------
// decode_gpu: host wrapper. Allocates device buffers, uploads the heatmaps,
// launches L blocks, downloads the L decoded landmarks, and reports kernel time.
//   Complexity on device: O(V) per block for the argmax scan (dominant) plus a
//   constant-size window; L blocks run concurrently across the SMs.
// ---------------------------------------------------------------------------
void decode_gpu(const HeatmapSet& hs, std::vector<Landmark>& out, float* kernel_ms) {
    const int L = hs.num_landmarks;
    const std::int64_t V = volume_voxels(hs.dims);
    const std::size_t total = static_cast<std::size_t>(V) * L;

    // Device buffers: the input heatmaps and the output landmark records.
    float* d_data = nullptr;
    GpuLandmark* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(L) * sizeof(GpuLandmark)));
    CUDA_CHECK(cudaMemcpy(d_data, hs.data.data(), total * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch: exactly L blocks (one landmark each) of THREADS_PER_BLOCK lanes.
    GpuTimer timer;
    timer.start();
    decode_kernel<<<L, THREADS_PER_BLOCK>>>(d_data, hs.dims, L, d_out);
    *kernel_ms = timer.stop_ms();       // blocks until the kernel finishes
    CUDA_CHECK_LAST("decode_kernel");   // catch launch + execution errors

    // Bring the decoded landmarks back and copy into the host Landmark type.
    std::vector<GpuLandmark> g(L);
    CUDA_CHECK(cudaMemcpy(g.data(), d_out,
                          static_cast<std::size_t>(L) * sizeof(GpuLandmark),
                          cudaMemcpyDeviceToHost));
    out.resize(L);
    for (int l = 0; l < L; ++l) {
        out[l].x = g[l].x; out[l].y = g[l].y; out[l].z = g[l].z;
        out[l].peak = g[l].peak;
        out[l].px = g[l].px; out[l].py = g[l].py; out[l].pz = g[l].pz;
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}
