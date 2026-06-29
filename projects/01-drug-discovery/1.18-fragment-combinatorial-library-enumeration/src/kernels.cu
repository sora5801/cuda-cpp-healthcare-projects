// ===========================================================================
// src/kernels.cu  --  Combinatorial enumeration kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.18 : Fragment / Combinatorial Library Enumeration
//
// WHAT THIS FILE DOES
//   enumerate_kernel: one GPU thread per product. It decodes its flat index into
//   per-slot synthon indices, sums the chosen synthons' additive descriptors,
//   applies the Lipinski + Veber filter, and -- if the product passes -- bumps a
//   global integer counter, adds the product MW to a global fixed-point sum, and
//   marks a per-product pass FLAG. The flag array lets the host recover the first
//   FIRST_K passing indices in deterministic order without serialising the GPU.
//
//   This is the GPU twin of enumerate_cpu() in reference_cpu.cpp. They call the
//   SAME per-product math (product_core.h), so they agree bit-for-bit. main.cu
//   runs both and verifies. See ../THEORY.md "GPU mapping".
//
// KEY CUDA TEACHING POINTS (and why they matter)
//   * CONSTANT memory for the synthon tables: read by every thread, written by
//     none, tiny -> the constant cache broadcasts a row to a whole warp in one
//     transaction (vs. a global load per thread).
//   * INTEGER atomics for the reductions: float atomicAdd is order-dependent and
//     therefore NON-deterministic; integer adds commute, so counting passes and
//     summing fixed-point MW gives a reproducible total that matches the CPU.
//
// READ THIS AFTER: kernels.cuh, product_core.h, util/cuda_check.cuh, timer.cuh.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cstdint>
#include <cstring>     // std::memcpy (flatten the per-slot tables)
#include <stdexcept>
#include <vector>

// product_core.h #undef's HD at its end, so re-establish the decorator here for
// any inline helpers we might add. (The shared functions are already compiled in
// as inline; this is just defensive so the macro is defined in this TU's scope.)
#ifndef HD
#define HD __host__ __device__
#endif

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide latency, many blocks resident for occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CONSTANT-MEMORY synthon descriptor tables.
//   We flatten the (jagged) per-slot tables into ONE fixed-capacity array plus a
//   small offset/size table. Constant memory is 64 KB total; a double row is 40
//   bytes, so MAX_SYNTHONS_CONST = 256 rows = 10 KB fits comfortably and is far
//   more than a teaching catalog needs. (A production library with 130k building
//   blocks would instead stream the tables from global memory -- see THEORY.)
// ---------------------------------------------------------------------------
static constexpr int MAX_SYNTHONS_CONST = 256;   // capacity across all slots

// c_desc[r*N_DESC + d] = descriptor d of the r-th synthon (in flattened order).
__constant__ double c_desc[MAX_SYNTHONS_CONST * N_DESC];
// c_off[k] = flattened row offset where slot k's synthons begin in c_desc.
__constant__ int    c_off[N_SLOTS];
// c_sizes[k] = number of synthons in slot k (also the mixed-radix radices).
__constant__ int    c_sizes[N_SLOTS];

// ---------------------------------------------------------------------------
// enumerate_kernel: one logical thread per product, grid-stride looped.
//   Thread (blockIdx.x, threadIdx.x) starts at p = block*blockDim + thread and
//   strides by the total thread count until p >= N -- so a fixed-size grid
//   covers any N.
//   Parameters:
//     N        : total number of products (= product of c_sizes).
//     d_count  : [1] global counter; atomicAdd(+1) per passing product.
//     d_sum_mw : [1] global int64 accumulator; atomicAdd(MW_milli) per pass.
//     d_flag   : [N] per-product byte; set to 1 iff product p passes (else 0).
//   Memory: c_desc/c_off/c_sizes from the constant cache; d_flag write is the
//   only global traffic on the fast path; atomics touch two scalars.
// ---------------------------------------------------------------------------
__global__ void enumerate_kernel(int64_t N,
                                 unsigned long long* __restrict__ d_count,
                                 unsigned long long* __restrict__ d_sum_mw,
                                 unsigned char* __restrict__ d_flag) {
    // Total number of threads in the grid -> the grid-stride step.
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Copy the tiny radix table into registers once (constant-cache hit anyway,
    // but this makes the decode loop register-only).
    const int sizes[N_SLOTS] = {c_sizes[0], c_sizes[1], c_sizes[2]};

    for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         p < N; p += stride) {

        // (1) Decode the flat index into per-slot synthon indices (odometer).
        int idx[N_SLOTS];
        decode_product_index(p, sizes, idx);

        // (2) Gather pointers to the chosen synthons' descriptor rows in c_desc.
        //     Row of slot k, synthon idx[k] lives at (c_off[k] + idx[k]) * N_DESC.
        ProductInputs pin;
        #pragma unroll
        for (int k = 0; k < N_SLOTS; ++k)
            pin.row[k] = &c_desc[(c_off[k] + idx[k]) * N_DESC];

        // (3) Sum descriptors (shared HD math) and (4) apply the filter.
        double desc[N_DESC];
        accumulate_descriptors(pin, desc);
        const int pass = passes_filter(desc);

        // (5) Record the per-product flag (always written, so d_flag is dense).
        d_flag[p] = static_cast<unsigned char>(pass);

        // (6) Reduce: count passes and accumulate fixed-point MW with INTEGER
        //     atomics. atomicAdd on unsigned long long is associative across the
        //     (nondeterministic) thread order, so the totals are reproducible and
        //     equal the CPU's (PATTERNS.md sec.3). We round MW the same way the
        //     CPU does (llround) so the fixed-point quanta match exactly.
        if (pass) {
            atomicAdd(d_count, 1ULL);
            const long long mw_milli = llround(desc[D_MW] * static_cast<double>(MW_FIXED_SCALE));
            atomicAdd(d_sum_mw, static_cast<unsigned long long>(mw_milli));
        }
    }
}

// ---------------------------------------------------------------------------
// enumerate_gpu: host wrapper. Uploads the synthon tables to constant memory,
// launches the kernel, reduces the device counters, and recovers the first
// FIRST_K passing product indices from the dense flag array (a host scan, so the
// preview is in canonical ascending order -- deterministic).
//   We time ONLY the kernel (CUDA events), not the H2D/D2H copies (discussed
//   separately in THEORY). All CUDA bookkeeping is hidden here; main.cu just
//   compares the returned EnumResult against the CPU one.
// ---------------------------------------------------------------------------
void enumerate_gpu(const SynthonLibrary& lib, EnumResult& out, float* kernel_ms) {
    const int64_t N = lib.num_products();

    // ---- Flatten the jagged per-slot tables into the constant-memory layout --
    // Build a host-side copy of c_desc / c_off / c_sizes, then upload once.
    int h_off[N_SLOTS];
    int h_sizes[N_SLOTS];
    int total_rows = 0;
    for (int k = 0; k < N_SLOTS; ++k) {
        h_off[k]   = total_rows;          // slot k starts after all earlier rows
        h_sizes[k] = lib.sizes[k];
        total_rows += lib.sizes[k];
    }
    // Guard the constant-memory capacity: fail LOUDLY rather than overrun the
    // __constant__ array (which would be undefined behaviour). A real library
    // would stream from global memory instead -- noted in THEORY "real world".
    if (total_rows > MAX_SYNTHONS_CONST)
        throw std::runtime_error("synthon catalog too large for constant memory "
            "(" + std::to_string(total_rows) + " rows > " +
            std::to_string(MAX_SYNTHONS_CONST) + "); use the global-memory path "
            "described in THEORY for production-scale libraries.");

    std::vector<double> h_desc(static_cast<std::size_t>(total_rows) * N_DESC);
    for (int k = 0; k < N_SLOTS; ++k) {
        // Copy slot k's contiguous descriptor block into its place in h_desc.
        const std::size_t bytes = lib.desc[k].size() * sizeof(double);
        std::memcpy(&h_desc[static_cast<std::size_t>(h_off[k]) * N_DESC],
                    lib.desc[k].data(), bytes);
    }

    // Upload the three constant tables (special copies that target the constant
    // bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_desc, h_desc.data(),
                                  h_desc.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_off, h_off, sizeof(h_off)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sizes, h_sizes, sizeof(h_sizes)));

    // ---- Device accumulators + the dense per-product flag array --------------
    unsigned long long* d_count = nullptr;   // [1] number of passing products
    unsigned long long* d_sum_mw = nullptr;  // [1] sum of passing MW (milli-g/mol)
    unsigned char*      d_flag = nullptr;    // [N] per-product pass flag (0/1)
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_mw, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_flag, static_cast<std::size_t>(N) * sizeof(unsigned char)));
    // Zero the two scalar accumulators before the kernel adds into them.
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_sum_mw, 0, sizeof(unsigned long long)));

    // ---- Launch: enough blocks to cover N, capped; grid-stride does the rest.
    int blocks = static_cast<int>((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;        // cap; the grid-stride loop covers any N
    GpuTimer timer;
    timer.start();
    enumerate_kernel<<<blocks, THREADS_PER_BLOCK>>>(N, d_count, d_sum_mw, d_flag);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("enumerate_kernel");     // catch launch + execution errors

    // ---- Copy results back ---------------------------------------------------
    unsigned long long h_count = 0, h_sum_mw = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(h_count), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_mw, d_sum_mw, sizeof(h_sum_mw), cudaMemcpyDeviceToHost));

    std::vector<unsigned char> h_flag(static_cast<std::size_t>(N));
    CUDA_CHECK(cudaMemcpy(h_flag.data(), d_flag,
                          static_cast<std::size_t>(N) * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

    // ---- Assemble the deterministic EnumResult -------------------------------
    out.n_pass = static_cast<int64_t>(h_count);
    out.sum_mw_pass_milli = static_cast<int64_t>(h_sum_mw);
    out.first_pass.clear();
    // Scan the flag array in ascending product order to find the first FIRST_K
    // passes -> identical ordering to the CPU reference (determinism).
    for (int64_t p = 0; p < N && static_cast<int>(out.first_pass.size()) < FIRST_K; ++p)
        if (h_flag[p]) out.first_pass.push_back(p);

    // ---- Free device memory --------------------------------------------------
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_sum_mw));
    CUDA_CHECK(cudaFree(d_flag));
}
