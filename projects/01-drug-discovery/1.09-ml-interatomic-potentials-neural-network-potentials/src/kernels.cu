// ===========================================================================
// src/kernels.cu  --  The per-atom NNP energy kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
//
// WHAT THIS FILE DOES
//   Implements the device kernel (nnp_energy_kernel) and the host glue
//   (nnp_energy_gpu) that uploads the structure + model, launches one thread per
//   atom, times the kernel, copies the per-atom energies back, and sums them.
//   It is the GPU twin of nnp_energy_cpu() in reference_cpu.cpp; both call the
//   SAME atomic_energy() from nnp.h, so they agree to floating-point round-off.
//   main.cu runs both and verifies.
//
// COMMENT DENSITY: kernel files target >= 1:1 comments-to-code (CLAUDE.md 6.2),
// because this is where the GPU-specific reasoning lives.
//
// READ THIS AFTER: kernels.cuh (the interface + the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// THE MODEL IN CONSTANT MEMORY.
//   c_params (ACSF hyperparameters) and c_net (the MLP weights) are read by
//   EVERY thread and written by NONE during a launch -> constant memory is ideal:
//   its dedicated cache broadcasts one address to all 32 lanes of a warp in a
//   single transaction, instead of each thread issuing its own global load. The
//   model is small and fixed-size at compile time, so it fits comfortably in the
//   64 KB constant bank (AtomicNet is ~ (16*8 + 16 + 16*16 + 16 + 16 + 1)*8 bytes
//   ~ 3.5 KB; AcsfParams is a handful of doubles). Filled by cudaMemcpyToSymbol
//   in nnp_energy_gpu(). This mirrors the constant-memory query in 1.12 Tanimoto.
// ---------------------------------------------------------------------------
__constant__ AcsfParams c_params;   // cutoff Rc, width eta, shell centers Rs[]
__constant__ AtomicNet  c_net;      // the per-atom MLP weights and biases

// 128 threads/block. Each thread runs the per-atom MLP, which uses two local
// arrays of N_HID doubles (h1, h2) plus a desc[N_DESC] -- i.e. a fair number of
// registers/local memory per thread. A moderate block size (128) keeps register
// pressure from crushing occupancy while still being a multiple of the 32-lane
// warp. (256 also works; 128 is a safe default for register-heavy kernels.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// nnp_energy_kernel: one thread computes one atom's energy E_i.
//   Launch config (set in nnp_energy_gpu):
//     grid  = ceil(n / THREADS_PER_BLOCK) blocks (capped; grid-stride covers more)
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: atom index i = blockIdx.x*blockDim.x + threadIdx.x, then
//   a grid-stride loop so a fixed-size grid handles an arbitrarily large system.
//
//   Memory touched:
//     * d_pos  : global memory, the 3n coordinates (read by the descriptor scan).
//     * c_params, c_net : constant memory (the model), broadcast to the warp.
//     * d_eatom[i] : global memory, this thread's single output write.
//   No shared memory, no atomics: the per-atom energies are fully independent,
//   so there is nothing to synchronize. (The final sum is done on the host in a
//   fixed order for determinism -- see nnp_energy_gpu.)
// ---------------------------------------------------------------------------
__global__ void nnp_energy_kernel(const double* __restrict__ d_pos, int n,
                                  double* __restrict__ d_eatom) {
    const int stride = blockDim.x * gridDim.x;                  // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // The ENTIRE physics is the shared host/device function from nnp.h. By
        // calling it here (device) and from reference_cpu.cpp (host) we guarantee
        // the two paths do byte-for-byte identical arithmetic. atomic_energy =
        // compute_descriptor (gather over neighbors within Rc) -> small MLP.
        d_eatom[i] = atomic_energy(d_pos, n, i, c_params, c_net);
    }
}

// ---------------------------------------------------------------------------
// nnp_energy_gpu: host wrapper. The canonical CUDA steps, with the (read-only)
//   model going to constant memory rather than a global buffer:
//     (a) upload the model to __constant__ symbols
//     (b) allocate + upload the coordinates (global); allocate the output
//     (c) launch one thread per atom (timed with CUDA events)
//     (d) copy per-atom energies back
//     (e) free device memory, then sum on the host in atom order
//   We time ONLY the kernel (step c), not the H2D/D2H copies, so the reported
//   figure is compute cost (the copies are discussed separately in THEORY).
// ---------------------------------------------------------------------------
double nnp_energy_gpu(const Structure& s, const AcsfParams& p, const AtomicNet& net,
                      std::vector<double>& e_atom, float* kernel_ms) {
    const int n = s.n;
    e_atom.assign(static_cast<std::size_t>(n), 0.0);
    const std::size_t pos_bytes   = static_cast<std::size_t>(3 * n) * sizeof(double);
    const std::size_t eatom_bytes = static_cast<std::size_t>(n)     * sizeof(double);

    // (a) Upload the model to the __constant__ symbols. cudaMemcpyToSymbol is a
    //     special copy that targets the constant bank (not ordinary global
    //     memory) -- the only way to fill a __constant__ variable from the host.
    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &p,   sizeof(AcsfParams)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_net,    &net, sizeof(AtomicNet)));

    // (b) Coordinates -> global memory; allocate the per-atom energy output.
    double* d_pos   = nullptr;   // [3*n] flat coordinates on the device
    double* d_eatom = nullptr;   // [n]   per-atom energies (kernel output)
    CUDA_CHECK(cudaMalloc(&d_pos,   pos_bytes));
    CUDA_CHECK(cudaMalloc(&d_eatom, eatom_bytes));
    CUDA_CHECK(cudaMemcpy(d_pos, s.pos.data(), pos_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n atoms one-thread-each, capped so the
    //     grid stays modest; the grid-stride loop in the kernel handles any
    //     larger n. Time only the kernel with CUDA events (GpuTimer).
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1)    blocks = 1;
    if (blocks > 1024) blocks = 1024;   // grid-stride covers n beyond this cap
    GpuTimer timer;
    timer.start();
    nnp_energy_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_pos, n, d_eatom);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("nnp_energy_kernel");   // catch launch + execution errors

    // (d) Copy the per-atom energies back to the host.
    CUDA_CHECK(cudaMemcpy(e_atom.data(), d_eatom, eatom_bytes, cudaMemcpyDeviceToHost));

    // (e) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_eatom));

    // Sum the per-atom energies on the HOST, in ascending atom order. We do the
    // reduction here (not with a parallel/atomic GPU reduction) ON PURPOSE: a
    // fixed-order double sum is deterministic and exactly matches the CPU
    // reference's loop order (PATTERNS.md sec 3: avoid order-dependent float
    // atomics when you want a reproducible, CPU-matching total).
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += e_atom[static_cast<std::size_t>(i)];
    return total;
}
