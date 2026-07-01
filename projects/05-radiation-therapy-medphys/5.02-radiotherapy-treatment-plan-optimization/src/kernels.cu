// ===========================================================================
// src/kernels.cu  --  GPU projected-gradient FMO optimizer (cuSPARSE SpMV)
// ---------------------------------------------------------------------------
// Project 5.2 : Radiotherapy Treatment-Plan Optimization
//
// WHAT THIS FILE DOES
//   The GPU twin of optimize_cpu() (reference_cpu.cpp). It keeps the sparse
//   dose-influence matrix D resident on the device in CSR form and runs the
//   projected-gradient loop entirely on the GPU:
//
//     x <- 0
//     repeat `iters`:
//        dose = D  * x            <- cusparseSpMV(NON_TRANSPOSE)   [library]
//        r    = residual(dose)    <- residual_kernel               [our kernel]
//        grad = D^T * r           <- cusparseSpMV(TRANSPOSE)       [library]
//        x    = max(0, x - eta g) <- update_kernel                 [our kernel]
//
//   The two SpMVs are the whole reason FMO is a GPU problem, so we use cuSPARSE
//   -- NVIDIA's tuned sparse library -- rather than a naive hand-rolled SpMV.
//   The comments below spell out exactly what each cuSPARSE call computes and
//   the data layout it expects, so it is not a black box (CLAUDE.md section 6.1.6).
//
// READ THIS AFTER: kernels.cuh (the interface + the big idea), fmo.h (the shared
//   per-voxel math), reference_cpu.cpp (the same algorithm, serial).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Silence the C4996 "type will be removed" deprecation warnings emitted by
// <cusparse.h> itself (CUDA 13.3 marks several legacy types like pruneInfo_t /
// cusparseSpMMOpPlan_t as deprecated). Those types belong to APIs we do NOT
// use -- we only touch the modern generic SpMV API (cusparseSpMV + descriptors)
// -- so this header-provided guard cleanly removes the noise WITHOUT hiding any
// warning in our own code. It must be defined BEFORE the include to take effect.
#define DISABLE_CUSPARSE_DEPRECATED
#include <cusparse.h>            // cuSPARSE generic SpMV API
#include <cstdio>
#include <cstdlib>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide memory latency, plenty of resident blocks.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CUSPARSE_CHECK: the cuSPARSE analogue of CUDA_CHECK (util/cuda_check.cuh).
//   cuSPARSE calls return a cusparseStatus_t, NOT a cudaError_t, so they need
//   their own guard. Every cuSPARSE call below is wrapped in this so a failed
//   handle/descriptor/SpMV aborts with a precise message instead of silently
//   producing garbage. cusparseGetErrorString maps the status to text.
// ---------------------------------------------------------------------------
#define CUSPARSE_CHECK(call)                                                    \
    do {                                                                        \
        cusparseStatus_t st__ = (call);                                         \
        if (st__ != CUSPARSE_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr,                                                \
                "[CUSPARSE_CHECK] %s:%d\n    call : %s\n    error: %s (code %d)\n",\
                __FILE__, __LINE__, #call,                                      \
                cusparseGetErrorString(st__), static_cast<int>(st__));          \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// residual_kernel: thread v owns voxel v.
//   Launch: grid = ceil(n_vox / block), block = THREADS_PER_BLOCK.
//   Map: v = blockIdx.x * blockDim.x + threadIdx.x.
//   Reads dose[v] and this voxel's VoxelSpec, writes resid[v] using the SAME
//   voxel_residual() the CPU calls (fmo.h) -> identical scalar math. Pure
//   element-wise: no shared memory, no atomics, no divergence beyond the guard.
// ---------------------------------------------------------------------------
__global__ void residual_kernel(int n_vox, const VoxelSpec* __restrict__ spec,
                                const float* __restrict__ dose,
                                float* __restrict__ resid) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vox) return;                          // guard the ragged last block
    resid[v] = voxel_residual(spec[v], dose[v]);     // shared HD-core (fmo.h)
}

// ---------------------------------------------------------------------------
// update_kernel: thread j owns beamlet j.
//   Launch: grid = ceil(n_beam / block), block = THREADS_PER_BLOCK.
//   Applies the projected gradient step in place: x[j] = max(0, x[j] - eta*g[j]).
//   project_nonneg() is the shared clamp (fmo.h) so CPU and GPU project the same
//   way. Independent per beamlet -> no atomics.
// ---------------------------------------------------------------------------
__global__ void update_kernel(int n_beam, float step,
                              const float* __restrict__ grad,
                              float* __restrict__ x) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_beam) return;
    x[j] = project_nonneg(x[j] - step * grad[j]);
}

// ---------------------------------------------------------------------------
// optimize_gpu: upload D once, run the projected-gradient loop on the device.
//   The five canonical CUDA steps, plus cuSPARSE setup:
//     (1) allocate device buffers + upload the CSR matrix, specs, and zero x.
//     (2) build cuSPARSE descriptors (matrix + the dense vectors it multiplies).
//     (3) loop: SpMV(D x) -> residual_kernel -> SpMV(D^T r) -> update_kernel.
//     (4) copy the optimized fluence back to the host.
//     (5) destroy descriptors and free device memory.
//   We time the whole loop with CUDA events (the reported figure).
// ---------------------------------------------------------------------------
void optimize_gpu(const Problem& p, std::vector<float>& x_out, float* total_ms) {
    const int   n_vox  = p.n_vox;
    const int   n_beam = p.n_beam;
    const int   nnz    = p.nnz();
    const std::size_t vox_bytes  = static_cast<std::size_t>(n_vox)  * sizeof(float);
    const std::size_t beam_bytes = static_cast<std::size_t>(n_beam) * sizeof(float);

    // --- (1) Device buffers. d_ prefix = DEVICE pointer (dereffing on host crashes).
    int   *d_row = nullptr, *d_col = nullptr;   // CSR structure
    float *d_val = nullptr;                      // CSR nonzero values
    float *d_x = nullptr, *d_dose = nullptr, *d_resid = nullptr, *d_grad = nullptr;
    VoxelSpec* d_spec = nullptr;                 // per-voxel objective specs

    CUDA_CHECK(cudaMalloc(&d_row,  static_cast<std::size_t>(n_vox + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col,  static_cast<std::size_t>(nnz)       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val,  static_cast<std::size_t>(nnz)       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spec, static_cast<std::size_t>(n_vox)     * sizeof(VoxelSpec)));
    CUDA_CHECK(cudaMalloc(&d_x,     beam_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad,  beam_bytes));
    CUDA_CHECK(cudaMalloc(&d_dose,  vox_bytes));
    CUDA_CHECK(cudaMalloc(&d_resid, vox_bytes));

    // Upload the resident matrix + specs ONCE (they never change during optim).
    CUDA_CHECK(cudaMemcpy(d_row, p.row_ptr.data(),
                          static_cast<std::size_t>(n_vox + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, p.col_idx.data(),
                          static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, p.values.data(),
                          static_cast<std::size_t>(nnz) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spec, p.voxels.data(),
                          static_cast<std::size_t>(n_vox) * sizeof(VoxelSpec), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, beam_bytes));   // start fluence at zero (x >= 0)

    // --- (2) cuSPARSE handle + descriptors ---------------------------------
    // A cusparseHandle_t holds the library context (stream, workspace pools).
    cusparseHandle_t handle = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Describe D as a CSR matrix. cusparseCreateCsr takes the three device CSR
    // arrays, the dimensions, and the index/value TYPES. It does NOT copy the
    // data -- it just records the layout so SpMV can traverse rows. Index base 0.
    cusparseSpMatDescr_t matD = nullptr;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matD, n_vox, n_beam, nnz,
        d_row, d_col, d_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,   // row_ptr / col_idx are int32
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));   // values are float32

    // Dense-vector descriptors. The FORWARD product y = D x needs x (length
    // n_beam) as input and dose (length n_vox) as output. The TRANSPOSE product
    // g = D^T r needs r (length n_vox) as input and grad (length n_beam) as
    // output. cuSPARSE lets us bind the SAME buffer to different descriptors, so
    // we create four vector descriptors over our four device buffers.
    cusparseDnVecDescr_t vecX = nullptr, vecDose = nullptr, vecR = nullptr, vecGrad = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX,    n_beam, d_x,     CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDose, n_vox,  d_dose,  CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecR,    n_vox,  d_resid, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecGrad, n_beam, d_grad,  CUDA_R_32F));

    // SpMV computes  y = alpha * op(A) * x + beta * y.  We want a plain product
    // (dose = D x, grad = D^T r), so alpha = 1, beta = 0 (beta=0 also means the
    // output need not be pre-initialized).
    const float alpha = 1.0f, beta = 0.0f;

    // cuSPARSE may need a scratch workspace whose size it reports up front. The
    // forward and transpose ops can need different sizes; we query BOTH and
    // allocate the max so one buffer serves both. CUSPARSE_SPMV_ALG_DEFAULT lets
    // the library pick its best CSR SpMV algorithm for these shapes.
    std::size_t bufFwd = 0, bufT = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matD, vecX, &beta, vecDose,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufFwd));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, matD, vecR, &beta, vecGrad,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufT));
    void* d_buf = nullptr;
    const std::size_t bufBytes = bufFwd > bufT ? bufFwd : bufT;
    if (bufBytes > 0) CUDA_CHECK(cudaMalloc(&d_buf, bufBytes));

    const int vox_blocks  = (n_vox  + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int beam_blocks = (n_beam + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // --- (3) The projected-gradient loop, timed as a whole -----------------
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < p.iters; ++it) {
        // dose = D * x   (forward SpMV; fluence -> dose in every voxel).
        CUSPARSE_CHECK(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matD, vecX, &beta, vecDose,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf));

        // r = residual(dose)   (per-voxel dF/dd_v, shared with the CPU).
        residual_kernel<<<vox_blocks, THREADS_PER_BLOCK>>>(n_vox, d_spec, d_dose, d_resid);

        // grad = D^T * r   (transpose SpMV; voxel residuals -> beamlet gradient).
        CUSPARSE_CHECK(cusparseSpMV(
            handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, matD, vecR, &beta, vecGrad,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf));

        // x = max(0, x - eta * grad)   (projected gradient step, in place).
        update_kernel<<<beam_blocks, THREADS_PER_BLOCK>>>(n_beam, p.step, d_grad, d_x);
    }
    *total_ms = timer.stop_ms();
    CUDA_CHECK_LAST("fmo optimization loop");   // catch any kernel launch/exec error

    // --- (4) Copy the optimized fluence back to the host -------------------
    x_out.assign(static_cast<std::size_t>(n_beam), 0.0f);
    CUDA_CHECK(cudaMemcpy(x_out.data(), d_x, beam_bytes, cudaMemcpyDeviceToHost));

    // --- (5) Tear down descriptors, handle, and device memory --------------
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecDose));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecR));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecGrad));
    CUSPARSE_CHECK(cusparseDestroySpMat(matD));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    if (d_buf) CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_row));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_spec));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_dose));
    CUDA_CHECK(cudaFree(d_resid));
}
