// ===========================================================================
// src/kernels.cuh  --  Public interface of the GPU reconstruction path
// ---------------------------------------------------------------------------
// Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
//
// WHAT THIS DECLARES
//   * Acquisition -- the "scan": image dims, the fully-sampled ground-truth
//     image (for scoring only), the UNDER-SAMPLED k-space we actually measured,
//     and the binary sampling MASK that says which frequencies were kept.
//   * ReconParams -- the unrolled-network hyper-parameters (stage count, the
//     denoiser strength lambda). A trained net would learn these; we fix them.
//   * recon_gpu() -- runs the whole unrolled reconstruction on the GPU and hands
//     back the reconstructed image plus a GPU-measured timing.
//
//   The GPU path is the twin of recon_cpu() (reference_cpu.cpp): identical math,
//   different executor. main.cu runs both and verifies they agree.
//
//   NOTE: this header is included only by .cu translation units. It has no
//   __global__ declarations (the kernels are file-local to kernels.cu), so it is
//   plain C++ -- but recon_cpu()'s declaration lives in reference_cpu.h to keep
//   the host-compiled reference free of any CUDA dependency.
//
// READ THIS AFTER: main.cu. READ BEFORE: kernels.cu (the implementations).
// The per-pixel math is in recon_core.h (stencil) and dft_core.h (transform).
// ===========================================================================
#pragma once

#include <vector>

// ---------------------------------------------------------------------------
// Acquisition: everything the reconstruction is given about one "scan".
//   In real MRI the scanner measures k-space along a trajectory that SKIPS lines
//   to go faster (acceleration). We model that with a per-frequency binary mask:
//   mask[k]==1 means "this frequency was measured", 0 means "skipped".
//   All arrays are row-major, length ny*nx, stored as structure-of-arrays so
//   that consecutive threads touch consecutive memory (coalesced GPU access).
// ---------------------------------------------------------------------------
struct Acquisition {
    int ny = 0;                    // image height in pixels
    int nx = 0;                    // image width in pixels
    std::vector<float> truth;      // [ny*nx] ground-truth image (scoring ONLY;
                                   //   a real scan never has this -- synthetic here)
    std::vector<float> kmeas_re;   // [ny*nx] measured k-space, real part
    std::vector<float> kmeas_im;   // [ny*nx] measured k-space, imag part
    std::vector<int>   mask;       // [ny*nx] 1 = frequency sampled, 0 = skipped

    int n() const { return ny * nx; }   // total pixel/frequency count
};

// ---------------------------------------------------------------------------
// ReconParams: the fixed "network" hyper-parameters of the unrolled recon.
//   stages : how many cascade stages (each = denoise + data-consistency). More
//            stages -> sharper reconstruction, up to a point (see Exercises).
//   lambda : denoiser strength in [0,1] (how far each stage moves toward D(x)).
//   A trained E2E-VarNet would LEARN both the number of unrolls and per-stage
//   weights; we fix them so the demo is deterministic and explainable.
// ---------------------------------------------------------------------------
struct ReconParams {
    int   stages = 12;
    float lambda = 0.5f;
};

// ---------------------------------------------------------------------------
// recon_gpu: run the full unrolled reconstruction on the GPU.
//   Inputs : acq (measurement + mask), p (hyper-parameters).
//   Output : recon -- [ny*nx] reconstructed image (row-major), resized inside.
//   Out-param kernel_ms -- total GPU time for all cascade-stage kernels (CUDA
//            events), a teaching artifact only (CLAUDE.md section 12).
//   The GPU result is verified against recon_cpu() in main.cu.
// ---------------------------------------------------------------------------
void recon_gpu(const Acquisition& acq, const ReconParams& p,
               std::vector<float>& recon, float* kernel_ms);
