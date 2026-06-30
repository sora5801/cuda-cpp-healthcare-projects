// ===========================================================================
// src/kernels.cuh  --  GPU DRR interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls render_drr_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by .cu
//   translation units (it contains a __global__ declaration, so the plain C++
//   compiler must never see it -- that is why the CPU reference lives in a
//   separate pure-C++ header, reference_cpu.h).
//
// THE BIG IDEA -- "gather", one thread per DRR pixel (PATTERNS.md section 1)
//   A DRR pixel is the line integral of attenuation along the ray from the X-ray
//   source through the volume to that detector pixel. Every pixel is INDEPENDENT
//   -- no pixel reads or writes another's data -- so we give each detector pixel
//   its own GPU thread, arranged in a 2-D grid that maps naturally onto the 2-D
//   detector panel. Each thread marches its own ray and writes one output value.
//   There are no atomics and no shared memory: it is a pure parallel gather.
//
//   The per-ray math is NOT here -- it is in the shared drr_core.h header
//   (integrate_ray + sample_trilinear), which both this kernel and the CPU
//   reference call, so their results match to float rounding.
//
//   PRODUCTION NOTE (kept as a comment, not code): a real DRR engine binds the
//   volume to a CUDA 3-D *texture* and replaces sample_trilinear() with a single
//   tex3D() call, so the hardware texture units do the tri-linear interpolation
//   essentially for free (and cache the 3-D neighbourhood). We deliberately do
//   the interpolation in plain device code so the learner can SEE every multiply
//   and add; THEORY.md "GPU mapping" describes the texture upgrade in full.
//
// READ THIS AFTER: drr_core.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // CtVolume, DrrGeometry, VolumeDesc (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// drr_kernel: thread (u, vrow) renders detector pixel (u, vrow).
//   d_mu : device pointer to the [nx*ny*nz] attenuation volume (row-major [z][y][x])
//   v    : volume shape (dims + spacing), passed by value into every thread
//   g    : DRR geometry (source, detector panel, step), passed by value
//   d_img: device output image [height*width], row-major [v][u]
//   Launch config is a 2-D grid of TILE x TILE blocks over the width x height
//   panel (see kernels.cu). __restrict__ promises d_mu and d_img do not alias,
//   letting the compiler keep the running integral in a register.
__global__ void drr_kernel(const float* __restrict__ d_mu,
                           VolumeDesc v, DrrGeometry g,
                           float* __restrict__ d_img);

// ---- Host wrapper --------------------------------------------------------
// render_drr_gpu: the host-callable "render the whole DRR on the GPU" function.
//   Uploads the attenuation volume, launches drr_kernel over a 2-D grid, copies
//   the rendered image back, and reports the measured KERNEL time (CUDA events)
//   via *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden.
//
//   vol       : loaded CT volume (host); its desc + mu[] are uploaded
//   g         : DRR geometry (same one passed to render_drr_cpu, for a fair check)
//   image     : host output, resized to width*height (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void render_drr_gpu(const CtVolume& vol, const DrrGeometry& g,
                    std::vector<float>& image, float* kernel_ms);
