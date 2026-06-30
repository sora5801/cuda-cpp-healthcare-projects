// ===========================================================================
// src/kernels.cuh  --  GPU ray-tracing interface (declarations + the big idea)
// ---------------------------------------------------------------------------
// Project 2.8 : GPU Molecular Visualization & Ray Tracing
//
// THE BIG IDEA  (docs/PATTERNS.md §1 -- "per-output-pixel gather")
//   Rendering an image is embarrassingly parallel: every PIXEL is independent,
//   reading the shared (read-only) scene and writing only its own output. So we
//   launch a 2-D grid of threads over the image and let thread (px,py) render
//   pixel (px,py) by calling the SHARED shade_pixel() from render_core.h -- the
//   exact same function the CPU reference loops over. Same code, same math, so
//   the two images agree (and after 8-bit quantization, agree EXACTLY).
//
//   This mirrors flagship 4.01 (CT backprojection), the other per-output-pixel
//   gather: a 2-D thread grid, each thread does an independent gather over the
//   scene, no shared memory or atomics required.
//
// WHERE THE ATOMS LIVE: CONSTANT MEMORY
//   Every thread reads EVERY atom (brute-force trace). When all threads in a
//   warp read the same address, CUDA __constant__ memory broadcasts it from a
//   cache in a single transaction -- ideal for a small, read-only scene shared
//   by the whole launch. (Same trick as flagship 1.12's constant-memory query.)
//   Constant memory is 64 KB total; we cap the scene at MAX_ATOMS atoms that fit
//   and fall back to a global-memory pointer for larger scenes (see kernels.cu).
//
// READ THIS AFTER: render_core.h, util/cuda_check.cuh, util/timer.cuh.
//   Then read kernels.cu (the kernel + host wrapper), then main.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Scene, Atom, Camera, RenderParams (pure C++, safe in .cu)

// Max atoms we can hold in constant memory. Each Atom is 5 floats + 1 int = 24
// bytes (with padding); 2048 * 24 = ~48 KB, comfortably under the 64 KB limit.
// Our demo molecule is far smaller; this headroom lets you drop in a small PDB.
static const int MAX_ATOMS = 2048;

// ---------------------------------------------------------------------------
// render_kernel: thread (px,py) renders one pixel into `image`.
//   cam     : orthographic camera (by value -> each thread's registers)
//   n_atoms : number of atoms (the scene itself lives in __constant__ memory)
//   rp      : shading parameters (by value)
//   image   : device buffer of width*height bytes (output); image[py*W+px]
//   The kernel just calls shade_pixel() + quantize8() -- all the physics is in
//   render_core.h, shared verbatim with the CPU reference.
// ---------------------------------------------------------------------------
__global__ void render_kernel(Camera cam, int n_atoms, RenderParams rp,
                              unsigned char* __restrict__ image);

// ---------------------------------------------------------------------------
// render_gpu: host wrapper. Uploads the scene to constant memory, launches the
//   2-D grid, copies the rendered byte image back, and reports the measured
//   KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly this.
//     scene     : the molecule + camera + shading params (host)
//     image     : host output, resized to width*height bytes (output parameter)
//     kernel_ms : out-param, milliseconds spent in the kernel (not the copies)
//   Throws std::runtime_error if the scene exceeds MAX_ATOMS (constant memory).
// ---------------------------------------------------------------------------
void render_gpu(const Scene& scene, std::vector<unsigned char>& image,
                float* kernel_ms);
