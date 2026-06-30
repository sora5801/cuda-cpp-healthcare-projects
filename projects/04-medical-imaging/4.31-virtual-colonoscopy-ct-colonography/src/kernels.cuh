// ===========================================================================
// src/kernels.cuh  --  GPU volume ray-casting interface
// ---------------------------------------------------------------------------
// Project 4.31 : Virtual Colonoscopy & CT Colonography
//
// THE BIG IDEA
//   The virtual-colonoscopy frame is a per-PIXEL GATHER: each output pixel casts
//   one independent ray into the CT volume and shades the first wall it hits.
//   So we launch a 2-D thread grid over the image -- thread (px,py) owns pixel
//   (px,py) -- and each thread calls the SAME cast_ray() the CPU reference uses
//   (from volume_render.h). No two threads touch the same output, so there are
//   no atomics and no shared memory: a clean, textbook gather (THEORY §4).
//
//   The volume is uploaded once to global memory. (In a production fly-through
//   it would live in a CUDA 3-D TEXTURE so the hardware does trilinear
//   interpolation and caches 3-D neighborhoods; we use plain global memory + a
//   hand-written trilinear blend so the CPU and GPU match bit-for-bit -- see
//   THEORY §6 for why the texture path would not.)
//
// READ THIS AFTER: volume_render.h, reference_cpu.h, util/cuda_check.cuh.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Scene, Camera (pure C++, safe inside a .cu)

// ---------------------------------------------------------------------------
// render_gpu(): upload the volume, launch the 2-D ray-casting grid, copy the
//   rendered image back. Mirrors render_cpu() so main.cu can run both and diff.
//     scene     : the loaded volume + camera + render parameters (host).
//     image     : out-param, resized to width*height, filled with the frame.
//     kernel_ms : out-param, GPU kernel time in ms (CUDA-event measured).
//   The host wrapper lives in kernels.cu alongside the kernel itself.
// ---------------------------------------------------------------------------
void render_gpu(const Scene& scene, std::vector<float>& image, float* kernel_ms);
