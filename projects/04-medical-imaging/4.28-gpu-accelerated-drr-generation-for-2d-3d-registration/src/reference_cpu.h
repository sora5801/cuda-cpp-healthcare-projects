// ===========================================================================
// src/reference_cpu.h  --  CT-volume loader + serial DRR reference interface
// ---------------------------------------------------------------------------
// Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
//
// WHAT THIS PROJECT COMPUTES
//   A Digitally Reconstructed Radiograph (DRR): a SIMULATED X-ray projection of a
//   3-D CT volume. For each detector pixel we trace a ray from the X-ray source
//   through the volume and integrate the linear attenuation along it
//   (Beer-Lambert). The result is the synthetic radiograph used in 2D/3D
//   registration -- lining up a daily portal/kV X-ray with the planning CT by
//   searching for the patient pose whose DRR best matches the real image.
//
// WHY A GPU
//   A 400x400 DRR through a 256^3 volume needs hundreds of millions of
//   tri-linear samples; a full registration needs 50-200 DRRs PER optimizer
//   iteration. Every detector pixel is INDEPENDENT, so the workload is
//   embarrassingly parallel: one GPU thread per pixel, each marching its own ray
//   (the "gather" pattern -- see docs/PATTERNS.md and flagship 4.01).
//
// ROLE OF THIS FILE
//   Declares the pure-C++ pieces (NO CUDA): the file loader, the synthetic-volume
//   helpers, and the serial DRR that the GPU result is verified against. The
//   per-ray PHYSICS lives in drr_core.h and is shared with the GPU kernel so the
//   two agree to float rounding.
//
// READ THIS AFTER: drr_core.h.   READ BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "drr_core.h"   // VolumeDesc, DrrGeometry, integrate_ray, hu_to_mu (pure C++ here)

// ---------------------------------------------------------------------------
// CtVolume: a loaded CT volume = its shape (VolumeDesc) + the voxel data.
//   `mu` holds LINEAR ATTENUATION COEFFICIENTS (1/mm), already converted from the
//   raw Hounsfield Units in the file by hu_to_mu(). We convert once at load time
//   so the hot ray loop integrates mu directly. Layout is row-major [z][y][x],
//   x fastest -- exactly what drr_core.h's indexing expects.
// ---------------------------------------------------------------------------
struct CtVolume {
    VolumeDesc desc{};          // nx,ny,nz and spacing sx,sy,sz (mm)
    std::vector<float> mu;      // [nx*ny*nz] attenuation, row-major [z][y][x]
};

// ---------------------------------------------------------------------------
// load_volume: read the tiny text CT format used by data/sample/ and
//              scripts/make_synthetic.py. Format (whitespace-separated):
//
//     nx ny nz sx sy sz            # header: dims (ints) then spacings in mm
//     hu hu hu ... (nx*ny*nz)      # Hounsfield Units, row-major [z][y][x]
//
//   The HU values are converted to mu via hu_to_mu() as they are read, so the
//   returned CtVolume is ready for ray-marching. Throws std::runtime_error on a
//   missing file or a truncated/malformed body, so demos fail loudly.
// ---------------------------------------------------------------------------
CtVolume load_volume(const std::string& path);

// ---------------------------------------------------------------------------
// make_demo_geometry: build a fixed cone-beam DRR geometry for a loaded volume.
//   Places a point source on one side of the volume and a flat detector on the
//   opposite side, both centered on the volume, sized so the whole volume
//   projects onto a `width` x `height` panel. This is the SINGLE pose the demo
//   renders; a registration loop would vary it (see THEORY.md "real world").
//   `step_mm` is the ray-march sampling step (smaller = more accurate, slower).
//   Kept in the host code (not the shared core) because it is setup, not per-ray
//   physics -- main.cu calls it once and hands the result to both CPU and GPU.
// ---------------------------------------------------------------------------
DrrGeometry make_demo_geometry(const VolumeDesc& v, int width, int height, float step_mm);

// ---------------------------------------------------------------------------
// render_drr_cpu: the SERIAL reference. Fills `image` (size width*height,
//   row-major [v][u]) by calling integrate_ray() for every detector pixel in a
//   double loop. This is the trusted baseline that the GPU kernel is checked
//   against; because it calls the SAME integrate_ray() as the kernel, agreement
//   is exact up to float rounding. O(width*height*n_steps).
// ---------------------------------------------------------------------------
void render_drr_cpu(const CtVolume& vol, const DrrGeometry& g, std::vector<float>& image);
