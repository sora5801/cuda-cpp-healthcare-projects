// ===========================================================================
// src/drr_core.h  --  The ONE TRUE per-ray physics, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
//
// WHY THIS HEADER EXISTS  (PATTERNS.md section 2: the __host__ __device__ core)
//   A Digitally Reconstructed Radiograph (DRR) is a SIMULATED X-ray: we shoot a
//   ray from the X-ray source, through a 3-D CT volume, to each detector pixel,
//   and integrate how much the tissue along that ray attenuates the beam. The
//   per-ray math (tri-linear sampling of the volume + the ray-marching integral)
//   must be *byte-for-byte identical* on the CPU reference and the GPU kernel,
//   otherwise "GPU == CPU" verification would only ever be approximate.
//
//   So we put that math here, ONCE, as `__host__ __device__` inline functions:
//     * reference_cpu.cpp includes this through the host compiler (cl.exe),
//     * kernels.cu includes this through nvcc,
//   and both call the SAME `integrate_ray()`. Verification then reduces to "did
//   the two loops visit the same pixels", and the per-pixel values match to
//   float rounding only.
//
//   HARD RULE for this file: NO CUDA-only constructs (no __global__, no
//   <cuda_runtime.h>, no texture<>). Only things the *host* compiler also
//   understands, decorated with the HD macro. That is what lets cl.exe compile it.
//
// READ THIS AFTER: nothing -- start here. THEN read reference_cpu.h (the loader
// and the CPU driver) and kernels.cuh (the GPU twin).
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// The HD macro idiom (PATTERNS.md section 2).
//   When compiled by nvcc, __CUDACC__ is defined, so DRR_HD expands to the CUDA
//   decorators that make a function callable from BOTH host and device. When
//   compiled by the plain host compiler (for reference_cpu.cpp), __CUDACC__ is
//   NOT defined, so DRR_HD expands to nothing and the functions are ordinary
//   inline C++ -- identical source, two compilers, one behavior.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define DRR_HD __host__ __device__
#else
#define DRR_HD
#endif

#include <cmath>   // std::floor, std::sqrt, std::exp (host); device gets the CUDA equivalents

// ---------------------------------------------------------------------------
// Vec3: a minimal 3-component float vector.
//   We roll our own (instead of CUDA's float3) so this header stays usable by
//   the host compiler with zero CUDA headers. All geometry below is in
//   "world millimetres": x,y to the side, z along the patient axis. The volume
//   occupies the axis-aligned box [0,nx*sx] x [0,ny*sy] x [0,nz*sz] mm.
// ---------------------------------------------------------------------------
struct Vec3 {
    float x, y, z;
};

// Small vector helpers -- each is HD so the same code runs on CPU and GPU.
DRR_HD inline Vec3 vec_add(Vec3 a, Vec3 b)            { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
DRR_HD inline Vec3 vec_sub(Vec3 a, Vec3 b)            { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
DRR_HD inline Vec3 vec_scale(Vec3 a, float s)         { return Vec3{a.x * s, a.y * s, a.z * s}; }
DRR_HD inline float vec_dot(Vec3 a, Vec3 b)           { return a.x * b.x + a.y * b.y + a.z * b.z; }
DRR_HD inline float vec_len(Vec3 a)                   { return std::sqrt(vec_dot(a, a)); }

// ---------------------------------------------------------------------------
// VolumeDesc: the SHAPE of the CT volume, with no pixel data.
//   We pass this small struct (by value) to every per-ray call so the math has
//   the grid dimensions and the mm-per-voxel spacing it needs to convert a world
//   point into a fractional voxel index. The actual voxel array (the heavy data)
//   is passed separately as a raw pointer, because on the GPU it lives in device
//   memory and on the CPU it lives in a std::vector -- but both are just
//   `const float*` to row-major [z][y][x] data.
//
//   Indexing convention (row-major, x fastest): voxel (ix,iy,iz) lives at
//       data[(iz * ny + iy) * nx + ix].
//   `data` holds LINEAR ATTENUATION COEFFICIENTS mu (units: 1/mm), already
//   converted from raw Hounsfield Units by hu_to_mu() (see below).
// ---------------------------------------------------------------------------
struct VolumeDesc {
    int   nx, ny, nz;     // voxel counts along x, y, z
    float sx, sy, sz;     // voxel spacing in mm along x, y, z
};

// Total number of voxels -- handy for allocation. size_t so 512^3 never overflows.
DRR_HD inline long long vol_count(const VolumeDesc& v) {
    return (long long)v.nx * v.ny * v.nz;
}

// ---------------------------------------------------------------------------
// hu_to_mu: convert a Hounsfield Unit (HU) to a linear attenuation coefficient
//           mu (1/mm) at a representative diagnostic energy.
//
//   HU is defined so that water = 0 HU and air = -1000 HU, via
//       HU = 1000 * (mu_tissue - mu_water) / mu_water
//   Inverting:  mu_tissue = mu_water * (1 + HU/1000).
//   We use mu_water ~ 0.019 /mm (a textbook value near ~70 keV effective energy)
//   and clamp negative results to 0 (air/vacuum cannot have negative attenuation).
//
//   WHY here: both the CPU reference and the GPU kernel must convert HU the same
//   way, so this lives in the shared core. The conversion is done ONCE when the
//   volume is loaded (see reference_cpu.cpp), so the ray loop integrates mu, not HU.
// ---------------------------------------------------------------------------
DRR_HD inline float hu_to_mu(float hu) {
    const float mu_water = 0.019f;                 // 1/mm, ~70 keV effective beam
    float mu = mu_water * (1.0f + hu * 0.001f);    // 0.001f = 1/1000
    return mu > 0.0f ? mu : 0.0f;                  // clamp: no negative attenuation
}

// ---------------------------------------------------------------------------
// sample_trilinear: read the volume at a CONTINUOUS world point `p` (mm), using
//                   tri-linear interpolation -- the software twin of what a CUDA
//                   3-D texture would do in hardware "for free".
//
//   Steps:
//     1. Convert world mm -> fractional voxel coordinates (divide by spacing).
//        We use a voxel-CENTERED convention: world (0,0,0) is the CENTER of
//        voxel (0,0,0), so fx = p.x/sx, and voxel ix has center at ix*sx.
//     2. Find the lower integer voxel (ix0,iy0,iz0) = floor(fx,fy,fz) and the
//        fractional offsets (tx,ty,tz) in [0,1) within that cell.
//     3. Blend the 8 surrounding voxels with weights (1-t) and t per axis.
//        Any neighbour that falls outside the grid contributes 0 (we treat the
//        outside as air / mu=0), which also makes rays that graze the edge safe.
//
//   Returns mu (1/mm) at p. This is the single hottest function in the whole
//   project -- on a real GPU it is replaced by one tex3D() call, which is why
//   DRR is such a natural texture-hardware workload (see THEORY.md "GPU mapping").
// ---------------------------------------------------------------------------
DRR_HD inline float sample_trilinear(const float* data, const VolumeDesc& v, Vec3 p) {
    // 1. world mm -> fractional voxel index (voxel-centered).
    float fx = p.x / v.sx;
    float fy = p.y / v.sy;
    float fz = p.z / v.sz;

    // 2. lower voxel + in-cell fraction.
    float fl_x = std::floor(fx), fl_y = std::floor(fy), fl_z = std::floor(fz);
    int ix0 = (int)fl_x, iy0 = (int)fl_y, iz0 = (int)fl_z;
    float tx = fx - fl_x, ty = fy - fl_y, tz = fz - fl_z;

    // Helper lambda would need <functional> on host; instead use a tiny static
    // fetch that returns 0 for out-of-bounds voxels (air outside the volume).
    // (Written as a nested block for clarity; the compiler inlines it.)
    // We read the 8 corners c000..c111 of the cell around (ix0,iy0,iz0).
    const int nx = v.nx, ny = v.ny, nz = v.nz;
    #define DRR_FETCH(IX, IY, IZ)                                                  \
        (((IX) >= 0 && (IX) < nx && (IY) >= 0 && (IY) < ny && (IZ) >= 0 && (IZ) < nz) \
            ? data[(((long long)(IZ) * ny + (IY)) * nx + (IX))]                    \
            : 0.0f)
    float c000 = DRR_FETCH(ix0,     iy0,     iz0);
    float c100 = DRR_FETCH(ix0 + 1, iy0,     iz0);
    float c010 = DRR_FETCH(ix0,     iy0 + 1, iz0);
    float c110 = DRR_FETCH(ix0 + 1, iy0 + 1, iz0);
    float c001 = DRR_FETCH(ix0,     iy0,     iz0 + 1);
    float c101 = DRR_FETCH(ix0 + 1, iy0,     iz0 + 1);
    float c011 = DRR_FETCH(ix0,     iy0 + 1, iz0 + 1);
    float c111 = DRR_FETCH(ix0 + 1, iy0 + 1, iz0 + 1);
    #undef DRR_FETCH

    // 3. tri-linear blend: interpolate along x, then y, then z. The order does
    //    not change the result (the form is symmetric), but we fix it so CPU and
    //    GPU evaluate the exact same float operations in the exact same order.
    float c00 = c000 * (1.0f - tx) + c100 * tx;
    float c10 = c010 * (1.0f - tx) + c110 * tx;
    float c01 = c001 * (1.0f - tx) + c101 * tx;
    float c11 = c011 * (1.0f - tx) + c111 * tx;
    float c0  = c00  * (1.0f - ty) + c10  * ty;
    float c1  = c01  * (1.0f - ty) + c11  * ty;
    return    c0   * (1.0f - tz) + c1   * tz;
}

// ---------------------------------------------------------------------------
// DrrGeometry: everything needed to turn a detector pixel (u,v) into a ray.
//
//   This is a simple CONE-BEAM model: a single point X-ray source, and a flat
//   panel detector. We describe the detector by its top-left ("origin") corner
//   and two edge vectors (`du` per column step, `dv` per row step), both in mm.
//   A detector pixel (u,v) sits at:  det = origin + u*du + v*dv.
//   The ray for that pixel goes from `source` to `det`.
//
//   To register a daily X-ray to the planning CT, an optimizer perturbs the
//   pose (rotation+translation of the patient relative to source/detector), which
//   here is baked into source/origin/du/dv. We keep ONE fixed pose for the demo
//   (the geometry is computed in main.cu from a few human-readable parameters);
//   THEORY.md "real world" explains the full 6-DOF registration loop.
// ---------------------------------------------------------------------------
struct DrrGeometry {
    Vec3 source;     // X-ray focal spot (mm, world coords)
    Vec3 origin;     // detector pixel (0,0) center (mm)
    Vec3 du;         // world displacement per +1 detector column (mm)
    Vec3 dv;         // world displacement per +1 detector row (mm)
    int  width;      // detector columns (DRR image width, pixels)
    int  height;     // detector rows (DRR image height, pixels)
    float step_mm;   // ray-march sampling step along the ray (mm)
};

// ---------------------------------------------------------------------------
// integrate_ray: THE CORE. Compute one DRR pixel value for detector pixel (u,v).
//
//   Algorithm (ray-marching / "ray-casting" DRR):
//     1. Build the ray source -> detector-pixel and its unit direction.
//     2. March along the ray in fixed steps of `step_mm`, from the source toward
//        the detector, sampling mu tri-linearly at each step and accumulating
//        the line integral  L = integral(mu ds) ~ sum(mu_i * step_mm).
//        We only accumulate while the sample point is inside (or near) the volume
//        box; sample_trilinear returns 0 outside, so stepping through empty space
//        simply adds nothing.
//     3. Return the integrated attenuation L (units: dimensionless, = mu*length).
//        The Beer-Lambert transmitted intensity is I = I0 * exp(-L); we return L
//        itself as the DRR pixel because intensity-based registration compares L
//        (or exp(-L)) consistently on both the DRR and the real X-ray. Reporting
//        L keeps the number interpretable ("total attenuation along this ray").
//
//   Determinism: the loop bound `n_steps` is computed from the fixed source->
//   detector distance and `step_mm`, so every ray takes a data-independent,
//   identical number of steps on CPU and GPU -> identical float summation order
//   -> the two results match to rounding. (PATTERNS.md section 3/4.)
//
//   Complexity: O(n_steps) per pixel; total O(width*height*n_steps). Each pixel
//   is INDEPENDENT -- that independence is exactly why one GPU thread per pixel
//   (the "gather" pattern, PATTERNS.md section 1) is the natural mapping.
// ---------------------------------------------------------------------------
DRR_HD inline float integrate_ray(const float* vol, const VolumeDesc& v,
                                  const DrrGeometry& g, int u, int vrow) {
    // 1. ray endpoints.
    Vec3 det = vec_add(g.origin, vec_add(vec_scale(g.du, (float)u),
                                         vec_scale(g.dv, (float)vrow)));
    Vec3 dir = vec_sub(det, g.source);          // source -> detector pixel
    float total_len = vec_len(dir);             // mm from source to this pixel
    if (total_len <= 0.0f) return 0.0f;         // degenerate guard
    Vec3 unit = vec_scale(dir, 1.0f / total_len);

    // 2. number of marching steps -- the SAME for every pixel of a given build
    //    is NOT required, but the count must match between CPU and GPU for this
    //    pixel. Both compute it from total_len and step_mm identically.
    int n_steps = (int)(total_len / g.step_mm);

    // March from the source toward the detector. We start half a step in
    // (the midpoint rule) so the quadrature is centered and a touch more accurate.
    float acc = 0.0f;                           // running line integral (sum mu*ds)
    for (int i = 0; i < n_steps; ++i) {
        float s = (i + 0.5f) * g.step_mm;       // distance along the ray (mm)
        Vec3 p = vec_add(g.source, vec_scale(unit, s));
        float mu = sample_trilinear(vol, v, p); // attenuation at this point (1/mm)
        acc += mu * g.step_mm;                   // rectangle of width step_mm
    }
    return acc;                                  // integral(mu ds): the DRR pixel value
}
