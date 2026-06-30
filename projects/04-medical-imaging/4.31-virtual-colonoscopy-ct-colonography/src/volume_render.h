// ===========================================================================
// src/volume_render.h  --  Shared __host__ __device__ ray-casting core
// ---------------------------------------------------------------------------
// Project 4.31 : Virtual Colonoscopy & CT Colonography
//
// WHY THIS FILE EXISTS  (PATTERNS.md §2 -- the single most useful idiom)
//   The per-pixel rendering math must run IDENTICALLY on the CPU reference and
//   the GPU kernel, otherwise "GPU matches CPU" verification is meaningless. So
//   we put every per-ray / per-sample primitive HERE as `__host__ __device__`
//   inline functions:
//       * trilinear sampling of the CT volume (the "3D texture" lookup, done by
//         hand so CPU and GPU agree bit-for-bit -- see THEORY §6),
//       * the central-difference gradient (the surface normal estimate),
//       * the Phong shading model,
//       * the single-ray volume ray-cast that composites along the ray.
//   reference_cpu.cpp loops these over every pixel; kernels.cu calls the SAME
//   `cast_ray()` from one thread per pixel. Same code -> same answer.
//
//   This header is included by BOTH the host compiler (via reference_cpu.cpp /
//   main.cu) AND nvcc (via kernels.cu). It must therefore contain NO __global__
//   kernels and NO CUDA-only types -- just plain math decorated with the HD
//   macro so the decorators vanish under the host compiler.
//
// THE PROBLEM, IN ONE BREATH
//   A CT colonography study is a 3-D grid of X-ray densities. The colon has been
//   inflated with air, so its hollow interior (the "lumen") reads as very low
//   density while the surrounding tissue wall reads higher. A virtual-colonoscopy
//   fly-through places a camera INSIDE the lumen and renders the inner wall, the
//   way an optical colonoscope would -- but non-invasively, from the CT. Each
//   output pixel is one ray cast from the camera into the volume; where the ray
//   crosses the air->wall density boundary we shade the surface. That per-pixel,
//   embarrassingly-parallel gather is the GPU teaching point of this project.
//
// READ THIS AFTER: reference_cpu.h (the Volume struct + scene description).
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// HD: the "host/device" decorator macro (PATTERNS.md §2).
//   * Under nvcc (__CUDACC__ defined) it expands to `__host__ __device__`, so
//     each inline function is compiled for BOTH the CPU and the GPU.
//   * Under a plain host compiler the decorators do not exist, so HD expands to
//     nothing and the very same source compiles as ordinary C++.
//   We deliberately avoid CUDA vector types (float3 etc.) so the host compiler
//   needs no CUDA headers; a tiny Vec3 below stands in and is trivially
//   constexpr-friendly.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#include <cmath>   // std::floor, std::sqrt, std::fmax, std::fmin (host + device)

// ---------------------------------------------------------------------------
// Vec3: a minimal 3-component vector used for world positions, ray directions,
//   and surface normals. We roll our own (rather than CUDA's float3) so this
//   header stays CUDA-free for the host compiler. Everything is float because
//   the rendered image is FP32 and FP32 trilinear interpolation matches between
//   CPU and GPU exactly (THEORY §5). All members public; it is plain data.
// ---------------------------------------------------------------------------
struct Vec3 {
    float x, y, z;
};

HD inline Vec3 vmake(float x, float y, float z) { Vec3 v; v.x = x; v.y = y; v.z = z; return v; }
HD inline Vec3 vadd(Vec3 a, Vec3 b)   { return vmake(a.x + b.x, a.y + b.y, a.z + b.z); }
HD inline Vec3 vsub(Vec3 a, Vec3 b)   { return vmake(a.x - b.x, a.y - b.y, a.z - b.z); }
HD inline Vec3 vscale(Vec3 a, float s){ return vmake(a.x * s, a.y * s, a.z * s); }
HD inline float vdot(Vec3 a, Vec3 b)  { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Normalize to unit length. A zero vector (e.g. a flat region with no gradient)
// would divide by zero, so we guard with a tiny epsilon and return it unchanged
// scaled to ~0 -- the caller treats a near-zero gradient as "no surface here".
HD inline Vec3 vnorm(Vec3 a) {
    float len = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    if (len < 1e-8f) return vmake(0.0f, 0.0f, 0.0f);
    float inv = 1.0f / len;
    return vmake(a.x * inv, a.y * inv, a.z * inv);
}

// ---------------------------------------------------------------------------
// VolumeView: a *non-owning* description of the CT density grid plus the few
//   scalars the renderer needs. Both the host vector<float> and the device
//   float* are wrapped in one of these (same struct, different `data` pointer),
//   so cast_ray() does not care whether it runs on the CPU or GPU.
//
//   Layout: data[(z*ny + y)*nx + x] -- x fastest, then y, then z (the standard
//   row-major volume layout). Densities are in synthetic "CT-ish" units where
//   air-filled lumen ~ 0 and tissue wall ~ 1 (see scripts/make_synthetic.py and
//   THEORY §1; real CT uses Hounsfield Units, air = -1000 HU, soft tissue ~ 0).
// ---------------------------------------------------------------------------
struct VolumeView {
    const float* data;   // density grid (host or device pointer; not owned)
    int   nx, ny, nz;    // grid dimensions in voxels
    float iso;           // iso-value of the lumen<->wall surface we render
    float step;          // ray-march step length, in voxel units (<1 = oversample)
    int   max_steps;     // hard cap on march steps so a ray always terminates
};

// ---------------------------------------------------------------------------
// clampi: keep an index inside [0, hi]. Used so trilinear sampling near the grid
//   border reads the edge voxel ("clamp to edge" addressing) instead of reading
//   out of bounds -- the same border behavior a CUDA texture in clamp mode gives.
// ---------------------------------------------------------------------------
HD inline int clampi(int v, int hi) {
    if (v < 0)  return 0;
    if (v > hi) return hi;
    return v;
}

// ---------------------------------------------------------------------------
// voxel(): fetch a single voxel with clamp-to-edge addressing. Centralizing the
//   index math here means the trilinear sampler and the gradient both use the
//   exact same out-of-range policy.
//   Index: (z*ny + y)*nx + x  (see VolumeView layout note).
// ---------------------------------------------------------------------------
HD inline float voxel(const VolumeView& V, int x, int y, int z) {
    x = clampi(x, V.nx - 1);
    y = clampi(y, V.ny - 1);
    z = clampi(z, V.nz - 1);
    return V.data[(size_t)(z * V.ny + y) * V.nx + x];
}

// ---------------------------------------------------------------------------
// sample_volume(): TRILINEAR interpolation of the density at a continuous point
//   p = (px,py,pz) in voxel coordinates. This is THE "3-D texture lookup" the
//   catalog mentions -- only here we do the 8-corner blend by hand so the CPU
//   and GPU produce identical bits (a hardware texture unit uses 9-bit fixed
//   fractional weights and would NOT match our FP32 reference; THEORY §6).
//
//   Steps:
//     1. Split each coordinate into an integer corner (x0) and a fraction (fx).
//     2. Read the 8 surrounding voxels.
//     3. Blend along x, then y, then z (the classic nested lerp).
//   Cost: 8 memory reads + ~7 fmas per sample. This function dominates the
//   project's runtime, which is exactly why the GPU's many parallel samplers win.
// ---------------------------------------------------------------------------
HD inline float sample_volume(const VolumeView& V, Vec3 p) {
    float fx = std::floor(p.x), fy = std::floor(p.y), fz = std::floor(p.z);
    int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;   // lower corner of the cell
    float tx = p.x - fx, ty = p.y - fy, tz = p.z - fz;  // fractional offsets [0,1)

    // The 8 voxel values at the cube corners around p.
    float c000 = voxel(V, x0,   y0,   z0  );
    float c100 = voxel(V, x0+1, y0,   z0  );
    float c010 = voxel(V, x0,   y0+1, z0  );
    float c110 = voxel(V, x0+1, y0+1, z0  );
    float c001 = voxel(V, x0,   y0,   z0+1);
    float c101 = voxel(V, x0+1, y0,   z0+1);
    float c011 = voxel(V, x0,   y0+1, z0+1);
    float c111 = voxel(V, x0+1, y0+1, z0+1);

    // Interpolate along x (4 edges), then y (2 edges), then z (1 result).
    // Written as a*(1-t) + b*t so the operation order matches the CPU reference
    // exactly -- determinism comes from doing the SAME fmas in the SAME order.
    float c00 = c000 * (1.0f - tx) + c100 * tx;
    float c10 = c010 * (1.0f - tx) + c110 * tx;
    float c01 = c001 * (1.0f - tx) + c101 * tx;
    float c11 = c011 * (1.0f - tx) + c111 * tx;
    float c0  = c00 * (1.0f - ty) + c10 * ty;
    float c1  = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

// ---------------------------------------------------------------------------
// gradient(): estimate the surface normal at p by CENTRAL DIFFERENCES of the
//   density field. The gradient of density points from low density (lumen air)
//   toward high density (tissue wall), i.e. INTO the wall; we negate it so the
//   normal points back toward the camera/light (out of the wall) for shading.
//   This is the "gradient-magnitude" surface estimate the catalog calls for; a
//   production renderer often precomputes a gradient volume, we compute it on
//   the fly (cheap, and avoids storing a second 3-channel volume).
//   h = 1 voxel: large enough to be numerically stable, small enough to be local.
// ---------------------------------------------------------------------------
HD inline Vec3 gradient(const VolumeView& V, Vec3 p) {
    const float h = 1.0f;
    float gx = sample_volume(V, vmake(p.x + h, p.y, p.z)) - sample_volume(V, vmake(p.x - h, p.y, p.z));
    float gy = sample_volume(V, vmake(p.x, p.y + h, p.z)) - sample_volume(V, vmake(p.x, p.y - h, p.z));
    float gz = sample_volume(V, vmake(p.x, p.y, p.z + h)) - sample_volume(V, vmake(p.x, p.y, p.z - h));
    // Negate: density increases into the wall, but we want the OUTWARD normal.
    return vmake(-gx, -gy, -gz);
}

// ---------------------------------------------------------------------------
// phong(): the classic local illumination model. Given a unit surface normal N,
//   a unit direction-to-light L, and a unit view direction Vd (here L == Vd: a
//   "headlamp" mounted on the virtual endoscope, which is how clinical CTC
//   fly-throughs are lit), return a grayscale intensity in [0,1]:
//
//       I = ambient + diffuse*max(N.L, 0) + specular*max(N.H, 0)^shininess
//
//   * ambient  : a small floor so surfaces facing away are dim, not pure black.
//   * diffuse  : Lambertian term -- brightest where the wall faces the light.
//   * specular : a tight highlight (Blinn-Phong half-vector H) that gives the
//                moist mucosal sheen radiologists are used to.
//   The numbers are tuned for a legible grayscale image, not for radiometric
//   accuracy (this is a teaching renderer; THEORY §7).
// ---------------------------------------------------------------------------
HD inline float phong(Vec3 N, Vec3 L, Vec3 Vd) {
    const float ambient = 0.15f, diffuse = 0.70f, specular = 0.35f, shininess = 16.0f;
    float ndotl = vdot(N, L);
    if (ndotl < 0.0f) ndotl = 0.0f;                 // light only the lit side
    // Blinn-Phong half vector H = normalize(L + V); N.H peaks at the mirror dir.
    Vec3  H = vnorm(vadd(L, Vd));
    float ndoth = vdot(N, H);
    if (ndoth < 0.0f) ndoth = 0.0f;
    float spec = std::pow(ndoth, shininess);
    float I = ambient + diffuse * ndotl + specular * spec;
    if (I > 1.0f) I = 1.0f;                          // clamp to displayable range
    return I;
}

// ---------------------------------------------------------------------------
// cast_ray(): the heart of the renderer -- march ONE ray and return its shaded
//   pixel value. This is what each GPU thread (and each CPU loop iteration) runs.
//
//   Algorithm (front-to-back first-hit iso-surface ray-cast):
//     1. Start at `origin` (the camera, inside the lumen) and step along `dir`
//        (a unit ray direction) by V.step voxels at a time.
//     2. At each step trilinearly sample the density. While we are still in the
//        air-filled lumen (density < iso) keep going. The first time the density
//        crosses `iso` we have hit the colon WALL.
//     3. On a hit, refine the crossing with one bisection step (so the surface
//        position is sub-step accurate -> smoother shading), estimate the normal
//        from the gradient, and Phong-shade with a headlamp light. Return that.
//     4. If we exhaust max_steps without a hit, the ray escaped into open lumen
//        / background -> return 0 (black). That black fraction is a deterministic,
//        verifiable statistic the demo reports.
//
//   Returns a grayscale intensity in [0,1]. No atomics, no shared memory -- each
//   ray is fully independent, which is precisely why this maps so cleanly onto
//   one-thread-per-pixel (THEORY §4).
// ---------------------------------------------------------------------------
HD inline float cast_ray(const VolumeView& V, Vec3 origin, Vec3 dir) {
    Vec3  p     = origin;
    float prev  = sample_volume(V, p);      // density at the camera (should be air)
    Vec3  stepv = vscale(dir, V.step);      // world advance per march step

    for (int i = 0; i < V.max_steps; ++i) {
        Vec3  np  = vadd(p, stepv);
        float cur = sample_volume(V, np);

        // Detect the air(low)->wall(high) crossing of the iso-surface.
        if (prev < V.iso && cur >= V.iso) {
            // One bisection refinement: linearly interpolate the crossing point
            // between p (below iso) and np (above iso) so the hit is sub-step
            // accurate. t in [0,1] is where density == iso along this segment.
            float denom = (cur - prev);
            float t = (denom != 0.0f) ? (V.iso - prev) / denom : 0.5f;
            Vec3 hit = vadd(p, vscale(stepv, t));

            // Surface normal from the density gradient at the hit point.
            Vec3 N = vnorm(gradient(V, hit));
            // Headlamp: the light and the view both point back along -dir (from
            // the surface toward the camera). This is the CTC lighting model.
            Vec3 toCam = vscale(dir, -1.0f);
            return phong(N, toCam, toCam);
        }
        // Advance one step and carry the density forward (1 sample/step, not 2).
        p = np;
        prev = cur;
    }
    return 0.0f;   // ray escaped without hitting a wall -> background (black)
}
