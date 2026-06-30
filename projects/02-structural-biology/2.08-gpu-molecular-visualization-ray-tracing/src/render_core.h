// ===========================================================================
// src/render_core.h  --  The ONE TRUE per-pixel renderer (CPU == GPU math)
// ---------------------------------------------------------------------------
// Project 2.8 : GPU Molecular Visualization & Ray Tracing
//
// WHY THIS HEADER EXISTS  (the "HD-macro" idiom, see docs/PATTERNS.md §2)
//   A ray tracer's per-pixel math is identical whether it runs on one CPU core
//   or one GPU thread. To guarantee the GPU image matches the CPU reference
//   BYTE-FOR-BYTE, we put every piece of per-ray physics here, ONCE, as
//   `__host__ __device__` inline functions:
//       * ray_sphere_t()    -- analytic ray/sphere intersection
//       * trace_nearest()   -- find the closest atom a ray hits
//       * occluded()        -- is a point shadowed from a direction? (any-hit)
//       * hemisphere_dir()  -- DETERMINISTIC ambient-occlusion sample direction
//       * shade_pixel()     -- the whole shading equation for one pixel
//   reference_cpu.cpp loops shade_pixel() over all pixels on the host; the CUDA
//   kernel (kernels.cu) calls the SAME shade_pixel() from one thread per pixel.
//   Same inputs + same operations -> the float results agree to ~1e-6, and once
//   we quantize luminance to a 0..255 byte they are EXACTLY equal (main.cu
//   verifies the byte image with zero tolerance -- see THEORY.md §verify).
//
// THE SCENE  (kept deliberately small + analytic so the demo is interpretable)
//   * Atoms are spheres: centre (x,y,z) in Angstrom, van-der-Waals radius r,
//     and a colour id (CPK-like). Real PDB atoms ARE drawn this way ("VDW" or
//     "space-filling" representation) -- this is exactly VMD's `VDW` style.
//   * The camera is ORTHOGRAPHIC, looking down -z (an axis view of the
//     molecule). Each pixel shoots one parallel ray; orthographic keeps the
//     geometry trivial so the learner sees the ray/sphere algebra clearly.
//     (THEORY.md §real-world: production uses a perspective camera + a BVH +
//     hardware ray tracing via OptiX; the math per ray is the same.)
//   * Lighting = ambient occlusion (soft "dirt in the crevices" shading that
//     makes 3-D molecular shape readable) + one directional light with a hard
//     shadow + Lambert diffuse. AO is THE reason VMD's ray-traced images look
//     so much more legible than flat OpenGL spheres.
//
// CUDA-CLEANLINESS RULE (PATTERNS.md §2)
//   This header is included by BOTH nvcc (kernels.cu) AND the plain host C++
//   compiler (reference_cpu.cpp). So it must contain NO CUDA-only constructs
//   (`__global__`, <<<>>>, kernel launches). Only `__host__ __device__` inline
//   helpers and plain structs live here. The HD macro below expands to nothing
//   under the host compiler.
//
// READ THIS AFTER: reference_cpu.h (Scene struct & loader).  Then read
//   kernels.cu (the GPU twin) and main.cu (the driver that verifies them).
// ===========================================================================
#pragma once

#include <cmath>     // sqrtf, fmaxf, fminf, floorf
#include <cstdint>   // uint32_t, uint8_t

// ---------------------------------------------------------------------------
// HD: the host/device decorator.
//   * Compiled by nvcc (__CUDACC__ defined): functions become callable from
//     BOTH host and device code.
//   * Compiled by cl.exe/g++: __host__/__device__ do not exist, so HD expands
//     to nothing and the functions are ordinary inline host functions.
// This single macro is what lets the CPU reference and the GPU kernel share the
// exact same source -> exact same arithmetic -> verifiable agreement.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Vec3: a minimal 3-component single-precision vector.
//   We roll our own (instead of CUDA's float3) so the type is identical on the
//   host and device and carries its own math -- no header drift between the two
//   compilers. FP32 is used throughout: it is what real-time renderers use, it
//   halves memory traffic vs FP64, and the GPU is far faster at it. The cost is
//   ~1e-6 rounding, which the final 8-bit quantization hides (THEORY §numerics).
// ---------------------------------------------------------------------------
struct Vec3 {
    float x, y, z;
};

// Construct a Vec3 (a free function so it is usable in both translation units).
HD inline Vec3 vec3(float x, float y, float z) { Vec3 v; v.x = x; v.y = y; v.z = z; return v; }

// Standard vector algebra. Each is one fused-multiply-add-friendly expression so
// the compiler can contract them identically on both sides.
HD inline Vec3  operator+(const Vec3& a, const Vec3& b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
HD inline Vec3  operator-(const Vec3& a, const Vec3& b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
HD inline Vec3  operator*(const Vec3& a, float s)       { return vec3(a.x * s, a.y * s, a.z * s); }
HD inline float dot(const Vec3& a, const Vec3& b)       { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD inline float length(const Vec3& a)                   { return sqrtf(dot(a, a)); }

// Normalize to unit length. Guards the zero vector (returns it unchanged) so a
// degenerate direction never produces a NaN that would diverge CPU vs GPU.
HD inline Vec3 normalize(const Vec3& a) {
    const float len = length(a);
    return (len > 0.0f) ? a * (1.0f / len) : a;
}

// ---------------------------------------------------------------------------
// Atom: one sphere in the scene. POD so it copies trivially to the GPU.
//   pos    : centre in Angstrom (world units).
//   radius : van-der-Waals radius in Angstrom (e.g. C~1.7, O~1.52, N~1.55).
//   color  : CPK colour id (0=C grey, 1=O red, 2=N blue, 3=H white, ...). The
//            id indexes the palette in shade_pixel(); we keep colours as ids so
//            the scene file stays human-readable (see data/README.md).
// ---------------------------------------------------------------------------
struct Atom {
    Vec3  pos;
    float radius;
    int   color;
};

// ---------------------------------------------------------------------------
// Camera: an orthographic view looking down -z onto the molecule.
//   The image plane spans [cx-half_w, cx+half_w] x [cy-half_h, cy+half_h] in
//   world units at z = z_plane; every primary ray points along (0,0,-1). A
//   pixel (px,py) maps to a world (x,y) by linear interpolation -- see
//   primary_ray(). Orthographic == "no perspective": parallel rays, so an
//   atom's on-screen size does not change with depth. This is the standard
//   default for inspecting a structure along a principal axis.
// ---------------------------------------------------------------------------
struct Camera {
    float cx, cy;        // world centre of the view (x,y)
    float half_w, half_h;// half-extent of the view in world units
    float z_plane;       // z of the image plane (rays start here, go to -z)
    int   width, height; // image resolution in pixels
};

// A ray: origin + (unit) direction. trace_nearest walks the scene along it.
struct Ray {
    Vec3 o;   // origin
    Vec3 d;   // direction (kept unit length)
};

// ---------------------------------------------------------------------------
// primary_ray: the orthographic ray for pixel (px, py).
//   We place a +0.5 pixel-centre offset and map the pixel grid linearly onto
//   the view rectangle. y is flipped (row 0 = top of image) so the output PPM
//   is right-way-up. Direction is straight down -z for every pixel (parallel
//   projection). This is pure geometry -- identical on host and device.
// ---------------------------------------------------------------------------
HD inline Ray primary_ray(const Camera& cam, int px, int py) {
    // u,v in [0,1): normalized pixel-centre coordinates across the image.
    const float u = (px + 0.5f) / cam.width;
    const float v = (py + 0.5f) / cam.height;
    const float wx = cam.cx + (2.0f * u - 1.0f) * cam.half_w;   // world x
    const float wy = cam.cy + (1.0f - 2.0f * v) * cam.half_h;   // world y (flipped)
    Ray r;
    r.o = vec3(wx, wy, cam.z_plane);
    r.d = vec3(0.0f, 0.0f, -1.0f);   // look down -z
    return r;
}

// ---------------------------------------------------------------------------
// ray_sphere_t: nearest positive intersection distance t of a ray with a sphere.
//   Solve |o + t d - c|^2 = r^2 with d unit length:
//       t^2 + 2 b t + cterm = 0,  b = d.(o-c),  cterm = |o-c|^2 - r^2.
//   discriminant = b^2 - cterm. If < 0 the ray misses. Otherwise the nearer
//   root is t = -b - sqrt(disc); if that is behind the origin we try -b + sqrt.
//   Returns a huge sentinel (NO_HIT) when there is no hit in front of `o`.
//   This is THE core geometric primitive of the whole renderer.
// ---------------------------------------------------------------------------
HD inline float ray_sphere_t(const Ray& ray, const Vec3& center, float radius) {
    const Vec3  oc    = ray.o - center;          // origin relative to sphere
    const float b     = dot(ray.d, oc);          // (d unit) -> half the usual 'b'
    const float cterm = dot(oc, oc) - radius * radius;
    const float disc  = b * b - cterm;           // discriminant of the quadratic
    if (disc < 0.0f) return 3.0e38f;             // miss: no real root -> NO_HIT
    const float sq = sqrtf(disc);
    float t = -b - sq;                           // near root
    if (t > 1.0e-4f) return t;                   // in front of the origin: use it
    t = -b + sq;                                 // else try the far root
    return (t > 1.0e-4f) ? t : 3.0e38f;          // 1e-4 epsilon avoids self-hit
}

// A distance bigger than any real scene hit, used as "no intersection".
HD inline float no_hit() { return 3.0e38f; }

// ---------------------------------------------------------------------------
// trace_nearest: closest atom hit by `ray`.
//   Brute force over all atoms -- O(n_atoms) per ray. For our small teaching
//   scenes (tens to a few thousand atoms) this is fine and crystal clear. The
//   *production* answer is a Bounding Volume Hierarchy (BVH) so a ray touches
//   O(log n) atoms; OptiX builds that BVH in hardware (THEORY §real-world).
//   Returns the nearest t and writes the winning atom index to *hit_idx
//   (-1 if the ray hits nothing).
// ---------------------------------------------------------------------------
HD inline float trace_nearest(const Ray& ray, const Atom* atoms, int n_atoms, int* hit_idx) {
    float best_t = no_hit();
    int   best_i = -1;
    for (int i = 0; i < n_atoms; ++i) {
        const float t = ray_sphere_t(ray, atoms[i].pos, atoms[i].radius);
        if (t < best_t) { best_t = t; best_i = i; }  // keep the nearest so far
    }
    *hit_idx = best_i;
    return best_t;
}

// ---------------------------------------------------------------------------
// occluded: ANY-HIT shadow/occlusion test.
//   Does a ray from `origin` along `dir` hit ANY atom within distance max_dist?
//   Unlike trace_nearest we can stop at the FIRST hit -- we only need a yes/no.
//   Used for (a) hard shadows toward the light and (b) ambient-occlusion
//   samples. `skip` is the surface atom we are shading, excluded so a point does
//   not shadow itself.
// ---------------------------------------------------------------------------
HD inline bool occluded(const Vec3& origin, const Vec3& dir, float max_dist,
                        const Atom* atoms, int n_atoms, int skip) {
    Ray r; r.o = origin; r.d = dir;
    for (int i = 0; i < n_atoms; ++i) {
        if (i == skip) continue;                 // don't self-shadow
        const float t = ray_sphere_t(r, atoms[i].pos, atoms[i].radius);
        if (t < max_dist) return true;           // first blocker -> occluded
    }
    return false;
}

// ---------------------------------------------------------------------------
// hemisphere_dir: a DETERMINISTIC ambient-occlusion sample direction.
//   Ambient occlusion estimates "how much of the sky is visible" at a surface
//   point by shooting several rays into the hemisphere around the normal and
//   counting how many escape. To keep the demo's stdout reproducible we MUST
//   NOT use a floating RNG whose summation order could vary -- instead we use a
//   FIXED low-discrepancy sequence: the (2-D) Hammersley point set. Sample k of
//   N is the deterministic pair
//       u = (k + 0.5)/N,         v = radical_inverse_base2(k).
//   We then map (u,v) to a COSINE-WEIGHTED hemisphere direction (more samples
//   near the normal, which is the correct AO weighting) and rotate it into the
//   frame around `n`. Because k, N and `n` fully determine the direction, the
//   host and device generate byte-identical sample sets.
//
//   radical_inverse_base2(k): mirror the bits of k about the binary point ->
//   a well-spread value in [0,1). This is the classic van der Corput sequence.
// ---------------------------------------------------------------------------
HD inline float radical_inverse_base2(uint32_t bits) {
    // Reverse the 32 bits of `bits` (standard bit-reversal), then scale to
    // [0,1). 2.3283064365386963e-10 == 1 / 2^32.
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
    bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
    bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
    return float(bits) * 2.3283064365386963e-10f;
}

HD inline Vec3 hemisphere_dir(const Vec3& n, int k, int n_samples) {
    // Hammersley (u,v) in the unit square.
    const float u = (k + 0.5f) / n_samples;
    const float v = radical_inverse_base2((uint32_t)k);

    // Cosine-weighted hemisphere sample in a canonical frame where +z == normal:
    //   r = sqrt(u) (disk radius), phi = 2*pi*v, z = sqrt(1-u).
    const float r   = sqrtf(u);
    const float phi = 6.2831853071795864769f * v;   // 2*pi*v
    const float lx  = r * cosf(phi);
    const float ly  = r * sinf(phi);
    const float lz  = sqrtf(fmaxf(0.0f, 1.0f - u));  // cosine term

    // Build an orthonormal basis (t, bvec, n) around the surface normal so we
    // can rotate the canonical sample onto the real hemisphere. The "t = n x
    // helper" trick picks a helper axis not parallel to n to avoid degeneracy.
    const Vec3 helper = (fabsf(n.x) > 0.9f) ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
    const Vec3 t      = normalize(vec3(n.y * helper.z - n.z * helper.y,
                                       n.z * helper.x - n.x * helper.z,
                                       n.x * helper.y - n.y * helper.x));  // n x helper
    const Vec3 bvec   = vec3(n.y * t.z - n.z * t.y,
                             n.z * t.x - n.x * t.z,
                             n.x * t.y - n.y * t.x);                       // n x t
    // World-space sample direction = lx*t + ly*bvec + lz*n.
    return normalize(t * lx + bvec * ly + n * lz);
}

// ---------------------------------------------------------------------------
// RenderParams: scalar shading knobs shared by CPU and GPU (so both shade
//   identically). Passed by value; tiny.
//     ao_samples : rays per pixel for ambient occlusion (more = smoother).
//     light      : UNIT direction TOWARD the directional light.
//     ambient    : floor brightness so fully-occluded pixels are not pure black.
//     ao_radius  : max distance an AO ray travels before it "escapes" (local
//                  occlusion only -- distant atoms shouldn't darken a point).
// ---------------------------------------------------------------------------
struct RenderParams {
    int   ao_samples;
    Vec3  light;
    float ambient;
    float ao_radius;
};

// ---------------------------------------------------------------------------
// cpk_luma: the base reflectance (0..1 grey) of a colour id.
//   We render a single-channel (luminance) image so the result is one number
//   per pixel -- easy to diff and to print. Each CPK id maps to a grey level
//   roughly matching its colour's brightness (white H bright, blue N darker).
//   (Extending to RGB is an Exercise; the structure is identical per channel.)
// ---------------------------------------------------------------------------
HD inline float cpk_luma(int color) {
    // index by colour id; clamp out-of-range ids to carbon-grey.
    switch (color) {
        case 0: return 0.55f;  // C  carbon  -> mid grey
        case 1: return 0.80f;  // O  oxygen  -> bright (stands in for red)
        case 2: return 0.45f;  // N  nitrogen-> darker (stands in for blue)
        case 3: return 0.95f;  // H  hydrogen-> near white
        case 4: return 0.70f;  // S  sulfur  -> yellowish, fairly bright
        default: return 0.55f;
    }
}

// ---------------------------------------------------------------------------
// shade_pixel: the COMPLETE per-pixel renderer. THE function both back-ends
//   call. Returns luminance in [0,1].
//
//   Steps (each is a teaching point in THEORY.md §algorithm):
//     1. Cast the primary ray; if it misses everything, return the background.
//     2. Compute the hit point and the outward surface normal (point - centre).
//     3. AMBIENT OCCLUSION: shoot ao_samples deterministic hemisphere rays;
//        the fraction that ESCAPE (are not occluded within ao_radius) is the
//        "openness" of this point. Crevices between atoms get few escapes ->
//        darker; exposed tops get many -> brighter. This is what gives the
//        image its 3-D, sculpted look.
//     4. DIRECT LIGHT + HARD SHADOW: Lambert term max(0, n.light), zeroed if a
//        shadow ray toward the light is occluded.
//     5. Combine: luminance = base_reflectance *
//                 (ambient + (1-ambient)*ao) * (ao_for_ambient + direct).
//        We fold AO into BOTH the ambient and a small self-term so even
//        unlit-side pixels still show shape. Clamp to [0,1].
// ---------------------------------------------------------------------------
HD inline float shade_pixel(const Camera& cam, int px, int py,
                            const Atom* atoms, int n_atoms,
                            const RenderParams& rp) {
    // (1) primary ray + nearest hit.
    const Ray ray = primary_ray(cam, px, py);
    int hit = -1;
    const float t = trace_nearest(ray, atoms, n_atoms, &hit);
    if (hit < 0) return 0.04f;   // background: a dark, non-black grey

    // (2) hit point and outward unit normal.
    const Vec3  p = ray.o + ray.d * t;
    const Atom& a = atoms[hit];
    const Vec3  n = normalize(p - a.pos);
    // Nudge the shading origin off the surface along the normal to dodge
    // self-intersection ("shadow acne") from finite float precision.
    const Vec3  surf = p + n * 1.0e-3f;

    // (3) ambient occlusion: fraction of hemisphere rays that escape.
    int escaped = 0;
    for (int s = 0; s < rp.ao_samples; ++s) {
        const Vec3 dir = hemisphere_dir(n, s, rp.ao_samples);
        if (!occluded(surf, dir, rp.ao_radius, atoms, n_atoms, hit))
            ++escaped;
    }
    // Guard ao_samples==0 (no AO requested) -> fully open.
    const float ao = (rp.ao_samples > 0)
                   ? (float)escaped / (float)rp.ao_samples
                   : 1.0f;

    // (4) direct directional light with a hard shadow.
    float diffuse = fmaxf(0.0f, dot(n, rp.light));   // Lambert cosine
    if (diffuse > 0.0f) {
        // Shadow ray toward the light; if blocked before "infinity", no direct.
        if (occluded(surf, rp.light, no_hit(), atoms, n_atoms, hit))
            diffuse = 0.0f;
    }

    // (5) combine. ambient floor keeps occluded crevices visible; AO modulates
    //     the ambient; direct light adds the bright, shadow-cut highlight.
    const float base = cpk_luma(a.color);
    float luma = base * (rp.ambient * ao + (1.0f - rp.ambient) * diffuse * ao + 0.15f * ao);
    // Clamp into displayable range.
    luma = fminf(1.0f, fmaxf(0.0f, luma));
    return luma;
}

// ---------------------------------------------------------------------------
// quantize8: map luminance [0,1] -> an 8-bit value 0..255.
//   We render in float but VERIFY (and store) the QUANTIZED byte image. Why:
//   the only CPU/GPU difference is ~1e-6 float rounding from FMA contraction;
//   after rounding to one of 256 levels those differences vanish, so the byte
//   images are EXACTLY equal and we can verify with zero tolerance (THEORY
//   §verify). +0.5f gives round-to-nearest. (Same identical op on both sides.)
// ---------------------------------------------------------------------------
HD inline unsigned char quantize8(float luma) {
    int q = (int)(luma * 255.0f + 0.5f);
    if (q < 0)   q = 0;
    if (q > 255) q = 255;
    return (unsigned char)q;
}
