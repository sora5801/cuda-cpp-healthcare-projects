// ===========================================================================
// src/reference_cpu.h  --  Scene description, volume loader, CPU reference
// ---------------------------------------------------------------------------
// Project 4.31 : Virtual Colonoscopy & CT Colonography
//
// WHAT THIS PROJECT COMPUTES
//   A single VIRTUAL COLONOSCOPY fly-through FRAME. The input is a small 3-D CT
//   volume of an air-distended colon (synthetic: a curved tube through soft
//   tissue, with one polyp bump on the wall -- the "known answer"). The output
//   is a shaded 2-D image, as seen by a virtual endoscope camera sitting INSIDE
//   the lumen looking down the tube. We render it by VOLUME RAY-CASTING: one ray
//   per output pixel, marched until it hits the air->wall iso-surface, then
//   Phong-shaded from the surface normal (the density gradient). See THEORY.md.
//
// WHY A GPU
//   Rendering is a per-PIXEL GATHER: every pixel's ray is independent and reads
//   many trilinear volume samples (8 voxels each). A clinical CTC fly-through is
//   a 512^3 volume rendered at 60 frames/s -- billions of samples per second,
//   hopeless serially but ideal for the GPU's thousands of parallel samplers and
//   its texture-interpolation hardware. We render ONE modest frame so the
//   geometry and the gather are easy to follow.
//
// FILE ROLES
//   * This header declares the scene/Volume types and the host-side functions.
//   * reference_cpu.cpp implements the loader + the SERIAL renderer (the trusted
//     baseline the GPU is checked against).
//   * volume_render.h holds the per-ray math shared by CPU and GPU (the real
//     teaching content). kernels.cu wraps the same cast_ray() in a CUDA kernel.
//
//   This header is pure C++ (no CUDA), so kernels.cu can include it safely.
//
// READ THIS AFTER: volume_render.h (the per-ray primitives this orchestrates).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "volume_render.h"   // VolumeView, Vec3, cast_ray (pure math, CUDA-free)

// ---------------------------------------------------------------------------
// Camera: the virtual endoscope. It sits at `eye` inside the lumen and looks
//   along `forward`; `right` and `up` span the image plane. `fov_scale` sets how
//   wide the field of view is (larger = more fish-eye, like a real colonoscope).
//   All vectors are in VOXEL coordinates (same space as the volume), which keeps
//   the ray math trivial -- no separate world<->voxel transform to get wrong.
// ---------------------------------------------------------------------------
struct Camera {
    Vec3  eye;
    Vec3  forward;     // unit
    Vec3  right;       // unit, image +x
    Vec3  up;          // unit, image +y
    float fov_scale;   // half-extent of the image plane at unit distance
};

// ---------------------------------------------------------------------------
// Scene: everything needed to render one frame -- the loaded density volume, its
//   dimensions, the render parameters (iso/step/max_steps), the output image
//   size, and the camera. load_scene() fills this from the sample file.
// ---------------------------------------------------------------------------
struct Scene {
    std::vector<float> vol;     // [nx*ny*nz] density grid (row-major; x fastest)
    int   nx = 0, ny = 0, nz = 0;
    float iso = 0.5f;           // lumen<->wall iso-value to render
    float step = 0.5f;          // ray-march step in voxels (oversample at 0.5)
    int   max_steps = 0;        // safety cap on steps per ray
    int   width = 0, height = 0;// output image dimensions (pixels)
    Camera cam;                 // the virtual endoscope

    // Build a VolumeView (the renderer's lightweight handle) over a given data
    // pointer -- host vector data for the CPU path, device pointer for the GPU
    // path. Centralizing this guarantees both paths see identical parameters.
    VolumeView view(const float* data) const {
        VolumeView V;
        V.data = data; V.nx = nx; V.ny = ny; V.nz = nz;
        V.iso = iso; V.step = step; V.max_steps = max_steps;
        return V;
    }
};

// ---------------------------------------------------------------------------
// pixel_ray(): map output pixel (px,py) to its world-space ray (origin,dir).
//   Marked HD so the SAME ray-generation runs on the CPU reference and inside
//   the GPU kernel -- identical rays are a precondition for identical images.
//     origin : the camera eye (same for every pixel).
//     dir    : unit direction through the pixel center on the image plane.
// ---------------------------------------------------------------------------
HD inline void pixel_ray(const Camera& cam, int px, int py, int width, int height,
                         Vec3& origin, Vec3& dir) {
    // Normalized image coords in [-1,1]*fov, with (0,0) at the image center. We
    // add 0.5 to hit pixel CENTERS (not corners), the usual sampling convention.
    float u = (2.0f * (px + 0.5f) / width  - 1.0f) * cam.fov_scale;
    float v = (2.0f * (py + 0.5f) / height - 1.0f) * cam.fov_scale;
    // Ray dir = forward + u*right + v*up, then normalized to unit length so the
    // march step length is the same in every direction.
    Vec3 d = vadd(cam.forward, vadd(vscale(cam.right, u), vscale(cam.up, v)));
    origin = cam.eye;
    dir = vnorm(d);
}

// ---------------------------------------------------------------------------
// load_scene(): read the synthetic CTC sample (text format, see data/README.md):
//     header : "<nx> <ny> <nz> <iso> <step> <max_steps> <width> <height>"
//     body   : nx*ny*nz density floats (x fastest, then y, then z)
//   It then derives the virtual-endoscope camera deterministically from the
//   volume size (placed in the lumen at the tube's mouth, looking down +z), so
//   the rendered frame -- and thus expected_output.txt -- is fully reproducible.
//   Throws std::runtime_error on any malformed input so the demo fails loudly.
// ---------------------------------------------------------------------------
Scene load_scene(const std::string& path);

// ---------------------------------------------------------------------------
// render_cpu(): the SERIAL reference renderer. Loops every pixel, builds its ray
//   with pixel_ray(), casts it with cast_ray() (the shared core), and stores the
//   shaded intensity. `image` is sized to width*height, row-major (x fastest).
//   This is the trusted baseline; main.cu checks the GPU image against it.
// ---------------------------------------------------------------------------
void render_cpu(const Scene& scene, std::vector<float>& image);
