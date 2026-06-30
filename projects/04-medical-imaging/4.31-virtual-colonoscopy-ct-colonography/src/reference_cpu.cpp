// ===========================================================================
// src/reference_cpu.cpp  --  Volume loader + serial reference renderer
// ---------------------------------------------------------------------------
// Project 4.31 : Virtual Colonoscopy & CT Colonography
//
// Compiled by the host C++ compiler only (no CUDA here). It provides:
//   * load_scene() -- parse the synthetic CTC sample and place the camera.
//   * render_cpu() -- the trusted SERIAL renderer (one ray per pixel, looped).
// The per-ray math itself lives in volume_render.h and is SHARED with the GPU
// kernel, so render_cpu() and the kernel compute identical images (THEORY §6).
//
// READ THIS AFTER: reference_cpu.h, volume_render.h.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_scene(): read the header + density grid, then build the endoscope camera.
//
//   The CAMERA is derived purely from the volume size so it is deterministic
//   (same file -> same camera -> same image -> stable expected_output.txt):
//     * eye      : centered in x/y, just inside the near face in z -> sitting in
//                  the air-filled lumen at the "mouth" of the tube.
//     * forward  : +z, i.e. looking straight down the colon (the fly-through
//                  direction). right/up complete a right-handed image basis.
//   We do NOT read the camera from the file: keeping it implicit means the demo
//   cannot drift if someone hand-edits the sample's geometry header.
// ---------------------------------------------------------------------------
Scene load_scene(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open volume file: " + path);

    Scene s;
    if (!(in >> s.nx >> s.ny >> s.nz >> s.iso >> s.step >> s.max_steps >> s.width >> s.height))
        throw std::runtime_error(
            "bad header (expected nx ny nz iso step max_steps width height) in " + path);
    if (s.nx <= 0 || s.ny <= 0 || s.nz <= 0)
        throw std::runtime_error("non-positive volume dimensions in " + path);
    if (s.width <= 0 || s.height <= 0)
        throw std::runtime_error("non-positive image dimensions in " + path);
    if (s.step <= 0.0f || s.max_steps <= 0)
        throw std::runtime_error("non-positive march parameters in " + path);

    // Read the nx*ny*nz density grid (x fastest, then y, then z).
    const size_t n = (size_t)s.nx * s.ny * s.nz;
    s.vol.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (!(in >> s.vol[i]))
            throw std::runtime_error("density grid truncated in " + path);
    }

    // ---- Build the deterministic virtual-endoscope camera -----------------
    Camera& c = s.cam;
    // Eye: centered in x/y, 2 voxels in from the near z face -> inside the lumen.
    c.eye     = vmake(s.nx * 0.5f, s.ny * 0.5f, 2.0f);
    c.forward = vmake(0.0f, 0.0f, 1.0f);   // look down the tube (+z)
    c.right   = vmake(1.0f, 0.0f, 0.0f);   // image +x
    c.up      = vmake(0.0f, 1.0f, 0.0f);   // image +y
    // fov_scale ~ tan(half-FOV). 1.0 gives a 90-degree-ish wide view, close to a
    // real colonoscope's wide angle so the whole lumen wall is visible.
    c.fov_scale = 1.0f;

    return s;
}

// ---------------------------------------------------------------------------
// render_cpu(): the serial baseline. For each output pixel (px,py) we generate
//   the ray with pixel_ray() and shade it with cast_ray() -- exactly the two
//   shared functions the GPU thread will call. Looping them here, on the CPU,
//   gives the reference image. O(width*height*max_steps*8) memory reads: this is
//   the cost the GPU parallelizes across pixels.
// ---------------------------------------------------------------------------
void render_cpu(const Scene& scene, std::vector<float>& image) {
    const int W = scene.width, H = scene.height;
    image.assign((size_t)W * H, 0.0f);

    // The VolumeView over the HOST data pointer. The GPU path makes the same view
    // over a DEVICE pointer; everything else is identical.
    const VolumeView V = scene.view(scene.vol.data());

    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            Vec3 origin, dir;
            pixel_ray(scene.cam, px, py, W, H, origin, dir);  // ray for this pixel
            float shade = cast_ray(V, origin, dir);           // march + shade
            image[(size_t)py * W + px] = shade;               // store intensity
        }
    }
}
