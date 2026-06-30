// ===========================================================================
// src/reference_cpu.h  --  Scene model, loader, and the CPU reference renderer
// ---------------------------------------------------------------------------
// Project 2.8 : GPU Molecular Visualization & Ray Tracing
//
// WHAT THIS PROJECT COMPUTES
//   Given a molecule as a list of atoms (centre + van-der-Waals radius +
//   colour), render a 2-D image of it as overlapping shaded spheres -- the
//   "space-filling" / "VDW" representation used by VMD, PyMOL and Mol*. Each
//   output pixel shoots a ray, finds the nearest atom it hits, and shades that
//   point with ambient occlusion + a directional light + a hard shadow. The
//   result is a greyscale image (one luminance byte per pixel).
//
// WHY A GPU  (docs/PATTERNS.md §1 -- "per-output-pixel gather")
//   Every output pixel is INDEPENDENT: it reads the shared scene and writes its
//   own pixel, with no communication between pixels. That is the textbook GPU
//   "gather" pattern (the same shape as 4.01 CT backprojection). So we give each
//   pixel its own thread. A frame is width*height*(1 + ao_samples + 1) rays;
//   even our tiny demo is well over a million ray/sphere tests -- embarrassingly
//   parallel and a perfect fit for the GPU. Real viewers push this to millions
//   of atoms at 30+ fps using hardware ray tracing (OptiX/RTX).
//
//   This header is PURE C++ (no CUDA): it is compiled by the host compiler for
//   the reference path AND #included by kernels.cu (nvcc tolerates plain C++).
//   The per-pixel physics lives in render_core.h (shared by CPU and GPU).
//
// READ THIS AFTER: render_core.h (the shared math).  Then: reference_cpu.cpp
//   (this file's implementation), kernels.cu (GPU twin), main.cu (driver).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "render_core.h"   // Atom, Camera, RenderParams, shade_pixel, quantize8

// One render problem: the molecule + the camera + the shading parameters.
//   atoms : the molecule, one Atom per row of the scene file.
//   cam   : orthographic view (auto-fitted to the molecule by load_scene()).
//   rp    : shading knobs (AO sample count, light direction, ambient floor).
// The image dimensions live inside `cam` (cam.width, cam.height).
struct Scene {
    std::vector<Atom> atoms;
    Camera            cam;
    RenderParams      rp;
};

// ---------------------------------------------------------------------------
// load_scene: parse the tiny text scene format (see data/README.md).
//   Header line:  "<n_atoms> <width> <height> <ao_samples>"
//   Then n_atoms lines: "<x> <y> <z> <radius> <color>"
//   After reading the atoms we AUTO-FIT the orthographic camera to the
//   molecule's bounding box (plus a margin) so any scene frames itself nicely,
//   and we set fixed, documented shading parameters so the output is
//   deterministic. Throws std::runtime_error on a malformed file so demos fail
//   loudly rather than render garbage.
// ---------------------------------------------------------------------------
Scene load_scene(const std::string& path);

// ---------------------------------------------------------------------------
// render_cpu: the trusted serial reference. Loops over every pixel (row-major)
//   and calls the shared shade_pixel(), storing the QUANTIZED 0..255 luminance.
//   `image` is resized to width*height bytes. This is the baseline the GPU
//   kernel is verified against (main.cu compares the two byte images).
// ---------------------------------------------------------------------------
void render_cpu(const Scene& scene, std::vector<unsigned char>& image);

// ---------------------------------------------------------------------------
// image_checksum: a fully deterministic 32-bit rolling checksum of a byte
//   image (FNV-1a). Printed to stdout as a compact, stable fingerprint of the
//   whole frame -- so expected_output.txt captures the ENTIRE image in one line
//   without dumping thousands of pixels.
// ---------------------------------------------------------------------------
unsigned int image_checksum(const std::vector<unsigned char>& image);

// ---------------------------------------------------------------------------
// write_pgm: save a byte image as a binary PGM (portable greymap) so the
//   learner can actually LOOK at the render. Pure host I/O; optional (the demo
//   verifies via the checksum, not the file). Returns false on write failure.
// ---------------------------------------------------------------------------
bool write_pgm(const std::string& path, const std::vector<unsigned char>& image,
               int width, int height);
