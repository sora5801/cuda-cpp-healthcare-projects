// ===========================================================================
// src/reference_cpu.cpp  --  Scene loader + serial reference renderer
// ---------------------------------------------------------------------------
// Project 2.8 : GPU Molecular Visualization & Ray Tracing
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU image is checked against. It is written to be
//   OBVIOUSLY correct -- one readable double loop over pixels, no parallelism,
//   no cleverness -- so that when the GPU and CPU images agree, we believe the
//   GPU. The per-pixel shading itself lives in render_core.h (shade_pixel), the
//   SAME function the GPU kernel calls, which is why agreement is exact.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: render_core.h, reference_cpu.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::max, std::min
#include <fstream>     // std::ifstream / std::ofstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_scene: read the tiny atom list and auto-fit an orthographic camera.
// ---------------------------------------------------------------------------
Scene load_scene(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open scene file: " + path);

    Scene sc;
    int n_atoms = 0, width = 0, height = 0, ao_samples = 0;
    // Header: number of atoms, image size, and AO sample count.
    if (!(in >> n_atoms >> width >> height >> ao_samples))
        throw std::runtime_error("bad header (expected: n_atoms width height ao_samples) in " + path);
    if (n_atoms <= 0 || width <= 0 || height <= 0 || ao_samples < 0)
        throw std::runtime_error("non-positive / invalid header values in " + path);

    // Read each atom row: x y z radius color.
    sc.atoms.reserve(static_cast<std::size_t>(n_atoms));
    for (int i = 0; i < n_atoms; ++i) {
        Atom a;
        float x, y, z, r;
        int color;
        if (!(in >> x >> y >> z >> r >> color))
            throw std::runtime_error("scene truncated at atom " + std::to_string(i) + " in " + path);
        a.pos = vec3(x, y, z);
        a.radius = r;
        a.color = color;
        sc.atoms.push_back(a);
    }

    // ---- Auto-fit the orthographic camera to the molecule's extent ---------
    // Compute the bounding box of every atom INCLUDING its radius, so spheres
    // are not clipped at the image edge. The camera then frames that box with a
    // small margin, keeping the aspect ratio square-ish (we use the larger of
    // the x/y half-extents for both, scaled by the pixel aspect).
    float min_x = sc.atoms[0].pos.x, max_x = min_x;
    float min_y = sc.atoms[0].pos.y, max_y = min_y;
    float max_z = sc.atoms[0].pos.z;
    for (const Atom& a : sc.atoms) {
        min_x = std::min(min_x, a.pos.x - a.radius);
        max_x = std::max(max_x, a.pos.x + a.radius);
        min_y = std::min(min_y, a.pos.y - a.radius);
        max_y = std::max(max_y, a.pos.y + a.radius);
        max_z = std::max(max_z, a.pos.z + a.radius);
    }
    const float cx = 0.5f * (min_x + max_x);     // view centre
    const float cy = 0.5f * (min_y + max_y);
    const float margin = 1.15f;                  // 15% padding around the box
    float half_w = 0.5f * (max_x - min_x) * margin;
    float half_h = 0.5f * (max_y - min_y) * margin;
    // Keep pixels square: stretch the smaller half-extent so world-per-pixel is
    // equal in x and y (no anisotropic squashing of the molecule).
    const float px_aspect = (float)width / (float)height;
    const float box_aspect = (half_h > 0.0f) ? (half_w / half_h) : 1.0f;
    if (box_aspect < px_aspect) half_w = half_h * px_aspect;  // pad width
    else                        half_h = half_w / px_aspect;  // pad height

    sc.cam.cx = cx; sc.cam.cy = cy;
    sc.cam.half_w = half_w; sc.cam.half_h = half_h;
    sc.cam.z_plane = max_z + 5.0f;   // start rays in front of the molecule (+z)
    sc.cam.width = width; sc.cam.height = height;

    // ---- Fixed, documented shading parameters (determinism) ----------------
    sc.rp.ao_samples = ao_samples;
    // Light from the upper-front-right, normalized. A fixed direction keeps the
    // image reproducible. (Exercise: animate it; rotate the molecule instead.)
    sc.rp.light = normalize(vec3(0.4f, 0.6f, 1.0f));
    sc.rp.ambient = 0.45f;     // floor brightness in shadowed crevices
    // AO rays travel ~3 atom radii before "escaping"; local occlusion only, so a
    // distant atom on the far side of the molecule does not darken this point.
    sc.rp.ao_radius = 6.0f;
    return sc;
}

// ---------------------------------------------------------------------------
// render_cpu: the serial baseline. One pass over pixels, row by row.
//   image[py*width + px] = quantized luminance of that pixel.
//   This is the function whose wall time (timed in main.cu) we compare with the
//   GPU kernel, and whose pixels we compare against the GPU's.
// ---------------------------------------------------------------------------
void render_cpu(const Scene& scene, std::vector<unsigned char>& image) {
    const int W = scene.cam.width, H = scene.cam.height;
    image.assign(static_cast<std::size_t>(W) * H, 0);

    const Atom* atoms = scene.atoms.data();
    const int   n     = static_cast<int>(scene.atoms.size());

    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            // The whole pixel is computed by the SHARED shade_pixel(): cast the
            // primary ray, find the nearest atom, do AO + light + shadow. The
            // GPU kernel calls this identical function -> identical result.
            const float luma = shade_pixel(scene.cam, px, py, atoms, n, scene.rp);
            image[static_cast<std::size_t>(py) * W + px] = quantize8(luma);
        }
    }
}

// ---------------------------------------------------------------------------
// image_checksum: FNV-1a over the byte image.
//   FNV-1a is a tiny, well-known, deterministic non-cryptographic hash: start
//   from a fixed basis, then for each byte XOR it in and multiply by a fixed
//   prime. Order-sensitive (so any pixel difference changes the hash) yet
//   trivial to compute identically anywhere -> a perfect one-line fingerprint
//   of an entire frame for expected_output.txt.
// ---------------------------------------------------------------------------
unsigned int image_checksum(const std::vector<unsigned char>& image) {
    unsigned int h = 2166136261u;            // FNV offset basis (32-bit)
    for (unsigned char b : image) {
        h ^= static_cast<unsigned int>(b);   // mix the byte in
        h *= 16777619u;                      // FNV prime (wraps mod 2^32)
    }
    return h;
}

// ---------------------------------------------------------------------------
// write_pgm: dump a binary PGM (P5) so the image can be opened in any viewer.
//   PGM is the simplest possible greyscale format: a tiny ASCII header
//   ("P5\n<width> <height>\n255\n") followed by width*height raw bytes.
// ---------------------------------------------------------------------------
bool write_pgm(const std::string& path, const std::vector<unsigned char>& image,
               int width, int height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.data()),
              static_cast<std::streamsize>(image.size()));
    return static_cast<bool>(out);
}
