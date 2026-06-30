// ===========================================================================
// src/reference_cpu.cpp  --  CT loader, demo geometry, serial DRR reference
// ---------------------------------------------------------------------------
// Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable double loop over detector
//   pixels, no parallelism -- so that when the GPU and CPU agree, we believe the
//   GPU. The per-ray physics it calls (integrate_ray) is the SAME code the GPU
//   kernel calls, so agreement is exact up to float rounding.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, drr_core.h. Compare against kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_volume: parse the tiny text CT format and convert HU -> mu on the fly.
//   Format:  "nx ny nz sx sy sz"  then  nx*ny*nz Hounsfield Units (row-major).
//   We validate aggressively because a silently-truncated volume would make the
//   DRR (and the GPU-vs-CPU comparison) meaningless.
// ---------------------------------------------------------------------------
CtVolume load_volume(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open CT volume file: " + path);

    CtVolume vol;
    VolumeDesc& d = vol.desc;
    // Header: three integer dims, then three float spacings (mm).
    if (!(in >> d.nx >> d.ny >> d.nz >> d.sx >> d.sy >> d.sz))
        throw std::runtime_error("bad header (expected nx ny nz sx sy sz) in " + path);
    if (d.nx <= 0 || d.ny <= 0 || d.nz <= 0)
        throw std::runtime_error("non-positive volume dimensions in " + path);
    if (d.sx <= 0.0f || d.sy <= 0.0f || d.sz <= 0.0f)
        throw std::runtime_error("non-positive voxel spacing in " + path);

    const long long n = vol_count(d);                  // total voxels (no overflow: long long)
    vol.mu.resize(static_cast<std::size_t>(n));
    for (long long i = 0; i < n; ++i) {
        float hu;                                      // raw Hounsfield Unit from file
        if (!(in >> hu))
            throw std::runtime_error("CT body truncated in " + path);
        // Convert HU -> linear attenuation mu (1/mm) ONCE, here, using the shared
        // core function so the kernel never has to repeat it. After this, mu[] is
        // ready for ray-marching.
        vol.mu[static_cast<std::size_t>(i)] = hu_to_mu(hu);
    }
    return vol;
}

// ---------------------------------------------------------------------------
// make_demo_geometry: a single, fixed cone-beam pose, computed deterministically
//   from the volume's physical extent. Conventions:
//     * The volume box spans [0, nx*sx] x [0, ny*sy] x [0, nz*sz] mm; its CENTER
//       is `c`. (Recall drr_core.h is voxel-centered, but for placing the source
//       and detector far outside the box this half-voxel detail is negligible.)
//     * We image along the world +X axis (a lateral view): the source sits on the
//       -X side, the detector panel on the +X side, both centered on c in (y,z).
//     * The detector panel is square-ish, sized to comfortably cover the volume's
//       y/z extent with a small margin, sampled by width x height pixels.
//   This mirrors a real fluoroscopy/portal setup closely enough to teach the
//   geometry while staying fully reproducible. A registration loop would rotate
//   and translate this pose; here it is fixed (THEORY.md "real world").
// ---------------------------------------------------------------------------
DrrGeometry make_demo_geometry(const VolumeDesc& v, int width, int height, float step_mm) {
    // Physical extent of the volume (mm) and its center.
    const float ex = v.nx * v.sx, ey = v.ny * v.sy, ez = v.nz * v.sz;
    const Vec3 c{0.5f * ex, 0.5f * ey, 0.5f * ez};

    DrrGeometry g{};
    g.width  = width;
    g.height = height;
    g.step_mm = step_mm;

    // Source-to-axis and axis-to-detector distances (mm). Generous so the cone
    // beam comfortably encloses the volume from both sides.
    const float src_dist = 1.5f * ex + 300.0f;   // source sits this far on -X of center
    const float det_dist = 1.5f * ex + 300.0f;   // detector this far on +X of center

    // Point X-ray source on the -X side, centered in y/z.
    g.source = Vec3{c.x - src_dist, c.y, c.z};

    // Detector panel physical size: cover the y/z extent with ~40% margin so the
    // whole projected volume lands inside the panel.
    const float panel_y = 1.4f * ey + 60.0f;     // mm, panel width  (maps to u / columns)
    const float panel_z = 1.4f * ez + 60.0f;     // mm, panel height (maps to v / rows)

    // Per-pixel edge vectors. Columns (u) run along +Y, rows (v) along +Z.
    g.du = Vec3{0.0f, panel_y / (width  - 1), 0.0f};
    g.dv = Vec3{0.0f, 0.0f, panel_z / (height - 1)};

    // Panel center on the +X side, centered in y/z; then back off to pixel (0,0)
    // (the top-left corner) by half the panel in each in-plane direction.
    const Vec3 panel_center{c.x + det_dist, c.y, c.z};
    g.origin = Vec3{
        panel_center.x,
        panel_center.y - 0.5f * panel_y,
        panel_center.z - 0.5f * panel_z
    };
    return g;
}

// ---------------------------------------------------------------------------
// render_drr_cpu: the trusted serial baseline. One pass over every detector
//   pixel; each calls the shared integrate_ray(). The pixel at (u, vrow) is
//   stored at image[vrow*width + u] (row-major [v][u]) -- the SAME layout the
//   GPU kernel writes, so max_abs_err compares like-for-like.
// ---------------------------------------------------------------------------
void render_drr_cpu(const CtVolume& vol, const DrrGeometry& g, std::vector<float>& image) {
    const int W = g.width, H = g.height;
    image.assign(static_cast<std::size_t>(W) * H, 0.0f);
    for (int vrow = 0; vrow < H; ++vrow) {            // detector row (maps to +Z)
        for (int u = 0; u < W; ++u) {                 // detector column (maps to +Y)
            // The whole DRR pixel computation is delegated to the shared core, so
            // CPU and GPU are guaranteed to run identical float arithmetic.
            image[static_cast<std::size_t>(vrow) * W + u] =
                integrate_ray(vol.mu.data(), vol.desc, g, u, vrow);
        }
    }
}
