// ===========================================================================
// src/reference_cpu.cpp  --  Trusted, plain-C++ landmark decoder + data loader
// ---------------------------------------------------------------------------
// Project 4.32 : GPU-Accelerated Landmark Detection
//
// ROLE
//   The readable baseline. It decodes each predicted heatmap into a landmark
//   coordinate the slow, obvious way -- a serial scan for the peak, then a
//   serial weighted-centroid over a small window. The GPU (kernels.cu) does the
//   same thing in parallel; main.cu runs both and checks they agree. Because
//   the per-voxel math lives in the shared landmark.h (argmax comparison,
//   fixed-point weights, the final division), the two implementations are exact
//   twins by construction, not by luck.
//
//   Complexity: O(L * V) where V = nx*ny*nz voxels -- every voxel is read once
//   for the argmax, then a tiny constant-size window for the centroid. This is
//   embarrassingly parallel across the L landmarks and across voxels, which is
//   precisely why the GPU version wins as V grows (a real 512^3 volume is 1.3e8
//   voxels PER landmark).
//
// READ THIS AFTER: landmark.h, reference_cpu.h. Then kernels.cu for the GPU twin.
// ===========================================================================
#include "reference_cpu.h"

#include <cstdint>     // std::uint64_t
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// load_heatmaps: parse the tiny text sample.
//
// FILE FORMAT (see data/README.md; all whitespace-separated):
//   line 1:  nx ny nz L
//   then, for each landmark l = 0..L-1:
//     a header line:  tx ty tz     (ground-truth coord the blob was built around)
//     then nx*ny*nz float intensities for that landmark's volume, in row-major
//     (x fastest, then y, then z) order.
//
// We read the whole file as a stream of tokens and interpret them positionally,
// which keeps the parser short and tolerant of how the generator wraps lines.
// ---------------------------------------------------------------------------
HeatmapSet load_heatmaps(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open heatmap file: " + path);

    HeatmapSet hs;
    // Header: grid dimensions + number of landmarks.
    if (!(in >> hs.dims.nx >> hs.dims.ny >> hs.dims.nz >> hs.num_landmarks))
        throw std::runtime_error("bad header (expected: nx ny nz L) in " + path);
    if (hs.dims.nx <= 0 || hs.dims.ny <= 0 || hs.dims.nz <= 0 || hs.num_landmarks <= 0)
        throw std::runtime_error("non-positive dimension/landmark count in " + path);

    const std::int64_t V = volume_voxels(hs.dims);   // voxels per heatmap
    const std::int64_t total = V * hs.num_landmarks; // total floats to read
    hs.data.resize(static_cast<std::size_t>(total));
    hs.truth_x.resize(hs.num_landmarks);
    hs.truth_y.resize(hs.num_landmarks);
    hs.truth_z.resize(hs.num_landmarks);

    // Read each landmark's ground-truth header then its dense volume.
    for (int l = 0; l < hs.num_landmarks; ++l) {
        if (!(in >> hs.truth_x[l] >> hs.truth_y[l] >> hs.truth_z[l]))
            throw std::runtime_error("missing truth coords for landmark "
                                     + std::to_string(l) + " in " + path);
        float* vol = hs.data.data() + static_cast<std::size_t>(l) * V;
        for (std::int64_t i = 0; i < V; ++i) {
            if (!(in >> vol[i]))
                throw std::runtime_error("heatmap truncated at landmark "
                                         + std::to_string(l) + " in " + path);
        }
    }
    return hs;
}

// ---------------------------------------------------------------------------
// decode_one: the reference decode for a SINGLE heatmap volume.
//   vol  : pointer to this landmark's nx*ny*nz intensities (row-major).
//   d    : the grid geometry.
//   Returns the decoded Landmark (integer peak + sub-voxel centroid).
//
// Step 1 -- ARGMAX: linear scan, keep the largest value and its (x,y,z). Ties
//   are broken by "first seen in row-major order" (a strict > keeps the first),
//   which is deterministic and MUST match the GPU's tie-break (see kernels.cu).
// Step 2 -- SOFT-ARGMAX: over the cube of half-width SOFTARGMAX_RADIUS around
//   the peak, accumulate integer weight sums (quantize_weight) times position.
//   Doing the sums in integers makes them order-independent so the GPU's atomic
//   version produces the identical totals.
// Step 3 -- finalize: divide the integer numerators by the integer denominator
//   (shared finalize_softargmax) to get the sub-voxel coordinate.
// ---------------------------------------------------------------------------
static Landmark decode_one(const float* vol, const VolumeDims& d) {
    Landmark lm{};

    // ---- Step 1: argmax over the whole volume -----------------------------
    float best = -3.4e38f;    // below any real heatmap value (~ -FLT_MAX)
    int bx = 0, by = 0, bz = 0;
    for (int z = 0; z < d.nz; ++z)
        for (int y = 0; y < d.ny; ++y)
            for (int x = 0; x < d.nx; ++x) {
                float v = vol[flat_index(x, y, z, d)];
                if (v > best) { best = v; bx = x; by = y; bz = z; }  // strict >
            }
    lm.peak = best;
    lm.px = bx; lm.py = by; lm.pz = bz;

    // ---- Step 2: soft-argmax integer accumulation over the window ---------
    // 64-bit accumulators: sum_w is the total weight (denominator), sum_wx/y/z
    // are weight*position (numerators). Integers => associative => match GPU.
    std::uint64_t sum_w = 0, sum_wx = 0, sum_wy = 0, sum_wz = 0;
    const int R = SOFTARGMAX_RADIUS;
    for (int z = bz - R; z <= bz + R; ++z) {
        if (z < 0 || z >= d.nz) continue;                 // clip window to volume
        for (int y = by - R; y <= by + R; ++y) {
            if (y < 0 || y >= d.ny) continue;
            for (int x = bx - R; x <= bx + R; ++x) {
                if (x < 0 || x >= d.nx) continue;
                std::uint32_t w = quantize_weight(vol[flat_index(x, y, z, d)]);
                sum_w  += w;
                sum_wx += static_cast<std::uint64_t>(w) * x;
                sum_wy += static_cast<std::uint64_t>(w) * y;
                sum_wz += static_cast<std::uint64_t>(w) * z;
            }
        }
    }

    // ---- Step 3: finalize into a sub-voxel coordinate ---------------------
    finalize_softargmax(sum_w, sum_wx, sum_wy, sum_wz, bx, by, bz,
                        lm.x, lm.y, lm.z);
    return lm;
}

// decode_cpu: decode every landmark in the set (serial loop over the L volumes).
void decode_cpu(const HeatmapSet& hs, std::vector<Landmark>& out) {
    out.resize(hs.num_landmarks);
    const std::int64_t V = volume_voxels(hs.dims);
    for (int l = 0; l < hs.num_landmarks; ++l) {
        const float* vol = hs.data.data() + static_cast<std::size_t>(l) * V;
        out[l] = decode_one(vol, hs.dims);
    }
}
