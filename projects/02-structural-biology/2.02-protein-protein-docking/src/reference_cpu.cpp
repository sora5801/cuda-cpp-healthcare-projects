// ===========================================================================
// src/reference_cpu.cpp  --  Loader, voxelization, brute-force correlation
// ---------------------------------------------------------------------------
// Project 2.2 : Protein-Protein Docking  (catalog id 2.2)
//
// ROLE IN THE PROJECT
//   The plain-C++ "ground truth". Everything here is written to be OBVIOUSLY
//   correct -- simple loops, no parallelism -- so that when the cuFFT result
//   agrees with it we trust the GPU. It provides:
//     * load_dock         -- parse the sample file.
//     * voxelize_receptor -- stamp the receptor shape grid (core + skin).
//     * voxelize_ligand   -- stamp the ligand occupancy grid.
//     * correlate_cpu     -- O(Ng^2) direct circular correlation S(t).
//     * argmax_grid       -- find the best-scoring translation.
//   The voxelization rule here is the SAME one the GPU kernel applies (kernels.cu
//   build_receptor_grid / build_ligand_grid), so the two grids are identical and
//   only the FFT introduces (tiny) round-off.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_dock : read the text sample (format documented in data/README.md).
//   We validate the header and every coordinate so a truncated or malformed
//   file throws rather than silently docking garbage.
// ---------------------------------------------------------------------------
DockData load_dock(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dock file: " + path);

    DockData d;
    // Header: n_recv n_lig N spacing  [true_tx true_ty true_tz]
    if (!(in >> d.n_recv >> d.n_lig >> d.N >> d.spacing) ||
        d.n_recv <= 0 || d.n_lig <= 0 || d.N <= 0 || d.spacing <= 0.0)
        throw std::runtime_error("bad header (expected 'n_recv n_lig N spacing') in " + path);

    // Optional known-answer translation (synthetic samples include it).
    // We peek: if three more ints sit on the header line they are the answer;
    // otherwise leave the NO_TRUTH sentinel so main.cu skips the science check.
    int tx, ty, tz;
    if (in >> tx >> ty >> tz) {
        d.true_tx = tx; d.true_ty = ty; d.true_tz = tz;
    } else {
        in.clear();             // not present -> clear failbit, keep NO_TRUTH
    }

    d.recv.resize(static_cast<std::size_t>(3) * d.n_recv);
    for (std::size_t i = 0; i < d.recv.size(); ++i)
        if (!(in >> d.recv[i])) throw std::runtime_error("receptor coords truncated in " + path);

    d.lig.resize(static_cast<std::size_t>(3) * d.n_lig);
    for (std::size_t i = 0; i < d.lig.size(); ++i)
        if (!(in >> d.lig[i])) throw std::runtime_error("ligand coords truncated in " + path);

    return d;
}

// ---------------------------------------------------------------------------
// choose_origin : pick the world coordinate of voxel 0 along each axis so that
//   the RECEPTOR sits centered in the grid with room for the ligand to slide.
//   We center the receptor's bounding-box center at the grid center (N/2). This
//   is a pure geometric helper used by both voxelize functions so they share a
//   frame. (The ligand grid uses the SAME origin -> a translation in voxels has
//   a direct physical meaning.)
// ---------------------------------------------------------------------------
static void choose_origin(const DockData& d, double origin[3]) {
    // Receptor bounding-box center (Angstrom).
    float lo[3] = { d.recv[0], d.recv[1], d.recv[2] };
    float hi[3] = { d.recv[0], d.recv[1], d.recv[2] };
    for (int i = 0; i < d.n_recv; ++i)
        for (int a = 0; a < 3; ++a) {
            float c = d.recv[static_cast<std::size_t>(3) * i + a];
            lo[a] = std::min(lo[a], c);
            hi[a] = std::max(hi[a], c);
        }
    // Place the receptor center at the grid center: origin = center - N/2 * spacing.
    const double half = 0.5 * d.N * d.spacing;
    for (int a = 0; a < 3; ++a) {
        double center = 0.5 * (lo[a] + hi[a]);
        origin[a] = center - half;
    }
}

// ---------------------------------------------------------------------------
// stamp_atoms : mark every voxel within `radius_vox` voxels of any atom as
//   "occupied" (occ[idx] = 1). A protein atom has a van-der-Waals radius of a
//   couple of Angstrom; we approximate each atom as a small cube/sphere of
//   occupied voxels. Order-independent (a simple OR) so CPU and GPU agree no
//   matter which thread/atom touches a voxel first.
//     occ      : N*N*N byte grid, must be pre-zeroed.
//     atoms    : [3*n] coordinates (Angstrom).
//     radius_vox: occupancy radius in voxels (integer).
// ---------------------------------------------------------------------------
static void stamp_atoms(int N, double spacing, const double origin[3],
                        const std::vector<float>& atoms, int n,
                        int radius_vox, std::vector<unsigned char>& occ) {
    const double r2 = static_cast<double>(radius_vox) * radius_vox;
    for (int i = 0; i < n; ++i) {
        // Atom center in voxel coordinates (continuous), then its integer voxel.
        int cx = world_to_voxel(atoms[static_cast<std::size_t>(3) * i + 0], origin[0], spacing);
        int cy = world_to_voxel(atoms[static_cast<std::size_t>(3) * i + 1], origin[1], spacing);
        int cz = world_to_voxel(atoms[static_cast<std::size_t>(3) * i + 2], origin[2], spacing);
        // Splat a small ball of radius `radius_vox` around the atom center.
        for (int dz = -radius_vox; dz <= radius_vox; ++dz)
            for (int dy = -radius_vox; dy <= radius_vox; ++dy)
                for (int dx = -radius_vox; dx <= radius_vox; ++dx) {
                    if (static_cast<double>(dx*dx + dy*dy + dz*dz) > r2) continue;  // ball, not cube
                    int x = cx + dx, y = cy + dy, z = cz + dz;
                    if (x < 0 || x >= N || y < 0 || y >= N || z < 0 || z >= N) continue;
                    occ[static_cast<std::size_t>(flat3(x, y, z, N))] = 1;
                }
    }
}

// ---------------------------------------------------------------------------
// build_shape_grid : turn an occupancy mask into the Katchalski-Katzir TWO-VALUE
//   shape function. Each occupied voxel is classified by its 6-neighborhood (von
//   Neumann): CORE (all 6 face-neighbors occupied -> buried interior) gets
//   CORE_VALUE (+contact); otherwise it is SKIN (on the surface) and gets
//   SKIN_PENALTY (-clash). Empty voxels stay 0. A voxel on the grid border is
//   treated as surface (its off-grid neighbor counts as empty).
//
//   WHY THE SAME RULE FOR BOTH PROTEINS: the docking score is the correlation of
//   the receptor shape with the ligand shape. Giving BOTH the core(+)/skin(-)
//   model means overlapping cores reward and overlapping surfaces penalize
//   clashes symmetrically -- the standard geometric complementarity score. With
//   our synthetic ligand (a displaced copy of the receptor), this makes the
//   correlation an AUTOCORRELATION whose unique peak is the known translation T.
// ---------------------------------------------------------------------------
static void build_shape_grid(int N, const std::vector<unsigned char>& occ,
                             std::vector<float>& g) {
    g.assign(static_cast<std::size_t>(N) * N * N, 0.0f);
    const int nb[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                std::size_t idx = static_cast<std::size_t>(flat3(x, y, z, N));
                if (!occ[idx]) continue;  // empty space stays 0
                bool buried = true;
                for (int k = 0; k < 6 && buried; ++k) {
                    int x2 = x + nb[k][0], y2 = y + nb[k][1], z2 = z + nb[k][2];
                    if (x2 < 0 || x2 >= N || y2 < 0 || y2 >= N || z2 < 0 || z2 >= N)
                        buried = false;
                    else if (!occ[static_cast<std::size_t>(flat3(x2, y2, z2, N))])
                        buried = false;
                }
                g[idx] = buried ? CORE_VALUE : SKIN_PENALTY;
            }
}

// ---------------------------------------------------------------------------
// voxelize_receptor : stamp the receptor onto the shared grid, centered.
//   Step 1: occupancy (a 2-voxel ball, ~3 Angstrom, around every receptor atom).
//   Step 2: classify core vs skin via build_shape_grid. Also returns the world
//           origin so the ligand uses the IDENTICAL frame.
// ---------------------------------------------------------------------------
void voxelize_receptor(const DockData& d, std::vector<float>& g, double origin_out[3]) {
    const int N = d.N;
    choose_origin(d, origin_out);
    std::vector<unsigned char> occ(static_cast<std::size_t>(N) * N * N, 0);
    stamp_atoms(N, d.spacing, origin_out, d.recv, d.n_recv, /*radius_vox=*/2, occ);
    build_shape_grid(N, occ, g);
}

// ---------------------------------------------------------------------------
// voxelize_ligand : stamp the ligand onto the grid in the SAME frame (origin)
//   using the SAME core/skin rule as the receptor (build_shape_grid). The ligand
//   sits at its own coordinates; the FFT then evaluates every translation t that
//   slides this whole grid over the receptor at once. Using the same shape model
//   for both is what makes the score a true geometric-complementarity correlation.
// ---------------------------------------------------------------------------
void voxelize_ligand(const DockData& d, const double origin[3], std::vector<float>& g) {
    const int N = d.N;
    std::vector<unsigned char> occ(static_cast<std::size_t>(N) * N * N, 0);
    stamp_atoms(N, d.spacing, origin, d.lig, d.n_lig, /*radius_vox=*/2, occ);
    build_shape_grid(N, occ, g);
}

// ---------------------------------------------------------------------------
// correlate_cpu : the trusted O(Ng^2) reference. For EVERY translation t we sum
//   over EVERY grid voxel x:  S(t) = sum_x R(x) * L(wrap(x - t)).
//   This is CIRCULAR (periodic) correlation -- exactly what the FFT computes --
//   so the two results are directly comparable. It is intentionally the naive
//   double loop: Ng outputs, each an Ng-term sum => O(Ng^2). Slow, but it is the
//   yardstick. (For the 32^3 sample that is ~10^9 multiply-adds -- a second or
//   two of CPU time, fine for teaching; a real 100^3 grid would be ~10^12 and is
//   why the FFT is mandatory in practice.)
// ---------------------------------------------------------------------------
void correlate_cpu(int N, const std::vector<float>& R, const std::vector<float>& L,
                   std::vector<float>& score) {
    score.assign(static_cast<std::size_t>(N) * N * N, 0.0f);
    for (int tz = 0; tz < N; ++tz)
        for (int ty = 0; ty < N; ++ty)
            for (int tx = 0; tx < N; ++tx) {
                double s = 0.0;  // accumulate in double for a clean reference
                for (int z = 0; z < N; ++z)
                    for (int y = 0; y < N; ++y)
                        for (int x = 0; x < N; ++x) {
                            float r = R[static_cast<std::size_t>(flat3(x, y, z, N))];
                            if (r == 0.0f) continue;  // skip empty receptor voxels (most of them)
                            int lx = wrap(x - tx, N), ly = wrap(y - ty, N), lz = wrap(z - tz, N);
                            float l = L[static_cast<std::size_t>(flat3(lx, ly, lz, N))];
                            s += static_cast<double>(r) * static_cast<double>(l);
                        }
                score[static_cast<std::size_t>(flat3(tx, ty, tz, N))] = static_cast<float>(s);
            }
}

// ---------------------------------------------------------------------------
// argmax_grid : linear scan for the maximum score and its voxel coordinates.
//   Deterministic: strict '>' keeps the FIRST (lowest-index) maximum on ties,
//   so CPU and GPU report the identical best translation.
// ---------------------------------------------------------------------------
std::size_t argmax_grid(int N, const std::vector<float>& score, int& bx, int& by, int& bz) {
    std::size_t best = 0;
    float bestv = score[0];
    for (std::size_t i = 1; i < score.size(); ++i)
        if (score[i] > bestv) { bestv = score[i]; best = i; }
    // Decompose the linear index back to (x,y,z) given idx = (z*N + y)*N + x.
    int z = static_cast<int>(best) / (N * N);
    int rem = static_cast<int>(best) % (N * N);
    int y = rem / N;
    int x = rem % N;
    bx = x; by = y; bz = z;
    return best;
}
