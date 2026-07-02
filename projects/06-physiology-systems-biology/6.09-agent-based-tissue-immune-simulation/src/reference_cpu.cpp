// ===========================================================================
// src/reference_cpu.cpp  --  Loader, deterministic cell placement, binning,
//                            and the serial ABM reference.
// ---------------------------------------------------------------------------
// Project 6.9 : Agent-Based Tissue / Immune Simulation
// Compiled by the HOST compiler only. All per-element physics lives in the
// shared abm_core.h, so this reference and the GPU kernels compute identically.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill, std::swap
#include <cmath>       // std::sqrt
#include <fstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// A tiny, self-contained LCG (linear congruential generator). We DO NOT use
// <random>, because different standard-library implementations produce
// different sequences from the same seed -- which would make the synthetic
// layout (and therefore the whole demo output) non-portable. This LCG is fully
// specified here, so the initial cell positions are identical on every machine.
// Constants are the well-known Numerical Recipes multiplier/increment.
// ---------------------------------------------------------------------------
namespace {
struct Lcg {
    unsigned int state;
    explicit Lcg(unsigned int seed) : state(seed) {}
    // Advance and return a uniform double in [0,1).
    double next() {
        state = 1664525u * state + 1013904223u;      // 32-bit LCG step
        // Use the top 24 bits for a clean [0,1) mantissa (avoids low-bit bias).
        return (state >> 8) / 16777216.0;             // /2^24
    }
};
}  // namespace

// ---------------------------------------------------------------------------
// load_abm: parse the one-line parameter file and then DETERMINISTICALLY place
// the cells. Tumor cells cluster near the domain centre (a small "tumor
// nodule"); immune cells start scattered across the domain -- so the demo shows
// immune cells chemotaxing inward over time (the mean immune->tumor distance
// shrinks). All placement uses the LCG seeded from the file, so the scenario is
// reproducible from the sample alone.
// ---------------------------------------------------------------------------
AbmParams load_abm(const std::string& path, Cells& cells) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ABM parameter file: " + path);

    AbmParams p;
    int n_tumor = 0, n_immune = 0;
    if (!(in >> p.gx >> p.gy >> p.dx >> p.steps >> p.dt >> p.D >> p.decay
             >> p.secretion >> p.radius >> p.k_rep >> p.chemotaxis
             >> p.seed >> n_tumor >> n_immune))
        throw std::runtime_error("bad parameters (expected 14 fields) in " + path);

    // Guard the CFL/stability condition for the explicit diffusion stencil:
    // D*dt/dx^2 must be <= 1/4 in 2-D or the field blows up. Fail loudly so a
    // learner who edits the sample sees WHY, instead of getting NaNs.
    const double cfl = p.D * p.dt / (p.dx * p.dx);
    if (p.gx <= 0 || p.gy <= 0 || p.steps < 0 || p.dx <= 0.0)
        throw std::runtime_error("invalid grid/time parameters in " + path);
    if (cfl > 0.25)
        throw std::runtime_error("unstable: D*dt/dx^2 = " + std::to_string(cfl) +
                                 " > 0.25 (reduce dt or D) in " + path);
    if (n_tumor < 0 || n_immune < 0)
        throw std::runtime_error("negative cell counts in " + path);

    const int n = n_tumor + n_immune;
    cells.n = n;
    cells.x.assign(n, 0.0);
    cells.y.assign(n, 0.0);
    cells.type.assign(n, CELL_TUMOR);

    const double W = p.width(), H = p.height();
    const double cxc = 0.5 * W, cyc = 0.5 * H;   // domain centre = tumor centre
    Lcg rng(p.seed);

    // Tumor cells: a blob near the centre (spread ~ 1/6 of the smaller side). We
    // approximate a bell shape via the sum of two uniforms (cheap, deterministic).
    const double tumor_spread = 0.16 * (W < H ? W : H);
    for (int k = 0; k < n_tumor; ++k) {
        const double rx = (rng.next() + rng.next() - 1.0);   // in (-1,1), ~triangular
        const double ry = (rng.next() + rng.next() - 1.0);
        cells.x[k] = cxc + rx * tumor_spread;
        cells.y[k] = cyc + ry * tumor_spread;
        cells.type[k] = CELL_TUMOR;
    }
    // Immune cells: scattered broadly across the domain (start far from tumor).
    for (int k = n_tumor; k < n; ++k) {
        cells.x[k] = 0.05 * W + 0.90 * W * rng.next();
        cells.y[k] = 0.05 * H + 0.90 * H * rng.next();
        cells.type[k] = CELL_IMMUNE;
    }
    // Clamp any stray placement back inside the domain (defensive).
    for (int k = 0; k < n; ++k) {
        if (cells.x[k] < 0.0) cells.x[k] = 0.0; else if (cells.x[k] > W) cells.x[k] = W;
        if (cells.y[k] < 0.0) cells.y[k] = 0.0; else if (cells.y[k] > H) cells.y[k] = H;
    }
    return p;
}

// ---------------------------------------------------------------------------
// build_bins: counting-sort the cells into a uniform bin grid. bin_size is set
// to the interaction diameter (2*radius) so every possible overlapping pair
// lies within the 3x3 neighbouring bins scanned in abm_move_cell(). Determinism:
// within a bin, cells appear in ASCENDING cell-index order (we fill `sorted`
// with a stable write cursor), which fixes the force-summation order for CPU
// and GPU alike -> exact agreement.
// ---------------------------------------------------------------------------
void build_bins(const AbmParams& p, const Cells& c, SpatialBins& bins) {
    // Bin side = interaction diameter (but at least dx so the grid is sane).
    double bs = 2.0 * p.radius;
    if (bs < p.dx) bs = p.dx;
    if (bs <= 0.0) bs = 1.0;
    bins.bin_size = bs;
    bins.bins_x = static_cast<int>(p.width()  / bs) + 1;
    bins.bins_y = static_cast<int>(p.height() / bs) + 1;
    const int nb = bins.bins_x * bins.bins_y;

    bins.bin_count.assign(nb, 0);
    bins.bin_start.assign(nb, 0);
    bins.sorted.assign(c.n, 0);

    // Which bin does cell i fall in? (clamped into range)
    auto bin_of = [&](int i) {
        int bx = static_cast<int>(c.x[i] / bs);
        int by = static_cast<int>(c.y[i] / bs);
        if (bx < 0) bx = 0; else if (bx >= bins.bins_x) bx = bins.bins_x - 1;
        if (by < 0) by = 0; else if (by >= bins.bins_y) by = bins.bins_y - 1;
        return by * bins.bins_x + bx;
    };
    // Pass 1: count how many cells fall in each bin.
    for (int i = 0; i < c.n; ++i) bins.bin_count[bin_of(i)]++;
    // Pass 2: prefix-sum the counts into start offsets.
    int acc = 0;
    for (int b = 0; b < nb; ++b) { bins.bin_start[b] = acc; acc += bins.bin_count[b]; }
    // Pass 3: scatter cell indices into `sorted` using a per-bin write cursor.
    // Iterating i ascending keeps each bin's members in ascending index order.
    std::vector<int> cursor(bins.bin_start);
    for (int i = 0; i < c.n; ++i) {
        const int b = bin_of(i);
        bins.sorted[cursor[b]++] = i;
    }
}

// ---------------------------------------------------------------------------
// summarize: fold a final (cells, field) state into the deterministic result.
//   * total_quanta : re-quantize the field to integer quanta and sum -> exact.
//   * peak         : the largest concentration and its grid location.
//   * mean_immune_tumor_dist : mean distance of immune cells to the tumor
//     centroid; this is the SCIENCE metric -- it decreases as immune cells
//     chemotax toward the tumor.
// ---------------------------------------------------------------------------
AbmResult summarize(const AbmParams& p, const std::vector<double>& x,
                    const std::vector<double>& y, const std::vector<int>& type,
                    const std::vector<double>& field) {
    AbmResult r;
    r.x = x; r.y = y;

    // Field totals + peak. Sum in integer quanta so the total is order-free.
    unsigned long long tot = 0;
    double peak = -1.0; int pc = 0, pr = 0;
    for (int row = 0; row < p.gy; ++row)
        for (int col = 0; col < p.gx; ++col) {
            const double v = field[abm_grid_idx(col, row, p.gx)];
            tot += abm_to_quanta(v);
            if (v > peak) { peak = v; pc = col; pr = row; }
        }
    r.total_quanta = tot;
    r.total_chemokine = abm_from_quanta(tot);
    r.peak_chemokine = peak < 0.0 ? 0.0 : peak;
    r.peak_col = pc; r.peak_row = pr;

    // Tumor centroid, then mean immune->tumor distance.
    double txc = 0.0, tyc = 0.0; int nt = 0, ni = 0;
    for (std::size_t i = 0; i < type.size(); ++i)
        if (type[i] == CELL_TUMOR) { txc += x[i]; tyc += y[i]; ++nt; }
    if (nt > 0) { txc /= nt; tyc /= nt; }
    double sumd = 0.0;
    for (std::size_t i = 0; i < type.size(); ++i)
        if (type[i] == CELL_IMMUNE) {
            const double dxv = x[i] - txc, dyv = y[i] - tyc;
            sumd += std::sqrt(dxv * dxv + dyv * dyv);
            ++ni;
        }
    r.n_tumor = nt; r.n_immune = ni;
    r.mean_immune_tumor_dist = ni > 0 ? sumd / ni : 0.0;
    return r;
}

// ---------------------------------------------------------------------------
// abm_cpu: the SERIAL reference. One timestep = secrete -> diffuse -> move,
// mirroring exactly what the three GPU kernels do (in the same order), so the
// results match. We keep the chemokine field in fixed-point quanta during the
// secretion scatter (so the CPU sum equals the GPU's atomic sum bit-for-bit),
// then convert to concentration before diffusing.
// ---------------------------------------------------------------------------
AbmResult abm_cpu(const AbmParams& p, const Cells& cells0,
                  std::vector<double>& field_out) {
    const int n = cells0.n;
    const int gc = p.grid_cells();

    // Working copies of the mutable state.
    std::vector<double> x = cells0.x, y = cells0.y;
    const std::vector<int>& type = cells0.type;
    std::vector<double> nx(n), ny(n);                  // next-step positions
    std::vector<double> field(gc, 0.0), field_new(gc, 0.0);
    std::vector<unsigned long long> quanta(gc, 0);     // secretion accumulator

    SpatialBins bins;

    for (int s = 0; s < p.steps; ++s) {
        // --- (1) SECRETE: each tumor cell adds fixed-point quanta to its cell.
        // Serial adds here == the GPU's atomicAdds (integer, order-free).
        std::fill(quanta.begin(), quanta.end(), 0ull);
        const double amount = p.secretion * p.dt;      // chemokine per tumor per step
        const unsigned long long q = abm_to_quanta(amount);
        for (int i = 0; i < n; ++i)
            if (type[i] == CELL_TUMOR) {
                const int col = abm_col_of(x[i], p.dx, p.gx);
                const int row = abm_row_of(y[i], p.dx, p.gy);
                quanta[abm_grid_idx(col, row, p.gx)] += q;
            }
        // Fold the freshly-secreted quanta into the concentration field.
        for (int c = 0; c < gc; ++c) field[c] += abm_from_quanta(quanta[c]);

        // --- (2) DIFFUSE: one explicit reaction-diffusion stencil step.
        for (int row = 0; row < p.gy; ++row)
            for (int col = 0; col < p.gx; ++col)
                abm_diffuse_cell(col, row, p, field.data(), field_new.data());
        field.swap(field_new);

        // --- (3) MOVE: bin the cells, then integrate positions (repulsion +
        // chemotaxis). We read `field` (post-diffusion) for the gradient.
        Cells snap; snap.n = n; snap.x = x; snap.y = y; snap.type = type;
        build_bins(p, snap, bins);
        for (int i = 0; i < n; ++i)
            abm_move_cell(i, p, x.data(), y.data(), type.data(), field.data(),
                          bins.bins_x, bins.bins_y, bins.bin_size,
                          bins.bin_start.data(), bins.bin_count.data(),
                          bins.sorted.data(), nx.data(), ny.data());
        x.swap(nx);
        y.swap(ny);
    }

    field_out = field;
    return summarize(p, x, y, type, field);
}
