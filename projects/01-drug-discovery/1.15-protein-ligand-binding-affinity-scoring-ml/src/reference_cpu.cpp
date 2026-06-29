// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial 3D-CNN forward pass + loader
// ---------------------------------------------------------------------------
// Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
//
// ROLE IN THE PROJECT
//   (1) load_complexes(): parse the tiny text dataset (format in data/README.md).
//   (2) score_cpu(): the obviously-correct serial forward pass the GPU kernels
//       are verified against. No parallelism, no cleverness -- one readable loop
//       per network stage -- so that when CPU and GPU agree, we trust the GPU.
//
//   Compiled by the host C++ compiler only (NO CUDA here). The shared math comes
//   from scoring_core.h (the HD-macro idiom): voxelization, the conv tap, ReLU,
//   and the deterministic weight generator are the SAME functions nvcc compiles
//   for the device, so the two implementations agree bit-for-bit.
//
// THE FORWARD PASS (one complex), stage by stage:
//   (a) VOXELIZE : splat each atom as a Gaussian blob into an [CIN][GRID^3] grid.
//   (b) CONV3D   : COUT filters of CIN x KSZ^3 taps slide over the grid (SAME
//                  padding -> output is also GRID^3 per output channel), then ReLU.
//   (c) POOL     : global average over the GRID^3 voxels of each output map ->
//                  one number per output channel (a COUT-vector).
//   (d) DENSE    : weighted sum of the COUT pooled features + bias -> raw score,
//                  mapped affinely into a plausible pKd range [2, 11].
//
// READ THIS AFTER: scoring_core.h, reference_cpu.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::lround
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_complexes: parse the batch text file. Format (see data/README.md):
//   line 1            : "<n>"                          number of complexes
//   then, per complex : "<m> <label_pKd>"              m = atom count
//                       m lines of "<x> <y> <z> <type> <is_ligand>"
//   `type` is 0..NTYPES-1 (C,N,O,S); `is_ligand` is 0 or 1.
// We validate aggressively so a malformed file fails loudly, not silently.
// ---------------------------------------------------------------------------
ComplexSet load_complexes(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open complex file: " + path);

    int n = 0;
    if (!(in >> n) || n <= 0)
        throw std::runtime_error("bad header (expected positive complex count) in " + path);

    ComplexSet cs;
    cs.n = n;
    cs.offset.reserve(static_cast<std::size_t>(n) + 1);
    cs.label.reserve(static_cast<std::size_t>(n));
    cs.offset.push_back(0);                    // CSR row pointer starts at 0

    for (int i = 0; i < n; ++i) {
        int m = 0;
        double label = 0.0;
        if (!(in >> m >> label) || m <= 0)
            throw std::runtime_error("bad complex record (expected '<m> <label>') in " + path);
        cs.label.push_back(label);
        for (int a = 0; a < m; ++a) {
            Atom at{};
            if (!(in >> at.x >> at.y >> at.z >> at.type >> at.is_ligand))
                throw std::runtime_error("unexpected end of atom data in " + path);
            if (at.type < 0 || at.type >= NTYPES)
                throw std::runtime_error("atom type out of range [0,NTYPES) in " + path);
            if (at.is_ligand != 0 && at.is_ligand != 1)
                throw std::runtime_error("is_ligand must be 0 or 1 in " + path);
            cs.atoms.push_back(at);
        }
        // CSR: the running total of atoms seen so far delimits the next complex.
        cs.offset.push_back(static_cast<int>(cs.atoms.size()));
    }
    return cs;
}

// ---------------------------------------------------------------------------
// score_one_cpu: the forward pass for a SINGLE complex (atoms in [a0,a1)).
//   `grid` and `feat` are scratch buffers the caller owns and reuses across
//   complexes (so we allocate them once, not n times). Returns the predicted pKd.
//
//   Every loop here has a one-to-one twin in kernels.cu; keeping the SAME
//   operation order (e.g. summing conv taps in c,dz,dy,dx order) is what makes
//   the GPU result match this one term-for-term (THEORY "How we verify").
// ---------------------------------------------------------------------------
static double score_one_cpu(const ComplexSet& cs, int a0, int a1,
                            std::vector<double>& grid, std::vector<double>& feat) {
    // --- (a) VOXELIZE: zero the grid, then deposit each atom's Gaussian blob ---
    std::fill(grid.begin(), grid.end(), 0.0);
    for (int a = a0; a < a1; ++a) {
        const Atom& at = cs.atoms[static_cast<std::size_t>(a)];
        const int c = channel_of(at.type, at.is_ligand);     // which feature channel
        // The atom's continuous position in voxel units (not yet rounded).
        const double fx = at.x / VOX_A, fy = at.y / VOX_A, fz = at.z / VOX_A;
        // Only voxels within GAUSS_CUTOFF of the atom get a nonzero contribution,
        // so we loop over the small integer box around the atom instead of all
        // GRID^3 voxels -- the same finite-support trick the GPU uses.
        const int rad = static_cast<int>(GAUSS_CUTOFF / VOX_A) + 1;
        const int cx = static_cast<int>(fx), cy = static_cast<int>(fy), cz = static_cast<int>(fz);
        for (int z = cz - rad; z <= cz + rad; ++z) {
            if (z < 0 || z >= GRID) continue;
            for (int y = cy - rad; y <= cy + rad; ++y) {
                if (y < 0 || y >= GRID) continue;
                for (int x = cx - rad; x <= cx + rad; ++x) {
                    if (x < 0 || x >= GRID) continue;
                    // Squared distance (in A^2) from the atom to THIS voxel center.
                    const double ddx = (x + 0.5) * VOX_A - at.x;
                    const double ddy = (y + 0.5) * VOX_A - at.y;
                    const double ddz = (z + 0.5) * VOX_A - at.z;
                    const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                    grid[static_cast<std::size_t>(grid_index(c, x, y, z))] += atom_contrib(r2);
                }
            }
        }
    }

    // --- (b) CONV3D + ReLU, then (c) global-average POOL fused in one sweep ---
    // For each output map oc we compute its conv response at every voxel, ReLU it,
    // and accumulate the mean directly into pooled[oc] (no need to store feat for
    // pooling -- but we keep `feat` available for clarity/extension). SAME padding:
    // taps that read outside the grid contribute 0.
    double pooled[COUT];
    for (int oc = 0; oc < COUT; ++oc) pooled[oc] = 0.0;

    for (int oc = 0; oc < COUT; ++oc) {
        double acc_map = 0.0;                         // sum of ReLU responses over voxels
        for (int z = 0; z < GRID; ++z)
        for (int y = 0; y < GRID; ++y)
        for (int x = 0; x < GRID; ++x) {
            double s = 0.0;                            // conv response at (x,y,z)
            // Convolution: sum over input channels and the KSZ^3 neighborhood.
            for (int ic = 0; ic < CIN; ++ic) {
                for (int dz = 0; dz < KSZ; ++dz) {
                    const int zz = z + dz - KSZ / 2;   // SAME-padding source z
                    if (zz < 0 || zz >= GRID) continue;
                    for (int dy = 0; dy < KSZ; ++dy) {
                        const int yy = y + dy - KSZ / 2;
                        if (yy < 0 || yy >= GRID) continue;
                        for (int dx = 0; dx < KSZ; ++dx) {
                            const int xx = x + dx - KSZ / 2;
                            if (xx < 0 || xx >= GRID) continue;
                            // Weight index must match kernels.cu EXACTLY:
                            //   ((oc*CIN + ic)*KSZ + dz)*KSZ + dy)*KSZ + dx
                            const uint64_t widx =
                                (((static_cast<uint64_t>(oc) * CIN + ic) * KSZ + dz) * KSZ + dy) * KSZ + dx;
                            const double w = lcg_weight(widx);
                            s += w * grid[static_cast<std::size_t>(grid_index(ic, xx, yy, zz))];
                        }
                    }
                }
            }
            acc_map += relu(s);                        // ReLU then accumulate for pooling
        }
        pooled[oc] = acc_map / static_cast<double>(VOX_PER_CH);   // global average
    }

    // --- (d) DENSE readout: bias + sum_oc(dense_w[oc] * pooled[oc]) -----------
    // The dense weights live just past the conv weights in the same LCG stream so
    // they are deterministic too. Index base = WCONV_SIZE.
    double raw = lcg_weight(static_cast<uint64_t>(WCONV_SIZE) + COUT);   // the bias term
    for (int oc = 0; oc < COUT; ++oc) {
        const double w = lcg_weight(static_cast<uint64_t>(WCONV_SIZE) + oc);
        raw += w * pooled[oc];
    }
    // Map the raw score into a plausible pKd window [2, 11] with a logistic
    // squash so the output is always a sensible affinity (THEORY "The math").
    const double squashed = 1.0 / (1.0 + std::exp(-raw));    // (0,1)
    return 2.0 + 9.0 * squashed;                             // [2, 11]
}

// ---------------------------------------------------------------------------
// score_cpu: run score_one_cpu over every complex in the batch. Scratch grid and
//   feature buffers are allocated ONCE and reused -- this is the serial baseline,
//   so clarity and a fair (allocation-free) timing both matter.
// ---------------------------------------------------------------------------
void score_cpu(const ComplexSet& cs, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(cs.n), 0.0);
    std::vector<double> grid(static_cast<std::size_t>(GRID_SIZE), 0.0);   // [CIN][GRID^3]
    std::vector<double> feat(static_cast<std::size_t>(FEAT_SIZE), 0.0);   // [COUT][GRID^3]
    for (int i = 0; i < cs.n; ++i) {
        out[static_cast<std::size_t>(i)] =
            score_one_cpu(cs, cs.offset[i], cs.offset[i + 1], grid, feat);
    }
}
