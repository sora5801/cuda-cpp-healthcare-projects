// ===========================================================================
// src/reference_cpu.cpp  --  Plain-C++ reference: DCC matrix + allosteric network
// ---------------------------------------------------------------------------
// Project 2.17 : Allosteric Network Analysis
//
// WHAT THIS FILE DOES
//   The readable, single-threaded baseline for the whole pipeline:
//     load_trajectory -> residue_means -> dcc_matrix_cpu -> build_contacts
//     -> shortest_paths -> reconstruct_path.
//   main.cu runs this DCC matrix AND the GPU DCC matrix and asserts they are
//   bit-identical (the shared dcc_core.h guarantees it). Everything downstream
//   of the matrix (contacts, shortest paths) is deterministic graph bookkeeping
//   shared by both code paths.
//
//   This file is compiled by the HOST C++ compiler, so it contains NO CUDA. It
//   includes dcc_core.h purely for the host-side expansion of dcc_pair() and
//   comm_weight() (the HD macro becomes empty here).
//
// READ THIS AFTER: reference_cpu.h (signatures), dcc_core.h (the per-pair math).
// READ THIS BEFORE: kernels.cu (the GPU twin of dcc_matrix_cpu).
// ===========================================================================
#include "reference_cpu.h"
#include "dcc_core.h"        // dcc_pair, comm_weight, coord_index (shared physics)

#include <fstream>           // std::ifstream
#include <sstream>           // std::istringstream
#include <stdexcept>         // std::runtime_error
#include <string>

// A large finite sentinel for "no path / unreachable". We use 1e30 rather than
// std::numeric_limits::infinity() so that adding two of them inside the
// Floyd-Warshall relaxation never produces a NaN/overflow surprise.
static const double INF = 1.0e30;

// ---------------------------------------------------------------------------
// load_trajectory: parse the structured sample file.
//
//   FILE FORMAT (see data/README.md and scripts/make_synthetic.py):
//     Lines beginning with '#' are comments, EXCEPT two annotation lines we
//     parse for the demo's narrative:
//         # SITE_ALLO <index>     -> the engineered allosteric site residue
//         # SITE_ACTIVE <index>   -> the engineered active site residue
//     The first non-comment line holds two integers:  N  T
//     Then T blocks follow, each of N lines, each line "x y z" for one residue.
//
//   We keep the parser deliberately simple and strict: any shape mismatch throws,
//   so a corrupted sample fails loudly rather than silently analyzing garbage.
// ---------------------------------------------------------------------------
Trajectory load_trajectory(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open trajectory file: " + path);

    Trajectory traj;
    std::string line;
    bool have_dims = false;          // have we read the "N T" header line yet?
    long expected = 0, got = 0;      // expected/observed coordinate-triple count

    while (std::getline(in, line)) {
        // Strip a trailing carriage return so Windows-edited files parse on Linux.
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Annotation comments carry the site indices; ordinary comments/blank skip.
        // We parse with istringstream (not sscanf) so the code is portable and
        // free of MSVC's "unsafe function" deprecation: read the leading '#', a
        // tag token, then the integer index.
        if (!line.empty() && line[0] == '#') {
            std::istringstream cs(line);
            std::string hash, tag;
            int idx = -1;
            if ((cs >> hash >> tag >> idx)) {
                if (tag == "SITE_ALLO")        traj.site_allo = idx;
                else if (tag == "SITE_ACTIVE") traj.site_active = idx;
            }
            continue;
        }
        if (line.empty()) continue;

        std::istringstream ss(line);
        if (!have_dims) {
            // First data line: the two dimensions.
            if (!(ss >> traj.N >> traj.T) || traj.N <= 0 || traj.T <= 0)
                throw std::runtime_error("trajectory header must be 'N T' with positive ints");
            have_dims = true;
            expected = static_cast<long>(traj.N) * traj.T;   // number of xyz triples
            traj.coords.reserve(static_cast<std::size_t>(expected) * 3);
            continue;
        }

        // A coordinate line: exactly three floats.
        float x, y, z;
        if (!(ss >> x >> y >> z))
            throw std::runtime_error("malformed coordinate line in trajectory: " + line);
        traj.coords.push_back(x);
        traj.coords.push_back(y);
        traj.coords.push_back(z);
        ++got;
    }

    if (!have_dims) throw std::runtime_error("trajectory file had no 'N T' header");
    if (got != expected)
        throw std::runtime_error("trajectory size mismatch: expected N*T coordinate triples");

    // Default the sites if the sample did not annotate them (keeps the demo robust).
    if (traj.site_allo < 0 || traj.site_allo >= traj.N) traj.site_allo = 0;
    if (traj.site_active < 0 || traj.site_active >= traj.N) traj.site_active = traj.N - 1;
    return traj;
}

// ---------------------------------------------------------------------------
// residue_means: <r_i> = average position of residue i over all T frames.
//   Accumulate in double, divide once at the end. This is the "equilibrium"
//   structure the displacements are measured against.
// ---------------------------------------------------------------------------
void residue_means(const Trajectory& traj, std::vector<double>& mean) {
    const int N = traj.N, T = traj.T;
    mean.assign(static_cast<std::size_t>(N) * 3, 0.0);

    // Sum every frame's contribution into the per-residue accumulator.
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < 3; ++c)
                mean[i * 3 + c] += traj.coords[coord_index(t, i, c, N)];

    // Convert sums to averages.
    const double invT = 1.0 / static_cast<double>(T);
    for (std::size_t k = 0; k < mean.size(); ++k) mean[k] *= invT;
}

// ---------------------------------------------------------------------------
// dcc_matrix_cpu: fill the N*N correlation matrix one entry at a time.
//   This triple-loop is O(N^2 * T) and is precisely the work the GPU spreads
//   across N*N threads in kernels.cu. We exploit symmetry (C[i][j]==C[j][i]) to
//   halve the dcc_pair() calls; the GPU version recomputes both for simplicity,
//   but the NUMBERS are identical because dcc_pair() is the shared truth.
// ---------------------------------------------------------------------------
void dcc_matrix_cpu(const Trajectory& traj, const std::vector<double>& mean,
                    std::vector<float>& C) {
    const int N = traj.N, T = traj.T;
    C.assign(static_cast<std::size_t>(N) * N, 0.0f);
    const float* coords = traj.coords.data();
    const double* m = mean.data();

    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            // One matrix entry from the shared physics; cast to float for an
            // apples-to-apples comparison with the GPU's float output.
            const float c = static_cast<float>(dcc_pair(coords, m, i, j, T, N));
            C[static_cast<std::size_t>(i) * N + j] = c;
            C[static_cast<std::size_t>(j) * N + i] = c;   // mirror (symmetric)
        }
    }
}

// ---------------------------------------------------------------------------
// build_contacts: the spatial scaffold the allosteric signal must travel along.
//   An edge between i and j is allowed only if their equilibrium Cα atoms are
//   within `cutoff` angstroms (they physically touch) or they are backbone
//   neighbors (|i-j| == 1, covalently bonded). This prevents the shortest path
//   from "teleporting" between distant correlated residues that are not in
//   contact -- communication must hop through the structure.
// ---------------------------------------------------------------------------
void build_contacts(const Trajectory& traj, const std::vector<double>& mean,
                    double cutoff, std::vector<char>& adj) {
    const int N = traj.N;
    adj.assign(static_cast<std::size_t>(N) * N, 0);
    const double cut2 = cutoff * cutoff;   // compare squared distances (skip sqrt)

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const double dx = mean[i * 3 + 0] - mean[j * 3 + 0];
            const double dy = mean[i * 3 + 1] - mean[j * 3 + 1];
            const double dz = mean[i * 3 + 2] - mean[j * 3 + 2];
            const double d2 = dx * dx + dy * dy + dz * dz;
            const bool neighbor = (j - i == 1);          // backbone bond
            if (d2 <= cut2 || neighbor) {
                adj[static_cast<std::size_t>(i) * N + j] = 1;
                adj[static_cast<std::size_t>(j) * N + i] = 1;  // undirected
            }
        }
    }
}

// ---------------------------------------------------------------------------
// shortest_paths: Floyd-Warshall on the -log|C| weighted contact graph.
//   Classic O(N^3) dynamic program: dist[i][j] is progressively relaxed by
//   allowing an intermediate vertex k. `next[i][j]` records the first hop on the
//   best i->j route so reconstruct_path() can walk it back out. Deterministic
//   and order-independent (we only ever take strict improvements), so it is a
//   trustworthy reference.
// ---------------------------------------------------------------------------
void shortest_paths(const std::vector<float>& C, const std::vector<char>& adj,
                    int N, std::vector<double>& dist, std::vector<int>& next) {
    dist.assign(static_cast<std::size_t>(N) * N, INF);
    next.assign(static_cast<std::size_t>(N) * N, -1);

    // Initialize from the direct edges of the contact graph.
    for (int i = 0; i < N; ++i) {
        dist[static_cast<std::size_t>(i) * N + i] = 0.0;       // zero cost to self
        next[static_cast<std::size_t>(i) * N + i] = i;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (adj[static_cast<std::size_t>(i) * N + j]) {
                // Edge weight = communication cost = -log|correlation|.
                const double w = comm_weight(C[static_cast<std::size_t>(i) * N + j]);
                dist[static_cast<std::size_t>(i) * N + j] = w;
                next[static_cast<std::size_t>(i) * N + j] = j;  // direct hop
            }
        }
    }

    // The relaxation: can routing through k shorten i->j?
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) {
            const double dik = dist[static_cast<std::size_t>(i) * N + k];
            if (dik >= INF) continue;                 // i can't even reach k: skip
            for (int j = 0; j < N; ++j) {
                const double through = dik + dist[static_cast<std::size_t>(k) * N + j];
                if (through < dist[static_cast<std::size_t>(i) * N + j]) {
                    dist[static_cast<std::size_t>(i) * N + j] = through;
                    // To get from i toward j, first step the way you'd step to k.
                    next[static_cast<std::size_t>(i) * N + j] =
                        next[static_cast<std::size_t>(i) * N + k];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// reconstruct_path: walk the `next` table from src to dst.
//   Returns [src, ..., dst]; empty if dst is unreachable. A guard caps the walk
//   at N hops so a malformed table can never spin forever.
// ---------------------------------------------------------------------------
std::vector<int> reconstruct_path(const std::vector<int>& next, int N,
                                  int src, int dst) {
    std::vector<int> path;
    if (next[static_cast<std::size_t>(src) * N + dst] < 0) return path;  // unreachable
    int at = src;
    path.push_back(at);
    int guard = 0;
    while (at != dst && guard++ <= N) {
        at = next[static_cast<std::size_t>(at) * N + dst];
        if (at < 0) { path.clear(); return path; }   // broken link
        path.push_back(at);
    }
    return path;
}
