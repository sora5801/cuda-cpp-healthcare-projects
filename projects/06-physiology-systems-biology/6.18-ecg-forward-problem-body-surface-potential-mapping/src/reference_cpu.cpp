// ===========================================================================
// src/reference_cpu.cpp  --  Plain-C++ reference for the ECG forward problem
// ---------------------------------------------------------------------------
// Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
//
// WHAT THIS FILE DOES
//   The serial, obvious, no-CUDA implementation of the whole pipeline:
//     * load_ecg                   : parse a sample file into ECGData.
//     * build_lead_field_reference : A[e][s] = potential at electrode e from a
//                                    unit source s  (calls ecg::dipole_potential).
//     * apply_forward_reference    : Phi = A * X, the textbook triple loop.
//   main.cu runs these AND the GPU twins and asserts the two agree. Because both
//   sides call the SAME ecg::dipole_potential (ecg_core.h), the lead field is
//   near bit-identical; the only tolerance we need is for the matrix multiply,
//   where the GPU sums the dot products in a different ORDER (floating-point add
//   is not associative -- a real, teachable effect; see main.cu's tolerances).
//
//   Compiled by the HOST C++ compiler (cl.exe), so NOTHING CUDA appears here.
//
// READ THIS AFTER: reference_cpu.h, ecg_core.h.  Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cstddef>     // std::size_t
#include <fstream>     // std::ifstream
#include <istream>     // std::istream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// next_data_line: read the next NON-EMPTY, NON-COMMENT line from a stream.
//   The sample format allows blank lines and '#'-prefixed comments for human
//   readability (see data/README.md). This helper skips them so the parser can
//   focus on data. Returns false at end-of-file.
// ---------------------------------------------------------------------------
static bool next_data_line(std::istream& in, std::string& out) {
    std::string line;
    while (std::getline(in, line)) {
        // Trim a leading UTF-8 BOM if present (Windows editors love to add it).
        if (line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }
        std::size_t i = line.find_first_not_of(" \t\r\n");   // first real char
        if (i == std::string::npos) continue;   // blank line -> skip
        if (line[i] == '#') continue;           // comment line -> skip
        out = line;
        return true;
    }
    return false;
}

// read_doubles: pull the next data line and parse exactly `n` doubles from it,
//   throwing a descriptive error if the line is short. Used for the header
//   counts and for every geometry / strength row.
static std::vector<double> read_doubles(std::istream& in, int n, const char* what) {
    std::string line;
    if (!next_data_line(in, line))
        throw std::runtime_error(std::string("unexpected EOF while reading ") + what);
    std::istringstream ss(line);
    std::vector<double> v;
    double x;
    while (ss >> x) v.push_back(x);
    if (static_cast<int>(v.size()) < n)
        throw std::runtime_error(std::string("too few values for ") + what);
    v.resize(n);
    return v;
}

// ---------------------------------------------------------------------------
// load_ecg: parse the sample file (format documented in data/README.md).
//   Layout (whitespace/newline separated, '#' comments and blanks allowed):
//     L S T                         (three ints: electrodes, sources, frames)
//     electrode positions           (L lines, each "x y z" in metres)
//     source positions              (S lines, each "x y z" in metres)
//     source directions             (S lines, each "dx dy dz"; normalized here)
//     source strengths              (S lines, each T numbers: the time series)
//   After parsing we compute expected_peak_lead from the geometry so the demo
//   has a deterministic ground-truth headline.
// ---------------------------------------------------------------------------
ECGData load_ecg(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ECG sample file: " + path);

    ECGData d;
    // ---- header: L S T ----------------------------------------------------
    std::vector<double> hdr = read_doubles(in, 3, "header (L S T)");
    d.L = static_cast<int>(hdr[0]);
    d.S = static_cast<int>(hdr[1]);
    d.T = static_cast<int>(hdr[2]);
    if (d.L <= 0 || d.S <= 0 || d.T <= 0)
        throw std::runtime_error("header L,S,T must all be positive");

    // ---- electrode positions [L x 3] --------------------------------------
    d.electrode.resize(d.L);
    for (int e = 0; e < d.L; ++e) {
        std::vector<double> p = read_doubles(in, 3, "electrode position");
        d.electrode[e] = ecg::Vec3{ p[0], p[1], p[2] };
    }

    // ---- source anchor positions [S x 3] ----------------------------------
    d.src_pos.resize(d.S);
    for (int s = 0; s < d.S; ++s) {
        std::vector<double> p = read_doubles(in, 3, "source position");
        d.src_pos[s] = ecg::Vec3{ p[0], p[1], p[2] };
    }

    // ---- source directions [S x 3] (normalized to unit length) ------------
    d.src_dir.resize(d.S);
    for (int s = 0; s < d.S; ++s) {
        std::vector<double> p = read_doubles(in, 3, "source direction");
        // Normalize here (shared ecg::normalize) so the "strength" carries the
        // magnitude and the lead-field entries are pure geometry.
        d.src_dir[s] = ecg::normalize(ecg::Vec3{ p[0], p[1], p[2] });
    }

    // ---- source strength time series X [S x T], row-major -----------------
    d.source_strength.assign(static_cast<std::size_t>(d.S) * d.T, 0.0);
    for (int s = 0; s < d.S; ++s) {
        std::vector<double> row = read_doubles(in, d.T, "source strength row");
        for (int t = 0; t < d.T; ++t)
            d.source_strength[static_cast<std::size_t>(s) * d.T + t] = row[t];
    }

    // ---- deterministic ground-truth: which electrode should swing most? ---
    // Reasoning: the electrode nearest the source with the largest strength
    // SWING (max-min over time) should record the largest peak-to-peak signal,
    // because the 1/dist^3 lead field makes the nearest strong source dominate.
    // We compute this purely from the input geometry (no results) so it is an
    // independent "truth" the computed Phi must recover (PATTERNS.md §6).
    int strongest_src = 0;
    double best_swing = -1.0;
    for (int s = 0; s < d.S; ++s) {
        double lo = d.source_strength[static_cast<std::size_t>(s) * d.T];
        double hi = lo;
        for (int t = 1; t < d.T; ++t) {
            double v = d.source_strength[static_cast<std::size_t>(s) * d.T + t];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        double swing = hi - lo;
        if (swing > best_swing) { best_swing = swing; strongest_src = s; }
    }
    // Nearest electrode to that strongest-swinging source.
    int nearest = 0;
    double best_d2 = -1.0;
    for (int e = 0; e < d.L; ++e) {
        double dx = d.electrode[e].x - d.src_pos[strongest_src].x;
        double dy = d.electrode[e].y - d.src_pos[strongest_src].y;
        double dz = d.electrode[e].z - d.src_pos[strongest_src].z;
        double dd = dx * dx + dy * dy + dz * dz;
        if (best_d2 < 0.0 || dd < best_d2) { best_d2 = dd; nearest = e; }
    }
    d.expected_peak_lead = nearest;

    return d;
}

// ---------------------------------------------------------------------------
// build_lead_field_reference: A [L x S] row-major, one entry per (electrode,
//   source). Each entry is the shared ecg::dipole_potential -- IDENTICAL to what
//   the GPU kernel computes, so A_cpu and A_gpu match to (near) the last bit.
//   Complexity O(L*S): tiny for the sample, but this is the step the catalog
//   calls "solve one BVP per electrode"; here the BVP is the analytic dipole
//   Green's function, done in closed form (THEORY.md §"real world").
// ---------------------------------------------------------------------------
void build_lead_field_reference(const ECGData& d, std::vector<double>& A) {
    A.assign(static_cast<std::size_t>(d.L) * d.S, 0.0);
    for (int e = 0; e < d.L; ++e) {
        for (int s = 0; s < d.S; ++s) {
            A[static_cast<std::size_t>(e) * d.S + s] =
                ecg::dipole_potential(d.electrode[e], d.src_pos[s], d.src_dir[s]);
        }
    }
}

// ---------------------------------------------------------------------------
// apply_forward_reference: Phi [L x T] = A [L x S] * X [S x T], row-major.
//   The classic i-s-t loop nest. Phi[e][t] = sum_s A[e][s] * X[s][t]. This is
//   exactly what cuBLAS DGEMM does on the GPU; we keep the naive version here so
//   the learner sees the O(L*S*T) work the library hides. We iterate s in the
//   OUTER loop so each A[e][s] is read once and streamed across the T-length row
//   -- the same accumulation order the CPU would prefer, and a fair baseline.
// ---------------------------------------------------------------------------
void apply_forward_reference(const std::vector<double>& A,
                             const std::vector<double>& X,
                             int L, int S, int T,
                             std::vector<double>& Phi) {
    Phi.assign(static_cast<std::size_t>(L) * T, 0.0);
    for (int e = 0; e < L; ++e) {
        for (int s = 0; s < S; ++s) {
            double a = A[static_cast<std::size_t>(e) * S + s];         // A[e][s]
            const double* xrow = &X[static_cast<std::size_t>(s) * T];  // X[s][*]
            double* prow       = &Phi[static_cast<std::size_t>(e) * T];// Phi[e][*]
            for (int t = 0; t < T; ++t)
                prow[t] += a * xrow[t];
        }
    }
}
