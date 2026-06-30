// ===========================================================================
// src/reference_cpu.cpp  --  Loader, position init, serial reverse diffusion
// ---------------------------------------------------------------------------
// Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- plain loops, no parallelism, no cleverness -- so that
//   when the GPU and CPU agree, we believe the GPU. The per-token denoising
//   math is NOT duplicated here: it lives once in cofold.h (denoise_token), and
//   both this file and the GPU kernel call it. That is what makes the two paths
//   numerically comparable (PATTERNS.md §2).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, cofold.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sqrt, std::log, std::cos
#include <cstdint>     // std::uint64_t
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <utility>     // std::swap

// ---------------------------------------------------------------------------
// load_complex: parse the sample file into a Complex.
//   Format (whitespace-separated; see data/README.md):
//     line 1 : n_protein n_ligand steps temp step_frac type_bias seed noise_scale
//     line k : type x* y* z*        (type in {0=protein, 1=ligand})
//   We validate aggressively so a corrupt file fails with a clear message
//   instead of silently producing garbage.
// ---------------------------------------------------------------------------
Complex load_complex(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open complex file: " + path);

    Complex C;
    CofoldParams& P = C.P;
    // Read the header. step_frac/temp/type_bias/noise_scale are doubles.
    if (!(in >> P.n_protein >> P.n_ligand >> P.steps >> P.temp
             >> P.step_frac >> P.type_bias >> P.seed >> P.noise_scale)) {
        throw std::runtime_error("bad header (expected 'n_protein n_ligand steps "
            "temp step_frac type_bias seed noise_scale') in " + path);
    }
    P.n_tokens = P.n_protein + P.n_ligand;

    // Sanity bounds: at least 2 tokens, a positive schedule, sane temperature.
    if (P.n_protein < 0 || P.n_ligand < 0 || P.n_tokens < 2)
        throw std::runtime_error("need >= 2 tokens in " + path);
    if (P.steps < 0 || P.temp <= 0.0 || P.step_frac <= 0.0 || P.step_frac > 1.0)
        throw std::runtime_error("invalid diffusion schedule in " + path);

    // Read one line per token: type then the 3 native coordinates.
    C.target.assign((std::size_t)P.n_tokens * D_POS, 0.0);
    C.types.assign((std::size_t)P.n_tokens, 0);
    for (int i = 0; i < P.n_tokens; ++i) {
        int type; double x, y, z;
        if (!(in >> type >> x >> y >> z))
            throw std::runtime_error("ran out of token rows in " + path);
        if (type != TYPE_PROTEIN && type != TYPE_LIGAND)
            throw std::runtime_error("token type must be 0 or 1 in " + path);
        C.types[i] = type;
        C.target[(std::size_t)i * D_POS + 0] = x;
        C.target[(std::size_t)i * D_POS + 1] = y;
        C.target[(std::size_t)i * D_POS + 2] = z;
    }
    return C;
}

// ---------------------------------------------------------------------------
// A tiny, fully deterministic PRNG so the initial noise is identical on every
// machine and every run (PATTERNS.md §3, "stdout must be identical every run").
//   splitmix64: a well-known 64-bit mixer. We seed it from CofoldParams.seed and
//   the token/axis index so each coordinate gets an independent stream. We do
//   NOT use std::mt19937/normal_distribution because their output is not
//   guaranteed identical across standard-library implementations -- a real
//   reproducibility gotcha worth teaching.
// ---------------------------------------------------------------------------
static inline std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;            // golden-ratio increment
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Draw one standard-normal sample from two uniforms via the Box-Muller
// transform: if u1,u2 ~ U(0,1] then sqrt(-2 ln u1) * cos(2 pi u2) ~ N(0,1).
// Deterministic given the integer key -> reproducible noise field.
static inline double gaussian_from_key(std::uint64_t key) {
    std::uint64_t s = key;
    // Two independent uniforms in (0,1]; +1 avoids log(0).
    const double u1 = (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
    const double u2 = (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
    const double r  = std::sqrt(-2.0 * std::log(u1 + 1.0e-300));
    const double PI = 3.14159265358979323846;
    return r * std::cos(2.0 * PI * u2);
}

// ---------------------------------------------------------------------------
// init_positions: x_T = x* + noise_scale * N(0,1), per coordinate. This is the
// fully-noised end of the forward diffusion that the reverse process undoes.
// The key mixes seed, token index and axis so neighbouring coordinates are not
// correlated.
// ---------------------------------------------------------------------------
void init_positions(const Complex& C, std::vector<double>& pos) {
    const CofoldParams& P = C.P;
    pos.assign((std::size_t)P.n_tokens * D_POS, 0.0);
    for (int i = 0; i < P.n_tokens; ++i) {
        for (int c = 0; c < D_POS; ++c) {
            const std::uint64_t key =
                (std::uint64_t)(std::uint32_t)P.seed * 0x100000001B3ULL
                + (std::uint64_t)i * 131ULL + (std::uint64_t)c * 17ULL;
            const double g = gaussian_from_key(key);
            const std::size_t idx = (std::size_t)i * D_POS + c;
            pos[idx] = C.target[idx] + P.noise_scale * g;
        }
    }
}

// ---------------------------------------------------------------------------
// simulate_cpu: the serial reverse-diffusion loop (the trusted baseline).
//   We double-buffer: read the FROZEN current positions, compute every token's
//   next position, then swap -- so within a step all tokens see the same input,
//   exactly mirroring the GPU's per-step kernel launch (no in-step races on
//   either side). This is the same double-buffer ("ping-pong") discipline as
//   the stencil flagship 14.02; here the per-token work is an attention pass,
//   not a 5-point Laplacian.
// ---------------------------------------------------------------------------
void simulate_cpu(const Complex& C, std::vector<double>& pos) {
    const CofoldParams& P = C.P;
    const std::size_t N = (std::size_t)P.n_tokens * D_POS;
    std::vector<double> buf(N);          // the destination (next) buffer
    double* cur = pos.data();            // source (current state)
    double* nxt = buf.data();            // destination (next state)

    for (int s = 0; s < P.steps; ++s) {
        // One attention pass: update every token from the frozen `cur` state.
        for (int i = 0; i < P.n_tokens; ++i)
            denoise_token(i, cur, C.target.data(), C.types.data(), P, nxt);
        std::swap(cur, nxt);             // the next state becomes current
    }
    // After the final swap, `cur` holds the latest positions. If that is the
    // local buffer (odd step count), copy it back so the caller sees the result.
    if (cur != pos.data())
        pos.assign(cur, cur + N);
}
