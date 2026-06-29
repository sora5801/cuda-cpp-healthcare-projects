// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is deliberately a
//   single readable serial loop over atoms -- no parallelism, no cleverness --
//   so that when the GPU and CPU agree we believe the GPU. The per-atom math it
//   calls (atomic_energy) lives in nnp.h and is SHARED with the GPU kernel, so
//   "agree" here means agree to floating-point round-off, not approximately.
//
//   This file ALSO builds the model (ACSF hyperparameters + the fixed MLP
//   weights). The weight generator is a tiny deterministic PRNG so the demo is
//   reproducible and self-contained; scripts/make_synthetic.py reproduces the
//   exact same numbers in Python for anyone who wants to inspect them.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, nnp.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cstdint>     // uint64_t for the PRNG state
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// splitmix64: a tiny, well-known deterministic PRNG (Sebastiano Vigna).
//   Given a 64-bit state it returns a well-mixed 64-bit value and advances the
//   state. We use it ONLY to manufacture reproducible "weights" -- it is not
//   security- or statistics-critical, just a portable way to get the SAME
//   pseudo-random doubles in C++ and in Python (make_synthetic.py mirrors it).
//   Integer math only -> identical results on every platform.
// ---------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;            // golden-ratio increment
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Map the next PRNG output to a double in the half-open interval [-0.5, 0.5).
//   Small, zero-centered weights keep the tanh activations in their responsive
//   region (not saturated), which is exactly how real nets are initialized.
//   Dividing a 53-bit mantissa slice by 2^53 gives a uniform [0,1) double, then
//   we shift to [-0.5, 0.5). The arithmetic matches make_synthetic.py exactly.
static inline double next_weight(uint64_t& state) {
    const uint64_t bits = splitmix64(state) >> 11;          // top 53 bits
    const double u = static_cast<double>(bits) / 9007199254740992.0;  // / 2^53 -> [0,1)
    return u - 0.5;                                          // -> [-0.5, 0.5)
}

// ---------------------------------------------------------------------------
// build_acsf_params: fixed, hand-chosen radial descriptor hyperparameters.
//   Rc = 5.0 Angstrom is a typical NNP cutoff (covers first/second neighbor
//   shells). The 8 shell centers Rs are spread from 0.8 to 4.0 Angstrom so the
//   descriptor resolves bond-length through non-bonded distances. eta sets each
//   Gaussian's width; 1.5 /Angstrom^2 gives shells ~0.5 Angstrom wide that
//   overlap smoothly. These are constants (no RNG) so they are obvious.
// ---------------------------------------------------------------------------
AcsfParams build_acsf_params() {
    AcsfParams p;
    p.Rc  = 5.0;    // cutoff radius (Angstrom)
    p.eta = 1.5;    // Gaussian width parameter (1/Angstrom^2)
    // Eight evenly spaced shell centers from 0.8 to 4.0 Angstrom.
    for (int s = 0; s < N_DESC; ++s)
        p.Rs[s] = 0.8 + s * (4.0 - 0.8) / (N_DESC - 1);   // 0.8, 1.257, ..., 4.0
    return p;
}

// ---------------------------------------------------------------------------
// build_atomic_net: deterministically "manufacture" the MLP weights.
//   Seed 0xA1MLP... is fixed so EVERY run produces identical weights -> the
//   output is reproducible and the GPU must match it. Bias terms are seeded the
//   same way. (A real NNP would load weights trained on ANI-1ccx / SPICE; we
//   substitute a fixed net so the demo runs offline -- see THEORY "real world".)
//   The exact draw ORDER (w1, b1, w2, b2, w3, b3) must match make_synthetic.py.
// ---------------------------------------------------------------------------
AtomicNet build_atomic_net() {
    AtomicNet net;
    uint64_t state = 0xA1A1A1A1A1A1A1A1ULL;   // fixed seed -> reproducible weights

    for (int i = 0; i < N_HID * N_DESC; ++i) net.w1[i] = next_weight(state);
    for (int i = 0; i < N_HID;          ++i) net.b1[i] = next_weight(state);
    for (int i = 0; i < N_HID * N_HID;  ++i) net.w2[i] = next_weight(state);
    for (int i = 0; i < N_HID;          ++i) net.b2[i] = next_weight(state);
    for (int i = 0; i < N_HID;          ++i) net.w3[i] = next_weight(state);
    net.b3 = next_weight(state);
    return net;
}

// ---------------------------------------------------------------------------
// load_structure: parse the simple text format documented in data/README.md.
//   Lines beginning with '#' (after optional whitespace) are comments. The first
//   non-comment token is the atom count n; the next 3n numbers are coordinates.
//   We validate aggressively so a truncated file is an explicit error, never a
//   silent zero-filled structure.
// ---------------------------------------------------------------------------
Structure load_structure(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open structure file: " + path);

    // Collect all whitespace-separated tokens, skipping '#' comment lines.
    std::vector<double> nums;
    std::string line;
    while (std::getline(in, line)) {
        // Find the first non-space character; skip blank and '#' comment lines.
        std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream ls(line);
        double v;
        while (ls >> v) nums.push_back(v);
    }

    if (nums.empty()) throw std::runtime_error("empty structure file: " + path);
    const int n = static_cast<int>(nums[0]);
    if (n <= 0) throw std::runtime_error("non-positive atom count in: " + path);
    if (static_cast<int>(nums.size()) < 1 + 3 * n)
        throw std::runtime_error("structure file too short (need 3 coords/atom): " + path);

    Structure s;
    s.n = n;
    s.pos.assign(nums.begin() + 1, nums.begin() + 1 + 3 * n);   // the 3n coordinates
    return s;
}

// ---------------------------------------------------------------------------
// nnp_energy_cpu: the serial reference. One readable loop over atoms; for each
//   atom call the shared atomic_energy() (descriptor + MLP, from nnp.h). The sum
//   is accumulated in a double in ascending atom order -- the SAME order the GPU
//   reduction uses on the host side -- so the total matches too.
//   Complexity: O(n^2) here (each atom scans all others for neighbors); a cell
//   list would make it O(n). See THEORY.md "real world".
// ---------------------------------------------------------------------------
double nnp_energy_cpu(const Structure& s, const AcsfParams& p, const AtomicNet& net,
                      std::vector<double>& e_atom) {
    e_atom.assign(static_cast<std::size_t>(s.n), 0.0);   // one energy per atom
    double total = 0.0;
    for (int i = 0; i < s.n; ++i) {
        // atomic_energy is the SHARED host/device function -> the GPU computes
        // the byte-identical value for this same atom in kernels.cu.
        const double Ei = atomic_energy(s.pos.data(), s.n, i, p, net);
        e_atom[static_cast<std::size_t>(i)] = Ei;
        total += Ei;                                     // accumulate the total
    }
    return total;
}
