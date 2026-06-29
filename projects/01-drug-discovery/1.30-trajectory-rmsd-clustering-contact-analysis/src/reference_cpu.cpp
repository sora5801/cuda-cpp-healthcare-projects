// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over frames, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree we believe
//   the GPU. Crucially it calls the EXACT SAME per-frame math (kabsch_rmsd,
//   frac_native_contacts, count_native_contacts) from rmsd_core.h that the GPU
//   kernel calls, so agreement is to ~machine epsilon, not "close enough".
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, rmsd_core.h. Compare with kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>      // std::floor
#include <fstream>    // std::ifstream
#include <sstream>    // std::istringstream
#include <stdexcept>  // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_trajectory: parse the simple whitespace text format (see data/README.md).
//   We validate that the file's atom count matches the compile-time N_ATOMS,
//   because the whole pipeline assumes a fixed per-frame layout (it lets the
//   inner loops unroll and lets a frame map cleanly onto one GPU thread).
//   Throwing on any inconsistency makes the demo fail loudly rather than read
//   garbage.
// ---------------------------------------------------------------------------
Trajectory load_trajectory(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open trajectory file: " + path);

    int n_frames = 0, n_atoms = 0, ref = 0;
    in >> n_frames >> n_atoms >> ref;
    if (!in) throw std::runtime_error("bad header (expected: n_frames n_atoms ref): " + path);
    if (n_atoms != N_ATOMS)
        throw std::runtime_error("file N_ATOMS != compiled N_ATOMS (" +
                                 std::to_string(n_atoms) + " vs " + std::to_string(N_ATOMS) + ")");
    if (n_frames <= 0) throw std::runtime_error("n_frames must be positive: " + path);
    if (ref < 0 || ref >= n_frames) throw std::runtime_error("ref frame out of range: " + path);

    Trajectory t;
    t.n_frames = n_frames;
    t.ref = ref;
    t.coords.resize(static_cast<std::size_t>(n_frames) * N_ATOMS * 3);

    // Read F*N atoms of "x y z". The flat index keeps frame-major, atom-major
    // order (matching frame_ptr() in rmsd_core.h).
    for (std::size_t k = 0; k < t.coords.size(); ++k) {
        double v;
        if (!(in >> v))
            throw std::runtime_error("trajectory truncated (not enough coordinates): " + path);
        t.coords[k] = v;
    }
    return t;
}

// ---------------------------------------------------------------------------
// analyze_trajectory_cpu: the serial twin of analyze_trajectory_gpu().
//   For each frame f we compute (1) optimal-superposition RMSD to the reference
//   and (2) the native-contact fraction Q. The reference's native-contact COUNT
//   is computed ONCE before the loop (it is the same for every frame), exactly
//   as the GPU wrapper does, so the two divide by the identical integer.
//   Complexity: O(F * (N + N^2)) -- per frame the QCP RMSD is O(N) to build the
//   covariance plus O(1) for the 4x4 eigenvalue, and the contact sweep is O(N^2).
// ---------------------------------------------------------------------------
void analyze_trajectory_cpu(const Trajectory& traj, FrameMetrics& out) {
    out.rmsd.assign(static_cast<std::size_t>(traj.n_frames), 0.0);
    out.qnc.assign(static_cast<std::size_t>(traj.n_frames), 0.0);

    const double* ref = frame_ptr(traj.coords.data(), traj.ref);
    const int native_total = count_native_contacts(ref);  // once: same for all f

    for (int f = 0; f < traj.n_frames; ++f) {
        const double* fr = frame_ptr(traj.coords.data(), f);
        // Both calls are the shared __host__ __device__ math from rmsd_core.h --
        // identical to what kernels.cu runs on the device.
        out.rmsd[static_cast<std::size_t>(f)] = kabsch_rmsd(fr, ref);
        out.qnc[static_cast<std::size_t>(f)]  = frac_native_contacts(fr, ref, native_total);
    }
}

// ---------------------------------------------------------------------------
// cluster_by_rmsd: the simplest deterministic conformational clustering -- bin
//   frames by their RMSD shell b = floor(rmsd / width). Real GROMOS clustering
//   uses a pairwise RMSD matrix + a fixed-radius neighbour rule (see THEORY);
//   this 1-D binning is its didactic stand-in and is fully order-independent, so
//   the histogram is byte-identical every run (important for the demo's stdout).
//   Complexity: O(F). counts is sized to the largest occupied bin + 1.
// ---------------------------------------------------------------------------
void cluster_by_rmsd(const FrameMetrics& m, double width, std::vector<int>& counts) {
    counts.clear();
    if (width <= 0.0) return;                      // guard: nonsensical bin width
    for (double r : m.rmsd) {
        const int b = static_cast<int>(std::floor(r / width));  // RMSD shell index
        if (b < 0) continue;                       // RMSD is >= 0, but be safe
        if (b >= static_cast<int>(counts.size()))
            counts.resize(static_cast<std::size_t>(b) + 1, 0);  // grow to fit bin b
        counts[static_cast<std::size_t>(b)] += 1;
    }
}
