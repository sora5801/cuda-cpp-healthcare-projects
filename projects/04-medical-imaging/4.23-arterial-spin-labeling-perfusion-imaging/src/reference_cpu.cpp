// ===========================================================================
// src/reference_cpu.cpp  --  ASL loader + serial per-voxel reference fit
// ---------------------------------------------------------------------------
// Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- a single readable loop over voxels, no parallelism, no
//   cleverness -- so that when the GPU and CPU agree we believe the GPU. The
//   actual per-voxel math (Buxton model + Gauss-Newton) lives in asl.h and is
//   shared verbatim with the kernel, so "agree" means "agree to round-off".
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, asl.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream (line-by-line parse)
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_asl: parse the tiny text ASL study (format in data/README.md).
//   The layout is line-oriented so a human can read the committed sample:
//     line 1:  n_voxels  n_plds  max_iters  f_init  att_init
//     line 2:  pld_0 ... pld_{n_plds-1}                     (the delay schedule)
//     next n_voxels lines:  true_cbf true_att s_0 ... s_{n_plds-1}
//   The acquisition constants (T1, alpha, ...) are the shared consensus defaults
//   from asl.h (asl_default_constants); the sample only carries the acquisition
//   TIMING (PLDs) and the per-voxel signals, exactly like a real ASL series.
//
//   Every read is checked; on any malformed field we throw so the demo fails
//   loudly instead of silently fitting garbage.
// ---------------------------------------------------------------------------
HostDataset load_asl(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ASL sample file: " + path);

    HostDataset ds;
    ds.consts = asl_default_constants();   // fixed acquisition constants (asl.h)

    std::string line;

    // --- line 1: header (voxel/pld counts + fit controls) ---
    if (!std::getline(in, line))
        throw std::runtime_error("ASL file empty (expected header line): " + path);
    {
        std::istringstream ss(line);
        if (!(ss >> ds.n_voxels >> ds.n_plds >> ds.max_iters
                 >> ds.f_init >> ds.att_init))
            throw std::runtime_error("bad header (want 'n_voxels n_plds max_iters "
                                     "f_init att_init') in " + path);
    }
    if (ds.n_voxels <= 0 || ds.n_plds <= 0 || ds.max_iters <= 0)
        throw std::runtime_error("non-positive counts in ASL header: " + path);

    // --- line 2: the PLD schedule (seconds), shared by all voxels ---
    ds.pld.resize(ds.n_plds);
    if (!std::getline(in, line))
        throw std::runtime_error("missing PLD schedule line in " + path);
    {
        std::istringstream ss(line);
        for (int j = 0; j < ds.n_plds; ++j)
            if (!(ss >> ds.pld[j]))
                throw std::runtime_error("bad PLD value in " + path);
    }

    // --- next n_voxels lines: ground truth + measured signal curve ---
    ds.true_cbf.resize(ds.n_voxels);
    ds.true_att.resize(ds.n_voxels);
    ds.signal.resize((size_t)ds.n_voxels * ds.n_plds);
    for (int v = 0; v < ds.n_voxels; ++v) {
        if (!std::getline(in, line))
            throw std::runtime_error("fewer voxel lines than n_voxels in " + path);
        std::istringstream ss(line);
        if (!(ss >> ds.true_cbf[v] >> ds.true_att[v]))
            throw std::runtime_error("bad ground-truth (cbf att) for a voxel in " + path);
        for (int j = 0; j < ds.n_plds; ++j)
            if (!(ss >> ds.signal[(size_t)v * ds.n_plds + j]))
                throw std::runtime_error("bad signal sample for a voxel in " + path);
    }
    return ds;
}

// ---------------------------------------------------------------------------
// fit_cpu: fit every voxel serially. Each voxel is an INDEPENDENT nonlinear
//   least-squares solve, so the reference is a plain loop here -- and one GPU
//   thread per voxel in kernels.cu. Both call the identical asl_fit_voxel().
//   Complexity: O(n_voxels * max_iters * n_plds); the timed baseline in main.cu.
// ---------------------------------------------------------------------------
void fit_cpu(const HostDataset& ds, std::vector<AslFit>& fits) {
    fits.assign(ds.n_voxels, AslFit{});
    const AslDataset view = ds.view();     // pointer-view over the host buffers
    for (int v = 0; v < ds.n_voxels; ++v)
        fits[v] = asl_fit_voxel(view, v);
}
