// ===========================================================================
// src/reference_cpu.cpp  --  RF loader + serial Delay-and-Sum baseline
// ---------------------------------------------------------------------------
// Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct: two plain nested loops over the image grid, delegating
//   the per-pixel math to das_pixel() in beamform.h -- the SAME function the
//   GPU kernel calls. When CPU and GPU agree we therefore believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: beamform.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cstddef>     // std::size_t
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_beamform: parse the text RF file (format documented in data/README.md).
//   We read a fixed 12-field header, then n_elements*n_samples RF floats. Each
//   `>>` is checked so a truncated or garbled file throws rather than silently
//   beamforming uninitialised memory.
// ---------------------------------------------------------------------------
BeamformProblem load_beamform(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open RF data file: " + path);

    BeamformProblem p;
    BeamformGeom& g = p.geom;
    // Header order MUST match make_synthetic.py and data/README.md exactly.
    if (!(in >> g.n_elements >> g.n_samples >> g.nx >> g.nz
             >> g.fs >> g.c >> g.pitch
             >> g.x_min >> g.z_min >> g.dx >> g.dz >> g.t0)) {
        throw std::runtime_error(
            "bad header (need: n_elements n_samples nx nz fs c pitch "
            "x_min z_min dx dz t0) in " + path);
    }
    if (g.n_elements <= 0 || g.n_samples <= 0 || g.nx <= 0 || g.nz <= 0) {
        throw std::runtime_error("non-positive geometry in " + path);
    }

    // Slurp the RF matrix: n_elements rows of n_samples floats, element-major.
    const std::size_t n = static_cast<std::size_t>(g.n_elements) * g.n_samples;
    p.rf.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(in >> p.rf[i])) {
            throw std::runtime_error("RF data truncated in " + path);
        }
    }
    return p;
}

// ---------------------------------------------------------------------------
// beamform_cpu: the serial reference. One pass over every output pixel; each
//   pixel calls das_pixel() (beamform.h), which loops over all elements summing
//   interpolated, delay-aligned RF samples. Complexity: O(nx * nz * n_elements)
//   -- this triple nest is exactly the work the GPU parallelises across pixels.
// ---------------------------------------------------------------------------
void beamform_cpu(const BeamformProblem& p, std::vector<float>& image) {
    const BeamformGeom& g = p.geom;
    image.assign(static_cast<std::size_t>(g.nx) * g.nz, 0.0f);  // zero the frame

    for (int iz = 0; iz < g.nz; ++iz) {            // depth rows (z)
        for (int ix = 0; ix < g.nx; ++ix) {        // lateral columns (x)
            // das_pixel returns the signed coherent sum for this pixel, using
            // the identical formula the kernel uses -> CPU==GPU to tight tol.
            image[static_cast<std::size_t>(iz) * g.nx + ix] =
                das_pixel(g, p.rf.data(), ix, iz);
        }
    }
}
