// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial per-window MC sampling (CPU baseline)
// ---------------------------------------------------------------------------
// Project 1.5 : Free Energy Perturbation / Thermodynamic Integration
//
// Compiled by the host C++ compiler ONLY (never nvcc). The model, RNG and the
// MC chain itself are in alchemy.h; here we only (a) parse the config file and
// (b) drive run_chain() once per lambda-window. The GPU kernel (kernels.cu)
// calls the SAME run_chain() from one thread per window, so these results match.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// load_config: read the 10 whitespace-separated fields of an AlchemyConfig.
//   Layout:  kA x0A kB x0B kT windows equil samples step x_init
//   Lines beginning with '#' are treated as comments and skipped (the sample
//   file carries a "SYNTHETIC" banner this way). We validate the physically /
//   numerically meaningful constraints so a typo in the file produces a clear
//   error rather than a silent NaN later.
// ---------------------------------------------------------------------------
AlchemyConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);

    // Concatenate all non-comment lines into one stream of tokens, so the data
    // may be laid out on one line or several and may be preceded by '#' banners.
    std::ostringstream body;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;  // blank/comment
        body << line << ' ';
    }
    std::istringstream toks(body.str());

    AlchemyConfig c;
    if (!(toks >> c.kA >> c.x0A >> c.kB >> c.x0B >> c.kT
              >> c.windows >> c.equil >> c.samples >> c.step >> c.x_init)) {
        throw std::runtime_error(
            "bad parameters (expected 'kA x0A kB x0B kT windows equil samples "
            "step x_init') in " + path);
    }
    // kA,kB > 0 : real springs; kT > 0 : positive temperature; windows >= 2 so
    // the lambda-grid spans [0,1]; samples > 0 so the average is defined.
    if (c.kA <= 0.0 || c.kB <= 0.0 || c.kT <= 0.0 ||
        c.windows < 2 || c.equil < 0 || c.samples <= 0 || c.step <= 0.0) {
        throw std::runtime_error("invalid alchemy parameters in " + path);
    }
    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu: the serial reference. Each window is an independent MC chain,
//   so this is a plain loop (one GPU thread per window in kernels.cu). We record
//   < dU/dlambda > for every window plus its accepted-move count.
// ---------------------------------------------------------------------------
void integrate_cpu(const AlchemyConfig& c,
                   std::vector<double>& dvals,
                   std::vector<long long>& accepted) {
    const int W = n_windows(c);
    dvals.assign(W, 0.0);
    accepted.assign(W, 0);
    for (int w = 0; w < W; ++w) {
        long long acc = 0;
        dvals[w] = run_chain(c, w, &acc);   // the one true sampler (alchemy.h)
        accepted[w] = acc;
    }
}
