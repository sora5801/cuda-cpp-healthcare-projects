// ===========================================================================
// src/util/io.hpp  --  Pure-C++ host helpers: timing, comparison, file I/O
// ---------------------------------------------------------------------------
// ROLE IN THE PROJECT
//   Shared host-side utilities that are needed by BOTH the GPU entry point
//   (main.cu) and the CPU reference (reference_cpu.cpp). Because reference_cpu
//   is compiled by the plain C++ compiler (cl.exe / g++), this header MUST stay
//   free of any CUDA constructs -- only <chrono>, <vector>, <fstream>, etc.
//   (GPU timing lives separately in util/timer.cuh.)
//
//   Copied verbatim into every project's src/util/ (documented duplication).
//
// WHAT'S HERE
//   * CpuTimer   -- a std::chrono wall-clock stopwatch for the CPU reference.
//   * max_abs_err / allclose -- compare two float arrays so we can VERIFY the
//                  GPU result against the CPU reference within a tolerance.
//   * read_floats / write_floats -- whitespace-separated float I/O for the
//                  tiny sample datasets in data/sample/.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <chrono>     // high_resolution_clock
#include <cmath>      // std::fabs
#include <cstddef>    // std::size_t
#include <fstream>    // std::ifstream / std::ofstream
#include <limits>     // std::numeric_limits
#include <iomanip>    // std::setprecision
#include <sstream>    // std::ostringstream
#include <stdexcept>  // std::runtime_error
#include <string>
#include <vector>

namespace util {

// ---------------------------------------------------------------------------
// CpuTimer: measure wall-clock time of a host computation.
//   We use steady_clock (monotonic; never jumps backwards on NTP adjustments)
//   so a duration is always non-negative and meaningful.
// ---------------------------------------------------------------------------
struct CpuTimer {
    std::chrono::steady_clock::time_point t0;
    void start() { t0 = std::chrono::steady_clock::now(); }
    double stop_ms() const {
        auto t1 = std::chrono::steady_clock::now();
        // duration<double, milli> converts the tick count to fractional ms.
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ---------------------------------------------------------------------------
// max_abs_err: the largest |a[i] - b[i]| over two equal-length arrays.
//   This single number is our headline correctness metric: if it is below the
//   project's documented tolerance, the GPU result "agrees with" the CPU one.
//   Returns +infinity on a length mismatch so the caller cannot mistake a
//   shape bug for agreement.
// ---------------------------------------------------------------------------
inline double max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > worst) worst = d;
    }
    return worst;
}

// allclose: convenience boolean wrapper around max_abs_err for a fixed atol.
inline bool allclose(const std::vector<float>& a, const std::vector<float>& b, double atol) {
    return max_abs_err(a, b) <= atol;
}

// ---------------------------------------------------------------------------
// read_floats: slurp every whitespace-separated number from a text file into a
//   flat vector<float>. The caller knows the layout (e.g. "first value is n").
//   Throws std::runtime_error if the file cannot be opened so demos fail loudly
//   instead of silently running on empty input.
// ---------------------------------------------------------------------------
inline std::vector<float> read_floats(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);
    std::vector<float> v;
    float x;
    while (in >> x) v.push_back(x);
    return v;
}

// write_floats: dump a vector<float> one-per-line at fixed precision (handy for
//   regenerating expected_output.txt or saving results).
inline void write_floats(const std::string& path, const std::vector<float>& v) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open output file: " + path);
    out << std::setprecision(6) << std::fixed;
    for (float x : v) out << x << "\n";
}

}  // namespace util
