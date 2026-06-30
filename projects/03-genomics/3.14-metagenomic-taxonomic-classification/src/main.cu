// ===========================================================================
// src/main.cu  --  Entry point: load, classify (CPU+GPU), verify, report
// ---------------------------------------------------------------------------
// Project 3.14 : Metagenomic Taxonomic Classification
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (reference genomes + reads from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted per-read taxon ids.
//   3. GPU classify   (kernels.cu)         -> the thing being taught.
//   4. VERIFY: the GPU's integer taxon ids match the CPU's EXACTLY (tolerance 0,
//      because both call the same __host__ __device__ classify_read core).
//   5. REPORT: a deterministic taxonomic abundance profile + accuracy vs the
//      synthetic ground truth to stdout; timings to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then kmer_core.h (the shared math), kernels.cuh ->
// kernels.cu (the GPU harness), then reference_cpu.* (the baseline + loader).
// ===========================================================================
#include <chrono>     // steady_clock for the CPU-reference stopwatch
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // classify_gpu, RefDatabase, ReadSet
#include "reference_cpu.h"    // load_problem, classify_cpu
#include "kmer_core.h"        // KMER_K, MAX_TAXA, TAXON_UNCLASSIFIED

static const char* PROJECT_ID   = "3.14";
static const char* PROJECT_NAME = "Metagenomic Taxonomic Classification";

// Default dataset if none is given on the command line.
static const char* DEFAULT_SAMPLE = "data/sample/metagenome_sample.txt";

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : DEFAULT_SAMPLE;
    RefDatabase db;
    ReadSet     reads;
    try {
        load_problem(path, db, reads);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    // A std::chrono stopwatch around the serial classifier. We avoid pulling in
    // util/io.hpp's float helpers here (the result is integer ids), so we time
    // inline with the standard library.
    std::vector<uint32_t> taxa_cpu;
    auto t0 = std::chrono::steady_clock::now();
    classify_cpu(reads, db, taxa_cpu);
    auto t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- 3. GPU classify (kernel timed inside the wrapper) ----------------
    std::vector<uint32_t> taxa_gpu;
    float gpu_kernel_ms = 0.0f;
    classify_gpu(reads, db, taxa_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: integer-exact agreement -------------------------------
    // Both sides ran the identical __host__ __device__ classify_read core, so the
    // taxon ids must match bit-for-bit. A single mismatch is a real bug, not a
    // floating-point rounding artifact -> the tolerance is exactly 0.
    int mismatches = 0;
    int first_bad = -1;
    if (taxa_cpu.size() != taxa_gpu.size()) {
        mismatches = reads.n_reads;   // shape bug
    } else {
        for (int i = 0; i < reads.n_reads; ++i) {
            if (taxa_cpu[i] != taxa_gpu[i]) {
                if (first_bad < 0) first_bad = i;
                ++mismatches;
            }
        }
    }
    const bool pass = (mismatches == 0);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // (a) Per-taxon abundance: how many reads were assigned to each taxon. This
    //     IS the metagenomic profile -- the headline scientific output. We scan
    //     taxon ids in ascending order so the listing is deterministic.
    std::vector<int> assigned(db.names.size(), 0);   // assigned[id] = read count
    int unclassified = 0;
    for (int i = 0; i < reads.n_reads; ++i) {
        uint32_t t = taxa_gpu[i];
        if (t == TAXON_UNCLASSIFIED || t >= db.names.size()) ++unclassified;
        else ++assigned[t];
    }
    // (b) Accuracy vs the synthetic ground truth: of the reads we DID classify,
    //     how many got the correct taxon? (Integer counts -> deterministic.)
    int correct = 0, classified = 0;
    for (int i = 0; i < reads.n_reads; ++i) {
        if (taxa_gpu[i] != TAXON_UNCLASSIFIED) {
            ++classified;
            if (taxa_gpu[i] == reads.truth[i]) ++correct;
        }
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("k-mer classification: %d reads vs %llu reference %d-mers (%d taxa)\n",
                reads.n_reads, (unsigned long long)db.num_kmers, KMER_K,
                (int)db.names.size() - 1);
    std::printf("taxonomic abundance profile (reads assigned per taxon):\n");
    // Fixed-width columns so the listing is tidy and byte-identical every run:
    //   "  taxon <id>  <name padded to 24>  <count> reads".
    for (std::size_t id = 1; id < db.names.size(); ++id) {
        std::printf("  taxon %zu  %-24s  %d reads\n", id, db.names[id].c_str(), assigned[id]);
    }
    // The unclassified bin uses the same column layout (taxon id column blanked).
    std::printf("  %-8s %-24s  %d reads\n", "(none)", "unclassified", unclassified);
    std::printf("accuracy on classified reads: %d/%d correct\n", correct, classified);
    std::printf("RESULT: %s (GPU taxon ids match CPU exactly; %d/%d reads agree)\n",
                pass ? "PASS" : "FAIL", reads.n_reads - mismatches, reads.n_reads);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (reads=%d, taxa=%d, table_capacity=%llu)\n",
                 path.c_str(), reads.n_reads, (int)db.names.size() - 1,
                 (unsigned long long)db.capacity);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins at clinical scale (millions of reads).\n");
    if (!pass)
        std::fprintf(stderr, "[verify] %d mismatch(es); first at read %d (cpu=%u gpu=%u)\n",
                     mismatches, first_bad,
                     first_bad >= 0 ? taxa_cpu[first_bad] : 0u,
                     first_bad >= 0 ? taxa_gpu[first_bad] : 0u);
    else
        std::fprintf(stderr, "[verify] all %d reads agree (integer taxon ids, tolerance 0)\n",
                     reads.n_reads);

    return pass ? 0 : 1;
}
