# Demo — 12.01 Mass-Spectrometry Proteomics Search

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/spectra_sample.txt` (1 query vs 1024 library spectra).
3. **Verify** that the GPU cosine scores match the CPU reference.
4. **Report** the **top-5 matches** and the **rank of the known target**.

stdout (top matches) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The query was derived from
library spectrum **7**, and the search recovers it at **rank 1** with cosine
≈ 0.993, while unrelated spectra score ≈ 0.3. `RESULT: PASS` means the GPU and CPU
cosine scores agree (here to `0`). The GPU scores all 1024 spectra in a fraction of
the CPU time; the gap explodes toward the 10⁶ peptides × 10⁵ spectra of real runs.

> The spectra are **synthetic** random peak patterns — a demonstration of the
> batched dot-product search pattern, not a real proteomics identification.
