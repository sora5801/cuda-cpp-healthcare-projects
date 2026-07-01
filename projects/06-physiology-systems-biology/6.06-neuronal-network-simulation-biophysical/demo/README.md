# Demo — 6.6 Neuronal Network Simulation (Biophysical)

## What this demonstrates

One command builds (if needed) and runs the biophysical network simulator on the
committed synthetic sample, then checks that the **GPU result matches the CPU
reference exactly**.

Run it:

```powershell
# Windows (PowerShell) — uses the Visual Studio Release build
./demo/run_demo.ps1
```

```bash
# Linux/macOS — uses the CMake build
./demo/run_demo.sh
```

## What you are looking at

The program simulates a **ring of 16 multi-compartment Hodgkin–Huxley neurons**
for 100 ms. Neuron *i* excites neuron *(i+1)*; one leading cell is depolarised at
*t = 0*, fires, and the spike propagates around the ring as a **travelling wave**.

`stdout` (diffed against `expected_output.txt`, so it is byte-identical every run):

- the network dimensions and wiring,
- a per-cell line `cN : spikes firstStep` for 8 evenly-spaced cells — notice the
  **first-spike step marches steadily upward** (c0 at step 9, c2 at 81, c4 at 153,
  …), which is the wave sweeping around the ring one synaptic delay per hop,
- network totals (total spikes, active cells, mean firing rate in Hz),
- a `RESULT: PASS` line — the GPU per-cell spike counts match the CPU reference
  **exactly** (integer crossing counts of identical double-precision voltages, so
  the tolerance is literally zero).

`stderr` (shown but **not** diffed — it varies per machine/run):

- CPU vs GPU timing and the number of per-step kernel launches,
- an honest note that one launch per timestep is *launch-bound* on a tiny 16-cell
  network (the GPU here is slower than the CPU) and only wins as the cell count
  grows into the thousands — a teaching artifact, never a benchmark claim,
- a short activity trace (first active step, peak simultaneous spikes),
- the verification residual (`mismatches = 0`).

## Expected output

See [`expected_output.txt`](./expected_output.txt) — captured from a real run on
an RTX 2080 (sm_75), CUDA 13.3, VS 2026. The `stdout` lines there must match
exactly; the `stderr` timing is not part of the comparison.
