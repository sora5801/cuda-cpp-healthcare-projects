# Demo — 6.7 Spiking Neural Network (Point-Neuron) Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/network.txt` — a small
   Brunel-style balanced network of 200 leaky integrate-and-fire (LIF)
   neurons (160 excitatory + 40 inhibitory), 3200 sparse synapses, 500
   timesteps of 0.1 ms (50 ms of biological time).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   the total, per-step, and per-neuron spike counts must match **exactly**
   (they are computed with identical shared physics and identical integer
   fixed-point synaptic accumulation), and final membrane potentials must
   agree to ~machine round-off. Prints a clear `PASS`/`FAIL`.
4. **Time** the GPU kernel loop (CUDA events) and the CPU baseline — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result (stdout)

```
6.7 -- Spiking Neural Network (Point-Neuron) Simulation
network: 200 neurons (160 exc + 40 inh), out_degree=16, 500 steps @ dt=0.10 ms (50.0 ms)
weights: w_exc=0.900  w_inh=-2.200  ext_kick=1.800 every 30
total spikes (GPU) = 943   mean rate = 94.300 Hz
population spike raster (step: count):
  [  0:  0] [ 71:  0] [142:  1] [213:  3] [285:  3] [356:  4] [427:  5] [499:  2]
most active neurons (id:spikes:type):
  [84:10:E] [16:9:E] [29:9:E] [59:9:E] [61:9:E]
RESULT: PASS (GPU spike counts match CPU exactly; final V within 1e-09 mV)
```

## How to read it

- **total spikes / mean rate** — the population activity summary. 943 spikes over
  200 neurons in 50 ms is ~94 Hz average; this is a deliberately brisk, short demo
  so activity is clearly visible (real cortex is sparser — see THEORY §Numerics).
- **population spike raster** — the number of neurons that fired at 8 evenly-spaced
  timesteps: a down-sampled peri-stimulus time histogram (PSTH). Activity ramps as
  recurrent excitation recruits neurons, then settles under inhibition.
- **most active neurons** — the top-5 firers, ordered by (count desc, id asc); `E`
  = excitatory, `I` = inhibitory. A stable fingerprint of which cells the dynamics
  drives hardest.
- **RESULT** — the GPU-vs-CPU agreement gate.

The timing on **stderr** shows the CPU and GPU times. On this tiny network the GPU
is *slower*: with 3 kernel launches per step and only 200 neurons, the run is
launch-bound. That is expected and honest — the GPU's advantage appears at the
`10^5`–`10^6`-neuron scale point-neuron simulators are actually built for.
