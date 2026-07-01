# Data — 6.6 Neuronal Network Simulation (Biophysical)

## Committed sample (`sample/network.txt`) — SYNTHETIC

The demo runs on a **tiny, fully synthetic** network configuration so it works
offline with zero downloads. It is **not** derived from any real recording or
morphology; it is generated deterministically by `scripts/make_synthetic.py`.

### Format

A single whitespace-separated line of 10 numbers:

```
ncell ncomp dt steps v_rest n_stim i_stim gAxial wSyn tauSyn
```

| field    | meaning                                                        | units      |
|----------|----------------------------------------------------------------|------------|
| `ncell`  | number of neurons in the ring                                  | count      |
| `ncomp`  | compartments per neuron (soma + dendrites); must be ≤ 8        | count      |
| `dt`     | integration timestep                                           | ms         |
| `steps`  | number of timesteps (run length = `steps*dt` ms)              | count      |
| `v_rest` | resting membrane voltage all compartments start at            | mV         |
| `n_stim` | number of leading cells given a startup depolarisation        | count      |
| `i_stim` | startup depolarisation added to those cells' soma at t=0      | mV         |
| `gAxial` | inter-compartment (axial) coupling conductance                 | mS/cm²     |
| `wSyn`   | excitatory synaptic conductance added per presynaptic spike    | mS/cm²     |
| `tauSyn` | synaptic conductance decay time constant                       | ms         |

The remaining Hodgkin–Huxley constants (gNa, gK, gL, reversal potentials,
capacitance, synaptic reversal, threshold) use their standard textbook values,
defined and commented in `src/neuron.h` (`HHParams`).

### The committed values

```
16 4 0.025 4000 -65 1 45 0.3 0.9 2
```

16 neurons, each a 4-compartment cable, integrated for 4000 steps of 0.025 ms
(= 100 ms of simulated activity). One leading cell is kicked with +45 mV, which
makes it fire; its spike excites its ring successor one synaptic delay later, and
the activity propagates all the way around the ring as a **travelling wave** —
an interpretable, verifiable result (each cell's first-spike step increases by a
roughly constant hop delay).

### Regenerate / resize

```bash
python scripts/make_synthetic.py                 # rewrites the committed sample
python scripts/make_synthetic.py --ncell 256 --steps 8000   # a bigger ring
```

## Real-world data (optional, not committed)

The catalog points at several real neuroscience resources. None is required for
the demo, none is redistributed here, and each carries its own license — respect
it and cite the original authors.

| Source | What it is | Link |
|--------|-----------|------|
| **NeuroMorpho.Org** | 200,000+ 3D neuronal reconstructions (SWC) across 900+ species | https://neuromorpho.org |
| **ModelDB** | Curated computational neuron models (NEURON/GENESIS files) | https://modeldb.science |
| **Allen Brain Cell Atlas** | Patch-seq morpho-electric single-cell data | https://portal.brain-map.org |
| **DANDI Archive** | Neurophysiology datasets in NWB format | https://dandiarchive.org |

`scripts/download_data.ps1` / `.sh` print these pointers and never bypass any
registration. Converting a real SWC morphology into this model's compartment
chain (parse the tree, collapse branches into compartments, order them for the
Hines solver) is left as an exercise — see the README.

> **Not for clinical use.** Educational material only. The synthetic network does
> not model any specific brain region, species, or patient.
