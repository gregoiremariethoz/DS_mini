# DS_mini

A minimal, self-contained Python implementation of **Direct Sampling (DS)** — a multiple-point geostatistical simulation algorithm.

> This demo is a Python port of the schematic MATLAB version described on the  
> [Gaia Lab website](https://wp.unil.ch/gaia/mps/ds-matlab/) (UNIL).

---

## What is Direct Sampling?

Direct Sampling reproduces complex spatial patterns by scanning a **Training Image (TI)** — a conceptual model of the expected spatial variability — and copying values to a simulation grid node by node. At each node, it looks for a TI location whose neighbourhood best matches what has already been simulated nearby.

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `template_half` | Half-size of the square search template (side = 2h+1) |
| `max_fraction` | Max fraction of TI to scan per node (controls speed vs quality) |
| `threshold` | Distance below which a match is accepted early (0 = exact) |

## Usage

```bash
# Install dependency (only numpy and matplotlib needed)
pip install numpy matplotlib

# Run the demo
python ds_mini.py
```

This generates a binary training image with random circular blobs, runs DS, and saves `ds_result.png`.

## Example output

| Training Image | DS Simulation |
|:--------------:|:-------------:|
| Random circular blobs | Reproduced patterns |

The simulation reproduces the shape, size, and connectivity of the circular structures in the TI without any explicit model — the TI is the model.

## Import as a module

```python
from ds_mini import generate_ti, ds_simulate, plot_results

ti  = generate_ti(size=(100, 100), n_objects=50, seed=42)
sim = ds_simulate(ti, sim_size=(80, 80), threshold=0.05)
plot_results(ti, sim)
```

## Algorithm steps

1. Visit simulation nodes in a random order
2. At each node, extract the **data event**: already-simulated values in the template neighbourhood
3. Scan TI locations at random; compute the **mismatch fraction** between TI and data event
4. Accept the TI value if mismatch ≤ threshold, otherwise keep scanning up to `max_fraction × |TI|`
5. Copy the best-matching TI central value to the simulation node

## References

- Mariethoz, G., Renard, P., Straubhaar, J. (2010). *The Direct Sampling method to perform multiple-point geostatistical simulations.* Water Resources Research, 46(11). https://doi.org/10.1029/2008WR007621
- Full C implementation: [DS_newera](https://github.com/gregoiremariethoz/DS_newera)
- Gaia Lab, University of Lausanne: https://wp.unil.ch/gaia/
