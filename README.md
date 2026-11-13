# Oxygen and the spatial reach of nitric oxide: model code and simulation data

This repository holds the model, the simulation code and the simulation outputs behind every figure in

> Thomas DD. *Oxygen determines where nitric oxide signals in tissue.* (submitted)

The model couples oxygen-dependent NO synthesis by the three NOS isoforms to a measured two-route cellular NO sink, one route oxygen-dependent and one oxygen-independent. It solves the coupled O2/NO transport problem in a perivascular slab, in tissue containing discrete NOS2-expressing cells, along a vessel, in a spherical tumor and in a vascularized tumor represented as a periodic unit cell.

Scripts that only draw figures are not included. Every number plotted in the paper is in `data/`.

## Requirements

- Python 3.11 or later, with `numpy` and `scipy`.
- Wolfram Language (Mathematica or `wolframscript`) for the finite-element model of Figures 5 and 6, `scripts/vascular_unit_cell_fem.wl`.

## Layout

| folder | contents |
|---|---|
| `src/` | the model: physical constants and kinetic parameters (`parameters.py`, `constants.py`), rate laws (`kinetics.py`), oxygen fields (`o2_field.py`), source placement (`source_distributions.py`) and the numerical solvers (`solver.py`) |
| `scripts/` | the simulations; each writes its outputs to `data/` |
| `data/` | the outputs used in the paper, one folder per figure; see [`data/README.md`](data/README.md) |

Every physical constant and kinetic parameter is defined once, in `src/parameters.py` and `src/constants.py`, and every script reads its values from there. Values and sources are listed in Table 1 and Supplementary Table S1 of the paper.

## Simulations, by figure

| figure | simulation code |
|---|---|
| 1 | closed-form rate laws in `src/kinetics.py` and `src/constants.py`; the perivascular solver in `src/solver.py` |
| 2 | tissue solver `scripts/tornado_v2.py` with `scripts/fft_field.py` and `scripts/tissue.py`; ensembles `scripts/gen_fig2_*.py`; 3-D fields `scripts/export_fig2_*_for_render.py`; `scripts/analyse_fixedtissue_paired.py`; `scripts/measure_realized_nn_spacing.py` |
| 3 | `scripts/gate4_vessel_pair.py`, `scripts/primary_functional_reach.py`, `scripts/gate16_spherical_periodic_row.py` |
| 4 | `scripts/spheroid_field.py`, `scripts/spheroid_ablation.py`, `scripts/spheroid_crossover.py`, `scripts/export_fig4_cutaways.py`, `scripts/gen_fig4_panels.py`, `scripts/refine_fig4_boundaries.py` |
| 5, 6 | `scripts/vascular_unit_cell_fem.wl` (finite-element periodic unit cell), which reads the constants written by `scripts/export_fem_constants.py`; `scripts/vascular_field.py` (finite-difference cross-section) |
| 7 | `scripts/tornado_v2.py` (one-at-a-time sensitivity analysis) |

`scripts/gate1_pair_cooperation.py` tests whether two NOS2-expressing cells interact beyond the superposition of their fields; it supports the text rather than a figure.

## Running

Scripts are run from the repository root and import `src/` themselves:

```
python scripts/spheroid_field.py
python scripts/tornado_v2.py
python scripts/export_fem_constants.py
wolframscript -file scripts/vascular_unit_cell_fem.wl 150
```

The last line solves one unit cell at 150 um vessel spacing. Several scripts run checks before writing output and stop if a check fails.

## Data not included

The experimental NO measurements that the paper reanalyzes (Figure 1b, f and g, and Supplementary Figure 1) are available from the author on request.

## License and citation

MIT; see `LICENSE`. If you use this code or data, please cite the paper. The archived release has its own DOI, given in the paper's Code availability statement.
