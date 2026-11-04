# Simulation data

One folder per figure. CSV files are plain text with a header row; lines starting with `#` are notes. JSON files hold one simulation result or the metadata of a 3-D field. A `.f32` file is a 3-D NO concentration field in nM, stored as little-endian float32 in C order; its `.json` gives the grid shape, bounds and spacing.

Scripts write their outputs to `data/`; the copies here are the ones used in the paper.

## Figure 1

| file | panel | content | produced by |
|---|---|---|---|
| `fig1_panelE_RAW.csv` | a | NO diffusion length against oxygen | closed form, from the parameters in `src/` |
| `fig1_panelG6_normalised.csv` | c | normalised steady-state NO against oxygen, three NOS isoforms | closed form, from the parameters in `src/` |
| `fig2_panelB_RAW.csv` | e | share of NO removal carried by each route against oxygen | closed form, from the parameters in `src/` |
| `fig1_panelD6_ORIGIN.csv` | h | reach of oxygen and of NO, four bars | closed form, from the parameters in `src/` |
| `fig1_panelF3_ORIGIN.csv` | i | NO against distance from the vessel, one curve per oxygen tension | closed form, from the parameters in `src/` |

## Figure 2

| file | panel | content | produced by |
|---|---|---|---|
| `oxygen_seed7_220uM.json` | a-f | 3-D NO field at 220 uM O2 (metadata) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_220uM.f32` | a-f | 3-D NO field at 220 uM O2 (float32 volume, layout in the .json) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_40uM.json` | a-f | 3-D NO field at 40 uM O2 (metadata) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_40uM.f32` | a-f | 3-D NO field at 40 uM O2 (float32 volume, layout in the .json) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_20uM.json` | a-f | 3-D NO field at 20 uM O2 (metadata) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_20uM.f32` | a-f | 3-D NO field at 20 uM O2 (float32 volume, layout in the .json) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_10uM.json` | a-f | 3-D NO field at 10 uM O2 (metadata) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_10uM.f32` | a-f | 3-D NO field at 10 uM O2 (float32 volume, layout in the .json) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_2uM.json` | a-f | 3-D NO field at 2 uM O2 (metadata) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_2uM.f32` | a-f | 3-D NO field at 2 uM O2 (float32 volume, layout in the .json) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_1uM.json` | a-f | 3-D NO field at 1 uM O2 (metadata) | scripts/export_fig2_fields_for_render.py |
| `oxygen_seed7_1uM.f32` | a-f | 3-D NO field at 1 uM O2 (float32 volume, layout in the .json) | scripts/export_fig2_fields_for_render.py |
| `fig2_oxygen_ensemble.csv` | g | per-realization coverage and volume against oxygen | scripts/gen_fig2_oxygen_ensemble.py |
| `fig2_panelD_oxygen_v2.csv` | g | volume above 1 nM against oxygen, mean and s.d. | scripts/gen_fig2_dose_and_absolute_volume.py |
| `fig2_panelD_criterion_v2.csv` | g | coverage of the focus above 0.5-10 nM against oxygen | scripts/gen_fig2_criterion_dependence.py |
| `fig2_panelI_bridging.csv` | h | nearest-neighbour gaps bridged only by the population | scripts/gen_fig2_* family |
| `fixedtissue_2pct.json` | i-k | 3-D NO field, 2% NOS2+ in the fixed tissue (metadata) | scripts/export_fig2_fixedtissue_for_render.py |
| `fixedtissue_2pct.f32` | i-k | 3-D NO field, 2% NOS2+ in the fixed tissue (float32 volume, layout in the .json) | scripts/export_fig2_fixedtissue_for_render.py |
| `fixedtissue_15pct.json` | i-k | 3-D NO field, 15% NOS2+ in the fixed tissue (metadata) | scripts/export_fig2_fixedtissue_for_render.py |
| `fixedtissue_15pct.f32` | i-k | 3-D NO field, 15% NOS2+ in the fixed tissue (float32 volume, layout in the .json) | scripts/export_fig2_fixedtissue_for_render.py |
| `fixedtissue_40pct.json` | i-k | 3-D NO field, 40% NOS2+ in the fixed tissue (metadata) | scripts/export_fig2_fixedtissue_for_render.py |
| `fixedtissue_40pct.f32` | i-k | 3-D NO field, 40% NOS2+ in the fixed tissue (float32 volume, layout in the .json) | scripts/export_fig2_fixedtissue_for_render.py |

## Figure 3

| file | panel | content | produced by |
|---|---|---|---|
| `fig3A_profiles.csv` | a | NO and O2 against distance from the vessel, three source depths | scripts/gate4_vessel_pair.py |
| `primary_functional_reach.csv` | b, c | signalling extent and inward/outward reach against source depth | scripts/primary_functional_reach.py |
| `gate16_refined_5um.csv` | d | widest sustaining spacing s_max_parallel against distance from the vessel | scripts/gate16_spherical_periodic_row.py |

## Figure 4

| file | panel | content | produced by |
|---|---|---|---|
| `fig4_cutaway_profiles.csv` | b | radial O2 and NO profiles, every sphere in the cutaway matrix | scripts/export_fig4_cutaways.py |
| `fig4_cutaway_markers.csv` | b | radius of maximal NO and contour radii, every sphere | scripts/export_fig4_cutaways.py |
| `fig4_D_ablation_profiles.csv` | a, d | radial profiles, full model and each coupling removed | scripts/spheroid_ablation.py |
| `fig4_D_ablation_markers.csv` | a, d | central NO, anoxic radius and sGC front, each ablation | scripts/spheroid_ablation.py |
| `fig4_B_delta.csv` | c | delta_d against NOS2+ fraction | scripts/gen_fig4_panels.py, scripts/refine_fig4_boundaries.py |

## Figure 5 and 6

| file | panel | content | produced by |
|---|---|---|---|
| `Fig5_surface_data.csv` | Fig 6b | mean tissue NO over spacing x NOS2+ fraction | summary of unit_cell_results |
| `Fig5_panelE_profiles.csv` | Fig 6c | NO averaged around the vessel against distance from the wall | summary of unit_cell_fields |
| `unit_cell_results/` (98 files) | Figures 5, 6 | unit-cell finite-element results, one file per vessel spacing x NOS2+ fraction x NOS3 state (`cell_<spacing>_f<fraction>_*.json`) | `scripts/vascular_unit_cell_fem.wl` |
| `unit_cell_fields/` (27 files) | Figures 5, 6 | unit-cell finite-element NO and O2 fields on a grid (`cellfield_<spacing>_f<fraction>.json`) | `scripts/vascular_unit_cell_fem.wl` |

## Figure 7

| file | panel | content | produced by |
|---|---|---|---|
| `tornado_v2_results.csv` | a-e | one-at-a-time sensitivity sweep, every output x parameter endpoint | scripts/tornado_v2.py |
| `Fig7_panelA_delivery_optimum_20260928.csv` | a | plotted bars for panel a | reshaped from tornado_v2_results.csv |
| `Fig7_panelB_reach_at_anoxia_20260928.csv` | b | plotted bars for panel b | reshaped from tornado_v2_results.csv |
| `Fig7_panelC_wall_no_20260928.csv` | c | plotted bars for panel c | reshaped from tornado_v2_results.csv |
| `Fig7_panelD_steady_state_no_20260928.csv` | d | plotted bars for panel d | reshaped from tornado_v2_results.csv |
| `Fig7_panelE_effective_mitochondrial_km_20260928.csv` | e | plotted bars for panel e | reshaped from tornado_v2_results.csv |
