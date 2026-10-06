# Notebook workflow

The canonical source is stored in `pipeline_steps/*.py`. Each chapter notebook
loads its required earlier steps into the same IPython namespace and then runs
its own visible step. This allows either stepwise review or a complete clean
run.

## MwC 01–05

| Step | Notebook | Purpose |
|---:|---|---|
| 1 | `01_mwc_01_setup_recompute.ipynb` | Validate the 144-node GCMI/PET bundle and fixed 1,030-edge structural mask; recompute MwC. |
| 2 | `02_mwc_02_subject_level.ipynb` | Extract the formal 12-cortical + 14-limbic subject-level MwC values and HC-referenced deviations. |
| 3 | `03_mwc_03_deviation_spatial.ipynb` | Create the 26-ROI MwC deviation heatmap and TRD1–TRD2 spatial-replication result. |
| 4 | `04_mwc_04_group_analysis.ipynb` | Run the nTRD-versus-TRD1 MwC analysis and longitudinal MwC statistics. |
| 5 | `05_mwc_05_figures_audit.ipynb` | Create the MwC figure components and verify the five-step MwC outputs. |

## Main 01–10

| Step | Notebook | Purpose |
|---:|---|---|
| 1 | `01_main_01_setup_load.ipynb` | Load and validate the 26 ROIs, 168 cross-network edges, clinical data, PET data, and MwC data. |
| 2 | `02_main_02_demographics_hc_reference.ipynb` | Reproduce cohort/treatment characteristics and construct HC references. |
| 3 | `03_main_03_baseline_discovery_sensitivity.ipynb` | Run baseline nTRD-versus-TRD1 regional, connectivity, MwC, and baseline-severity-adjusted analyses. |
| 4 | `04_main_04_longitudinal_paired_clinical.ipynb` | Run iTBS, rTMS, and sham paired analyses and HDRS-17 correlations. |
| 5 | `05_main_05_interactions_mwc.ipynb` | Run active-versus-sham group-by-time models and formal longitudinal MwC analyses. |
| 6 | `06_main_06_spatial_replication_report.ipynb` | Reproduce TRD1–TRD2 regional, connectivity, and MwC spatial-replication statistics. |
| 7 | `07_main_07_core_figures_heatmaps.ipynb` | Create heatmaps, rainclouds, and clinical-correlation panels. Longitudinal panels are selected dynamically from active-arm edges with paired `p < .01`. |
| 8 | `08_main_08_circle_diagrams_checks.ipynb` | Create the four cross-sectional 26-node/168-edge circle diagrams. |
| 9 | `09_main_09_five_global_indexes.ipynb` | Compute and plot the five global metabolic indexes. |
| 10 | `10_main_10_longitudinal_sgc_figures_audit.ipynb` | Create longitudinal circle diagrams, sensitivity glass brains, and the final workflow audit. |

## Execution

Run `notebooks/00_run_all_steps.ipynb` for MwC 01–05 followed by Main 01–10.
All 15 chapter notebooks and the run-all notebook are stored with outputs for
direct GitHub rendering. Every notebook ends with a **Results preview** section
containing its key tables and, where applicable, inline figure previews. The
full-resolution files remain under `outputs/`.

`scripts/94_execute_notebook_result_previews.py` regenerates the inline Results
previews after the analysis outputs have been created.
