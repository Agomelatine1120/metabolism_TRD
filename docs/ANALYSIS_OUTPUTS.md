# Analysis-to-output map

This file links each analysis stage to its principal generated tables, plotting
data, and figure components. Paths are under `outputs/main_analyses/` unless
they begin with `outputs/mwc/`.

## Analysis scope

- The formal analysis space contains 12 cortical and 14 limbic ROIs, including
  bilateral SGC: 26 regional/MwC features and 168 cortical-to-limbic edges.
- Baseline analyses include regional SUVR, metabolic connectivity, MwC, five
  global indices, cross-cohort spatial comparisons, and baseline-HDRS-17
  sensitivity models.
- Longitudinal analyses include paired rTMS, iTBS, and sham tests;
  active-versus-sham group-by-time models; and HDRS-17 associations.
- The workflow generates individual analytical figure components. Final
  multi-panel placement, sizing, lettering, and manually positioned anatomical
  arrows are separate layout operations.

## Main analytical outputs

| Analysis | Step | Principal generated files |
|---|---|---|
| Regional SUVR deviations and group models | Main 03, 06, 07 | `01_TRD1_discovery_agesex/regional_all.xlsx`; `04_figures/nTRD_TRD1_TRD2_regional26_strip_heatmap.*`; `04_figures/spatial_correlation_scatter.*` |
| Metabolic-connectivity deviations and group models | Main 03, 06, 07 | `01_TRD1_discovery_agesex/edge_all.xlsx`; `04_figures/nTRD_TRD1_TRD2_corticolimbic12x14_heatmap.*`; `04_figures/spatial_correlation_scatter.*` |
| MwC deviations and group models | MwC 02–05; Main 06–07 | `outputs/mwc/figures/Figure_nTRD_TRD1_TRD2_normativeSC_MwC_26ROI_deviation_heatmap.*`; `Figure_selected_unilateral_MwC_deviation_raincloud.*`; `Figure_TRD1_TRD2_normativeSC_MwC_26ROI_spatial_correlation.*` |
| Cross-sectional circle diagrams | Main 08 | `04_figures/corticolimbic_circular_summary.*`; `corticolimbic_circular_plot_data.xlsx` |
| Longitudinal circle diagrams | Main 10 | `04_figures/corticolimbic_circular_longitudinal_summary.*`; `corticolimbic_circular_longitudinal_effects_data.xlsx` |
| Five global metabolic indices | Main 09 | `10_metabolic_signature_five_metrics/Figure_five_metric_nTRD_TRD_raincloud.*`; `Figure_five_metric_active_pre_post_raincloud.*`; associated workbooks |
| Paired treatment changes and HDRS-17 correlations | Main 04, 07 | `04_figures/TRD2_prepost_raincloud.*`; `TRD2_prepost_HDRS_correlation_panel_green_reference.*`; plotting workbooks |
| Active-versus-sham group-by-time effects | Main 05 | `05_active_vs_sham_interaction/active_treatment_vs_sham_group_by_time.xlsx` |
| Baseline-HDRS-17-adjusted analyses | Main 03, 07, 10 | `08_TRD1_vs_nTRD_HDRS_sensitivity/TRD1_vs_nTRD_baseline_HDRS_sensitivity.xlsx`; `04_figures/nTRD_TRD_HDRS_sensitivity_top10_raincloud.*`; adjusted circle outputs |
| Sensitivity circuit glass brains | Main 10 | `13_sensitivity_mPFC_striatopallidal_glass_brain/figures/` and source-value tables |
| Regional-SUVR glass brain | Main 10 | `15_nonTRD_TRD_significant_regional_SUVR_glass_brain/figures/` and source-value tables |
| MwC glass brain selected by the configured threshold | Main 10 | `11_SGC_26ROI_figures/figures/Figure_TRD_vs_nonTRD_significant_MwC_nilearn_glass_brain.*` |

## Dynamic longitudinal plotting rule

Main 07 does not use a hard-coded ROI or edge list. It reads the complete paired
test tables, selects every active-arm metabolic-connectivity edge meeting
paired-sample `p < .01`, and orders the selected edges by treatment arm and the
formal cortical/limbic anatomical order.

The exact selected order and source values are stored in
`outputs/main_analyses/04_figures/TRD2_prepost_raincloud_data.xlsx`, sheet
`panel_order`. The validation step compares this sheet directly with the
complete all-edge paired-test outputs.

## Locked input-population audit

The input package is population-restricted before analysis. The clinical input
exactly equals the 169-participant roster; PET and full GCMI/MwC inputs exactly
equal their 255 permitted subject-sessions. The specialized nTRD/TRD1 MwC
input exactly equals its prespecified 60-participant baseline subset. These are equality checks rather
than subset checks against larger source files.

The machine-readable audit is `manifests/validation.json`. It verifies
the locked populations, expected statistical tables, spatial statistics,
global-index values, dynamically selected longitudinal panels, heatmap/circle
source values, sensitivity values, and existence plus nontrivial dimensions of
the required analytical figures.

## Statistical table sources

| Table content | Reporting rule | Generated source |
|---|---|---|
| Cohort characteristics | All configured variables | Main 02: `outputs/main_analyses/demographic_and_clinical_summary.xlsx` |
| Treatment-group characteristics and HDRS-17 reduction | PET-paired samples | Main 02: `outputs/main_analyses/demographic_and_clinical_summary.xlsx` |
| Regional SUVR | Uncorrected `p < .01` | Main 03: `01_TRD1_discovery_agesex/regional_uncorrected_p_lt_0p01.xlsx` |
| Metabolic connectivity | Uncorrected `p < .01` | Main 03: `01_TRD1_discovery_agesex/edge_uncorrected_p_lt_0p01.xlsx` |
| MwC | Primary `p < .01`; complete values retained in the full table | MwC 04–05 and Main 05 |
| Paired and group-by-time analyses | Configured primary and exploratory thresholds | Main 04–05 |
| Baseline-severity-adjusted analyses | Primary `p < .01`; exploratory `p < .05` | Main 03: `08_TRD1_vs_nTRD_HDRS_sensitivity/TRD1_vs_nTRD_baseline_HDRS_sensitivity.xlsx` |

The repository also retains full unthresholded regional, edge, MwC,
interaction, and clinical-correlation tables; plotting workbooks; spatial-null
archives; and validation/checksum files.
