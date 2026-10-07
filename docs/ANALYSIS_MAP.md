# Formal analysis map

The formal universe is 12 cortical and 14 limbic ROIs, including bilateral
SGC: 26 regional/MwC features and 168 cortical-to-limbic metabolic-
connectivity edges.

## MwC 01–05

| Step | Inputs | Outputs |
|---:|---|---|
| 1 | 253-session 144-node GCMI/PET bundle; 1,030-edge normative-SC mask | Recomputed 144-node and extracted 26-ROI MwC tables; calculation validation |
| 2 | MwC 01; clinical roster | Subject-level 26-ROI wide/long tables and HC-referenced deviations |
| 3 | MwC 02 | 26-ROI deviation heatmap, group vectors, spatial-replication statistics and figure |
| 4 | MwC 02–03; age, sex; treatment/session data | Baseline and longitudinal MwC statistical tables |
| 5 | MwC 03–04 | MwC raincloud panel and output audit |

## Main 01–10

| Step | Inputs | Outputs |
|---:|---|---|
| 1 | Clinical, PET, 12×14 ROI definition, 26-ROI MwC | Validated analysis objects and initialized output directories |
| 2 | Main 01 | Demographic/clinical tables and HC-reference functions |
| 3 | Main 02 | Baseline regional, connectivity, MwC, and baseline-severity-adjusted results |
| 4 | Main 03; paired TRD2 PET | Paired regional/connectivity results and HDRS-17 partial Spearman results |
| 5 | Main 04; longitudinal MwC | Group-by-time and longitudinal MwC results |
| 6 | Main 03–05 | TRD1–TRD2 spatial-replication statistics and plotting data |
| 7 | Main 03–06 | Regional/connectivity/MwC heatmaps and dynamic paired-`p < .01` longitudinal panels |
| 8 | Main 03–07 | Four cross-sectional circle diagrams and plotting workbook |
| 9 | Main 03–08 | Five global indexes, statistical workbooks, and raincloud panels |
| 10 | Main 01–09 | Four longitudinal circle diagrams, sensitivity Nilearn panels, and final audit |

See `docs/NOTEBOOK_WORKFLOW.md` for execution order and
`docs/ANALYSIS_OUTPUTS.md` for the analysis-to-output mapping.
