# Data dictionary

## Clinical workbook

`inputs/core/clinical_data.xlsx` contains one worksheet and one row per
analyzed participant.

| Field | Meaning |
|---|---|
| `subject_key` | Cross-file public key `P001`–`P169` |
| `analysis_cohort` | `HC`, `nTRD`, `TRD1`, or `TRD2` |
| `analysis_arm` | `iTBS`, `rTMS`, `sham`, or blank when not applicable |
| `sex` | Encoded sex covariate used by the analysis |
| `age` | Age in years |
| `education_years` | Years of education |
| `hdrs17_pre`, `hdrs17_post` | HDRS-17 baseline and post-treatment scores |
| `tiv` | Total intracranial volume |
| `msm_t`, `msm_total` | MSM measures used in the reported analyses |

## Locked analysis population

`inputs/core/analysis_population_roster.csv` contains exactly the same 169
public keys as the clinical workbook, their analysis cohort/arm, and four
binary session flags: `pet_ses_01`, `pet_ses_02`, `mwc_ses_01`, and
`mwc_ses_02`. These flags define the exact permitted subject/session records in
each bundled subject-level input.

## PET workbook

`inputs/core/AAL3_PET_wbSUVR_Table.xlsx` contains 144 retained AAL3 rows.
Subject/session columns follow
`swsub_P###_ses-##_run_01_trc_fdg_ac_pet`. The retained/excluded lists and the
PET-grid atlas are stored beside the workbook.

## Formal ROI definition

`inputs/core/corticolimbic_12x14_ROI_definition.csv` defines 12 cortical and 14
limbic targets. Their Cartesian product defines the 168 metabolic-connectivity
edges. The same 26 targets are used for regional SUVR and MwC reporting.

## MwC arrays

`session_gcmi_pet_AAL3_144.npz` stores, for each analyzed session, the public
subject key, session, 144-node PET vector, and 144×144 GCMI matrix needed for
MwC calculation. `normative_SC_top10_mask.npz` stores the fixed binary
structural mask. MwC is calculated in
`scripts/03_compute_normative_sc_mwc.py`.
