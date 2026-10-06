# Bundled inputs

All subject-level inputs are minimized, pseudonymized, and restricted to the
participants in the locked analysis population. They are not broader source
exports that are filtered only after loading. The private identifier crosswalk
is not included.

## `core/`

- `clinical_data.xlsx`: 187 analyzed participants and only the 11 fields used
  by the workflow: public key, cohort, treatment arm, sex, age, education,
  pre/post HDRS-17, TIV, MSM-T, and MSM-total.
- `AAL3_PET_wbSUVR_Table.xlsx`: 144 retained AAL3 regions by 291 analyzed
  public participant-sessions.
- `retained_AAL3_144_ROIs.csv`: retained AAL3 index and label list.
- `excluded_AAL3_ROIs.csv`: 22 excluded atlas regions with voxel counts on the
  PET grid.
- `AAL3_144_space-PET_res-2_dseg.nii.gz`: the bundled 144-region PET-grid atlas
  used by the Nilearn figures.
- `corticolimbic_12x14_ROI_definition.csv`: the formal 12 cortical and 14
  limbic targets, including bilateral SGC.
- `analysis_population_roster.csv`: the locked 187-participant roster and the
  exact PET/MwC session-presence flags against which every subject-level input
  is validated.

## `mwc/normative_sc/`

- `session_gcmi_pet_AAL3_144.npz`: 291 public subject-sessions containing the
  144-node GCMI/PET arrays required to calculate MwC. The GCMI matrices were
  derived from corrected BOLD signals without individual grey-matter masking.
- `normative_SC_top10_mask.npz`: fixed 144-node normative structural-
  connectivity mask with 1,030 undirected edges.
- `subject_level_AAL3_144_normativeSC10_MwC_wide.csv.gz`: independently stored
  144-node MwC reference values used for calculation verification; metadata are
  limited to `subject_key` and `session`.
- `subject_level_corticolimbic26_normativeSC10_MwC_wide.csv`: corresponding
  formal 26-target reference values, again with only subject/session metadata.
- `validation_report.json`: dimensional and calculation checks.

All identifiers follow `P###`. `clinical_data.xlsx` exactly equals the locked
187-participant population. PET and full MwC session inputs exactly equal the
session flags in `analysis_population_roster.csv`; the nTRD/TRD1-only MwC file
exactly equals that prespecified baseline subset. The reproduction validator
fails if any bundled input contains an unlisted participant or session.
