"""Auto-extracted pipeline step: MwC 02 — subject-level 26-ROI extraction.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 6
roi = pd.read_csv(ROI_FILE).reset_index(drop=True)
roi["node_index"] = np.arange(1, len(roi) + 1)
roi["ROI_full"] = [f"AAL3_{int(row.AAL3_ID):03d}_{row.ROI}" for row in roi.itertuples()]
assert len(roi) == N_ROIS

with AAL_VALIDATION_FILE.open(encoding="utf-8") as stream:
    aal_validation = json.load(stream)
assert aal_validation["status"] == "PASS"
assert aal_validation["checks"]["retained_nodes_144"]
assert aal_validation["checks"]["fixed_structural_edges_1030"]
assert mwc_calculation_validation["checks"]["all_144_nodes_computed_before_26_extraction"]

mwc_source = pd.read_csv(MWC_WIDE_FILE)
full_source = pd.read_csv(MWC_FULL_FILE)
assert len(mwc_source) == len(full_source) == 255
key_columns = ["subject_key", "session"]
assert mwc_source[key_columns].equals(full_source[key_columns])

mwc_columns = [f"MwC_{name}" for name in roi.ROI]
full_columns = [
    f"MwC_AAL3_{int(aal3_id):03d}_{name}"
    for aal3_id, name in zip(roi.AAL3_ID, roi.ROI)
]
assert set(mwc_columns).issubset(mwc_source)
assert set(full_columns).issubset(full_source)
all144_extraction_error = float(np.nanmax(np.abs(
    mwc_source[mwc_columns].to_numpy(float)
    - full_source[full_columns].to_numpy(float)
)))
assert all144_extraction_error < 1e-10


def _metadata_norm_key(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r'P(\d{3})', text)
    if not match:
        raise ValueError(f'Expected a deidentified P### subject key, got {text!r}')
    return f'P{int(match.group(1)):03d}'


clinical_all_mwc = pd.read_excel(CLINICAL_FILE)
clinical_all_mwc['subject_key'] = clinical_all_mwc['subject_key'].map(_metadata_norm_key)
clinical_all_mwc['age'] = pd.to_numeric(clinical_all_mwc['age'], errors='coerce')
clinical_all_mwc['sex'] = pd.to_numeric(clinical_all_mwc['sex'], errors='coerce')
clinical_all_mwc['education'] = pd.to_numeric(clinical_all_mwc['education_years'], errors='coerce')
clinical_all_mwc['hdrs_pre'] = pd.to_numeric(clinical_all_mwc['hdrs17_pre'], errors='coerce')
clinical_all_mwc['hdrs_post'] = pd.to_numeric(clinical_all_mwc['hdrs17_post'], errors='coerce')
clinical_lookup = clinical_all_mwc.set_index('subject_key')
expected_clinical_counts = {"HC": 23, "nTRD": 25, "TRD1": 35, "TRD2": 86}
assert clinical_all_mwc["analysis_cohort"].value_counts().to_dict() == expected_clinical_counts

mwc_source = mwc_source.copy()
mwc_source['subject_key'] = mwc_source['subject_key'].map(_metadata_norm_key)
mwc_source.insert(0, "session_row", np.arange(len(mwc_source), dtype=int))
clinical_session_mask = mwc_source["subject_key"].isin(clinical_lookup.index)
mwc_source = mwc_source.loc[clinical_session_mask].reset_index(drop=True)

for column in ['analysis_cohort', 'analysis_arm', 'age', 'sex', 'education', 'hdrs_pre', 'hdrs_post']:
    mapped = mwc_source['subject_key'].map(clinical_lookup[column])
    mwc_source[column] = mapped

metadata_columns = [
    "session_row", "subject_key", "session", "analysis_cohort", "analysis_arm",
    "age", "sex", "education", "hdrs_pre", "hdrs_post",
]
mwc_wide = mwc_source[metadata_columns + mwc_columns].copy()
observed_node_mwc = mwc_wide[mwc_columns].to_numpy(float)

matrix_frame = pd.DataFrame(
    observed_node_mwc, index=mwc_wide.session_row, columns=roi.node_index,
)
matrix_frame.index.name = "session_row"
mwc_long = (
    matrix_frame.reset_index()
    .melt(id_vars="session_row", var_name="node_index", value_name="MwC")
    .merge(
        roi[["node_index", "network", "AAL3_ID", "ROI", "ROI_full"]],
        on="node_index", how="left", validate="many_to_one",
    )
    .merge(mwc_wide[metadata_columns], on="session_row", how="left", validate="many_to_one")
    .sort_values(["session_row", "node_index"])
    .reset_index(drop=True)
)

ALIASES = {
    "Frontal_Sup_Medial": "dmPFC",
    "Frontal_Med_Orb": "vmPFC",
    "OFCmed": "medOFC",
}


def display_roi_name(roi_name: str) -> str:
    for source_name, alias in ALIASES.items():
        if roi_name == source_name or roi_name.startswith(source_name + "_"):
            return roi_name.replace(source_name, alias, 1)
    return roi_name


mwc_long["display_roi"] = mwc_long.ROI.map(display_roi_name)
mwc_wide.to_csv(OUTPUT_DIR / "subject_level_corticolimbic26_normativeSC10_MwC_wide.csv", index=False)
mwc_long.to_csv(OUTPUT_DIR / "subject_level_corticolimbic26_normativeSC10_MwC_long.csv", index=False)

# HC-referenced deviations are regenerated from the normative-SC-constrained MwC values.
baseline_hc = mwc_wide.loc[
    mwc_wide.session.eq("ses-01") & mwc_wide.analysis_cohort.eq("HC")
].copy()
assert len(baseline_hc) == 23
hc_mean = baseline_hc[mwc_columns].mean()
hc_sd = baseline_hc[mwc_columns].std(ddof=1)
assert np.isfinite(hc_sd).all() and (hc_sd > 0).all()
baseline_groups = mwc_wide.loc[
    mwc_wide.session.eq("ses-01")
    & mwc_wide.analysis_cohort.isin(["nTRD", "TRD1"])
].copy()
assert baseline_groups.groupby("analysis_cohort").size().to_dict() == {"TRD1": 35, "nTRD": 25}
deviation = baseline_groups[[
    "session_row", "subject_key", "session", "analysis_cohort", "analysis_arm",
    "age", "sex", "education",
]].copy()
for source_column, roi_name in zip(mwc_columns, roi.ROI):
    deviation[f"MwC_deviation_{roi_name}"] = (
        baseline_groups[source_column].to_numpy(float) - hc_mean[source_column]
    ) / hc_sd[source_column]
deviation["display_group"] = deviation.analysis_cohort.map({"nTRD": "nTRD", "TRD1": "TRD"})
deviation.to_csv(
    OUTPUT_DIR / "subject_level_nTRD_TRD1_corticolimbic26_normativeSC10_MwC_HC_reference_deviation.csv",
    index=False,
)

display(pd.DataFrame({
    "quantity": [
        "upstream paired sessions", "clinical sessions", "ROIs per session",
        "long-table rows", "all-144 to 26 extraction maximum absolute error",
    ],
    "value": [255, len(mwc_wide), N_ROIS, len(mwc_long), all144_extraction_error],
}))
display(roi[["node_index", "network", "AAL3_ID", "ROI"]])
