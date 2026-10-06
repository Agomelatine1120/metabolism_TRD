"""Auto-extracted pipeline step: Main 03 — baseline discovery and HDRS-17 sensitivity.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 15
TRD1_OUT = OUTPUT_ROOT / "01_TRD1_discovery_agesex"
trd1_result = run_trd1_vs_ntrd(
    reference_covariates=["age", "sex"],
    glm_covariates=["age", "sex"],
    output_dir=TRD1_OUT,
)
trd1_counts = {
    "edge_uncorrected_p_lt_0p01": len(trd1_result["edge_reported"]),
    "regional_uncorrected_p_lt_0p01": len(trd1_result["regional_reported"]),
}
trd1_counts


# Source cell 17
SENSITIVITY_OUT = OUTPUT_ROOT / "08_TRD1_vs_nTRD_HDRS_sensitivity"
SENSITIVITY_OUT.mkdir(parents=True, exist_ok=True)
MWC_DEVIATION_FILE = MWC_DEVIATION_INPUT_FILE


def run_hdrs_sensitivity(trd1_context):
    case = trd1_context["case_rows"]
    ntrd = trd1_context["ntrd"]
    keys = list(case["subject_key"]) + list(ntrd["subject_key"])
    group = np.array([1] * len(case) + [0] * len(ntrd), dtype=float)
    covariates = clinical.set_index("subject_key")[[
        "age", "sex", "hdrs_pre"
    ]]

    def analyze(values, labels, modality):
        rows = []
        for feature_index, label in enumerate(labels):
            frame = pd.DataFrame({
                "subject_key": keys,
                "group": group,
                "y": values[:, feature_index],
            }).join(covariates, on="subject_key")
            primary = fit_group_glm(frame, "y", ["age", "sex"])
            adjusted = fit_group_glm(
                frame, "y", ["age", "sex", "hdrs_pre"]
            )
            row = {
                "modality": modality,
                "TRD1_deviation": float(frame.loc[frame["group"] == 1, "y"].mean()),
                "nTRD_deviation": float(frame.loc[frame["group"] == 0, "y"].mean()),
                "primary_t_group": primary["t"],
                "primary_effect_size_partial_r": primary["partial_r"],
                "primary_p_uncorrected": primary["p"],
                "primary_n": primary["n"],
                "HDRS_adjusted_t_group": adjusted["t"],
                "HDRS_adjusted_effect_size_partial_r": adjusted["partial_r"],
                "HDRS_adjusted_p_uncorrected": adjusted["p"],
                "HDRS_adjusted_n": adjusted["n"],
            }
            if modality == "edge":
                row.update({"ROI_1": label[0], "ROI_2": label[1]})
            else:
                row.update({"ROI": label})
            rows.append(row)
        result = pd.DataFrame(rows)
        result["primary_q_fdr_BH_audit_only"] = bh_fdr(result["primary_p_uncorrected"])
        result["HDRS_adjusted_q_fdr_BH_audit_only"] = bh_fdr(
            result["HDRS_adjusted_p_uncorrected"]
        )
        result["primary_p_lt_0p01"] = result["primary_p_uncorrected"] < GROUP_P_THRESHOLD
        result["HDRS_adjusted_p_lt_0p01"] = (
            result["HDRS_adjusted_p_uncorrected"] < GROUP_P_THRESHOLD
        )
        result["selection_change"] = np.select(
            [
                result["primary_p_lt_0p01"] & result["HDRS_adjusted_p_lt_0p01"],
                result["primary_p_lt_0p01"] & ~result["HDRS_adjusted_p_lt_0p01"],
                ~result["primary_p_lt_0p01"] & result["HDRS_adjusted_p_lt_0p01"],
            ],
            ["retained", "lost_after_HDRS_adjustment", "new_after_HDRS_adjustment"],
            default="not_selected",
        )
        return result.sort_values("HDRS_adjusted_p_uncorrected").reset_index(drop=True)

    edge_values = np.vstack([trd1_context["case_edge"], trd1_context["ntrd_edge"]])
    regional_values = np.vstack([
        trd1_context["case_regional"].to_numpy(float),
        trd1_context["ntrd_regional"].to_numpy(float),
    ])
    mwc_source = pd.read_csv(MWC_DEVIATION_FILE)
    mwc_source["subject_key"] = mwc_source["subject_key"].map(norm_key)
    mwc_source = mwc_source.loc[
        mwc_source["session"].eq("ses-01")
        & mwc_source["analysis_cohort"].isin(["nTRD", "TRD1"])
    ].copy()
    assert not mwc_source["subject_key"].duplicated().any()
    assert mwc_source["analysis_cohort"].value_counts().to_dict() == {"TRD1": 35, "nTRD": 25}
    mwc_feature_columns = [c for c in mwc_source if c.startswith("MwC_deviation_")]
    assert len(mwc_feature_columns) == 26
    mwc_aligned = mwc_source.set_index("subject_key").loc[keys]
    assert np.array_equal(
        mwc_aligned["analysis_cohort"].to_numpy(),
        np.where(group == 1, "TRD1", "nTRD"),
    )
    mwc_values = mwc_aligned[mwc_feature_columns].to_numpy(float)
    mwc_labels = [column.removeprefix("MwC_deviation_") for column in mwc_feature_columns]

    return (
        analyze(edge_values, CROSS_EDGE_NAMES, "edge"),
        analyze(regional_values, ROI_COLUMNS, "regional"),
        analyze(mwc_values, mwc_labels, "MwC"),
    )


(sensitivity_edge_all, sensitivity_regional_all, sensitivity_mwc_all) = run_hdrs_sensitivity(trd1_result)
sensitivity_edge_p001 = sensitivity_edge_all[sensitivity_edge_all["HDRS_adjusted_p_lt_0p01"]].copy()
sensitivity_regional_p001 = sensitivity_regional_all[sensitivity_regional_all["HDRS_adjusted_p_lt_0p01"]].copy()
sensitivity_mwc_p001 = sensitivity_mwc_all[sensitivity_mwc_all["HDRS_adjusted_p_lt_0p01"]].copy()


def sensitivity_summary_row(modality, result_all, result_p001):
    return {
        "modality": modality,
        "tested_n": len(result_all),
        "primary_p_lt_0p01": int(result_all["primary_p_lt_0p01"].sum()),
        "HDRS_adjusted_p_lt_0p01": len(result_p001),
        "retained": int((result_all["selection_change"] == "retained").sum()),
        "lost": int((result_all["selection_change"] == "lost_after_HDRS_adjustment").sum()),
        "new": int((result_all["selection_change"] == "new_after_HDRS_adjustment").sum()),
    }


sensitivity_summary = pd.DataFrame([
    sensitivity_summary_row("edge", sensitivity_edge_all, sensitivity_edge_p001),
    sensitivity_summary_row("regional", sensitivity_regional_all, sensitivity_regional_p001),
    sensitivity_summary_row("MwC", sensitivity_mwc_all, sensitivity_mwc_p001),
])

sensitivity_file = SENSITIVITY_OUT / "TRD1_vs_nTRD_baseline_HDRS_sensitivity.xlsx"
with pd.ExcelWriter(sensitivity_file, engine="openpyxl") as writer:
    sensitivity_summary.to_excel(writer, sheet_name="summary", index=False)
    sensitivity_edge_all.to_excel(writer, sheet_name="edge_all", index=False)
    sensitivity_edge_p001.to_excel(writer, sheet_name="edge_HDRS_adjusted_p001", index=False)
    sensitivity_regional_all.to_excel(writer, sheet_name="regional_all", index=False)
    sensitivity_regional_p001.to_excel(writer, sheet_name="region_HDRS_adjusted_p001", index=False)
    sensitivity_mwc_all.to_excel(writer, sheet_name="MwC_all", index=False)
    sensitivity_mwc_p001.to_excel(writer, sheet_name="MwC_HDRS_adjusted_p001", index=False)
style_excel_workbook(sensitivity_file)
display(sensitivity_summary)


# Source cell 19
TRD2_OUT = OUTPUT_ROOT / "02_TRD2_deviation_prepost_agesex"
trd2_context = build_deviation_context(
    case_cohort="TRD2",
    reference_covariates=["age", "sex"],
    output_dir=TRD2_OUT,
)
print("TRD2 subject-level deviations:", trd2_context["case_edge"].shape, trd2_context["case_regional"].shape)
