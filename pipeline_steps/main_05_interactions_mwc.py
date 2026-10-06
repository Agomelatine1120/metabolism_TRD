"""Auto-extracted pipeline step: Main 05 — group-by-time interactions and formal MwC.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 23
from statsmodels.stats.diagnostic import het_breuschpagan

INTERACTION_OUT = OUTPUT_ROOT / "05_active_vs_sham_interaction"
INTERACTION_OUT.mkdir(parents=True, exist_ok=True)


def paired_stats(pre, post):
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    mask = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[mask], post[mask]
    delta = post - pre
    test = stats.ttest_rel(post, pre)
    low, high = paired_mean_ci95(delta[:, None])
    return {
        "n": len(delta),
        "pre_mean": float(pre.mean()),
        "post_mean": float(post.mean()),
        "post_minus_pre": float(delta.mean()),
        "post_minus_pre_CI95_low": float(low[0]),
        "post_minus_pre_CI95_high": float(high[0]),
        "paired_t": float(test.statistic),
        "paired_p_uncorrected": float(test.pvalue),
        "Cohen_dz": float(delta.mean() / delta.std(ddof=1)),
    }


def active_vs_sham_row(active_result, modality, selected_row):
    if modality == "edge":
        feature = (selected_row["ROI_1"], selected_row["ROI_2"])
        feature_index = CROSS_EDGE_NAMES.index(feature)
        active_pre = active_result["edge_pre"][:, feature_index]
        active_post = active_result["edge_post"][:, feature_index]
        sham_pre = sham_result["edge_pre"][:, feature_index]
        sham_post = sham_result["edge_post"][:, feature_index]
        identifiers = {"ROI_1": feature[0], "ROI_2": feature[1]}
    else:
        feature = selected_row["ROI"]
        feature_index = ROI_COLUMNS.index(feature)
        active_pre = active_result["regional_pre"][:, feature_index]
        active_post = active_result["regional_post"][:, feature_index]
        sham_pre = sham_result["regional_pre"][:, feature_index]
        sham_post = sham_result["regional_post"][:, feature_index]
        identifiers = {"ROI": feature}

    active_change = active_post - active_pre
    sham_change = sham_post - sham_pre
    active_stats = paired_stats(active_pre, active_post)
    sham_stats = paired_stats(sham_pre, sham_post)
    frame = pd.DataFrame({
        "change": np.concatenate([active_change, sham_change]),
        "group": np.concatenate([np.ones(len(active_change)), np.zeros(len(sham_change))]),
        "age": np.concatenate([active_result["age"], sham_result["age"]]),
        "sex": np.concatenate([active_result["sex"], sham_result["sex"]]),
        "education": np.concatenate([active_result["education"], sham_result["education"]]),
    }).dropna()
    design = sm.add_constant(frame[["group", "age", "sex"]], has_constant="add")
    fit = sm.OLS(frame["change"], design).fit()
    ci_low, ci_high = fit.conf_int().loc["group"]
    t_value = float(fit.tvalues["group"])
    row = {
        "active_arm": active_result["arm"],
        "modality": modality,
        **identifiers,
        "active_n": active_stats["n"],
        "active_pre_mean_deviation": active_stats["pre_mean"],
        "active_post_mean_deviation": active_stats["post_mean"],
        "active_post_minus_pre": active_stats["post_minus_pre"],
        "active_change_CI95_low": active_stats["post_minus_pre_CI95_low"],
        "active_change_CI95_high": active_stats["post_minus_pre_CI95_high"],
        "active_Cohen_dz": active_stats["Cohen_dz"],
        "active_paired_t": active_stats["paired_t"],
        "active_paired_p_uncorrected": active_stats["paired_p_uncorrected"],
        "sham_n": sham_stats["n"],
        "sham_pre_mean_deviation": sham_stats["pre_mean"],
        "sham_post_mean_deviation": sham_stats["post_mean"],
        "sham_post_minus_pre": sham_stats["post_minus_pre"],
        "sham_change_CI95_low": sham_stats["post_minus_pre_CI95_low"],
        "sham_change_CI95_high": sham_stats["post_minus_pre_CI95_high"],
        "sham_Cohen_dz": sham_stats["Cohen_dz"],
        "sham_paired_t": sham_stats["paired_t"],
        "sham_paired_p_uncorrected": sham_stats["paired_p_uncorrected"],
        "interaction_beta_active_minus_sham": float(fit.params["group"]),
        "interaction_CI95_low": float(ci_low),
        "interaction_CI95_high": float(ci_high),
        "interaction_t": t_value,
        "interaction_effect_size_partial_r": signed_partial_r(t_value, fit.df_resid),
        "interaction_df_resid": float(fit.df_resid),
        "interaction_p_uncorrected": float(fit.pvalues["group"]),
        "interaction_n": int(fit.nobs),
        "interaction_residual_shapiro_W_descriptive": float(stats.shapiro(fit.resid).statistic),
        "interaction_residual_shapiro_p_descriptive": float(stats.shapiro(fit.resid).pvalue),
        "interaction_breusch_pagan_p_descriptive": float(
            het_breuschpagan(fit.resid, fit.model.exog)[1]
        ),
    }
    return row


interaction_rows = []
for active_result in [itbs_result, rtms_result]:
    for modality in ["regional", "edge"]:
        for _, selected_row in active_result[f"{modality}_selected"].iterrows():
            interaction_rows.append(
                active_vs_sham_row(active_result, modality, selected_row)
            )
interaction_all = pd.DataFrame(interaction_rows)
interaction_all["interaction_q_fdr_BH_within_arm_modality"] = np.nan
for (_, _), indices in interaction_all.groupby(["active_arm", "modality"]).groups.items():
    interaction_all.loc[indices, "interaction_q_fdr_BH_within_arm_modality"] = bh_fdr(
        interaction_all.loc[indices, "interaction_p_uncorrected"]
    )
interaction_all = interaction_all.sort_values(
    ["active_arm", "modality", "interaction_p_uncorrected"]
).reset_index(drop=True)
interaction_p05 = interaction_all[
    interaction_all["interaction_p_uncorrected"] < 0.05
].copy()

interaction_file = INTERACTION_OUT / "active_treatment_vs_sham_group_by_time.xlsx"
with pd.ExcelWriter(interaction_file, engine="openpyxl") as writer:
    interaction_all.to_excel(writer, sheet_name="all_active_paired_p05", index=False)
    interaction_p05.to_excel(writer, sheet_name="interaction_p_lt_0p05", index=False)
    for arm in ["iTBS", "rTMS"]:
        interaction_all[interaction_all["active_arm"] == arm].to_excel(
            writer, sheet_name=f"{arm}_all", index=False
        )
style_excel_workbook(interaction_file)
display(interaction_all)


# Source cell 25
import importlib.util

MWC_ANALYSIS_OUT = OUTPUT_ROOT / "07_age_sex_adjusted_MwC"
_mwc_spec = importlib.util.spec_from_file_location("metafl_adjusted_mwc", MWC_ANALYSIS_SCRIPT)
if _mwc_spec is None or _mwc_spec.loader is None:
    raise ImportError(MWC_ANALYSIS_SCRIPT)
_mwc_module = importlib.util.module_from_spec(_mwc_spec)
_mwc_spec.loader.exec_module(_mwc_module)
_mwc_return_code = _mwc_module.main(
    MWC_ADJUSTED_WIDE_FILE,
    ROI_FILE,
    MWC_BASELINE_HC3_FILE,
    MWC_VALIDATION_FILE,
    MWC_ANALYSIS_OUT,
)
assert _mwc_return_code == 0
mwc_analysis_validation = json.loads((MWC_ANALYSIS_OUT / "validation_report.json").read_text(encoding="utf-8"))
assert mwc_analysis_validation.get("status") == "PASS"

MWC_TABLE_OUT = MWC_ANALYSIS_OUT / "tables"
mwc_paired_all = pd.read_csv(MWC_TABLE_OUT / "iTBS_rTMS_sham_paired_all_26.csv")
mwc_paired_p05 = pd.read_csv(MWC_TABLE_OUT / "iTBS_rTMS_paired_p_lt_0p05.csv")
mwc_interaction_all = pd.read_csv(MWC_TABLE_OUT / "iTBS_rTMS_vs_sham_interaction_adjusted_all_26.csv")
mwc_hdrs_all = pd.read_csv(MWC_TABLE_OUT / "iTBS_rTMS_HDRS_partial_spearman_adjusted_all_26.csv")

COMPLETE_ACTIVE_OUT = OUTPUT_ROOT / "06_active_paired_complete_results"
COMPLETE_ACTIVE_OUT.mkdir(parents=True, exist_ok=True)

def prefixed(frame, prefix, keys):
    return frame.rename(columns={column: f"{prefix}{column}" for column in frame if column not in keys})

regional_complete_parts = []
for arm, result in [("iTBS", itbs_result), ("rTMS", rtms_result)]:
    paired = result["regional_selected"].copy()
    if paired.empty:
        continue
    paired.insert(0, "arm", arm)
    paired["paired_df"] = result["n"] - 1
    paired["paired_q_fdr_BH_26_regions"] = bh_fdr(result["regional_all"]["paired_p_uncorrected"])[
        [result["regional_all"].index[result["regional_all"]["ROI"].eq(roi)][0] for roi in paired["ROI"]]
    ]
    paired["paired_change_shapiro_W_descriptive"] = [
        float(stats.shapiro(result["regional_delta"][:, ROI_COLUMNS.index(roi)]).statistic)
        for roi in paired["ROI"]
    ]
    paired["paired_change_shapiro_p_descriptive"] = [
        float(stats.shapiro(result["regional_delta"][:, ROI_COLUMNS.index(roi)]).pvalue)
        for roi in paired["ROI"]
    ]
    interaction = interaction_all.loc[
        interaction_all.active_arm.eq(arm) & interaction_all.modality.eq("regional")
    ].drop(columns=["active_arm", "modality"])
    hdrs = result["regional_hdrs_all"].copy()
    regional_complete_parts.append(
        prefixed(paired, "paired_", ["arm", "ROI"])
        .merge(prefixed(interaction, "interaction_", ["ROI"]), on="ROI", how="left")
        .merge(prefixed(hdrs, "HDRS_", ["ROI"]), on="ROI", how="left")
    )
regional_active_complete = (
    pd.concat(regional_complete_parts, ignore_index=True)
    if regional_complete_parts else pd.DataFrame(columns=["modality"])
)
if "modality" not in regional_active_complete:
    regional_active_complete.insert(0, "modality", "Regional SUVR")

mwc_interaction_for_join = mwc_interaction_all.rename(columns={"active_arm": "arm"})
mwc_active_complete = (
    prefixed(mwc_paired_p05, "paired_", ["arm", "ROI", "display_roi"])
    .merge(
        prefixed(mwc_interaction_for_join, "interaction_", ["arm", "ROI", "display_roi"]),
        on=["arm", "ROI", "display_roi"], how="left",
    )
    .merge(
        prefixed(mwc_hdrs_all, "HDRS_", ["arm", "ROI", "display_roi"]),
        on=["arm", "ROI", "display_roi"], how="left",
    )
)
mwc_active_complete.insert(0, "modality", "MwC")

regional_active_complete.to_csv(
    COMPLETE_ACTIVE_OUT / "Regional_SUVR_iTBS_rTMS_paired_p_lt_0p05_complete.csv", index=False
)
mwc_active_complete.to_csv(
    COMPLETE_ACTIVE_OUT / "MwC_iTBS_rTMS_paired_p_lt_0p05_complete.csv", index=False
)
pd.concat([regional_active_complete, mwc_active_complete], ignore_index=True, sort=False).to_csv(
    COMPLETE_ACTIVE_OUT / "Regional_SUVR_and_MwC_iTBS_rTMS_paired_p_lt_0p05_complete.csv", index=False
)

regional_subject_rows = []
for arm, result in [("iTBS", itbs_result), ("rTMS", rtms_result)]:
    for roi in result["regional_selected"]["ROI"]:
        index = ROI_COLUMNS.index(roi)
        for source_arm, source in [(arm, result), ("sham", sham_result)]:
            for subject_index, subject_key in enumerate(source["subject_keys"]):
                regional_subject_rows.append({
                    "modality": "Regional SUVR", "selected_for_arm": arm,
                    "analysis_arm": source_arm, "ROI": roi,
                    "subject_key": subject_key,
                    "pre_deviation": source["regional_pre"][subject_index, index],
                    "post_deviation": source["regional_post"][subject_index, index],
                    "post_minus_pre": source["regional_delta"][subject_index, index],
                    "HDRS_improvement_pre_minus_post": source["hdrs_improvement"][subject_index],
                    "age": source["age"][subject_index], "sex": source["sex"][subject_index],
                    "education": source["education"][subject_index],
                })
pd.DataFrame(regional_subject_rows).to_csv(
    COMPLETE_ACTIVE_OUT / "Regional_SUVR_selected_subject_level_values.csv", index=False
)

def adjusted_mwc_paired_rows(arm, roi_name, selected_for_arm):
    subset = mwc_adjusted.loc[
        mwc_adjusted.analysis_cohort.eq("TRD2") & mwc_adjusted.analysis_arm.eq(arm)
    ].copy()
    keys = sorted(
        set(subset.loc[subset.session.eq("ses-01"), "subject_key"])
        & set(subset.loc[subset.session.eq("ses-02"), "subject_key"])
    )
    feature = f"MwC_{roi_name}"
    pre = subset.loc[subset.session.eq("ses-01")].set_index("subject_key").loc[keys]
    post = subset.loc[subset.session.eq("ses-02")].set_index("subject_key").loc[keys]
    rows = []
    for key in keys:
        rows.append({
            "modality": "MwC", "selected_for_arm": selected_for_arm,
            "analysis_arm": arm, "ROI": roi_name, "subject_key": key,
            "pre_deviation": pre.at[key, feature], "post_deviation": post.at[key, feature],
            "post_minus_pre": post.at[key, feature] - pre.at[key, feature],
            "HDRS_improvement_pre_minus_post": pre.at[key, "hdrs_pre"] - pre.at[key, "hdrs_post"],
            "age": pre.at[key, "age"], "sex": pre.at[key, "sex"],
            "education": pre.at[key, "education"],
        })
    return rows

mwc_subject_rows = []
for row in mwc_paired_p05.itertuples(index=False):
    mwc_subject_rows.extend(adjusted_mwc_paired_rows(row.arm, row.ROI, row.arm))
    mwc_subject_rows.extend(adjusted_mwc_paired_rows("sham", row.ROI, row.arm))
pd.DataFrame(mwc_subject_rows).to_csv(
    COMPLETE_ACTIVE_OUT / "MwC_selected_subject_level_values.csv", index=False
)

display(regional_active_complete)
display(mwc_active_complete)
