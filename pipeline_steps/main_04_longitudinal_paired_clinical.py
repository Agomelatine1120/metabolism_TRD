"""Auto-extracted pipeline step: Main 04 — longitudinal paired tests and clinical associations.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 21
def add_hdrs_correlations(selected, value_matrix, feature_lookup, hdrs, age, sex, q_column):
    rows = []
    for _, selected_row in selected.iterrows():
        index = feature_lookup(selected_row)
        r_value, p_value, n_used = partial_spearman_age_sex(
            value_matrix[:, index], hdrs, age, sex
        )
        row = selected_row.to_dict()
        row.update({
            "HDRS_partial_Spearman_r": r_value,
            "HDRS_p_uncorrected": p_value,
            "n_used": n_used,
        })
        rows.append(row)

    columns = list(selected.columns) + [
        "HDRS_partial_Spearman_r", "HDRS_p_uncorrected", "n_used", q_column,
        "HDRS_primary_FDR_q_lt_0p05", "HDRS_exploratory_uncorrected_p_lt_0p05",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    result[q_column] = bh_fdr(result["HDRS_p_uncorrected"])
    result["HDRS_primary_FDR_q_lt_0p05"] = result[q_column] < FDR_THRESHOLD
    result["HDRS_exploratory_uncorrected_p_lt_0p05"] = (
        result["HDRS_p_uncorrected"] < CLINICAL_P_THRESHOLD
    )
    return result.sort_values([q_column, "HDRS_p_uncorrected"]).reset_index(drop=True)


def save_prepost_and_hdrs(all_tests, selected, hdrs_results, output_dir, stem, q_column, include_hdrs=True):
    all_tests.to_excel(output_dir / f"{stem}_pairttest_all.xlsx", index=False)
    selected.to_excel(
        output_dir / f"{stem}_pairttest_uncorrected_p_lt_0p05.xlsx", index=False
    )
    hdrs_fdr = hdrs_results[hdrs_results["HDRS_primary_FDR_q_lt_0p05"]].copy()
    hdrs_exploratory = hdrs_results[
        hdrs_results["HDRS_exploratory_uncorrected_p_lt_0p05"]
    ].copy()
    if include_hdrs:
        hdrs_results.to_excel(
            output_dir / f"{stem}_HDRS_partial_spearman_all_prepost_p_lt_0p05.xlsx",
            index=False,
        )
        hdrs_fdr.to_excel(
            output_dir / f"{stem}_HDRS_partial_spearman_FDR_q_lt_0p05_primary.xlsx",
            index=False,
        )
        hdrs_exploratory.to_excel(
            output_dir / f"{stem}_HDRS_partial_spearman_uncorrected_p_lt_0p05_exploratory.xlsx",
            index=False,
        )
    return hdrs_fdr, hdrs_exploratory


def paired_mean_ci95(delta):
    delta = np.asarray(delta, dtype=float)
    n = np.sum(np.isfinite(delta), axis=0)
    mean = np.nanmean(delta, axis=0)
    sem = stats.sem(delta, axis=0, nan_policy="omit")
    critical = stats.t.ppf(0.975, np.maximum(n - 1, 1))
    return mean - critical * sem, mean + critical * sem


def run_arm_prepost(arm, context, output_dir, include_hdrs=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_clinical = clinical[
        (clinical["analysis_cohort"] == "TRD2")
        & (clinical["analysis_arm"] == arm)
    ].copy()
    required = ROI_COLUMNS + ["age", "sex", "education", "hdrs_improvement"]
    pre = arm_clinical.merge(pet_pre, on="subject_key", how="inner").dropna(subset=required)
    post = arm_clinical.merge(pet_post, on="subject_key", how="inner").dropna(subset=required)
    paired = sorted(set(pre["subject_key"]).intersection(post["subject_key"]))
    pre = pre[pre["subject_key"].isin(paired)].sort_values("subject_key").reset_index(drop=True)
    post = post[post["subject_key"].isin(paired)].sort_values("subject_key").reset_index(drop=True)
    assert len(pre) == EXPECTED_PAIRED_ARMS[arm]
    assert pre["subject_key"].tolist() == post["subject_key"].tolist()

    edge_pre = individual_edge_deviation(
        pre,
        context["reference_values"],
        context["reference_covariates"],
        context["pcc_reference"],
        context["denominator"],
        context["reference_covariate_names"],
    )
    edge_post = individual_edge_deviation(
        post,
        context["reference_values"],
        context["reference_covariates"],
        context["pcc_reference"],
        context["denominator"],
        context["reference_covariate_names"],
    )
    edge_delta = edge_post - edge_pre
    t_edge, p_edge = stats.ttest_rel(edge_post, edge_pre, axis=0, nan_policy="omit")
    edge_ci_low, edge_ci_high = paired_mean_ci95(edge_delta)

    edge_all = pd.DataFrame({
        "ROI_1": [pair[0] for pair in CROSS_EDGE_NAMES],
        "ROI_2": [pair[1] for pair in CROSS_EDGE_NAMES],
        "pre_deviation": np.nanmean(edge_pre, axis=0),
        "pre_SD_deviation": np.nanstd(edge_pre, axis=0, ddof=1),
        "post_deviation": np.nanmean(edge_post, axis=0),
        "post_SD_deviation": np.nanstd(edge_post, axis=0, ddof=1),
        "post_minus_pre": np.nanmean(edge_delta, axis=0),
        "post_minus_pre_CI95_low": edge_ci_low,
        "post_minus_pre_CI95_high": edge_ci_high,
        "paired_t": t_edge,
        "effect_size_Cohen_dz": t_edge / np.sqrt(len(pre)),
        "paired_p_uncorrected": p_edge,
    }).sort_values("paired_p_uncorrected").reset_index(drop=True)
    edge_selected = edge_all[
        edge_all["paired_p_uncorrected"] < PREPOST_P_THRESHOLD
    ].copy()

    hdrs = pre["hdrs_improvement"].to_numpy(float)
    age = pre["age"].to_numpy(float)
    sex = pre["sex"].to_numpy(float)
    edge_lookup = {pair: index for index, pair in enumerate(CROSS_EDGE_NAMES)}
    edge_hdrs_all = add_hdrs_correlations(
        edge_selected if include_hdrs else edge_selected.iloc[0:0],
        edge_delta,
        lambda row: edge_lookup[(row["ROI_1"], row["ROI_2"])],
        hdrs,
        age,
        sex,
        "HDRS_q_fdr_BH_within_prepost_significant_edges",
    )
    edge_hdrs_fdr, edge_hdrs_exploratory = save_prepost_and_hdrs(
        edge_all,
        edge_selected,
        edge_hdrs_all,
        output_dir,
        "edge",
        "HDRS_q_fdr_BH_within_prepost_significant_edges",
        include_hdrs=include_hdrs,
    )

    pre_design = sm.add_constant(pre[["age", "sex"]], has_constant="add").reindex(
        columns=context["regional_hc_design_columns"]
    )
    post_design = sm.add_constant(post[["age", "sex"]], has_constant="add").reindex(
        columns=context["regional_hc_design_columns"]
    )
    regional_pre = (
        pre[ROI_COLUMNS].to_numpy(float) - pre_design.to_numpy(float) @ context["regional_beta"]
    ) / context["reference_sd"].to_numpy(float)
    regional_post = (
        post[ROI_COLUMNS].to_numpy(float) - post_design.to_numpy(float) @ context["regional_beta"]
    ) / context["reference_sd"].to_numpy(float)
    regional_delta = regional_post - regional_pre
    t_regional, p_regional = stats.ttest_rel(
        regional_post, regional_pre, axis=0, nan_policy="omit"
    )
    regional_ci_low, regional_ci_high = paired_mean_ci95(regional_delta)
    regional_all = pd.DataFrame({
        "ROI": ROI_COLUMNS,
        "pre_deviation": np.nanmean(regional_pre, axis=0),
        "pre_SD_deviation": np.nanstd(regional_pre, axis=0, ddof=1),
        "post_deviation": np.nanmean(regional_post, axis=0),
        "post_SD_deviation": np.nanstd(regional_post, axis=0, ddof=1),
        "post_minus_pre": np.nanmean(regional_delta, axis=0),
        "post_minus_pre_CI95_low": regional_ci_low,
        "post_minus_pre_CI95_high": regional_ci_high,
        "paired_t": t_regional,
        "effect_size_Cohen_dz": t_regional / np.sqrt(len(pre)),
        "paired_p_uncorrected": p_regional,
    }).sort_values("paired_p_uncorrected").reset_index(drop=True)
    regional_selected = regional_all[
        regional_all["paired_p_uncorrected"] < PREPOST_P_THRESHOLD
    ].copy()
    regional_lookup = {roi: index for index, roi in enumerate(ROI_COLUMNS)}
    regional_hdrs_all = add_hdrs_correlations(
        regional_selected if include_hdrs else regional_selected.iloc[0:0],
        regional_delta,
        lambda row: regional_lookup[row["ROI"]],
        hdrs,
        age,
        sex,
        "HDRS_q_fdr_BH_within_prepost_significant_regions",
    )
    regional_hdrs_fdr, regional_hdrs_exploratory = save_prepost_and_hdrs(
        regional_all,
        regional_selected,
        regional_hdrs_all,
        output_dir,
        "regional",
        "HDRS_q_fdr_BH_within_prepost_significant_regions",
        include_hdrs=include_hdrs,
    )

    return {
        "arm": arm,
        "n": len(pre),
        "edge_all": edge_all,
        "edge_selected": edge_selected,
        "edge_hdrs_all": edge_hdrs_all,
        "edge_hdrs_fdr": edge_hdrs_fdr,
        "edge_hdrs_exploratory": edge_hdrs_exploratory,
        "regional_all": regional_all,
        "regional_selected": regional_selected,
        "regional_hdrs_all": regional_hdrs_all,
        "regional_hdrs_fdr": regional_hdrs_fdr,
        "regional_hdrs_exploratory": regional_hdrs_exploratory,
        "subject_keys": pre["subject_key"].to_numpy(),
        "hdrs_improvement": hdrs,
        "age": age,
        "sex": sex,
        "education": pre["education"].to_numpy(float),
        "edge_pre": edge_pre,
        "edge_post": edge_post,
        "edge_delta": edge_delta,
        "regional_pre": regional_pre,
        "regional_post": regional_post,
        "regional_delta": regional_delta,
    }


itbs_result = run_arm_prepost("iTBS", trd2_context, TRD2_OUT / "03_iTBS_prepost")
rtms_result = run_arm_prepost("rTMS", trd2_context, TRD2_OUT / "04_rTMS_prepost")
sham_result = run_arm_prepost("sham", trd2_context, TRD2_OUT / "05_sham_prepost", include_hdrs=False)

LONGITUDINAL_EDGE_FILE = TRD2_OUT / "longitudinal_high_quality_PET_subject_matrices_cross168.npz"
_paired_results = [itbs_result, rtms_result, sham_result]
np.savez_compressed(
    LONGITUDINAL_EDGE_FILE,
    subject_keys=np.concatenate([result["subject_keys"] for result in _paired_results]),
    arms=np.concatenate([np.repeat(result["arm"], result["n"]) for result in _paired_results]),
    edge_pre=np.vstack([result["edge_pre"] for result in _paired_results]),
    edge_post=np.vstack([result["edge_post"] for result in _paired_results]),
    edge_roi1=np.asarray([edge[0] for edge in CROSS_EDGE_NAMES], dtype=object),
    edge_roi2=np.asarray([edge[1] for edge in CROSS_EDGE_NAMES], dtype=object),
)
assert sum(EXPECTED_PAIRED_ARMS.values()) == sum(result["n"] for result in _paired_results)
print("Longitudinal edge archive:", LONGITUDINAL_EDGE_FILE)


prepost_counts = pd.DataFrame([
    {
        "arm": result["arm"],
        "n": result["n"],
        "edge_prepost_uncorrected_p_lt_0p05": len(result["edge_selected"]),
        "edge_HDRS_FDR_q_lt_0p05_primary": len(result["edge_hdrs_fdr"]),
        "edge_HDRS_uncorrected_p_lt_0p05_exploratory": len(result["edge_hdrs_exploratory"]),
        "regional_prepost_uncorrected_p_lt_0p05": len(result["regional_selected"]),
        "regional_HDRS_FDR_q_lt_0p05_primary": len(result["regional_hdrs_fdr"]),
        "regional_HDRS_uncorrected_p_lt_0p05_exploratory": len(result["regional_hdrs_exploratory"]),
    }
    for result in [itbs_result, rtms_result, sham_result]
])
prepost_counts.to_excel(TRD2_OUT / "prepost_and_HDRS_summary_counts.xlsx", index=False)
display(prepost_counts)
