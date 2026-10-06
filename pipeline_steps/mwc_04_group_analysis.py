"""Auto-extracted pipeline step: MwC 04 — final nTRD versus TRD analysis.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 12
SELECTED_ROIS = {
    "SFG_L": "MwC_deviation_Frontal_Sup_2_L",
    "MFG_L": "MwC_deviation_Frontal_Mid_2_L",
    "vmPFC_R": "MwC_deviation_Frontal_Med_Orb_R",
    "PAL_L": "MwC_deviation_Pallidum_L",
    "HPC_R": "MwC_deviation_Hippocampus_R",
    "AMY_R": "MwC_deviation_Amygdala_R",
}

baseline = mwc_adjusted_26.loc[
    mwc_adjusted_26.session.eq("ses-01")
    & mwc_adjusted_26.analysis_cohort.isin(["nTRD", "TRD1"])
].copy()
assert baseline.groupby("analysis_cohort").size().to_dict() == {"TRD1": 35, "nTRD": 25}
for deviation_metric in SELECTED_ROIS.values():
    source_metric = deviation_metric.replace("MwC_deviation_", "MwC_", 1)
    baseline[deviation_metric] = baseline[source_metric].to_numpy(float)


def adjusted_group_test(frame: pd.DataFrame, region: str, metric: str) -> dict[str, object]:
    data = frame[[metric, "analysis_cohort", "age", "sex"]].dropna().copy()
    data["trd"] = data.analysis_cohort.eq("TRD1").astype(float)
    data["age_c"] = data.age - data.age.mean()
    design = sm.add_constant(data[["trd", "age_c", "sex"]], has_constant="add")
    model = sm.OLS(data[metric], design).fit(cov_type="HC3")
    ci = model.conf_int().loc["trd"].to_numpy(float)
    ntrd = data.loc[data.analysis_cohort.eq("nTRD"), metric].to_numpy(float)
    trd = data.loc[data.analysis_cohort.eq("TRD1"), metric].to_numpy(float)
    return {
        "region": region,
        "metric": metric,
        "comparison": "TRD - nTRD",
        "covariates": "age + sex",
        "n_nTRD": len(ntrd),
        "n_TRD": len(trd),
        "nTRD_mean_HC_z": float(np.mean(ntrd)),
        "nTRD_sd_HC_z": float(np.std(ntrd, ddof=1)),
        "TRD_mean_HC_z": float(np.mean(trd)),
        "TRD_sd_HC_z": float(np.std(trd, ddof=1)),
        "beta_TRD_minus_nTRD_HC_SD": float(model.params["trd"]),
        "hc3_se": float(model.bse["trd"]),
        "ci95_low_TRD_minus_nTRD_HC_SD": float(ci[0]),
        "ci95_high_TRD_minus_nTRD_HC_SD": float(ci[1]),
        "p_adjusted": float(model.pvalues["trd"]),
    }


adjusted_statistics = pd.DataFrame(
    [adjusted_group_test(baseline, region, metric) for region, metric in SELECTED_ROIS.items()]
)
adjusted_statistics.to_csv(
    OUTPUT_DIR / "ntrd_vs_trd_selected_unilateral_mwc_deviation_age_sex.csv",
    index=False,
)

assert len(adjusted_statistics) == len(SELECTED_ROIS) == 6
assert adjusted_statistics[["p_adjusted", "beta_TRD_minus_nTRD_HC_SD"]].notna().all().all()
expected_complete_cases = {}
for region, metric in SELECTED_ROIS.items():
    complete = baseline[[metric, "analysis_cohort", "age", "sex"]].dropna()
    counts = complete.groupby("analysis_cohort").size().to_dict()
    expected_complete_cases[region] = {
        "n_nTRD": int(counts.get("nTRD", 0)),
        "n_TRD": int(counts.get("TRD1", 0)),
    }

observed_complete_cases = adjusted_statistics.set_index("region")[["n_nTRD", "n_TRD"]].astype(int)
for region, counts in expected_complete_cases.items():
    assert observed_complete_cases.loc[region].to_dict() == counts
assert (observed_complete_cases >= 20).all().all()

display(adjusted_statistics.round(6))
