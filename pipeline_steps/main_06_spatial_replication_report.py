"""Auto-extracted pipeline step: Main 06 — spatial replication and integrated report.

Do not edit generated copies independently of the source notebooks.
"""


# Source cell 27
import itertools
import math

from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

SPATIAL_OUT = OUTPUT_ROOT / "03_TRD1_TRD2_spatial_correlation"
SPATIAL_OUT.mkdir(parents=True, exist_ok=True)
FIGURE_OUT = OUTPUT_ROOT / "04_figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)
N_SPATIAL_PERMUTATIONS = 100_000
SPATIAL_SEED = 20260821
SPATIAL_PERMUTATION_SEED = 20260830
SPATIAL_ALPHA = 0.05
PERMUTATION_BATCH_SIZE = 1_000


def rowwise_pearson(x, y):
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    return np.divide(
        numerator, denominator,
        out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0,
    )


def spatial_point_metrics(first, second):
    pearson = pearsonr(first, second)
    same = ((first > 0) & (second > 0)) | ((first < 0) & (second < 0))
    abs_first, abs_second = np.abs(first), np.abs(second)
    weighted_signed_dsc = (
        2 * np.sum(np.minimum(abs_first, abs_second) * same)
        / (abs_first.sum() + abs_second.sum())
    )
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "same_direction_proportion": float(same.mean()),
        "same_direction_count": int(same.sum()),
        "weighted_signed_dsc": float(weighted_signed_dsc),
    }


def permutation_metric_batch(first, permuted_second):
    first_rows = np.broadcast_to(first, permuted_second.shape)
    same = ((first_rows > 0) & (permuted_second > 0)) | (
        (first_rows < 0) & (permuted_second < 0)
    )
    abs_first, abs_second = np.abs(first_rows), np.abs(permuted_second)
    return pd.DataFrame({
        "pearson_r": rowwise_pearson(first_rows, permuted_second),
        "same_direction_proportion": same.mean(axis=1),
        "weighted_signed_dsc": (
            2 * (np.minimum(abs_first, abs_second) * same).sum(axis=1)
            / (abs_first.sum(axis=1) + abs_second.sum(axis=1))
        ),
    })


def expand_bilateral_pair_permutations(pair_permutations, pair_offset=0):
    pair_permutations = np.asarray(pair_permutations, dtype=int)
    hemispheres = np.array([0, 1], dtype=int)
    return (2 * pair_permutations[..., :, None] + hemispheres + pair_offset).reshape(
        len(pair_permutations), -1
    )


def regional_pair_permutation_null(first, second, n_permutations, rng):
    """Permute 13 bilateral ROI identities while preserving left/right pairing."""
    chunks, completed = [], 0
    n_pairs = len(second) // 2
    assert n_pairs == 13 and len(second) == 26
    while completed < n_permutations:
        size = min(PERMUTATION_BATCH_SIZE, n_permutations - completed)
        pair_orders = np.argsort(rng.random((size, n_pairs)), axis=1)
        feature_orders = expand_bilateral_pair_permutations(pair_orders)
        metrics = permutation_metric_batch(first, second[feature_orders])
        metrics.insert(0, "permutation_iteration", np.arange(completed + 1, completed + size + 1))
        chunks.append(metrics)
        completed += size
    return pd.concat(chunks, ignore_index=True)


def connectivity_endpoint_permutation_null(first, second):
    """Exactly permute six frontal and seven limbic bilateral endpoint pairs."""
    first = np.asarray(first, dtype=float)
    second_matrix = np.asarray(second, dtype=float).reshape(12, 14)
    row_pair_orders = np.asarray(list(itertools.permutations(range(6))), dtype=np.int16)
    column_pair_orders = np.asarray(list(itertools.permutations(range(7))), dtype=np.int16)
    row_endpoint_orders = expand_bilateral_pair_permutations(row_pair_orders)
    column_endpoint_orders = expand_bilateral_pair_permutations(column_pair_orders)
    chunks, completed = [], 0
    for row_order in row_endpoint_orders:
        row_reordered = second_matrix[row_order, :]
        permuted = np.transpose(row_reordered[:, column_endpoint_orders], (1, 0, 2)).reshape(
            len(column_endpoint_orders), -1
        )
        metrics = permutation_metric_batch(first, permuted)
        metrics.insert(0, "permutation_iteration", np.arange(completed + 1, completed + len(metrics) + 1))
        chunks.append(metrics)
        completed += len(metrics)
    result = pd.concat(chunks, ignore_index=True)
    assert len(result) == math.factorial(6) * math.factorial(7)
    return result


def empirical_upper_tail_p(null_values, observed, exact):
    null_values = np.asarray(null_values, dtype=float)
    offset = 0 if exact else 1
    return float((np.count_nonzero(null_values >= observed) + offset) / (len(null_values) + offset))


def upper_tail_significance_threshold(null_values, alpha, exact):
    ordered = np.sort(np.asarray(null_values, dtype=float))
    support, first_indices = np.unique(ordered, return_index=True)
    offset = 0 if exact else 1
    upper_tail_p = (len(ordered) - first_indices + offset) / (len(ordered) + offset)
    eligible = support[upper_tail_p < alpha]
    return float(eligible[0]) if len(eligible) else np.nan


# Scientific sequential palettes progress from light to dark across the three panels.
# Blues: ColorBrewer Blues; orange-yellow: colorblind-safe sequential Oranges.
SPATIAL_BLUE_SEQUENCE = ["#6BAED6", "#3182BD", "#08519C"]
SPATIAL_ORANGE_SEQUENCE = ["#FDB863", "#E69F00", "#B35806"]
modalities = {
    "Regional": {
        "display_name": "Regional SUVR",
        "labels": pd.DataFrame({"ROI": ROI_COLUMNS}),
        "trd1": trd1_result["case_regional"].to_numpy(float),
        "trd2": trd2_context["case_regional"].to_numpy(float),
        "color": SPATIAL_BLUE_SEQUENCE[0], "agreement_color": SPATIAL_ORANGE_SEQUENCE[0], "unit": "ROIs",
    },
    "Connectivity": {
        "display_name": "Metabolic edge",
        "labels": pd.DataFrame({
            "ROI_1": [edge[0] for edge in CROSS_EDGE_NAMES],
            "ROI_2": [edge[1] for edge in CROSS_EDGE_NAMES],
        }),
        "trd1": trd1_result["case_edge"],
        "trd2": trd2_context["case_edge"],
        "color": SPATIAL_BLUE_SEQUENCE[1], "agreement_color": SPATIAL_ORANGE_SEQUENCE[1], "unit": "edges",
    },
    "MwC": {
        "display_name": "MwC",
        "labels": pd.DataFrame({"ROI": ROI_COLUMNS}),
        "trd1": mwc_group_values["TRD1"],
        "trd2": mwc_group_values["TRD2"],
        "color": SPATIAL_BLUE_SEQUENCE[2], "agreement_color": SPATIAL_ORANGE_SEQUENCE[2], "unit": "ROIs",
    },
}

summary_rows, permutation_threshold_rows = [], []
spatial_vectors, spatial_permutation_inference = {}, {}
for modality, item in modalities.items():
    mean1, mean2 = np.nanmean(item["trd1"], axis=0), np.nanmean(item["trd2"], axis=0)
    point = spatial_point_metrics(mean1, mean2)
    if modality in {"Regional", "MwC"}:
        permutation_seed = SPATIAL_PERMUTATION_SEED + (1 if modality == "MwC" else 0)
        permutation_null = regional_pair_permutation_null(
            mean1, mean2, N_SPATIAL_PERMUTATIONS, np.random.default_rng(permutation_seed)
        )
        exact_permutation = False
        permutation_scheme = "Monte Carlo permutation of 13 bilateral ROI-pair labels; left/right pairing preserved"
    else:
        permutation_seed = np.nan
        permutation_null = connectivity_endpoint_permutation_null(mean1, mean2)
        exact_permutation = True
        permutation_scheme = "Exact independent permutations of six bilateral frontal and seven bilateral limbic node-pair labels"

    np.savez_compressed(
        SPATIAL_OUT / f"{modality.lower()}_structure_preserving_null.npz",
        **{column: permutation_null[column].to_numpy(float) for column in [
            "pearson_r", "same_direction_proportion", "weighted_signed_dsc"
        ]},
    )
    vectors = item["labels"].copy()
    vectors.insert(0, "feature_index", np.arange(1, len(vectors) + 1))
    vectors["TRD1_mean_deviation"] = mean1
    vectors["TRD2_mean_deviation"] = mean2
    vectors["same_direction"] = ((mean1 > 0) & (mean2 > 0)) | ((mean1 < 0) & (mean2 < 0))
    spatial_vectors[modality] = vectors
    spatial_permutation_inference[modality] = {}

    for label, column, p_key in [
        ("Pearson correlation", "pearson_r", "pearson_p"),
        ("Same-direction proportion", "same_direction_proportion", None),
        ("Magnitude-weighted signed DSC", "weighted_signed_dsc", None),
    ]:
        null_values = permutation_null[column].to_numpy(float)
        empirical_p = empirical_upper_tail_p(null_values, point[column], exact_permutation)
        critical_value = upper_tail_significance_threshold(null_values, SPATIAL_ALPHA, exact_permutation)
        inference = {
            "empirical_p_one_sided": empirical_p,
            "null_mean": float(np.mean(null_values)),
            "null_95th_percentile": float(np.quantile(null_values, 0.95, method="higher")),
            "significance_threshold_alpha_0_05": critical_value,
            "threshold_operator": ">=",
            "threshold_exceeded": bool(np.isfinite(critical_value) and point[column] >= critical_value),
            "significant_at_alpha_0_05": bool(empirical_p < SPATIAL_ALPHA),
        }
        spatial_permutation_inference[modality][column] = inference
        summary_rows.append({
            "modality": modality, "display_name": item["display_name"],
            "n_TRD1": len(item["trd1"]), "n_TRD2": len(item["trd2"]),
            "n_features": item["trd1"].shape[1], "metric": label,
            "estimate": point[column],
            "parametric_p_descriptive": np.nan if p_key is None else point[p_key],
            **inference,
            "same_direction_count": point["same_direction_count"] if column == "same_direction_proportion" else np.nan,
            "permutation_resamples": len(permutation_null),
            "permutation_exact": exact_permutation,
            "permutation_seed": permutation_seed,
            "permutation_scheme": permutation_scheme,
        })
        if column in {"same_direction_proportion", "weighted_signed_dsc"}:
            permutation_threshold_rows.append({
                "modality": modality, "metric": label, "estimate": point[column], **inference,
                "critical_same_direction_count": int(round(critical_value * len(mean1)))
                if column == "same_direction_proportion" and np.isfinite(critical_value) else np.nan,
                "critical_same_direction_percent": float(100 * critical_value)
                if column == "same_direction_proportion" and np.isfinite(critical_value) else np.nan,
                "n_features": len(mean1), "alpha": SPATIAL_ALPHA, "alternative": "greater",
                "permutation_resamples": len(permutation_null), "permutation_exact": exact_permutation,
                "permutation_seed": permutation_seed, "permutation_scheme": permutation_scheme,
            })

spatial_summary = pd.DataFrame(summary_rows)
spatial_permutation_thresholds = pd.DataFrame(permutation_threshold_rows)
assert len(spatial_permutation_thresholds) == 6
assert spatial_permutation_thresholds["empirical_p_one_sided"].between(0, 1).all()
spatial_summary.to_csv(SPATIAL_OUT / "spatial_correlation_summary.csv", index=False)
spatial_permutation_thresholds.to_csv(SPATIAL_OUT / "spatial_similarity_permutation_thresholds.csv", index=False)
spatial_vectors["Regional"].to_csv(SPATIAL_OUT / "regional_mean_deviation_vectors.csv", index=False)
spatial_vectors["Connectivity"].to_csv(SPATIAL_OUT / "connectivity_mean_deviation_vectors.csv", index=False)
spatial_vectors["MwC"].to_csv(SPATIAL_OUT / "mwc_mean_deviation_vectors.csv", index=False)

figure_dsc_rows = []
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=False)
fig.patch.set_facecolor("white")
for ax, (modality, item) in zip(axes, modalities.items()):
    x, y = np.nanmean(item["trd1"], axis=0), np.nanmean(item["trd2"], axis=0)
    point = spatial_point_metrics(x, y)
    p = spatial_permutation_inference[modality]["pearson_r"]["empirical_p_one_sided"]
    if p < 0.001:
        p_text, stars = "< 0.001", "***"
    elif p < 0.01:
        p_text, stars = f"= {p:.3f}", "**"
    elif p < 0.05:
        p_text, stars = f"= {p:.3f}", "*"
    else:
        p_text, stars = f"= {p:.3f}", ""
    sns.regplot(
        x=x, y=y, ax=ax, ci=95, n_boot=1000, seed=SPATIAL_SEED,
        color=item["color"], scatter_kws={"s": 14, "alpha": 0.8, "edgecolor": "none"},
        line_kws={"lw": 1.5},
    )
    ax.set_title(item["display_name"], fontsize=15, fontweight="bold", pad=8)
    ax.text(0.03, 0.95, f"r = {point['pearson_r']:.3f},  p {p_text}{stars}",
            transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")
    ax.text(0.03, 0.86, rf"$\mathbf{{DSC}}_{{\mathbf{{weighted}}}}$ = {point['weighted_signed_dsc']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")
    ax.set_facecolor("white")
    ax.set_box_aspect(1)
    ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    proportion, count, feature_n = point["same_direction_proportion"], point["same_direction_count"], len(x)
    inset = ax.inset_axes([0.67, 0.035, 0.30, 0.30])
    inset.set_axis_off(); inset.set_xlim(0, 1); inset.set_ylim(0, 1); inset.patch.set_alpha(0)
    inset.text(0.5, 0.78, f"{100 * proportion:.1f}%", ha="center", va="center",
               fontsize=15, fontweight="bold", color=item["agreement_color"])
    inset.add_patch(Rectangle((0.08, 0.49), 0.84, 0.10, facecolor="#F1E6DA", edgecolor="none"))
    inset.add_patch(Rectangle((0.08, 0.49), 0.84 * proportion, 0.10,
                              facecolor=item["agreement_color"], edgecolor="none"))
    inset.text(0.5, 0.20, f"({count}/{feature_n}\n{item['unit']})", ha="center", va="center",
               fontsize=14, fontweight="medium", linespacing=1.05, color="#333333")
    figure_dsc_rows.append({
        "modality": modality, "pearson_r": point["pearson_r"],
        "pearson_p_descriptive": point["pearson_p"], "pearson_p_empirical_one_sided": p,
        "DSC_weighted": point["weighted_signed_dsc"],
        "DSC_weighted_p_empirical_one_sided": spatial_permutation_inference[modality]["weighted_signed_dsc"]["empirical_p_one_sided"],
        "same_direction_n": count, "feature_n": feature_n,
        "same_direction_percent": 100 * proportion,
        "same_direction_p_empirical_one_sided": spatial_permutation_inference[modality]["same_direction_proportion"]["empirical_p_one_sided"],
    })

fig.subplots_adjust(left=0.01, right=0.995, top=0.90, bottom=0.03, wspace=0.04)
spatial_png = FIGURE_OUT / "spatial_correlation_scatter.png"
spatial_pdf = FIGURE_OUT / "spatial_correlation_scatter.pdf"
fig.savefig(spatial_png, dpi=600, facecolor="white")
fig.savefig(spatial_pdf, dpi=600, facecolor="white")
plt.close(fig)
pd.DataFrame(figure_dsc_rows).to_csv(SPATIAL_OUT / "spatial_correlation_figure_DSC_values.csv", index=False)
display(spatial_summary)
display(spatial_permutation_thresholds)


# Source cell 29
summary = pd.DataFrame([
    {"analysis": "TRD1 vs nTRD edge", "family_or_tested_n": 168,
     "reporting_rule": "uncorrected p<0.01", "reported_n": len(trd1_result["edge_reported"])},
    {"analysis": "TRD1 vs nTRD regional", "family_or_tested_n": 26,
     "reporting_rule": "uncorrected p<0.01", "reported_n": len(trd1_result["regional_reported"])},
    {"analysis": "HDRS-adjusted sensitivity edge", "family_or_tested_n": 168,
     "reporting_rule": "uncorrected p<0.01", "reported_n": len(sensitivity_edge_p001)},
    {"analysis": "HDRS-adjusted sensitivity regional", "family_or_tested_n": 26,
     "reporting_rule": "uncorrected p<0.01", "reported_n": len(sensitivity_regional_p001)},
])

for result in [itbs_result, rtms_result, sham_result]:
    for modality in ["edge", "regional"]:
        rows = [{
            "analysis": f"{result['arm']} {modality} pre-post",
            "family_or_tested_n": 168 if modality == "edge" else 26,
            "reporting_rule": "paired uncorrected p<0.05",
            "reported_n": len(result[f"{modality}_selected"]),
        }]
        if result["arm"] != "sham":
            rows.extend([
                {
                    "analysis": f"{result['arm']} {modality} HDRS",
                    "family_or_tested_n": len(result[f"{modality}_hdrs_all"]),
                    "reporting_rule": "BH-FDR q<0.05 primary",
                    "reported_n": len(result[f"{modality}_hdrs_fdr"]),
                },
                {
                    "analysis": f"{result['arm']} {modality} HDRS",
                    "family_or_tested_n": len(result[f"{modality}_hdrs_all"]),
                    "reporting_rule": "uncorrected p<0.05 exploratory",
                    "reported_n": len(result[f"{modality}_hdrs_exploratory"]),
                },
            ])
        summary = pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)

summary = pd.concat([summary, pd.DataFrame([{
    "analysis": "Active treatment vs sham group-by-time",
    "family_or_tested_n": len(interaction_all),
    "reporting_rule": "ordinary OLS uncorrected p<0.05",
    "reported_n": len(interaction_p05),
}])], ignore_index=True)

combined_file = OUTPUT_ROOT / "all_reported_results.xlsx"
with pd.ExcelWriter(combined_file, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    demographic_table.to_excel(writer, sheet_name="demographic", index=False)
    baseline_hdrs_table.to_excel(writer, sheet_name="baseline_HDRS", index=False)
    msm_table.to_excel(writer, sheet_name="MSM_scores", index=False)
    hdrs_reduction_table.to_excel(writer, sheet_name="HDRS_reduction", index=False)
    stimulation_table.to_excel(writer, sheet_name="stimulation_paired", index=False)
    trd1_result["edge_reported"].to_excel(writer, sheet_name="TRD1_edges_p001", index=False)
    trd1_result["regional_reported"].to_excel(writer, sheet_name="TRD1_regions_p001", index=False)
    sensitivity_summary.to_excel(writer, sheet_name="HDRS_sensitivity_summary", index=False)
    sensitivity_edge_p001.to_excel(writer, sheet_name="HDRS_sens_edge_p001", index=False)
    sensitivity_regional_p001.to_excel(writer, sheet_name="HDRS_sens_region_p001", index=False)
    for result in [itbs_result, rtms_result, sham_result]:
        arm = result["arm"]
        result["edge_selected"].to_excel(writer, sheet_name=f"{arm}_edge_prepost_p05", index=False)
        result["regional_selected"].to_excel(writer, sheet_name=f"{arm}_region_prepost_p05", index=False)
        if arm != "sham":
            result["edge_hdrs_fdr"].to_excel(writer, sheet_name=f"{arm}_edge_HDRS_FDR", index=False)
            result["edge_hdrs_exploratory"].to_excel(writer, sheet_name=f"{arm}_edge_HDRS_p05", index=False)
            result["regional_hdrs_fdr"].to_excel(writer, sheet_name=f"{arm}_region_HDRS_FDR", index=False)
            result["regional_hdrs_exploratory"].to_excel(writer, sheet_name=f"{arm}_region_HDRS_p05", index=False)
    interaction_all.to_excel(writer, sheet_name="active_vs_sham_interaction", index=False)
    spatial_summary.to_excel(writer, sheet_name="TRD1_TRD2_spatial", index=False)
style_excel_workbook(combined_file)
print("Combined workbook:", combined_file)
display(summary)
