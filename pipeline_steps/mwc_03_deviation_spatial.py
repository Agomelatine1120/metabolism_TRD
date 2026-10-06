"""MwC 03 — formal 26-ROI deviation maps and spatial replication."""


# Source cell 10
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

mwc_adjusted_26 = pd.read_csv(MWC_ADJUSTED_26_FILE)
mwc_adjusted_columns = [f"MwC_{name}" for name in roi.ROI]
assert len(mwc_adjusted_columns) == 26
assert set(mwc_adjusted_columns).issubset(mwc_adjusted_26.columns)
assert len(mwc_adjusted_26) == 291

mwc_baseline_26 = mwc_adjusted_26.loc[
    mwc_adjusted_26.session.eq("ses-01")
    & mwc_adjusted_26.analysis_cohort.isin(["nTRD", "TRD1", "TRD2"])
].copy()
assert mwc_baseline_26.groupby("analysis_cohort").size().to_dict() == {
    "TRD1": 35, "TRD2": 104, "nTRD": 25,
}
mwc_group_means_26 = {
    group: mwc_baseline_26.loc[
        mwc_baseline_26.analysis_cohort.eq(group), mwc_adjusted_columns
    ].mean().to_numpy(float)
    for group in ["nTRD", "TRD1", "TRD2"]
}

trd1_vector = mwc_group_means_26["TRD1"]
trd2_vector = mwc_group_means_26["TRD2"]
pearson_r, pearson_p = stats.pearsonr(trd1_vector, trd2_vector)
same_direction = float(np.mean(np.sign(trd1_vector) == np.sign(trd2_vector)))
weighted_dsc = float(
    2 * np.sum(np.minimum(np.abs(trd1_vector), np.abs(trd2_vector)))
    / np.sum(np.abs(trd1_vector) + np.abs(trd2_vector))
)
mwc_spatial_summary = pd.DataFrame([{
    "n_ROIs": 26, "cortical_ROIs": 12, "limbic_ROIs": 14,
    "TRD1_n": 35, "TRD2_n": 104,
    "pearson_r": float(pearson_r), "pearson_p_two_sided": float(pearson_p),
    "same_direction_agreement": same_direction, "DSC_weighted": weighted_dsc,
    "MwC_input": "HC-only age/sex residual z scores; 26 ROIs extracted after all-144 computation",
}])
mwc_spatial_summary.to_csv(MWC_SPATIAL_SUMMARY_FILE, index=False)

mwc_mean_rows = []
for group, values in mwc_group_means_26.items():
    for roi_row, value in zip(roi.itertuples(), values):
        mwc_mean_rows.append({
            "group": group, "network": roi_row.network,
            "AAL3_ID": int(roi_row.AAL3_ID), "ROI": roi_row.ROI,
            "mean_age_sex_adjusted_MwC_deviation_z": float(value),
        })
pd.DataFrame(mwc_mean_rows).to_csv(
    OUTPUT_DIR / "nTRD_TRD1_TRD2_normativeSC_MwC_26ROI_mean_deviations.csv",
    index=False,
)

heatmap_cmap = LinearSegmentedColormap.from_list(
    "scientific_blue_red",
    ["#2441B5", "#6FA8FD", "#D9D6D9", "#FE8869", "#BF0323"], N=256,
)
cortical_index = roi.index[roi.network.eq("cortical")].to_numpy()
limbic_index = roi.index[roi.network.eq("limbic")].to_numpy()
short_labels = [display_roi_name(name).replace("_L", "-L").replace("_R", "-R") for name in roi.ROI]
offset = len(limbic_index) - len(cortical_index)
matrices = {}
for group, values in mwc_group_means_26.items():
    matrix = np.full((2, 14), np.nan)
    matrix[0, offset:] = values[cortical_index]
    matrix[1, :] = values[limbic_index]
    matrices[group] = matrix
vmax = float(np.nanmax(np.abs(np.concatenate([value.ravel() for value in matrices.values()]))))

fig, axes = plt.subplots(3, 1, figsize=(12.0, 4.8), sharex=True)
for ax, group in zip(axes, ["nTRD", "TRD1", "TRD2"]):
    sns.heatmap(
        matrices[group], mask=np.isnan(matrices[group]), ax=ax, cmap=heatmap_cmap,
        vmin=-vmax, vmax=vmax, center=0, annot=True, fmt=".2f",
        annot_kws={"fontsize": 7}, linewidths=.5, linecolor="white",
        cbar=group == "TRD2", yticklabels=["Cortical", "Limbic"], xticklabels=False,
    )
    ax.set_title(group, loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", rotation=0)
for column, index in enumerate(cortical_index, start=offset):
    axes[0].text(column + .5, -.10, short_labels[index], rotation=25,
                 ha="left", va="bottom", fontsize=8, clip_on=False)
axes[0].set_xticks(np.arange(14) + .5)
axes[0].set_xticklabels([short_labels[i] for i in limbic_index], rotation=25, ha="left")
fig.subplots_adjust(left=.10, right=.94, top=.92, bottom=.16, hspace=.32)
fig.savefig(MWC_DEVIATION_HEATMAP, dpi=300, facecolor="white")
fig.savefig(MWC_DEVIATION_HEATMAP.with_suffix(".pdf"), facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 5.0))
ax.scatter(trd1_vector, trd2_vector, s=46, color="#447AEC", edgecolor="white", linewidth=.6)
fit = np.polyfit(trd1_vector, trd2_vector, 1)
xline = np.linspace(trd1_vector.min(), trd1_vector.max(), 100)
ax.plot(xline, np.polyval(fit, xline), color="#2441B5", linewidth=2)
for x, y, label in zip(trd1_vector, trd2_vector, short_labels):
    ax.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
ax.set_xlabel("TRD1 mean MwC deviation (HC z)")
ax.set_ylabel("TRD2 mean MwC deviation (HC z)")
ax.set_title(f"26-ROI spatial replication: r = {pearson_r:.3f}, p = {pearson_p:.3g}")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(MWC_SPATIAL_FIGURE, dpi=300, facecolor="white")
fig.savefig(MWC_SPATIAL_FIGURE.with_suffix(".pdf"), facecolor="white")
plt.close(fig)

assert int(mwc_spatial_summary.iloc[0].n_ROIs) == 26
display(mwc_spatial_summary.round(6))
display(Image(filename=str(MWC_DEVIATION_HEATMAP), width=1800))
display(Image(filename=str(MWC_SPATIAL_FIGURE), width=900))
