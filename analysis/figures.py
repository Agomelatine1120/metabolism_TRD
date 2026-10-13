from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from core import FIGURES, TABLES, global_index_results, load_data, spatial_results


BLUE = "#6FB6E8"
ORANGE = "#E8B06A"
GREEN = "#0AA17A"
PURPLE = "#A34BA8"
CMAP = LinearSegmentedColormap.from_list("blue_white_red", ["#2459B8", "#F7F7F7", "#C43A3A"])


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def strip_heatmap(groups, labels, name):
    means = np.vstack([np.nanmean(groups[key], axis=0) for key in ["nTRD", "TRD1", "TRD2"]])
    fig, axes = plt.subplots(3, 1, figsize=(11.4, 4.9), gridspec_kw={"hspace": .42})
    limit = max(1.0, np.nanmax(np.abs(means)))
    for ax, row, title in zip(axes, means, ["nTRD", "TRD1", "TRD2"]):
        matrix = np.full((2, 14), np.nan)
        matrix[0, 2:] = row[:12]
        matrix[1, :] = row[12:]
        sns.heatmap(matrix, ax=ax, cmap=CMAP, center=0, vmin=-limit, vmax=limit, cbar=False,
                    mask=~np.isfinite(matrix), linewidths=.7, linecolor="white",
                    xticklabels=False, yticklabels=False)
        ax.text(-.10, 1.14, title, transform=ax.transAxes, ha="left", va="center", fontsize=10, fontweight="bold")
        ax.text(-.10, .75, "Frontal", transform=ax.transAxes, ha="left", va="center", fontsize=8)
        ax.text(-.10, .25, "Limbic", transform=ax.transAxes, ha="left", va="center", fontsize=8)
        for x, label in zip(np.arange(2, 14) + .5, labels[:12]):
            ax.text(x, 1.03, label, rotation=25, ha="left", va="bottom", fontsize=6.2,
                    transform=ax.get_xaxis_transform())
        for x, label in zip(np.arange(14) + .5, labels[12:]):
            ax.text(x, -.10, label, rotation=25, ha="right", va="top", fontsize=6.2,
                    transform=ax.get_xaxis_transform())
        ax.tick_params(left=False, bottom=False)
    fig.subplots_adjust(left=.12, right=.96, top=.96, bottom=.04)
    color_axis = fig.add_axes([.972, .12, .012, .30])
    plt.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), cmap=CMAP), cax=color_axis)
    color_axis.tick_params(labelsize=6)
    save(fig, name)


def connectivity_heatmaps(groups, frontal, limbic):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.35), constrained_layout=True)
    means = [np.nanmean(groups[key], axis=0).reshape(12, 14) for key in ["nTRD", "TRD1", "TRD2"]]
    limit = max(1.0, max(np.nanmax(np.abs(value)) for value in means))
    for ax, matrix, title in zip(axes, means, ["nTRD", "TRD1", "TRD2"]):
        sns.heatmap(matrix, ax=ax, cmap=CMAP, center=0, vmin=-limit, vmax=limit, cbar=ax is axes[-1],
                    annot=True, fmt=".2f", annot_kws={"fontsize": 3.8}, linewidths=.2, linecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_yticklabels(frontal, rotation=0, fontsize=6)
        ax.set_xticklabels(limbic, rotation=45, ha="right", fontsize=6)
    save(fig, "metabolic_connectivity_deviation_matrices")


def circle_panel(ax, node_values, edge_values, labels, title):
    n = len(labels)
    angles = np.linspace(np.pi * .1, np.pi * 1.9, n, endpoint=False)
    xy = np.column_stack([np.cos(angles), np.sin(angles)])
    edge_matrix = np.zeros((n, n))
    k = 0
    for i in range(12):
        for j in range(12, 26):
            edge_matrix[i, j] = edge_matrix[j, i] = edge_values[k]
            k += 1
    limit = max(np.nanmax(np.abs(edge_values)), 1e-8)
    for i in range(n):
        for j in range(i + 1, n):
            value = edge_matrix[i, j]
            if value == 0 or not np.isfinite(value):
                continue
            color = "#2B6DAA" if value < 0 else "#C74343"
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=color,
                    alpha=.08 + .55 * abs(value) / limit, lw=.25 + 1.5 * abs(value) / limit, zorder=1)
    node_limit = max(np.nanmax(np.abs(node_values)), 1e-8)
    colors = [CMAP((value / node_limit + 1) / 2) for value in node_values]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=55, edgecolor="white", linewidth=.6, zorder=3)
    for (x, y), label, angle in zip(xy, labels, angles):
        r = 1.15
        ax.text(r*x, r*y, label, fontsize=5.6, rotation=np.degrees(angle)-90,
                ha="center", va="center", rotation_mode="anchor")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")


def circles(data):
    meta = data["meta"]
    baseline = meta.session.eq("ses-01")
    masks = {name: baseline & meta.analysis_cohort.eq(name) for name in ["nTRD", "TRD1", "TRD2"]}
    panels = [
        ("nTRD deviation", np.nanmean(data["regional"][masks["nTRD"]], 0), np.nanmean(data["edge"][masks["nTRD"]], 0)),
        ("TRD1 deviation", np.nanmean(data["regional"][masks["TRD1"]], 0), np.nanmean(data["edge"][masks["TRD1"]], 0)),
        ("TRD2 deviation", np.nanmean(data["regional"][masks["TRD2"]], 0), np.nanmean(data["edge"][masks["TRD2"]], 0)),
        ("TRD1 − nTRD", np.nanmean(data["regional"][masks["TRD1"]], 0)-np.nanmean(data["regional"][masks["nTRD"]], 0),
         np.nanmean(data["edge"][masks["TRD1"]], 0)-np.nanmean(data["edge"][masks["nTRD"]], 0)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1))
    for ax, (title, nodes, edges) in zip(axes, panels): circle_panel(ax, nodes, edges, data["roi_labels"], title)
    save(fig, "corticolimbic_circular_summary")

    longitudinal = []
    deltas = {}
    for arm in ["rTMS", "iTBS", "sham"]:
        selected = meta.analysis_cohort.eq("TRD2") & meta.analysis_arm.eq(arm)
        frame = meta.loc[selected]
        pre_ids = frame.loc[frame.session.eq("ses-01")].sort_values("participant_id").index
        post_ids = frame.loc[frame.session.eq("ses-02")].sort_values("participant_id").index
        reg = data["regional"][post_ids] - data["regional"][pre_ids]
        edge = data["edge"][post_ids] - data["edge"][pre_ids]
        deltas[arm] = (reg, edge)
        longitudinal.append((f"{arm} post − pre", np.nanmean(reg, 0), np.nanmean(edge, 0)))
    active_reg = np.vstack([deltas["rTMS"][0], deltas["iTBS"][0]])
    active_edge = np.vstack([deltas["rTMS"][1], deltas["iTBS"][1]])
    longitudinal.append(("Active − sham", np.nanmean(active_reg,0)-np.nanmean(deltas["sham"][0],0),
                         np.nanmean(active_edge,0)-np.nanmean(deltas["sham"][1],0)))
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1))
    for ax, (title, nodes, edges) in zip(axes, longitudinal): circle_panel(ax, nodes, edges, data["roi_labels"], title)
    save(fig, "corticolimbic_circular_longitudinal_summary")


def longitudinal_panels(data):
    tests = pd.read_csv(TABLES / "longitudinal_all_660_tests.csv")
    selected = tests.loc[
        tests.modality.eq("Metabolic connectivity") & tests.analysis_arm.isin(["rTMS", "iTBS"]) & tests.paired_p_lt_0p01
    ].sort_values(["analysis_arm", "paired_p_uncorrected"])
    meta = data["meta"]
    count = len(selected)
    fig, axes = plt.subplots(1, count, figsize=(1.9*count, 3.25), squeeze=False)
    corr_fig, corr_axes = plt.subplots(1, count, figsize=(1.9*count, 3.25), squeeze=False)
    for axis, caxis, row in zip(axes[0], corr_axes[0], selected.itertuples(index=False)):
        arm = row.analysis_arm
        mask = meta.analysis_cohort.eq("TRD2") & meta.analysis_arm.eq(arm)
        frame = meta.loc[mask]
        pre_idx = frame.loc[frame.session.eq("ses-01")].sort_values("participant_id").index
        post_idx = frame.loc[frame.session.eq("ses-02")].sort_values("participant_id").index
        x = data["edge"][pre_idx, row.feature_index]; y = data["edge"][post_idx, row.feature_index]
        color = BLUE if arm == "iTBS" else ORANGE
        for before, after in zip(x, y): axis.plot([0,1],[before,after], color="#B7B7B7", lw=.5, alpha=.55)
        axis.scatter(np.zeros(len(x)), x, s=10, color=color, alpha=.8)
        axis.scatter(np.ones(len(y)), y, s=10, color=color, alpha=.8)
        axis.set_xticks([0,1],["Pre","Post"], fontsize=6); axis.tick_params(labelsize=6)
        axis.set_title(f"{arm}\n{row.feature}\n$p$={row.paired_p_uncorrected:.3f}", fontsize=7)
        delta = y-x; hdrs = meta.loc[pre_idx,"hdrs_improvement"].to_numpy(float)
        caxis.scatter(delta, hdrs, s=12, facecolor="white", edgecolor=GREEN, linewidth=.8)
        valid=np.isfinite(delta)&np.isfinite(hdrs)
        if valid.sum()>1:
            slope,intercept=np.polyfit(delta[valid],hdrs[valid],1);xx=np.linspace(delta[valid].min(),delta[valid].max(),50)
            caxis.plot(xx,intercept+slope*xx,color=GREEN,lw=1.2)
        caxis.set_title(f"r={row.HDRS_partial_Spearman_r_age_sex:.2f}\n$p$={row.HDRS_p_uncorrected:.3f}",fontsize=7)
        caxis.tick_params(labelsize=6)
    save(fig, "TRD2_prepost_raincloud_paired_p_lt_0p01")
    save(corr_fig, "TRD2_prepost_HDRS_correlation_paired_p_lt_0p01")


def spatial_plot(data):
    summary = spatial_results(data)
    meta=data["meta"];trd1=meta.analysis_cohort.eq("TRD1")&meta.session.eq("ses-01");trd2=meta.analysis_cohort.eq("TRD2")&meta.session.eq("ses-01")
    fig,axes=plt.subplots(1,3,figsize=(9,3))
    for ax,(modality,values),row in zip(axes,[("Regional SUVR",data["regional"]),("Metabolic connectivity",data["edge"]),("MwC",data["mwc_z"])],summary.itertuples(index=False)):
        x=np.nanmean(values[trd1],0);y=np.nanmean(values[trd2],0);valid=np.isfinite(x)&np.isfinite(y)
        ax.scatter(x[valid],y[valid],s=18,alpha=.75,color=BLUE,edgecolor="white",linewidth=.3)
        slope,intercept=np.polyfit(x[valid],y[valid],1);xx=np.linspace(x[valid].min(),x[valid].max(),50);ax.plot(xx,intercept+slope*xx,color=ORANGE)
        ax.set_title(f"{modality}\nr={row.pearson_r:.2f}",fontsize=9);ax.set_xlabel("TRD1");ax.set_ylabel("TRD2")
    save(fig,"spatial_correlation_scatter")


def _raincloud_axis(ax, arrays, colors, title, p_value, seed):
    rng = np.random.default_rng(seed)
    for position, (label, raw) in enumerate(arrays.items()):
        values = np.asarray(raw, float)
        values = values[np.isfinite(values)]
        violin = ax.violinplot(values, positions=[position + .16], widths=.48, showextrema=False)
        body = violin["bodies"][0]
        body.set_facecolor(colors[label]); body.set_edgecolor("none"); body.set_alpha(.48)
        ax.scatter(position - .14 + rng.normal(0, .035, len(values)), values,
                   s=9, color=colors[label], alpha=.75, edgecolor="none")
        ax.boxplot(values, positions=[position], widths=.10, patch_artist=True, showfliers=False,
                   boxprops={"facecolor":"white", "edgecolor":"#555555", "linewidth":.7},
                   medianprops={"color":"#222222", "linewidth":.9},
                   whiskerprops={"color":"#555555", "linewidth":.7},
                   capprops={"color":"#555555", "linewidth":.7})
    combined = np.concatenate([np.asarray(value, float) for value in arrays.values()])
    combined = combined[np.isfinite(combined)]
    span = max(float(np.ptp(combined)), .5)
    top = float(combined.max()) + .20 * span
    ax.plot([0,0,1,1], [top, top+.03*span, top+.03*span, top], color="#555555", lw=.7)
    label = "$p<.001$" if p_value < .001 else f"$p={p_value:.3f}$"
    ax.text(.5, top+.05*span, label, ha="center", va="bottom", fontsize=7)
    ax.set_ylim(float(combined.min())-.12*span, float(combined.max())+.34*span)
    ax.set_xticks([0,1], list(arrays), fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)


def global_index_figures(data):
    metrics, group_tests, paired_tests = global_index_results(data)
    specifications = [
        ("MBI_z", "MBI"), ("D_SUVR", "$D_{SUVR}$"), ("D_MC", "$D_{MC}$"),
        ("D_MwC", "$D_{MwC}$"), ("D_metabolic_signature", r"$D_{metabolic\ signature}$"),
    ]
    baseline = metrics.loc[metrics.session.eq("ses-01")]
    fig, axes = plt.subplots(1, 5, figsize=(12.35, 3.4))
    for index, (ax, (metric, title)) in enumerate(zip(axes, specifications)):
        arrays = {
            "nTRD": baseline.loc[baseline.analysis_cohort.eq("nTRD"), metric].to_numpy(float),
            "TRD": baseline.loc[baseline.analysis_cohort.eq("TRD1"), metric].to_numpy(float),
        }
        p = float(group_tests.loc[group_tests.metric.eq(metric), "p_uncorrected"].iloc[0])
        _raincloud_axis(ax, arrays, {"nTRD":GREEN, "TRD":PURPLE}, title, p, 20261010+index)
    fig.subplots_adjust(left=.05, right=.995, bottom=.15, top=.84, wspace=.38)
    save(fig, "Figure_five_metric_nTRD_TRD_raincloud")

    active = metrics.loc[metrics.analysis_cohort.eq("TRD2") & metrics.analysis_arm.isin(["rTMS","iTBS"])]
    fig, axes = plt.subplots(1, 5, figsize=(12.35, 3.4))
    for index, (ax, (metric, title)) in enumerate(zip(axes, specifications)):
        wide = active.pivot(index="participant_id", columns="session", values=metric).dropna()
        arrays = {"Pre":wide["ses-01"].to_numpy(float), "Post":wide["ses-02"].to_numpy(float)}
        p = float(paired_tests.loc[paired_tests.metric.eq(metric), "p_uncorrected"].iloc[0])
        _raincloud_axis(ax, arrays, {"Pre":BLUE, "Post":ORANGE}, title, p, 20261110+index)
    fig.subplots_adjust(left=.05, right=.995, bottom=.15, top=.84, wspace=.38)
    save(fig, "Figure_five_metric_active_pre_post_raincloud")


def run_all_figures():
    sns.set_theme(style="whitegrid", context="paper")
    data=load_data();meta=data["meta"];baseline=meta.session.eq("ses-01")
    groups={cohort:data["regional"][baseline&meta.analysis_cohort.eq(cohort)] for cohort in ["nTRD","TRD1","TRD2"]}
    strip_heatmap(groups,data["roi_labels"],"regional_SUVR_deviation_heatmap")
    groups={cohort:data["mwc_z"][baseline&meta.analysis_cohort.eq(cohort)] for cohort in ["nTRD","TRD1","TRD2"]}
    strip_heatmap(groups,data["roi_labels"],"MwC_deviation_heatmap")
    groups={cohort:data["edge"][baseline&meta.analysis_cohort.eq(cohort)] for cohort in ["nTRD","TRD1","TRD2"]}
    connectivity_heatmaps(groups,data["roi_labels"][:12],data["roi_labels"][12:])
    circles(data);longitudinal_panels(data);spatial_plot(data);global_index_figures(data)
    from ten_edge_direction_and_clinical import main as ten_edge_main
    ten_edge_main()
    return sorted(str(path.relative_to(FIGURES)) for path in FIGURES.rglob("*.png"))


if __name__ == "__main__":
    print(run_all_figures())
