from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import itertools
import json
import math
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
OUTPUT = ROOT / "outputs"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
MI_FLOOR_BITS = 0.01
PET_SCALE = 100.0
SPATIAL_PERMUTATION_SEED = 20260830
N_SPATIAL_PERMUTATIONS = 100_000
PERMUTATION_BATCH_SIZE = 1_000

ALIASES = {
    "Frontal_Sup_2": "SFG", "Frontal_Mid_2": "MFG",
    "Frontal_Sup_Medial": "dmPFC", "Frontal_Med_Orb": "vmPFC",
    "OFCmed": "mOFC", "OFClat": "lOFC", "Hippocampus": "HPC",
    "ParaHippocampal": "PHG", "Amygdala": "AMY", "Caudate": "CAU",
    "Putamen": "PUT", "Pallidum": "PAL", "ACC_sub": "SGC",
}


def roi_label(name: str) -> str:
    match = re.fullmatch(r"(.+)_([LR])", name)
    if not match:
        return name
    region, side = match.groups()
    return f"{'Lt.' if side == 'L' else 'Rt.'}{ALIASES.get(region, region)}"


def partial_corr_matrix(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), covariates])
    beta = np.linalg.pinv(design.T @ design) @ design.T @ values
    residuals = values - design @ beta
    return np.clip(np.corrcoef(residuals, rowvar=False), -1.0, 1.0)


def partial_spearman(x, y, age, sex):
    x, y, age, sex = map(lambda v: np.asarray(v, dtype=float), (x, y, age, sex))
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(age) & np.isfinite(sex)
    x, y, age, sex = x[valid], y[valid], age[valid], sex[valid]
    n = len(x)
    if n < 6:
        return np.nan, np.nan, n
    covariates = np.column_stack([np.ones(n), stats.rankdata(age), stats.rankdata(sex)])
    projection = np.eye(n) - covariates @ np.linalg.pinv(covariates)
    xr = projection @ stats.rankdata(x)
    yr = projection @ stats.rankdata(y)
    r = float(np.clip(np.corrcoef(xr, yr)[0, 1], -0.999999999, 0.999999999))
    df = n - 4
    t = r * np.sqrt(df / (1.0 - r * r))
    return r, float(2 * stats.t.sf(abs(t), df)), n


def calculate_mwc(gcmi: np.ndarray, pet: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weighted = np.asarray(gcmi, dtype=float).copy()
    weighted[weighted < MI_FLOOR_BITS] = 0.0
    np.fill_diagonal(weighted, 0.0)
    weighted[~mask] = 0.0
    row_sums = weighted.sum(axis=1)
    valid = row_sums > 0
    weights = np.divide(weighted, row_sums[:, None], out=np.zeros_like(weighted), where=row_sums[:, None] > 0)
    result = PET_SCALE * (weights @ np.asarray(pet, dtype=float))
    result[~valid] = np.nan
    return result


def fit_reference(values: np.ndarray, age: np.ndarray, sex: np.ndarray):
    design = np.column_stack([np.ones(len(values)), age, sex])
    beta = np.full((3, values.shape[1]), np.nan)
    sd = np.full(values.shape[1], np.nan)
    for index in range(values.shape[1]):
        valid = np.isfinite(values[:, index]) & np.isfinite(age) & np.isfinite(sex)
        beta[:, index] = np.linalg.pinv(design[valid]) @ values[valid, index]
        residual = values[valid, index] - design[valid] @ beta[:, index]
        sd[index] = residual.std(ddof=1)
    return beta, sd


def reference_deviation(values, age, sex, beta, sd):
    design = np.column_stack([np.ones(len(values)), age, sex])
    return (values - design @ beta) / sd


def loo_reference_deviation(values, age, sex):
    output = np.full_like(values, np.nan, dtype=float)
    for index in range(len(values)):
        keep = np.arange(len(values)) != index
        beta, sd = fit_reference(values[keep], age[keep], sex[keep])
        output[index] = reference_deviation(values[[index]], age[[index]], sex[[index]], beta, sd)[0]
    return output


def edge_deviation(target, target_age, target_sex, hc, hc_age, hc_sex, cortical, limbic):
    positions = {name: index for index, name in enumerate(cortical + limbic)}
    pairs = [(first, second) for first in cortical for second in limbic]
    triangle = np.triu_indices(len(positions), 1)
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(*triangle))}
    edge_ids = [lookup[tuple(sorted((positions[a], positions[b])))] for a, b in pairs]
    covariates = np.column_stack([hc_age, hc_sex])
    reference = partial_corr_matrix(hc, covariates)
    denominator = (1.0 - reference[triangle] ** 2) / (len(hc) - 1)
    denominator[np.abs(denominator) < 1e-12] = 1e-12
    output = []
    for values, age, sex in zip(target, target_age, target_sex):
        perturbed = partial_corr_matrix(
            np.vstack([hc, values]),
            np.vstack([covariates, [age, sex]]),
        )
        output.append(((perturbed - reference)[triangle] / denominator)[edge_ids])
    return np.asarray(output), pairs


def loo_edge_deviation(hc, age, sex, cortical, limbic):
    output = []
    for index in range(len(hc)):
        keep = np.arange(len(hc)) != index
        values, _ = edge_deviation(hc[[index]], age[[index]], sex[[index]], hc[keep], age[keep], sex[keep], cortical, limbic)
        output.append(values[0])
    return np.asarray(output)


@lru_cache(maxsize=1)
def load_data() -> dict:
    roi = pd.read_csv(INPUT / "corticolimbic_12x14_ROI_definition.csv")
    with np.load(INPUT / "session_pet_gcmi_AAL3_140.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(INPUT / "normative_SC_top10_mask.npz", allow_pickle=False) as archive:
        mask = np.asarray(archive["sc_top10_mask"], dtype=bool)
    if arrays["pet_suvr"].shape != (321, 140) or arrays["gcmi_bits"].shape != (321, 140, 140):
        raise ValueError("Expected 321 sessions in the 140-ROI analysis space")
    if mask.shape != (140, 140) or np.count_nonzero(np.triu(mask, 1)) != 973:
        raise ValueError("Expected the 140-node, 973-edge normative SC mask")
    meta = pd.DataFrame({
        "participant_id": arrays["participant_id"].astype(str),
        "analysis_cohort": arrays["analysis_cohort"].astype(str),
        "analysis_arm": arrays["analysis_arm"].astype(str),
        "session": arrays["session"].astype(str),
        "age": arrays["age"].astype(float), "sex": arrays["sex"].astype(float),
        "education": arrays["education"].astype(float),
        "hdrs_pre": arrays["hdrs_pre"].astype(float),
        "hdrs_post": arrays["hdrs_post"].astype(float),
        "hdrs_improvement": arrays["hdrs_improvement"].astype(float),
        "mwc_candidate": arrays["mwc_candidate"].astype(bool),
    })
    mwc = np.full((len(meta), 140), np.nan)
    for index in np.flatnonzero(meta.mwc_candidate.to_numpy()):
        mwc[index] = calculate_mwc(arrays["gcmi_bits"][index], arrays["pet_suvr"][index], mask)
    position = {name: index for index, name in enumerate(arrays["roi_names"].astype(str))}
    target_names = roi.ROI.astype(str).tolist()
    target_positions = [position[name] for name in target_names]
    cortical = roi.loc[roi.network.eq("cortical"), "ROI"].astype(str).tolist()
    limbic = roi.loc[roi.network.eq("limbic"), "ROI"].astype(str).tolist()
    pet26 = arrays["pet_suvr"][:, target_positions]
    mwc26 = mwc[:, target_positions]

    hc = meta.analysis_cohort.eq("HC").to_numpy()
    reg_beta, reg_sd = fit_reference(pet26[hc], meta.age.to_numpy()[hc], meta.sex.to_numpy()[hc])
    regional = reference_deviation(pet26, meta.age.to_numpy(), meta.sex.to_numpy(), reg_beta, reg_sd)
    regional[hc] = loo_reference_deviation(pet26[hc], meta.age.to_numpy()[hc], meta.sex.to_numpy()[hc])
    mwc_beta, mwc_sd = fit_reference(mwc26[hc], meta.age.to_numpy()[hc], meta.sex.to_numpy()[hc])
    mwc_z = reference_deviation(mwc26, meta.age.to_numpy(), meta.sex.to_numpy(), mwc_beta, mwc_sd)
    mwc_z[hc] = loo_reference_deviation(mwc26[hc], meta.age.to_numpy()[hc], meta.sex.to_numpy()[hc])
    edge, edge_names = edge_deviation(
        pet26, meta.age.to_numpy(), meta.sex.to_numpy(), pet26[hc],
        meta.age.to_numpy()[hc], meta.sex.to_numpy()[hc], cortical, limbic,
    )
    edge[hc] = loo_edge_deviation(
        pet26[hc], meta.age.to_numpy()[hc], meta.sex.to_numpy()[hc], cortical, limbic,
    )
    return {
        "roi": roi, "meta": meta, "regional": regional, "edge": edge,
        "mwc": mwc, "mwc_z": mwc_z, "edge_names": edge_names,
        "roi_names": target_names, "roi_labels": [roi_label(name) for name in target_names],
        "edge_labels": [f"{roi_label(a)}–{roi_label(b)}" for a, b in edge_names],
    }


def feature_model(values, metadata, names, modality, sensitivity=False):
    rows = []
    for index, name in enumerate(names):
        frame = metadata[["analysis_cohort", "age", "sex", "hdrs_pre"]].copy()
        frame["outcome"] = values[:, index]
        required = ["outcome", "age", "sex"] + (["hdrs_pre"] if sensitivity else [])
        frame = frame.dropna(subset=required)
        frame["group_TRD1"] = frame.analysis_cohort.eq("TRD1").astype(float)
        predictors = ["group_TRD1", "age", "sex"] + (["hdrs_pre"] if sensitivity else [])
        fit = sm.OLS(frame.outcome, sm.add_constant(frame[predictors], has_constant="add")).fit()
        low, high = fit.conf_int().loc["group_TRD1"]
        t = float(fit.tvalues["group_TRD1"])
        rows.append({
            "modality": modality, "feature_index": index, "feature": name,
            "model": "age + sex + baseline HDRS-17" if sensitivity else "age + sex",
            "n": int(fit.nobs), "n_nTRD": int(frame.analysis_cohort.eq("nTRD").sum()),
            "n_TRD1": int(frame.analysis_cohort.eq("TRD1").sum()),
            "nTRD_mean_deviation": float(frame.loc[frame.analysis_cohort.eq("nTRD"), "outcome"].mean()),
            "TRD1_mean_deviation": float(frame.loc[frame.analysis_cohort.eq("TRD1"), "outcome"].mean()),
            "TRD1_minus_nTRD_beta": float(fit.params["group_TRD1"]),
            "CI95_low": float(low), "CI95_high": float(high), "t": t,
            "df_resid": float(fit.df_resid),
            "partial_r": float(np.sign(t) * np.sqrt(t * t / (t * t + fit.df_resid))),
            "p_uncorrected": float(fit.pvalues["group_TRD1"]),
        })
    result = pd.DataFrame(rows)
    result["p_lt_0p01"] = result.p_uncorrected.lt(.01)
    result["p_lt_0p05"] = result.p_uncorrected.lt(.05)
    return result.sort_values("p_uncorrected", kind="stable").reset_index(drop=True)


def baseline_results(data=None):
    data = load_data() if data is None else data
    meta = data["meta"]
    select = meta.analysis_cohort.isin(["nTRD", "TRD1"]) & meta.session.eq("ses-01")
    indices = np.flatnonzero(select)
    metadata = meta.iloc[indices].reset_index(drop=True)
    families = {
        "Regional SUVR": (data["regional"][indices], data["roi_labels"]),
        "Metabolic connectivity": (data["edge"][indices], data["edge_labels"]),
        "MwC": (data["mwc_z"][indices], data["roi_labels"]),
    }
    primary = {name: feature_model(values, metadata, labels, name, False) for name, (values, labels) in families.items()}
    sensitivity = {name: feature_model(values, metadata, labels, name, True) for name, (values, labels) in families.items()}
    return primary, sensitivity


def _aligned_arm(meta, values, arm):
    selected = meta.analysis_cohort.eq("TRD2") & meta.analysis_arm.eq(arm)
    frame = meta.loc[selected].copy()
    current = values[selected]
    pre_order = frame.loc[frame.session.eq("ses-01")].sort_values("participant_id").index
    post_order = frame.loc[frame.session.eq("ses-02")].sort_values("participant_id").index
    if meta.loc[pre_order, "participant_id"].tolist() != meta.loc[post_order, "participant_id"].tolist():
        raise ValueError("Pre/post participant mismatch")
    return meta.loc[pre_order].reset_index(drop=True), values[pre_order], values[post_order]


def longitudinal_results(data=None):
    data = load_data() if data is None else data
    meta = data["meta"]
    raw_roi = data["roi"].analysis_roi.astype(str).tolist()
    raw_lookup = dict(zip(data["roi_names"], raw_roi))
    families = {
        "Regional SUVR": (data["regional"], data["roi_labels"], [(name, name) for name in raw_roi]),
        "Metabolic connectivity": (data["edge"], data["edge_labels"], [(raw_lookup[a], raw_lookup[b]) for a, b in data["edge_names"]]),
        "MwC": (data["mwc_z"], [f"{label} MwC" for label in data["roi_labels"]], [(name, name) for name in raw_roi]),
    }
    rows = []
    for modality, (values, labels, roi_pairs) in families.items():
        arm_data = {arm: _aligned_arm(meta, values, arm) for arm in ["rTMS", "iTBS", "sham"]}
        for arm, (metadata, pre, post) in arm_data.items():
            for index, label in enumerate(labels):
                valid = np.isfinite(pre[:, index]) & np.isfinite(post[:, index])
                x, y = pre[valid, index], post[valid, index]
                delta = y - x
                paired = stats.ttest_rel(y, x)
                paired_sem = stats.sem(delta)
                paired_low, paired_high = stats.t.interval(.95, len(delta)-1, loc=delta.mean(), scale=paired_sem)
                roi_1, roi_2 = roi_pairs[index]
                row = {
                    "modality": modality, "analysis_arm": arm, "feature_index": index,
                    "ROI_1": roi_1, "ROI_2": roi_2, "feature": label, "n": len(delta), "pre_mean_deviation": float(x.mean()),
                    "pre_SD_deviation": float(x.std(ddof=1)), "post_mean_deviation": float(y.mean()),
                    "post_SD_deviation": float(y.std(ddof=1)), "post_minus_pre": float(delta.mean()),
                    "post_minus_pre_CI95_low": float(paired_low), "post_minus_pre_CI95_high": float(paired_high),
                    "paired_t": float(paired.statistic), "paired_df": len(delta)-1,
                    "effect_size_Cohen_dz": float(delta.mean()/delta.std(ddof=1)),
                    "paired_p_uncorrected": float(paired.pvalue),
                }
                if arm in ["rTMS", "iTBS"]:
                    sham_meta, sham_pre, sham_post = arm_data["sham"]
                    sham_valid = np.isfinite(sham_pre[:, index]) & np.isfinite(sham_post[:, index])
                    sham_delta = sham_post[sham_valid, index] - sham_pre[sham_valid, index]
                    frame = pd.DataFrame({
                        "change": np.r_[delta, sham_delta],
                        "group": np.r_[np.ones(len(delta)), np.zeros(len(sham_delta))],
                        "age": np.r_[metadata.loc[valid, "age"], sham_meta.loc[sham_valid, "age"]],
                        "sex": np.r_[metadata.loc[valid, "sex"], sham_meta.loc[sham_valid, "sex"]],
                    }).dropna()
                    fit = sm.OLS(frame.change, sm.add_constant(frame[["group", "age", "sex"]], has_constant="add")).fit()
                    interaction_low, interaction_high = fit.conf_int().loc["group"]
                    interaction_t = float(fit.tvalues["group"])
                    r, p, n_used = partial_spearman(
                        delta, metadata.loc[valid, "hdrs_improvement"], metadata.loc[valid, "age"], metadata.loc[valid, "sex"]
                    )
                    row.update({
                        "active_n": len(delta), "active_post_minus_pre": float(delta.mean()),
                        "active_Cohen_dz": float(delta.mean()/delta.std(ddof=1)),
                        "sham_n": len(sham_delta), "sham_post_minus_pre": float(sham_delta.mean()),
                        "sham_Cohen_dz": float(sham_delta.mean()/sham_delta.std(ddof=1)),
                        "interaction_beta_active_minus_sham": float(fit.params["group"]),
                        "interaction_CI95_low": float(interaction_low), "interaction_CI95_high": float(interaction_high),
                        "interaction_t": interaction_t, "interaction_df_resid": float(fit.df_resid),
                        "interaction_effect_size_partial_r": float(np.sign(interaction_t) * np.sqrt(interaction_t**2/(interaction_t**2+fit.df_resid))),
                        "interaction_p_uncorrected": float(fit.pvalues["group"]),
                        "interaction_n": int(fit.nobs),
                        "HDRS_partial_Spearman_r_age_sex": r,
                        "HDRS_p_uncorrected": p, "n_used": n_used,
                    })
                rows.append(row)
    result = pd.DataFrame(rows)
    result["paired_p_lt_0p01"] = result.paired_p_uncorrected.lt(.01)
    result["paired_p_lt_0p05"] = result.paired_p_uncorrected.lt(.05)
    result["imaging_change_definition"] = "post minus pre HC-referenced deviation"
    result["HDRS_improvement_definition"] = "pre minus post"
    return result


def _rowwise_pearson(x, y):
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def _point_spatial_metrics(first, second):
    pearson = stats.pearsonr(first, second)
    same = ((first > 0) & (second > 0)) | ((first < 0) & (second < 0))
    absolute_first, absolute_second = np.abs(first), np.abs(second)
    weighted = 2 * np.sum(np.minimum(absolute_first, absolute_second) * same) / (absolute_first.sum() + absolute_second.sum())
    return {
        "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "same_direction_proportion": float(same.mean()), "same_direction_count": int(same.sum()),
        "weighted_signed_dsc": float(weighted),
    }


def _permutation_spatial_metrics(first, permuted_second):
    first_rows = np.broadcast_to(first, permuted_second.shape)
    same = ((first_rows > 0) & (permuted_second > 0)) | ((first_rows < 0) & (permuted_second < 0))
    absolute_first, absolute_second = np.abs(first_rows), np.abs(permuted_second)
    weighted = 2 * (np.minimum(absolute_first, absolute_second) * same).sum(axis=1) / (absolute_first.sum(axis=1) + absolute_second.sum(axis=1))
    return _rowwise_pearson(first_rows, permuted_second), same.mean(axis=1), weighted


def _expand_hemisphere_pairs(pair_permutations):
    return (2 * pair_permutations[..., :, None] + np.array([0, 1], dtype=int)).reshape(len(pair_permutations), -1)


def _regional_spatial_null(first, second, seed):
    rng = np.random.default_rng(seed)
    chunks = [[], [], []]
    completed = 0
    pair_count = len(second) // 2
    while completed < N_SPATIAL_PERMUTATIONS:
        size = min(PERMUTATION_BATCH_SIZE, N_SPATIAL_PERMUTATIONS - completed)
        orders = _expand_hemisphere_pairs(np.argsort(rng.random((size, pair_count)), axis=1))
        values = _permutation_spatial_metrics(first, second[orders])
        for index, current in enumerate(values):
            chunks[index].append(current)
        completed += size
    return tuple(np.concatenate(value) for value in chunks)


def _connectivity_spatial_null(first, second):
    matrix = np.asarray(second, float).reshape(12, 14)
    row_orders = _expand_hemisphere_pairs(np.asarray(list(itertools.permutations(range(6))), dtype=np.int16))
    column_orders = _expand_hemisphere_pairs(np.asarray(list(itertools.permutations(range(7))), dtype=np.int16))
    total = math.factorial(6) * math.factorial(7)
    output = [np.empty(total, float) for _ in range(3)]
    completed = 0
    for row_order in row_orders:
        permuted = np.transpose(matrix[row_order, :][:, column_orders], (1, 0, 2)).reshape(len(column_orders), -1)
        values = _permutation_spatial_metrics(first, permuted)
        end = completed + len(column_orders)
        for index, current in enumerate(values):
            output[index][completed:end] = current
        completed = end
    return tuple(output)


def spatial_results(data=None):
    data = load_data() if data is None else data
    meta = data["meta"]
    trd1 = meta.analysis_cohort.eq("TRD1") & meta.session.eq("ses-01")
    trd2 = meta.analysis_cohort.eq("TRD2") & meta.session.eq("ses-01")
    rows = []
    families = [("Regional", "Regional SUVR", data["regional"]), ("Connectivity", "Metabolic edge", data["edge"]), ("MwC", "MwC", data["mwc_z"])]
    for index, (modality, display, values) in enumerate(families):
        left = np.nanmean(values[trd1], axis=0)
        right = np.nanmean(values[trd2], axis=0)
        valid = np.isfinite(left) & np.isfinite(right)
        left, right = left[valid], right[valid]
        point = _point_spatial_metrics(left, right)
        null = _connectivity_spatial_null(left, right) if modality == "Connectivity" else _regional_spatial_null(left, right, SPATIAL_PERMUTATION_SEED + index)
        exact = modality == "Connectivity"
        empirical = (np.count_nonzero(null[0] >= point["pearson_r"]) + (0 if exact else 1)) / (len(null[0]) + (0 if exact else 1))
        rows.append({
            "modality": modality, "display_name": display, "n_features": int(valid.sum()), **point,
            "pearson_p_empirical_one_sided": float(empirical),
            "permutation_resamples": len(null[0]), "permutation_exact": exact,
        })
    return pd.DataFrame(rows)


def global_index_results(data=None):
    """Build the five subject-level global indices and their formal tests."""
    data = load_data() if data is None else data
    meta = data["meta"].copy().reset_index(drop=True)
    metrics = meta[[
        "participant_id", "analysis_cohort", "analysis_arm", "session",
        "age", "sex", "education",
    ]].copy()
    regional = np.asarray(data["regional"], float)
    edge = np.asarray(data["edge"], float)
    mwc = np.asarray(data["mwc_z"], float)
    metrics["MBI_z"] = np.nanmean(regional[:, 12:], axis=1) - np.nanmean(regional[:, :12], axis=1)
    metrics["D_SUVR"] = np.sqrt(np.nanmean(regional ** 2, axis=1))
    metrics["D_MC"] = np.sqrt(np.nanmean(edge ** 2, axis=1))
    # The formal definition is complete-case across all 26 MwC ROIs.
    metrics["D_MwC"] = np.sqrt(np.mean(mwc ** 2, axis=1))
    hc = metrics.analysis_cohort.eq("HC")
    for domain in ["SUVR", "MC", "MwC"]:
        column = f"D_{domain}"
        reference = metrics.loc[hc, column]
        metrics[f"ZD_{domain}"] = (metrics[column] - reference.mean()) / reference.std(ddof=1)
    metrics["D_metabolic_signature"] = metrics[["ZD_SUVR", "ZD_MC", "ZD_MwC"]].mean(axis=1)

    names = ["MBI_z", "D_SUVR", "D_MC", "D_MwC", "D_metabolic_signature"]
    baseline = metrics.loc[metrics.session.eq("ses-01")].copy()
    group_rows = []
    for name in names:
        frame = baseline.loc[baseline.analysis_cohort.isin(["nTRD", "TRD1"])].dropna(subset=[name, "age", "sex"]).copy()
        frame["group_TRD1"] = frame.analysis_cohort.eq("TRD1").astype(float)
        fit = sm.OLS(frame[name], sm.add_constant(frame[["group_TRD1", "age", "sex"]], has_constant="add")).fit()
        low, high = fit.conf_int().loc["group_TRD1"]
        t_value = float(fit.tvalues["group_TRD1"])
        group_rows.append({
            "metric": name, "comparison": "TRD1 - nTRD", "n_TRD1": int(frame.analysis_cohort.eq("TRD1").sum()),
            "n_nTRD": int(frame.analysis_cohort.eq("nTRD").sum()),
            "TRD1_mean": float(frame.loc[frame.analysis_cohort.eq("TRD1"), name].mean()),
            "nTRD_mean": float(frame.loc[frame.analysis_cohort.eq("nTRD"), name].mean()),
            "beta": float(fit.params["group_TRD1"]), "CI95_low": float(low), "CI95_high": float(high),
            "t": t_value, "df_resid": float(fit.df_resid),
            "partial_r": float(np.sign(t_value) * np.sqrt(t_value ** 2 / (t_value ** 2 + fit.df_resid))),
            "p_uncorrected": float(fit.pvalues["group_TRD1"]),
        })
    group = pd.DataFrame(group_rows)

    active = metrics.loc[
        metrics.analysis_cohort.eq("TRD2") & metrics.analysis_arm.isin(["rTMS", "iTBS"])
    ]
    paired_rows = []
    for name in names:
        wide = active.pivot(index="participant_id", columns="session", values=name).dropna(subset=["ses-01", "ses-02"])
        pre = wide["ses-01"].to_numpy(float)
        post = wide["ses-02"].to_numpy(float)
        delta = post - pre
        test = stats.ttest_rel(post, pre)
        sem = stats.sem(delta)
        low, high = stats.t.interval(.95, len(delta) - 1, loc=delta.mean(), scale=sem)
        paired_rows.append({
            "metric": name, "n": len(delta), "pre_mean": float(pre.mean()), "post_mean": float(post.mean()),
            "post_minus_pre": float(delta.mean()), "CI95_low": float(low), "CI95_high": float(high),
            "t": float(test.statistic), "df": len(delta)-1,
            "effect_size_Cohen_dz": float(delta.mean()/delta.std(ddof=1)),
            "p_uncorrected": float(test.pvalue),
        })
    paired = pd.DataFrame(paired_rows)
    return metrics, group, paired


def write_outputs() -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_data()
    participants = data["meta"].drop_duplicates("participant_id")
    cohort_summary = (
        participants.groupby(["analysis_cohort", "analysis_arm"], dropna=False)
        .participant_id.nunique().rename("participants").reset_index()
    )
    cohort_summary.to_csv(TABLES / "cohort_summary.csv", index=False)
    primary, sensitivity = baseline_results(data)
    for modality, stem in [("Regional SUVR", "regional"), ("Metabolic connectivity", "connectivity"), ("MwC", "MwC")]:
        primary[modality].to_csv(TABLES / f"primary_{stem}_all.csv", index=False)
        primary[modality].loc[primary[modality].p_lt_0p01].to_csv(TABLES / f"primary_{stem}_p_lt_0p01.csv", index=False)
        sensitivity[modality].to_csv(TABLES / f"sensitivity_{stem}_all.csv", index=False)
        sensitivity[modality].loc[sensitivity[modality].p_lt_0p05].to_csv(TABLES / f"sensitivity_{stem}_p_lt_0p05.csv", index=False)
    longitudinal = longitudinal_results(data)
    longitudinal.to_csv(TABLES / "longitudinal_all_660_tests.csv", index=False)
    longitudinal.loc[longitudinal.analysis_arm.isin(["rTMS", "iTBS"]) & longitudinal.paired_p_lt_0p01].to_csv(TABLES / "active_all_modalities_p_lt_001.csv", index=False)
    longitudinal.loc[longitudinal.analysis_arm.isin(["rTMS", "iTBS"]) & longitudinal.paired_p_lt_0p05].to_csv(TABLES / "active_all_modalities_p_lt_005.csv", index=False)
    spatial = spatial_results(data)
    spatial.to_csv(TABLES / "spatial_correlation_summary.csv", index=False)
    metrics, group_indexes, paired_indexes = global_index_results(data)
    metrics.to_csv(TABLES / "five_global_indexes_deidentified.csv.gz", index=False, compression="gzip")
    group_indexes.to_csv(TABLES / "global_indexes_nTRD_TRD_statistics.csv", index=False)
    paired_indexes.to_csv(TABLES / "global_indexes_active_pre_post_statistics.csv", index=False)
    summary = {
        "status": "PASS",
        "primary_p_lt_0p01": {key: int(value.p_lt_0p01.sum()) for key, value in primary.items()},
        "active_paired_p_lt_0p01": int((longitudinal.analysis_arm.isin(["rTMS", "iTBS"]) & longitudinal.paired_p_lt_0p01).sum()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
