#!/usr/bin/env python3
"""Plot every MwC region meeting the configured p < .01 reporting rule.

The selected regions are read directly from the complete 26-ROI MwC results;
no ROI is preselected in this script. Colours are categorical and do not
encode effect magnitude.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

_MPL_CACHE = Path("/private/tmp/metafl_table5_mwc_mpl")
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
from nilearn import plotting
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("METAFL_ROOT", Path(__file__).resolve().parents[1])).resolve()
ATLAS_FILE = PROJECT_ROOT / "inputs/core/AAL3_144_space-PET_res-2_dseg.nii.gz"
MWC_RESULT_FILE = (
    PROJECT_ROOT
    / "outputs/main_analyses/07_age_sex_adjusted_MwC/tables/"
    / "baseline_nTRD_vs_TRD1_adjusted_all_26.csv"
)
SIGNIFICANCE_THRESHOLD = 0.01

# Exact palette anchors used by the regional glass-brain figures.
SFG_LIGHT_BLUE = "#BACFEA"
MFG_DEEP_BLUE = "#9AC3F8"
HPC_LIGHT_PINK = "#EAD8DE"
PAL_DEEP_PINK = "#D2A1B1"
COLOR_BY_ROI = {
    "Frontal_Sup_2_L": SFG_LIGHT_BLUE,
    "Frontal_Sup_2_R": SFG_LIGHT_BLUE,
    "Frontal_Mid_2_L": MFG_DEEP_BLUE,
    "Hippocampus_R": HPC_LIGHT_PINK,
    "Pallidum_R": PAL_DEEP_PINK,
}
DISPLAY_BY_ROI = {
    "Frontal_Sup_2_L": "Lt. SFG",
    "Frontal_Sup_2_R": "Rt. SFG",
    "Frontal_Mid_2_L": "Lt. MFG",
    "Hippocampus_R": "Rt. HPC",
    "Pallidum_R": "Rt. PAL",
}
ENCODED_VALUE_BY_COLOR = {
    SFG_LIGHT_BLUE: 1,
    MFG_DEEP_BLUE: 2,
    HPC_LIGHT_PINK: 3,
    PAL_DEEP_PINK: 4,
}
REGION_CMAP = ListedColormap(
    [SFG_LIGHT_BLUE, MFG_DEEP_BLUE, HPC_LIGHT_PINK, PAL_DEEP_PINK],
    name="MetaFL_Table5_MwC_categories",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, default=ATLAS_FILE)
    parser.add_argument("--mwc-results", type=Path, default=MWC_RESULT_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/main_analyses/11_SGC_26ROI_figures",
    )
    args = parser.parse_args()

    atlas_file = args.atlas.expanduser().resolve()
    result_file = args.mwc_results.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    figure_dir = output / "figures"
    mask_dir = output / "masks"
    table_dir = output / "tables"
    for directory in [figure_dir, mask_dir, table_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    all_results = pd.read_csv(result_file)
    required_columns = {
        "AAL3_ID", "ROI", "nTRD_mean_age_sex_adjusted_z",
        "TRD1_mean_age_sex_adjusted_z", "adjusted_beta_TRD1_minus_nTRD_HC_SD",
        "hc3_t", "hc3_p_uncorrected",
    }
    if not required_columns <= set(all_results.columns):
        raise ValueError(f"MwC results lack required columns: {required_columns - set(all_results.columns)}")
    selected = (
        all_results.loc[all_results["hc3_p_uncorrected"] < SIGNIFICANCE_THRESHOLD]
        .sort_values("hc3_p_uncorrected")
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("No MwC regions meet the configured p < .01 reporting rule")
    if selected["ROI"].duplicated().any():
        raise ValueError("Table 5 selection contains duplicate ROI labels")
    missing_style = set(selected["ROI"]) - set(COLOR_BY_ROI)
    if missing_style:
        raise ValueError(f"No figure colour assigned to dynamically selected ROI(s): {missing_style}")

    atlas_img = nib.load(atlas_file)
    atlas = np.asanyarray(atlas_img.dataobj)
    categorical = np.zeros(atlas.shape, dtype=np.uint8)
    label_counts: dict[int, int] = {}
    selected_rows = []
    for row in selected.itertuples(index=False):
        aal_id = int(row.AAL3_ID)
        roi = str(row.ROI)
        region_mask = atlas == aal_id
        voxel_count = int(region_mask.sum())
        if voxel_count <= 0:
            raise ValueError(f"Selected AAL3 label is absent from atlas: {aal_id} {roi}")
        color = COLOR_BY_ROI[roi]
        encoded_value = ENCODED_VALUE_BY_COLOR[color]
        categorical[region_mask] = encoded_value
        label_counts[aal_id] = voxel_count
        selected_rows.append({
            "AAL3_ID": aal_id,
            "ROI": roi,
            "display_label": DISPLAY_BY_ROI[roi],
            "voxel_count_on_PET_grid": voxel_count,
            "encoded_value": encoded_value,
            "display_color": color,
            "nTRD_mean_age_sex_adjusted_z": float(row.nTRD_mean_age_sex_adjusted_z),
            "TRD1_mean_age_sex_adjusted_z": float(row.TRD1_mean_age_sex_adjusted_z),
            "TRD1_minus_nTRD_HC_SD": float(row.adjusted_beta_TRD1_minus_nTRD_HC_SD),
            "hc3_t": float(row.hc3_t),
            "hc3_p_uncorrected": float(row.hc3_p_uncorrected),
            "direction": "lower in TRD" if row.adjusted_beta_TRD1_minus_nTRD_HC_SD < 0 else "higher in TRD",
            "selection_rule": "HC3 uncorrected p < 0.01",
        })

    expected_union = sum(label_counts.values())
    observed_union = int(np.count_nonzero(categorical))
    if observed_union != expected_union:
        raise ValueError("Selected AAL3 labels overlap or the union is inconsistent")

    header = atlas_img.header.copy()
    header.set_data_dtype(np.uint8)
    mask_path = mask_dir / "AAL3_MwC_p_lt_0p01_union_binary.nii.gz"
    categorical_path = mask_dir / "AAL3_MwC_p_lt_0p01_categorical_labels.nii.gz"
    nib.save(nib.Nifti1Image((categorical > 0).astype(np.uint8), atlas_img.affine, header), mask_path)
    nib.save(nib.Nifti1Image(categorical, atlas_img.affine, header), categorical_path)

    region_table = pd.DataFrame(selected_rows)
    region_file = table_dir / "TRD1_vs_nTRD_MwC_p_lt_0p01_figure_values.csv"
    region_table.to_csv(region_file, index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(11.8, 4.15), facecolor="white")
    display = plotting.plot_glass_brain(
        nib.load(categorical_path),
        display_mode="lyrz",
        plot_abs=False,
        threshold=0.5,
        colorbar=False,
        cmap=REGION_CMAP,
        vmin=1.0,
        vmax=4.0,
        black_bg=False,
        annotate=True,
        alpha=1.0,
        radiological=False,
        figure=fig,
    )
    fig.suptitle(
        "Metabolic-weighted centrality differences: TRD versus nonTRD",
        x=0.5,
        y=0.975,
        fontsize=16,
        fontweight="bold",
        color="#3F4650",
    )
    legend_handles = [
        Patch(
            facecolor=row.display_color,
            edgecolor="none",
            label=f"{row.display_label}  (p = {row.hc3_p_uncorrected:.4f})",
        )
        for row in region_table.itertuples(index=False)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(5, len(legend_handles)),
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        handleheight=0.85,
        columnspacing=1.3,
    )

    stem = figure_dir / "Figure_TRD_vs_nonTRD_significant_MwC_nilearn_glass_brain"
    for suffix, kwargs in [(".png", {"dpi": 600}), (".pdf", {}), (".svg", {})]:
        display.savefig(stem.with_suffix(suffix), **kwargs)
    display.close()

    selected_ids = region_table["AAL3_ID"].astype(int).tolist()
    report = {
        "status": "PASS",
        "scope": "All MwC regions selected dynamically at HC3 uncorrected p < 0.01",
        "selection_source": str(result_file),
        "selection_source_sha256": sha256(result_file),
        "selection_threshold": SIGNIFICANCE_THRESHOLD,
        "selected_region_count": len(region_table),
        "selected_AAL3_IDs_in_p_order": selected_ids,
        "selected_regions_in_p_order": region_table["display_label"].tolist(),
        "atlas": str(atlas_file),
        "atlas_sha256": sha256(atlas_file),
        "atlas_shape": list(atlas_img.shape),
        "union_voxel_count": observed_union,
        "rendering": {
            "library": "nilearn.plotting.plot_glass_brain",
            "display_mode": "lyrz",
            "categorical_colour_encoding": True,
            "effect_magnitude_not_encoded": True,
            "lower_MwC_palette": [SFG_LIGHT_BLUE, MFG_DEEP_BLUE],
            "higher_MwC_palette": [HPC_LIGHT_PINK, PAL_DEEP_PINK],
        },
        "checks": {
            "selection_is_dynamic_p_lt_0p01": bool((selected["hc3_p_uncorrected"] < .01).all()),
            "at_least_one_region_selected": len(selected) >= 1,
            "all_selected_labels_nonempty": all(value > 0 for value in label_counts.values()),
            "union_count_matches_sum": observed_union == expected_union,
            "selected_table_matches_rendered_labels": set(selected["AAL3_ID"].astype(int)) == set(selected_ids),
            "PNG_exists": stem.with_suffix(".png").is_file(),
            "PDF_exists": stem.with_suffix(".pdf").is_file(),
            "SVG_exists": stem.with_suffix(".svg").is_file(),
        },
        "outputs": {
            "PNG": str(stem.with_suffix(".png")),
            "PDF": str(stem.with_suffix(".pdf")),
            "SVG": str(stem.with_suffix(".svg")),
            "NIfTI_mask": str(mask_path),
            "NIfTI_categorical_colour_map": str(categorical_path),
            "region_table": str(region_file),
        },
    }
    if not all(report["checks"].values()):
        report["status"] = "FAIL"
    report_file = output / "significant_MwC_glass_brain_validation_report.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
