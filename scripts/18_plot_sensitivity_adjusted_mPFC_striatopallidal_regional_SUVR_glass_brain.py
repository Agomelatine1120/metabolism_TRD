#!/usr/bin/env python3
"""Plot the baseline-severity-adjusted medial prefrontal–striatal circuit.

Every displayed ROI is coloured by the adjusted regional-SUVR group beta
(TRD1 minus nTRD) from the formal sensitivity model.  The figure does not use
metabolic-connectivity or MwC values and does not apply a display p threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MPL_CACHE = Path("/private/tmp/metafl_sensitivity_mpf_striatal_glassbrain_mpl")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("METAFL_ROOT", Path(__file__).resolve().parents[1])).resolve()
ATLAS_FILE = PROJECT_ROOT / "inputs/core/AAL3_144_space-PET_res-2_dseg.nii.gz"
RESULT_FILE = PROJECT_ROOT / (
    "outputs/main_analyses/11_SGC_26ROI_figures/tables/"
    "baseline_severity_adjusted_TRD1_vs_nTRD_regional26_SGC.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/main_analyses/13_sensitivity_mPFC_striatopallidal_glass_brain"
)

# Exact AAL3-166 IDs used by the formal regional-SUVR analysis.
ROI_SPECS = [
    (21, "AAL3_021_Frontal_Med_Orb_L", "Lt. vmPFC", "medial prefrontal"),
    (22, "AAL3_022_Frontal_Med_Orb_R", "Rt. vmPFC", "medial prefrontal"),
    (19, "AAL3_019_Frontal_Sup_Medial_L", "Lt. dmPFC", "medial prefrontal"),
    (20, "AAL3_020_Frontal_Sup_Medial_R", "Rt. dmPFC", "medial prefrontal"),
    (77, "AAL3_077_Putamen_L", "Lt. putamen", "striatal-pallidal"),
    (78, "AAL3_078_Putamen_R", "Rt. putamen", "striatal-pallidal"),
    (79, "AAL3_079_Pallidum_L", "Lt. pallidum", "striatal-pallidal"),
    (80, "AAL3_080_Pallidum_R", "Rt. pallidum", "striatal-pallidal"),
]
VALUE_COLUMN = "adjusted_group_beta_TRD1_minus_nTRD"
COLOR_LIMIT = 2.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed(value: float) -> str:
    return f"{value:+.2f}".replace("-", "−")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, default=ATLAS_FILE)
    parser.add_argument("--results", type=Path, default=RESULT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    atlas_file = args.atlas.expanduser().resolve()
    result_file = args.results.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    figure_dir = output / "figures"
    mask_dir = output / "masks"
    table_dir = output / "tables"
    for directory in (figure_dir, mask_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(result_file)
    if results["ROI"].duplicated().any():
        raise ValueError("The formal regional result table contains duplicate ROIs")
    indexed = results.set_index("ROI")

    rows = []
    values_by_id: dict[int, float] = {}
    for aal3_id, roi, label, circuit_component in ROI_SPECS:
        if roi not in indexed.index:
            raise ValueError(f"Missing formal sensitivity result for {roi}")
        source = indexed.loc[roi]
        value = float(source[VALUE_COLUMN])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite adjusted beta for {roi}")
        values_by_id[aal3_id] = value
        rows.append(
            {
                "AAL3_ID": aal3_id,
                "ROI": roi,
                "display_label": label,
                "circuit_component": circuit_component,
                "value_encoded_in_colour": value,
                "value_definition": "adjusted regional-SUVR beta (TRD1 minus nTRD)",
                "standard_error": float(source["standard_error"]),
                "ci95_low": float(source["ci95_low"]),
                "ci95_high": float(source["ci95_high"]),
                "t": float(source["t"]),
                "df_resid": float(source["df_resid"]),
                "p_uncorrected": float(source["p_uncorrected"]),
                "nominal_p_lt_0p05": bool(float(source["p_uncorrected"]) < 0.05),
                "model": str(source["model"]),
                "n_total": int(source["n_total"]),
                "n_TRD1": int(source["n_TRD1"]),
                "n_nTRD": int(source["n_nTRD"]),
            }
        )
    value_table = pd.DataFrame(rows)

    atlas_img = nib.load(atlas_file)
    atlas = np.asanyarray(atlas_img.dataobj)
    if atlas.ndim != 3 or not np.isfinite(atlas).all():
        raise ValueError("Atlas must be a finite 3D label image")
    if not np.array_equal(atlas, np.rint(atlas)):
        raise ValueError("Atlas labels must be integers")

    label_counts = {
        aal3_id: int(np.count_nonzero(atlas == aal3_id))
        for aal3_id in values_by_id
    }
    if not all(count > 0 for count in label_counts.values()):
        raise ValueError(f"One or more AAL3 labels are absent: {label_counts}")

    value_map = np.zeros(atlas.shape, dtype=np.float32)
    observed = np.zeros(atlas.shape, dtype=np.uint8)
    for aal3_id, value in values_by_id.items():
        region = atlas == aal3_id
        value_map[region] = value
        observed[region] = 1

    value_table["voxel_count_1mm"] = value_table["AAL3_ID"].map(label_counts)
    value_table["color_scale_vmin"] = -COLOR_LIMIT
    value_table["color_scale_vmax"] = COLOR_LIMIT
    value_table["color_map"] = "ColorCET CET_D1A"
    value_table["display_threshold"] = 1e-6
    value_table_file = table_dir / (
        "sensitivity_adjusted_mPFC_striatopallidal_regional_SUVR_values.csv"
    )
    value_table.to_csv(value_table_file, index=False)

    value_header = atlas_img.header.copy()
    value_header.set_data_dtype(np.float32)
    mask_header = atlas_img.header.copy()
    mask_header.set_data_dtype(np.uint8)
    value_map_file = mask_dir / (
        "AAL3_bilateral_vmPFC_dmPFC_putamen_pallidum_"
        "sensitivity_adjusted_regional_SUVR_beta.nii.gz"
    )
    observed_mask_file = mask_dir / (
        "AAL3_bilateral_vmPFC_dmPFC_putamen_pallidum_observed_mask.nii.gz"
    )
    nib.save(
        nib.Nifti1Image(value_map, atlas_img.affine, header=value_header),
        value_map_file,
    )
    nib.save(
        nib.Nifti1Image(observed, atlas_img.affine, header=mask_header),
        observed_mask_file,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(12.0, 4.5), facecolor="white")
    display = plotting.plot_glass_brain(
        nib.load(value_map_file),
        display_mode="lyrz",
        plot_abs=False,
        threshold=1e-6,
        colorbar=True,
        cbar_tick_format="%.1f",
        cmap=cc.cm.CET_D1A,
        vmin=-COLOR_LIMIT,
        vmax=COLOR_LIMIT,
        symmetric_cbar=True,
        black_bg=False,
        annotate=True,
        alpha=1.0,
        radiological=False,
        resampling_interpolation="nearest",
        figure=fig,
        axes=(0.015, 0.19, 0.95, 0.66),
    )
    fig.suptitle(
        "Medial prefrontal–striatal circuit after depressive-severity adjustment",
        x=0.5,
        y=0.975,
        fontsize=17,
        fontweight="bold",
        color="#3F4650",
    )
    fig.text(
        0.5,
        0.900,
        "Regional SUVR: adjusted β (TRD1 − nTRD)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4A4A4A",
    )
    by_label = value_table.set_index("display_label")["value_encoded_in_colour"]
    fig.text(
        0.5,
        0.105,
        (
            f"vmPFC   L {signed(by_label['Lt. vmPFC'])}   R {signed(by_label['Rt. vmPFC'])}"
            f"      |      dmPFC   L {signed(by_label['Lt. dmPFC'])}   "
            f"R {signed(by_label['Rt. dmPFC'])}"
        ),
        ha="center",
        va="center",
        fontsize=9.4,
        color="#333333",
    )
    fig.text(
        0.5,
        0.047,
        (
            f"Putamen   L {signed(by_label['Lt. putamen'])}   R {signed(by_label['Rt. putamen'])}"
            f"      |      Pallidum   L {signed(by_label['Lt. pallidum'])}   "
            f"R {signed(by_label['Rt. pallidum'])}"
        ),
        ha="center",
        va="center",
        fontsize=9.4,
        color="#333333",
    )
    if getattr(display, "_cbar", None) is not None:
        display._cbar.set_ticks([-2.4, -1.2, 0.0, 1.2, 2.4])
        display._cbar.set_label("Adjusted regional-SUVR β", fontsize=9, labelpad=8)
        display._cbar.ax.tick_params(labelsize=8)

    stem = figure_dir / (
        "Figure_sensitivity_adjusted_mPFC_striatopallidal_"
        "regional_SUVR_nilearn_glass_brain"
    )
    for suffix, kwargs in [
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ]:
        display.savefig(stem.with_suffix(suffix), **kwargs)
    display.close()

    encoded_checks = {
        str(aal3_id): bool(
            np.allclose(value_map[atlas == aal3_id], value, rtol=0, atol=1e-6)
        )
        for aal3_id, value in values_by_id.items()
    }
    report = {
        "status": "PASS",
        "scope": (
            "Bilateral vmPFC, dmPFC, putamen and pallidum coloured by the "
            "formal sensitivity-adjusted regional-SUVR group beta"
        ),
        "comparison": "TRD1 minus nTRD",
        "covariates": ["age", "sex", "baseline HDRS-17"],
        "all_eight_prespecified_ROIs_displayed": True,
        "display_p_threshold_applied": False,
        "atlas": str(atlas_file),
        "atlas_sha256": sha256(atlas_file),
        "atlas_shape": list(atlas_img.shape),
        "atlas_affine": atlas_img.affine.tolist(),
        "result_table": str(result_file),
        "result_table_sha256": sha256(result_file),
        "nilearn_version": __import__("nilearn").__version__,
        "rendering": {
            "function": "nilearn.plotting.plot_glass_brain",
            "display_mode": "lyrz",
            "plot_abs": False,
            "threshold": 1e-6,
            "vmin": -COLOR_LIMIT,
            "vmax": COLOR_LIMIT,
            "symmetric_cbar": True,
            "resampling_interpolation": "nearest",
            "colormap": "ColorCET CET_D1A perceptually uniform blue-red diverging",
            "negative_colour": "blue (lower regional SUVR in TRD)",
            "positive_colour": "red (higher regional SUVR in TRD)",
        },
        "AAL3_labels": {
            str(row["AAL3_ID"]): {
                "ROI": row["ROI"],
                "display_label": row["display_label"],
                "adjusted_beta": row["value_encoded_in_colour"],
                "p_uncorrected": row["p_uncorrected"],
                "voxel_count": label_counts[int(row["AAL3_ID"])],
            }
            for row in rows
        },
        "checks": {
            "all_target_labels_nonempty": all(count > 0 for count in label_counts.values()),
            "all_target_values_encoded_exactly": all(encoded_checks.values()),
            "observed_mask_voxels_match_label_sum": int(observed.sum()) == sum(label_counts.values()),
            "outside_observed_mask_is_zero": bool(np.all(value_map[observed == 0] == 0)),
            "PNG_exists": stem.with_suffix(".png").is_file(),
            "PDF_exists": stem.with_suffix(".pdf").is_file(),
            "SVG_exists": stem.with_suffix(".svg").is_file(),
            "value_map_exists": value_map_file.is_file(),
            "observed_mask_exists": observed_mask_file.is_file(),
            "value_table_exists": value_table_file.is_file(),
        },
        "outputs": {
            "PNG": str(stem.with_suffix(".png")),
            "PDF": str(stem.with_suffix(".pdf")),
            "SVG": str(stem.with_suffix(".svg")),
            "regional_SUVR_value_map": str(value_map_file),
            "observed_ROI_mask": str(observed_mask_file),
            "figure_input_values": str(value_table_file),
        },
    }
    if not all(report["checks"].values()):
        report["status"] = "FAIL"
    report_file = output / "validation_report.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
