#!/usr/bin/env python3
"""Validate the bundled inputs, numerical results, and stored notebooks."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
EXCLUDED_PARTS = {".git", ".matplotlib", ".venv", "__pycache__", ".ipynb_checkpoints"}


def release_files() -> list[Path]:
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and not set(path.relative_to(ROOT).parts).intersection(EXCLUDED_PARTS)
        and path.name != ".DS_Store"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_inventory(path: Path, files: list[Path]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["repository_path", "bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        for item in files:
            writer.writerow(
                {
                    "repository_path": item.relative_to(ROOT).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256(item),
                }
            )


def main() -> int:
    MANIFESTS.mkdir(exist_ok=True)
    checks: dict[str, bool] = {}

    clinical = pd.read_excel(ROOT / "inputs/core/clinical_data.xlsx")
    roster = pd.read_csv(ROOT / "inputs/core/analysis_population_roster.csv")
    pet = pd.read_excel(ROOT / "inputs/core/AAL3_PET_wbSUVR_Table.xlsx")
    roi = pd.read_csv(ROOT / "inputs/core/corticolimbic_12x14_ROI_definition.csv")
    excluded = pd.read_csv(ROOT / "inputs/core/excluded_AAL3_ROIs.csv")
    contract = json.loads((ROOT / "inputs/core/analysis_contract.json").read_text())
    gcmi = np.load(ROOT / "inputs/mwc/normative_sc/session_gcmi_pet_AAL3_144.npz", allow_pickle=True)

    expected_keys = [f"P{index:03d}" for index in range(1, 169)]
    checks["clinical_population"] = (
        len(clinical) == 168
        and clinical.subject_key.astype(str).tolist() == expected_keys
        and clinical.analysis_cohort.value_counts().to_dict()
        == {"TRD2": 85, "TRD1": 35, "nTRD": 25, "HC": 23}
        and clinical.loc[clinical.analysis_cohort.eq("TRD2"), "analysis_arm"].value_counts().to_dict()
        == {"iTBS": 31, "rTMS": 28, "sham": 26}
    )
    checks["roster_population"] = (
        roster.subject_key.astype(str).tolist() == expected_keys
        and roster[["pet_ses_01", "pet_ses_02", "mwc_ses_01", "mwc_ses_02"]].sum().to_dict()
        == {"pet_ses_01": 168, "pet_ses_02": 85, "mwc_ses_01": 168, "mwc_ses_02": 85}
    )
    pet_columns = pet.columns[1:].astype(str)
    pet_keys = pet_columns.str.extract(r"(P\d{3})", expand=False)
    checks["pet_input"] = pet.shape == (144, 254) and pet_keys.notna().all() and set(pet_keys) == set(expected_keys)
    gcmi_keys = np.asarray(gcmi["subject_key"]).astype(str)
    checks["gcmi_input"] = (
        np.asarray(gcmi["gcmi_raw"]).shape == (253, 144, 144)
        and np.asarray(gcmi["pet_wbSUVR"]).shape == (253, 144)
        and set(gcmi_keys) == set(expected_keys)
        and all(re.fullmatch(r"P\d{3}", key) for key in gcmi_keys)
    )
    checks["roi_definition"] = (
        len(roi) == 26
        and roi.network.value_counts().to_dict() == {"limbic": 14, "cortical": 12}
        and len(excluded) == 22
        and "voxel_count_on_PET_grid" in excluded.columns
    )
    checks["analysis_contract"] = contract == {
        "status": "PASS",
        "participants": {"HC": 23, "nTRD": 25, "TRD1": 35, "rTMS": 28, "iTBS": 31, "sham": 26},
        "sessions": 253,
        "input_policy": "Only final analysed participants and sessions are included",
        "PET": "GM > 0.3 individual-grey-matter-threshold wbSUVR; 144 retained AAL3 ROIs",
        "BOLD": "AAL3 spatial-mean time series without individual GM masking, represented by session-level GCMI matrices",
        "PET_and_fMRI_required_for_TRD2": True,
        "models": {
            "HC_reference": "age + sex",
            "group": "age + sex",
            "sensitivity": "age + sex + baseline HDRS-17",
            "paired": "paired t-test without covariates",
            "group_by_time": "age + sex",
            "clinical": "partial Spearman controlling age + sex",
        },
    }

    main_output = ROOT / "outputs/main_analyses"
    regional = pd.read_excel(main_output / "01_TRD1_discovery_agesex/regional_uncorrected_p_lt_0p01.xlsx")
    edges = pd.read_excel(main_output / "01_TRD1_discovery_agesex/edge_uncorrected_p_lt_0p01.xlsx")
    mwc = pd.read_csv(main_output / "07_age_sex_adjusted_MwC/tables/baseline_nTRD_vs_TRD1_adjusted_all_26.csv")
    checks["primary_significance_counts"] = (len(regional), len(edges), int((mwc.hc3_p_uncorrected < .01).sum())) == (8, 19, 3)

    sensitivity_file = main_output / "08_TRD1_vs_nTRD_HDRS_sensitivity/TRD1_vs_nTRD_baseline_HDRS_sensitivity.xlsx"
    sensitivity_counts = tuple(
        int((pd.read_excel(sensitivity_file, sheet_name=sheet).HDRS_adjusted_p_uncorrected < .05).sum())
        for sheet in ("regional_all", "edge_all", "MwC_all")
    )
    checks["sensitivity_significance_counts"] = sensitivity_counts == (4, 18, 2)

    paired_counts = {}
    for arm, folder in (("iTBS", "03_iTBS_prepost"), ("rTMS", "04_rTMS_prepost")):
        table = pd.read_excel(main_output / f"02_TRD2_deviation_prepost_agesex/{folder}/edge_pairttest_all.xlsx")
        paired_counts[arm] = int((table.paired_p_uncorrected < .01).sum())
    checks["dynamic_longitudinal_selection"] = paired_counts == {"iTBS": 2, "rTMS": 6}

    required_figures = [
        "04_figures/nTRD_TRD1_TRD2_regional26_strip_heatmap.png",
        "04_figures/nTRD_TRD1_TRD2_corticolimbic12x14_heatmap.png",
        "04_figures/nTRD_TRD1_TRD2_age_sex_adjusted_MwC_26ROI_strip_heatmap.png",
        "04_figures/corticolimbic_circular_summary.png",
        "04_figures/corticolimbic_circular_longitudinal_summary.png",
        "04_figures/TRD2_prepost_raincloud.png",
        "04_figures/TRD2_prepost_HDRS_correlation_panel_green_reference.png",
        "04_figures/spatial_correlation_scatter.png",
        "10_metabolic_signature_five_metrics/Figure_five_metric_nTRD_TRD_raincloud.png",
        "10_metabolic_signature_five_metrics/Figure_five_metric_active_pre_post_raincloud.png",
        "11_SGC_26ROI_figures/figures/Figure_TRD_vs_nonTRD_significant_MwC_nilearn_glass_brain.png",
        "15_nonTRD_TRD_significant_regional_SUVR_glass_brain/figures/Figure_TRD_vs_nonTRD_significant_regional_SUVR_combined_nilearn_glass_brain.png",
    ]
    checks["required_figures"] = all((main_output / path).is_file() and (main_output / path).stat().st_size > 10_000 for path in required_figures)

    notebook_rows = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
        preview = next((cell for cell in code_cells if cell.get("id") == "results-preview"), None)
        notebook_rows.append(
            {
                "notebook": path.relative_to(ROOT).as_posix(),
                "code_cells": len(code_cells),
                "executed_code_cells": sum(cell.execution_count is not None for cell in code_cells),
                "stored_outputs": len(outputs),
                "errors": sum(output.output_type == "error" for output in outputs),
                "results_preview_last": preview is not None and code_cells[-1].get("id") == "results-preview",
            }
        )
    checks["executed_notebooks"] = (
        len(notebook_rows) == 16
        and all(row["code_cells"] == row["executed_code_cells"] for row in notebook_rows)
        and all(row["stored_outputs"] > 0 and row["errors"] == 0 and row["results_preview_last"] for row in notebook_rows)
    )

    paths = [path.relative_to(ROOT).as_posix() for path in release_files()]
    permitted_main_output_directories = {
        "00_setup",
        "01_TRD1_discovery_agesex",
        "02_TRD2_deviation_prepost_agesex",
        "03_TRD1_TRD2_spatial_correlation",
        "04_figures",
        "05_active_vs_sham_interaction",
        "06_active_paired_complete_results",
        "07_age_sex_adjusted_MwC",
        "08_TRD1_vs_nTRD_HDRS_sensitivity",
        "10_metabolic_signature_five_metrics",
        "11_SGC_26ROI_figures",
        "13_sensitivity_mPFC_striatopallidal_glass_brain",
        "15_nonTRD_TRD_significant_regional_SUVR_glass_brain",
    }
    observed_main_output_directories = {
        path.name for path in main_output.iterdir() if path.is_dir()
    }
    checks["clean_file_tree"] = (
        observed_main_output_directories == permitted_main_output_directories
        and not any(Path(path).suffix.lower() == ".docx" for path in paths)
    )
    checks["no_conversion_tools"] = not (ROOT / "tools").exists()

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "population": {"HC": 23, "nTRD": 25, "TRD1": 35, "rTMS": 28, "iTBS": 31, "sham": 26},
        "sessions": 253,
        "models": contract["models"],
        "results": {
            "primary_p_lt_0p01": {"regional_SUVR": 8, "metabolic_connectivity": 19, "MwC": 3},
            "sensitivity_p_lt_0p05": {"regional_SUVR": 4, "metabolic_connectivity": 18, "MwC": 2},
            "active_paired_p_lt_0p01_edges": paired_counts,
        },
        "checks": checks,
        "notebooks": notebook_rows,
    }
    (MANIFESTS / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    write_inventory(MANIFESTS / "input_inventory.csv", [path for path in release_files() if "inputs" in path.relative_to(ROOT).parts])
    write_inventory(MANIFESTS / "output_inventory.csv", [path for path in release_files() if "outputs" in path.relative_to(ROOT).parts])
    write_inventory(MANIFESTS / "notebook_inventory.csv", sorted((ROOT / "notebooks").glob("*.ipynb")))
    checksum_path = MANIFESTS / "SHA256SUMS.txt"
    checksum_files = [path for path in release_files() if path != checksum_path]
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in checksum_files),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
