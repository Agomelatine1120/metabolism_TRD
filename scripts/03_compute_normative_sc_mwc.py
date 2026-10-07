#!/usr/bin/env python3
"""Recompute AAL3-144 dTOR-985 normative-SC-constrained MwC.

The calculation universe is fixed at 144 retained AAL3 nodes. For every
participant/session, bias-corrected GCMI values below 0.01 bits are removed,
the remaining GCMI matrix is constrained by the fixed group-normative dTOR-985
top-10% structural mask (1,030 undirected edges), and each non-isolated target
row is normalized to sum to one. MwC is then calculated for all 144 nodes as
100 * W @ PET. The 26 prespecified cortico-limbic targets are selected only
afterward, with no re-thresholding or re-normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


N_NODES = 144
N_SESSIONS = 253
MI_FLOOR_BITS = 0.01
PET_SCALE = 100.0
EXPECTED_SC_EDGES = 1_030
EXPECTED_POSSIBLE_EDGES = N_NODES * (N_NODES - 1) // 2
REFERENCE_TOLERANCE = 2e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_session_mwc(
    gcmi_raw: np.ndarray,
    pet_wbsuvr: np.ndarray,
    structural_mask: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Return all-144 MwC, isolated-node count, and row-sum error."""
    gcmi = np.asarray(gcmi_raw, dtype=float)
    pet = np.asarray(pet_wbsuvr, dtype=float)
    if gcmi.shape != (N_NODES, N_NODES) or pet.shape != (N_NODES,):
        raise ValueError(f"Unexpected GCMI/PET shape: {gcmi.shape}/{pet.shape}")
    if not np.isfinite(gcmi).all() or not np.isfinite(pet).all():
        raise ValueError("GCMI and PET inputs must be finite")
    if float(np.max(np.abs(gcmi - gcmi.T))) > 1e-5:
        raise ValueError("GCMI is not symmetric within float32 tolerance")

    weighted = (gcmi + gcmi.T) / 2.0
    weighted[weighted < MI_FLOOR_BITS] = 0.0
    np.fill_diagonal(weighted, 0.0)
    weighted[~structural_mask] = 0.0

    row_sums = weighted.sum(axis=1)
    valid = row_sums > 0
    weights = np.divide(
        weighted,
        row_sums[:, None],
        out=np.zeros_like(weighted),
        where=row_sums[:, None] > 0,
    )
    # Missing PET values intentionally propagate through this product. NumPy
    # can otherwise emit platform-specific BLAS warnings even when the finite
    # values reproduce the frozen reference within tolerance.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mwc = PET_SCALE * (weights @ pet)
    mwc[~valid] = np.nan
    row_error = (
        float(np.max(np.abs(weights[valid].sum(axis=1) - 1.0)))
        if valid.any()
        else 0.0
    )
    return mwc, int(np.count_nonzero(~valid)), row_error


def recompute(base: Path) -> dict[str, Path]:
    base = base.expanduser().resolve()
    normative_root = base / "inputs" / "mwc" / "normative_sc"
    output_root = base / "outputs" / "mwc" / "recomputed_normative_sc"
    output_root.mkdir(parents=True, exist_ok=True)

    bundle_file = normative_root / "session_gcmi_pet_AAL3_144.npz"
    mask_file = normative_root / "normative_SC_top10_mask.npz"
    reference_full_file = (
        normative_root / "subject_level_AAL3_144_normativeSC10_MwC_wide.csv.gz"
    )
    reference_26_file = (
        normative_root / "subject_level_corticolimbic26_normativeSC10_MwC_wide.csv"
    )
    roi_file = base / "inputs" / "core" / "corticolimbic_12x14_ROI_definition.csv"
    required = [bundle_file, mask_file, reference_full_file, reference_26_file, roi_file]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MwC calculation inputs:\n" + "\n".join(missing))

    with np.load(mask_file, allow_pickle=False) as payload:
        structural_mask = np.asarray(payload["sc_top10_mask"], dtype=bool)
        mask_ids = np.asarray(payload["aal3_ids"], dtype=int)
        mask_names = np.asarray(payload["roi_names"], dtype=str)
    if structural_mask.shape != (N_NODES, N_NODES):
        raise ValueError("Structural mask is not 144 x 144")
    if not np.array_equal(structural_mask, structural_mask.T):
        raise ValueError("Structural mask is not symmetric")
    if np.any(np.diag(structural_mask)):
        raise ValueError("Structural mask diagonal must be zero")
    retained_sc_edges = int(np.count_nonzero(np.triu(structural_mask, 1)))
    if retained_sc_edges != EXPECTED_SC_EDGES:
        raise ValueError(f"Expected {EXPECTED_SC_EDGES} SC edges, found {retained_sc_edges}")

    with np.load(bundle_file, allow_pickle=False) as payload:
        gcmi_all = np.asarray(payload["gcmi_raw"], dtype=np.float32)
        pet_all = np.asarray(payload["pet_wbSUVR"], dtype=np.float32)
        bundle_ids = np.asarray(payload["aal3_ids"], dtype=int)
        bundle_names = np.asarray(payload["roi_names"], dtype=str)
        subject_key = np.asarray(payload["subject_key"], dtype=str)
        identity = pd.DataFrame(
            {
                "subject_key": subject_key,
                "session": np.asarray(payload["session"], dtype=str),
            }
        )
    if gcmi_all.shape != (N_SESSIONS, N_NODES, N_NODES):
        raise ValueError(f"Unexpected bundled GCMI shape: {gcmi_all.shape}")
    if pet_all.shape != (N_SESSIONS, N_NODES):
        raise ValueError(f"Unexpected bundled PET shape: {pet_all.shape}")
    if not np.array_equal(bundle_ids, mask_ids) or not np.array_equal(bundle_names, mask_names):
        raise ValueError("AAL3 IDs/names do not align between session bundle and SC mask")

    reference_full = pd.read_csv(reference_full_file)
    reference_26 = pd.read_csv(reference_26_file)
    if len(reference_full) != N_SESSIONS or len(reference_26) != N_SESSIONS:
        raise ValueError(f"Frozen normative-SC reference tables must contain {N_SESSIONS} sessions")
    key_columns = ["subject_key", "session"]
    for column in key_columns:
        if not np.array_equal(
            identity[column].astype(str).to_numpy(),
            reference_full[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"Session bundle order does not match reference column {column}")

    mwc_all = np.full((N_SESSIONS, N_NODES), np.nan, dtype=float)
    isolated_counts = np.zeros(N_SESSIONS, dtype=int)
    maximum_row_error = 0.0
    for index in range(N_SESSIONS):
        mwc, isolated, row_error = calculate_session_mwc(
            gcmi_all[index], pet_all[index], structural_mask
        )
        mwc_all[index] = mwc
        isolated_counts[index] = isolated
        maximum_row_error = max(maximum_row_error, row_error)

    full_columns = [
        f"MwC_AAL3_{int(aal3_id):03d}_{roi_name}"
        for aal3_id, roi_name in zip(mask_ids, mask_names)
    ]
    if full_columns != [column for column in reference_full if column.startswith("MwC_AAL3_")]:
        raise ValueError("AAL3-144 output column order does not match the validated reference")
    metadata_columns = [column for column in reference_full if column not in full_columns]
    recomputed_full = pd.concat(
        [reference_full[metadata_columns].copy(), pd.DataFrame(mwc_all, columns=full_columns)],
        axis=1,
    )

    roi = pd.read_csv(roi_file)
    if len(roi) != 26 or roi.groupby("network").size().to_dict() != {
        "cortical": 12,
        "limbic": 14,
    }:
        raise ValueError("Cortico-limbic definition must contain 12 cortical and 14 limbic ROIs")
    full_lookup = {int(aal3_id): index for index, aal3_id in enumerate(mask_ids)}
    offsets = np.asarray([full_lookup[int(aal3_id)] for aal3_id in roi.AAL3_ID], dtype=int)
    selected_columns = [f"MwC_{name}" for name in roi.ROI]
    selected_values = mwc_all[:, offsets]
    selected_metadata = [column for column in reference_26 if not column.startswith("MwC_")]
    recomputed_26 = pd.concat(
        [reference_26[selected_metadata].copy(), pd.DataFrame(selected_values, columns=selected_columns)],
        axis=1,
    )

    reference_full_values = reference_full[full_columns].to_numpy(float)
    reference_26_values = reference_26[selected_columns].to_numpy(float)
    full_error = float(np.nanmax(np.abs(mwc_all - reference_full_values)))
    selected_error = float(np.nanmax(np.abs(selected_values - reference_26_values)))
    if full_error >= REFERENCE_TOLERANCE or selected_error >= REFERENCE_TOLERANCE:
        raise ValueError(
            f"Recomputed MwC differs from validated reference: {full_error}/{selected_error}"
        )
    if not np.array_equal(np.isnan(mwc_all), np.isnan(reference_full_values)):
        raise ValueError("Recomputed full-144 missingness differs from validated reference")

    full_output = output_root / "subject_level_AAL3_144_normativeSC10_MwC_wide.csv.gz"
    selected_output = (
        output_root / "subject_level_corticolimbic26_normativeSC10_MwC_wide.csv"
    )
    report_output = output_root / "calculation_validation_report.json"
    recomputed_full.to_csv(
        full_output, index=False, compression="gzip", float_format="%.12g"
    )
    recomputed_26.to_csv(selected_output, index=False, float_format="%.12g")

    checks = {
        "sessions_253": len(recomputed_full) == N_SESSIONS,
        "retained_AAL3_nodes_144": len(full_columns) == N_NODES,
        "possible_undirected_edges_10296": EXPECTED_POSSIBLE_EDGES == 10_296,
        "fixed_dTOR_normative_SC_edges_1030": retained_sc_edges == EXPECTED_SC_EDGES,
        "structural_mask_symmetric": bool(np.array_equal(structural_mask, structural_mask.T)),
        "structural_mask_diagonal_zero": bool(not np.any(np.diag(structural_mask))),
        "gcmi_floor_bits_0p01": MI_FLOOR_BITS == 0.01,
        "row_normalization_error_below_1e_12": maximum_row_error < 1e-12,
        "all_144_nodes_computed_before_26_extraction": bool(
            np.allclose(selected_values, mwc_all[:, offsets], equal_nan=True)
        ),
        "isolated_nodes_remain_missing": int(np.isnan(mwc_all).sum())
        == int(isolated_counts.sum()),
        "full_144_matches_validated_reference": full_error < REFERENCE_TOLERANCE,
        "selected_26_matches_validated_reference": selected_error < REFERENCE_TOLERANCE,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "metric": "MwC only",
            "structural_connectome": (
                "dTOR-985 group-normative direct streamline-count top-10% mask"
            ),
            "structural_edges": retained_sc_edges,
            "functional_weight": "bias-corrected GCMI >= 0.01 bits within fixed SC mask",
            "normalization": "row-normalize each non-isolated target",
            "formula": "MwC = 100 * row-normalized SC-constrained GCMI @ PET wbSUVR",
            "target_extraction": "26 targets selected after complete AAL3-144 computation",
            "missing_policy": "isolated nodes remain missing; no imputation or zero fill",
        },
        "checks": checks,
        "audit": {
            "maximum_abs_error_vs_reference_full_144": full_error,
            "maximum_abs_error_vs_reference_selected_26": selected_error,
            "maximum_valid_row_sum_error": maximum_row_error,
            "total_missing_MwC_cells": int(np.isnan(mwc_all).sum()),
        },
        "inputs": {
            "session_bundle": bundle_file.relative_to(base).as_posix(),
            "session_bundle_sha256": sha256(bundle_file),
            "structural_mask": mask_file.relative_to(base).as_posix(),
            "structural_mask_sha256": sha256(mask_file),
        },
        "outputs": {
            "full_144": full_output.relative_to(base).as_posix(),
            "selected_26": selected_output.relative_to(base).as_posix(),
        },
    }
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "PASS":
        raise RuntimeError(f"MwC calculation validation failed: {checks}")
    return {
        "full_144": full_output,
        "selected_26": selected_output,
        "validation": report_output,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path.cwd())
    args = parser.parse_args()
    outputs = recompute(args.base)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    print("PASS: AAL3-144 normative-SC-constrained MwC recomputed before 26-ROI extraction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
