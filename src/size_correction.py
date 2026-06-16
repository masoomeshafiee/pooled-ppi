#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml


CorrectionMethod = Literal["pooled_ppi_fixed", "fit_from_data"]
CorrectionType = Literal["subtract", "ratio"]
ResolveReplicates = Literal[
    "mean",
    "median",
    "max",
    "min",
    "first",
    "geo_mean",
]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = {
        "protein_a",
        "protein_b",
        "protein_a_length",
        "protein_b_length",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    swap_mask = df["protein_a"].astype(str) > df["protein_b"].astype(str)

    df["protein_1"] = np.where(swap_mask, df["protein_b"], df["protein_a"])
    df["protein_2"] = np.where(swap_mask, df["protein_a"], df["protein_b"])

    df["protein_1_length"] = np.where(
        swap_mask,
        df["protein_b_length"],
        df["protein_a_length"],
    )

    df["protein_2_length"] = np.where(
        swap_mask,
        df["protein_a_length"],
        df["protein_b_length"],
    )

    return df


def fit_linear_baseline(
    df: pd.DataFrame,
    score_column: str,
    root_to_use: float,
) -> tuple[float, float]:
    """
    Python equivalent of the paper's correction model:

    score ~ (protein_1_length + protein_2_length) ^ root_to_use

    The R code uses robustbase::lmrob. Here we use ordinary least squares
    for portability. If you want robust fitting later, we can add statsmodels.
    """
    unique_pairs = (
        df.dropna(subset=[score_column])
        .drop_duplicates(subset=["protein_1", "protein_2"])
        .copy()
    )

    x = (
        unique_pairs["protein_1_length"].astype(float)
        + unique_pairs["protein_2_length"].astype(float)
    ) ** root_to_use

    y = unique_pairs[score_column].astype(float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        raise ValueError("Need at least two valid unique pairs to fit correction model.")

    slope, intercept = np.polyfit(x[valid], y[valid], deg=1)

    return float(intercept), float(slope)


def apply_size_correction(
    df: pd.DataFrame,
    score_column: str,
    method: CorrectionMethod,
    root_to_use: float,
    correction_type: CorrectionType,
    fixed_intercept: float,
    fixed_slope: float,
) -> tuple[pd.DataFrame, dict]:
    df = canonicalize_pairs(df)

    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found.")

    if method == "pooled_ppi_fixed":
        intercept = fixed_intercept
        slope = fixed_slope

    elif method == "fit_from_data":
        intercept, slope = fit_linear_baseline(
            df=df,
            score_column=score_column,
            root_to_use=root_to_use,
        )

    else:
        raise ValueError(f"Unsupported correction method: {method}")

    corrected = df.copy()

    corrected["pair_tokens"] = (
        corrected["protein_1_length"].astype(float)
        + corrected["protein_2_length"].astype(float)
    )

    corrected["size_feature"] = corrected["pair_tokens"] ** root_to_use

    corrected["size_baseline"] = (
        intercept + slope * corrected["size_feature"]
    )

    if correction_type == "subtract":
        corrected["score_size_corrected"] = (
            corrected[score_column] - corrected["size_baseline"]
        )

    elif correction_type == "ratio":
        corrected["score_size_corrected"] = (
            corrected[score_column] / corrected["size_baseline"]
        )

    else:
        raise ValueError(f"Unsupported correction type: {correction_type}")

    model_info = {
        "method": method,
        "score_column": score_column,
        "root_to_use": root_to_use,
        "correction_type": correction_type,
        "intercept": intercept,
        "slope": slope,
        "formula": (
            f"{score_column} ~ intercept + slope * "
            f"(protein_1_length + protein_2_length)^{root_to_use}"
        ),
        "n_observations": len(corrected),
        "n_unique_pairs": corrected[["protein_1", "protein_2"]]
        .drop_duplicates()
        .shape[0],
    }

    return corrected, model_info


def geometric_mean(values: pd.Series) -> float:
    values = values.astype(float)
    values = values[values > 0]

    if len(values) == 0:
        return np.nan

    return float(np.exp(np.mean(np.log(values))))


def resolve_replicates(values: pd.Series, method: ResolveReplicates) -> float:
    if method == "mean":
        return float(values.mean())

    if method == "median":
        return float(values.median())

    if method == "max":
        return float(values.max())

    if method == "min":
        return float(values.min())

    if method == "first":
        return float(values.iloc[0])

    if method == "geo_mean":
        return geometric_mean(values)

    raise ValueError(f"Unsupported replicate resolution method: {method}")


def aggregate_corrected_scores(
    corrected_df: pd.DataFrame,
    raw_score_column: str,
    resolve_method: ResolveReplicates,
) -> pd.DataFrame:
    rows = []

    grouped = corrected_df.groupby(["protein_1", "protein_2"], sort=False)

    for (protein_1, protein_2), group in grouped:
        rows.append(
            {
                "protein_1": protein_1,
                "protein_2": protein_2,
                "protein_1_length": int(group["protein_1_length"].iloc[0]),
                "protein_2_length": int(group["protein_2_length"].iloc[0]),
                "pair_tokens": float(group["pair_tokens"].iloc[0]),
                "n_observations": int(len(group)),
                "raw_score_resolved": resolve_replicates(
                    group[raw_score_column],
                    resolve_method,
                ),
                "raw_score_mean": float(group[raw_score_column].mean()),
                "raw_score_median": float(group[raw_score_column].median()),
                "raw_score_max": float(group[raw_score_column].max()),
                "raw_score_min": float(group[raw_score_column].min()),
                "corrected_score_resolved": resolve_replicates(
                    group["score_size_corrected"],
                    resolve_method,
                ),
                "corrected_score_mean": float(group["score_size_corrected"].mean()),
                "corrected_score_median": float(group["score_size_corrected"].median()),
                "corrected_score_max": float(group["score_size_corrected"].max()),
                "corrected_score_min": float(group["score_size_corrected"].min()),
                "pools": ";".join(sorted(set(group["pool_name"].astype(str)))),
            }
        )

    aggregated = pd.DataFrame(rows)

    aggregated = aggregated.sort_values(
        by=["corrected_score_resolved", "raw_score_resolved"],
        ascending=False,
    ).reset_index(drop=True)

    return aggregated


def size_corrector(config: dict) -> None:
    

    input_path = Path(config["size_correction_input"]["pair_scores_tsv"])

    corrected_observations_tsv = Path(
        config["size_correction_output"]["corrected_observations_tsv"]
    )
    corrected_aggregated_tsv = Path(
        config["size_correction_output"]["corrected_aggregated_tsv"]
    )
    correction_model_tsv = Path(config["size_correction_output"]["correction_model_tsv"])

    correction_cfg = config["size_correction"]
    aggregation_cfg = config["aggregation"]

    score_column = correction_cfg.get("score_column", "chain_pair_iptm")
    method = correction_cfg.get("method", "pooled_ppi_fixed")
    root_to_use = float(correction_cfg.get("root_to_use", 0.5))
    correction_type = correction_cfg.get("type_of_correction", "subtract")
    fixed_intercept = float(correction_cfg.get("fixed_intercept", 0.04))
    fixed_slope = float(correction_cfg.get("fixed_slope", 0.0044))

    resolve_method = aggregation_cfg.get("resolve_replicates", "mean")

    raw_df = pd.read_csv(input_path, sep="\t")

    corrected_df, model_info = apply_size_correction(
        df=raw_df,
        score_column=score_column,
        method=method,
        root_to_use=root_to_use,
        correction_type=correction_type,
        fixed_intercept=fixed_intercept,
        fixed_slope=fixed_slope,
    )

    aggregated_df = aggregate_corrected_scores(
        corrected_df=corrected_df,
        raw_score_column=score_column,
        resolve_method=resolve_method,
    )

    corrected_observations_tsv.parent.mkdir(parents=True, exist_ok=True)
    corrected_aggregated_tsv.parent.mkdir(parents=True, exist_ok=True)
    correction_model_tsv.parent.mkdir(parents=True, exist_ok=True)

    corrected_df.to_csv(corrected_observations_tsv, sep="\t", index=False)
    aggregated_df.to_csv(corrected_aggregated_tsv, sep="\t", index=False)
    pd.DataFrame([model_info]).to_csv(correction_model_tsv, sep="\t", index=False)

    print("Done.")
    print(f"Raw observations: {len(raw_df)}")
    print(f"Unique pairs: {len(aggregated_df)}")
    print(f"Correction method: {method}")
    print(f"Correction model: intercept={model_info['intercept']}, slope={model_info['slope']}")
    print(f"Corrected observations: {corrected_observations_tsv}")
    print(f"Corrected aggregated scores: {corrected_aggregated_tsv}")
    print(f"Model info: {correction_model_tsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    size_corrector(config)