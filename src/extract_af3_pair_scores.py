#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import re


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_chain_mapping(mapping_path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path, sep="\t")

    required = {"pool_name","entity_index","pseudo_chain_id", "protein_id", "protein_length"}
    missing = required - set(mapping.columns)

    if missing:
        raise ValueError(f"Chain mapping is missing columns: {sorted(missing)}")

    return mapping


def find_summary_jsons_in_folder(prediction_dir: Path) -> list[Path]:
    return sorted(prediction_dir.rglob("*summary_confidences*.json"))


def read_json_from_zip(zip_path: Path, member_name: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member_name) as handle:
            return json.load(handle)


def find_summary_jsons_in_zip(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        return sorted(
            name
            for name in archive.namelist()
            if name.endswith(".json") and "summary_confidences" in name
        )


def read_json_file(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_pool_name_from_summary_name(name: str) -> str:
    """
    Handles names like:
    fold_pool_00001_summary_confidences.json
    some/folder/fold_pool_00001_summary_confidences.json
    """
    stem = Path(name).name

    ste_without_fold = stem.replace("fold_", "")
    pool_name = re.sub(r"_summary_confidences_.*\.json$", "", ste_without_fold)
    
    return pool_name


def choose_top_summary_json(summary_json_names: list[str]) -> str:
    """
    For AF3 server/local outputs, the top-ranked summary confidence file is usually:
    <job_name>_summary_confidences.json

    Sample-level files may be nested under seed/sample subdirectories.
    For now, prefer the shortest path, which usually corresponds to the top-ranked file.
    """
    if not summary_json_names:
        raise ValueError("No summary_confidences JSON found.")

    return sorted(summary_json_names, key=lambda x: (x.count("/"), len(x)))[0]


def get_chain_pair_matrix(summary: dict[str, Any]) -> np.ndarray:
    if "chain_pair_iptm" not in summary:
        raise KeyError(
            "summary_confidences JSON does not contain 'chain_pair_iptm'. "
            f"Available keys: {sorted(summary.keys())}"
        )

    matrix = np.asarray(summary["chain_pair_iptm"], dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"chain_pair_iptm must be a square 2D matrix. Got shape {matrix.shape}"
        )

    return matrix


def get_optional_matrix(summary: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in summary:
        return None

    matrix = np.asarray(summary[key], dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    return matrix


def extract_pairs_from_summary(
    pool_name: str,
    summary: dict[str, Any],
    mapping_df: pd.DataFrame,
    source_file: str,
    include_self_pairs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_mapping = mapping_df[mapping_df["pool_name"].astype(str) == pool_name].copy()

    if pool_mapping.empty:
        raise ValueError(f"No chain mapping found for pool_name={pool_name}")

    pool_mapping = pool_mapping.sort_values("pseudo_chain_id")

    pseudo_chain_ids = pool_mapping["pseudo_chain_id"].astype(str).tolist()

    #chain_ids = pool_mapping["chain_id"].astype(str).tolist()
    protein_ids = pool_mapping["protein_id"].astype(str).tolist()
    protein_lengths = pool_mapping["protein_length"].astype(int).tolist()

    iptm = get_chain_pair_matrix(summary)

    n_chains = len(pseudo_chain_ids)

    if iptm.shape[0] != n_chains:
        raise ValueError(
            f"{pool_name}: chain_pair_iptm shape {iptm.shape} does not match "
            f"{n_chains} chains in mapping."
        )

    pae = get_optional_matrix(summary, "chain_pair_pae_min")
    pair_plddt = get_optional_matrix(summary, "chain_pair_plddt")

    global_metrics = {
        "ranking_score": summary.get("ranking_score"),
        "iptm": summary.get("iptm"),
        "ptm": summary.get("ptm"),
        "fraction_disordered": summary.get("fraction_disordered"),
        "has_clash": summary.get("has_clash"),
    }

    rows = []

    if include_self_pairs:
        pair_indices = (
            (i, j)
            for i in range(n_chains)
            for j in range(i, n_chains)
        )
    else:
        pair_indices = combinations(range(n_chains), 2)
    for i, j in pair_indices:
        protein_a = protein_ids[i]
        protein_b = protein_ids[j]

        rows.append(
            {
                "pool_name": pool_name,
                "source_file": source_file,
                "chain_a": pseudo_chain_ids[i],
                "chain_b": pseudo_chain_ids[j],
                "protein_a": protein_a,
                "protein_b": protein_b,
                "protein_a_length": protein_lengths[i],
                "protein_b_length": protein_lengths[j],
                "pair_length_sum": protein_lengths[i] + protein_lengths[j],
                "chain_pair_iptm": float(iptm[i, j]),
                "chain_pair_pae_min": float(pae[i, j]) if pae is not None else np.nan,
                "chain_pair_plddt": (
                    float(pair_plddt[i, j]) if pair_plddt is not None else np.nan
                ),
                **global_metrics,
            }
        )
        print(f"Extracted pair: {protein_a} - {protein_b}, ipTM={iptm[i, j]:.3f}")

    summary_row = {
        "pool_name": pool_name,
        "source_file": source_file,
        "n_chains": n_chains,
        "n_pairs": len(rows),
        **global_metrics,
    }

    return rows, summary_row


def extract_from_zip(
    zip_path: Path,
    mapping_df: pd.DataFrame,
    include_self_pairs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_names = find_summary_jsons_in_zip(zip_path)
    selected_summary = choose_top_summary_json(summary_names)

    summary = read_json_from_zip(zip_path, selected_summary)
    pool_name = infer_pool_name_from_summary_name(selected_summary)

    pair_rows, summary_row = extract_pairs_from_summary(
        pool_name=pool_name,
        summary=summary,
        mapping_df=mapping_df,
        source_file=str(zip_path),
        include_self_pairs=include_self_pairs,
    )

    return pair_rows, [summary_row]


def extract_from_folder(
    predictions_dir: Path,
    mapping_df: pd.DataFrame,
    include_self_pairs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows_all = []
    summary_rows_all = []

    summary_paths = find_summary_jsons_in_folder(predictions_dir)
    #print(list(summary_paths))

    if not summary_paths:
        raise FileNotFoundError(
            f"No summary_confidences JSON files found in {predictions_dir}"
        )

    # Prefer top-level summary files; skip nested sample summaries for now
    # top_level_summary_paths = [
    #     path for path in summary_paths
    #     if path.name.endswith("_summary_confidences.json")
    # ]
    # print(list(top_level_summary_paths))

    for summary_path in summary_paths:
        
        summary = read_json_file(summary_path)
        pool_name = infer_pool_name_from_summary_name(summary_path.name)
        print(f"Processing pool: {pool_name} from file: {summary_path}")

        pair_rows, summary_row = extract_pairs_from_summary(
            pool_name=pool_name,
            summary=summary,
            mapping_df=mapping_df,
            source_file=str(summary_path),
            include_self_pairs=include_self_pairs,
        )

        pair_rows_all.extend(pair_rows)
        summary_rows_all.append(summary_row)

    return pair_rows_all, summary_rows_all


def score_extractor(config: dict) -> None:

    predictions_dir = Path(config["af3_result_input"]["predictions_dir"])
    mapping_path = Path(config["af3_result_input"]["chain_mapping_tsv"])

    pair_scores_tsv = Path(config["af3_pair_scores_output"]["pair_scores_tsv"])
    pool_summary_tsv = Path(config["af3_pool_summary_output"]["pool_summary_tsv"])

    include_self_pairs = bool(config["extraction_options"].get("include_self_pairs", False))

    mapping_df = load_chain_mapping(mapping_path)


    pair_rows_all = []
    summary_rows_all = []

    zip_files = sorted(predictions_dir.glob("*.zip"))

    if zip_files:
        for zip_path in zip_files:
            pair_rows, summary_rows = extract_from_zip(
                zip_path=zip_path,
                mapping_df=mapping_df,
                include_self_pairs=include_self_pairs,
            )

            pair_rows_all.extend(pair_rows)
            summary_rows_all.extend(summary_rows)
    else:
        pair_rows_all, summary_rows_all = extract_from_folder(
            predictions_dir=predictions_dir,
            mapping_df=mapping_df,
            include_self_pairs=include_self_pairs,
        )
        print(f"Extracted pair scores from folder: {predictions_dir} ({len(summary_rows_all)} pairs)")

    if not pair_rows_all:
        raise RuntimeError("No pair scores were extracted.")

    pair_scores_tsv.parent.mkdir(parents=True, exist_ok=True)
    pool_summary_tsv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(pair_rows_all).to_csv(pair_scores_tsv, sep="\t", index=False)
    pd.DataFrame(summary_rows_all).to_csv(pool_summary_tsv, sep="\t", index=False)

    print("Done.")
    print(f"Pair scores extracted: {len(pair_rows_all)}")
    print(f"Pools parsed: {len(summary_rows_all)}")
    print(f"Pair scores TSV: {pair_scores_tsv}")
    print(f"Pool summary TSV: {pool_summary_tsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    score_extractor(config)