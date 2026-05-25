from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_protein_sequence(sequence: str, json_name: str, entity_index: int) -> int:
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError(
            f"{json_name}: entity {entity_index} has an empty protein sequence."
        )

    sequence = sequence.upper().strip()
    invalid = sorted(set(sequence) - VALID_AA)

    if invalid:
        raise ValueError(
            f"{json_name}: entity {entity_index} contains invalid amino acid "
            f"characters: {invalid}. AlphaFold Server supports only 20 standard amino acids."
        )

    return len(sequence)


def get_protein_chain(entry: dict, json_name: str, entity_index: int) -> dict:
    if "proteinChain" not in entry:
        raise ValueError(
            f"{json_name}: entity {entity_index} must contain 'proteinChain'."
        )

    protein_chain = entry["proteinChain"]

    if not isinstance(protein_chain, dict):
        raise ValueError(
            f"{json_name}: entity {entity_index} 'proteinChain' must be a dictionary."
        )

    return protein_chain


def validate_protein_chain(entry: dict, json_name: str, entity_index: int) -> dict:
    protein_chain = get_protein_chain(entry, json_name, entity_index)

    sequence = protein_chain.get("sequence")
    sequence_length = validate_protein_sequence(sequence, json_name, entity_index)

    count = protein_chain.get("count", 1)

    if not isinstance(count, int) or count < 1:
        raise ValueError(
            f"{json_name}: entity {entity_index} count must be a positive integer."
        )

    use_structure_template = protein_chain.get("useStructureTemplate", True)

    if not isinstance(use_structure_template, bool):
        raise ValueError(
            f"{json_name}: entity {entity_index} useStructureTemplate must be boolean."
        )

    total_entity_length = sequence_length * count

    return {
        "entity_index": entity_index,
        "sequence_length": sequence_length,
        "count": count,
        "total_entity_length": total_entity_length,
        "useStructureTemplate": use_structure_template,
    }


def validate_json_file(json_path: Path, max_pool_size: int) -> dict:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    required_fields = {"name", "sequences", "dialect", "version"}
    missing = required_fields - set(payload)

    if missing:
        raise ValueError(f"{json_path.name}: missing fields: {sorted(missing)}")

    if payload["dialect"] != "alphafoldserver":
        raise ValueError(
            f"{json_path.name}: dialect must be 'alphafoldserver', "
            f"got {payload['dialect']!r}."
        )

    version = payload.get("version", 1)
    if version not in (None, 1):
        raise ValueError(
            f"{json_path.name}: AlphaFold Server version must be undefined/null or 1."
        )

    if "modelSeeds" in payload:
        raise ValueError(
            f"{json_path.name}: modelSeeds should not be included in AlphaFold Server JSON."
        )

    sequences = payload["sequences"]

    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"{json_path.name}: sequences must be a non-empty list.")

    entity_rows = []

    for entity_index, entry in enumerate(sequences, start=1):
        entity_rows.append(
            validate_protein_chain(
                entry=entry,
                json_name=json_path.name,
                entity_index=entity_index,
            )
        )

    total_length = sum(row["total_entity_length"] for row in entity_rows)

    if total_length > max_pool_size:
        raise ValueError(
            f"{json_path.name}: total length {total_length} exceeds "
            f"max_pool_size {max_pool_size}."
        )

    counts = [row["count"] for row in entity_rows]
    use_templates = [row["useStructureTemplate"] for row in entity_rows]

    return {
        "json_file": json_path.name,
        "pool_name": payload["name"],
        "n_entities": len(entity_rows),
        "total_length": total_length,
        "counts": ";".join(map(str, counts)),
        "useStructureTemplate": ";".join(map(str, use_templates)),
    }


def validate_against_mapping(validation_df: pd.DataFrame, mapping_path: Path) -> None:
    if not mapping_path.exists():
        return

    mapping_df = pd.read_csv(mapping_path, sep="\t")

    required_mapping_cols = {"pool_name", "entity_index", "protein_id", "count"}
    missing_cols = required_mapping_cols - set(mapping_df.columns)

    if missing_cols:
        raise ValueError(
            f"{mapping_path.name}: missing required columns: {sorted(missing_cols)}"
        )

    expected_pools = set(mapping_df["pool_name"].astype(str))
    json_pools = set(validation_df["pool_name"].astype(str))

    missing_jsons = expected_pools - json_pools
    extra_jsons = json_pools - expected_pools

    if missing_jsons:
        raise ValueError(f"Missing JSONs for pools: {sorted(missing_jsons)[:10]}")

    if extra_jsons:
        raise ValueError(f"Extra JSONs not in mapping: {sorted(extra_jsons)[:10]}")

    mapping_summary = (
        mapping_df.groupby("pool_name", as_index=False)
        .agg(
            expected_n_entities=("entity_index", "count"),
            expected_total_length=("total_residues_for_entity", "sum"),
        )
    )

    merged = validation_df.merge(
        mapping_summary,
        on="pool_name",
        how="left",
    )

    bad_entity_counts = merged[
        merged["n_entities"] != merged["expected_n_entities"]
    ]

    if not bad_entity_counts.empty:
        examples = bad_entity_counts[
            ["pool_name", "n_entities", "expected_n_entities"]
        ].head(10)
        raise ValueError(
            "Mismatch between JSON entity counts and mapping TSV:\n"
            f"{examples.to_string(index=False)}"
        )

    bad_lengths = merged[
        merged["total_length"] != merged["expected_total_length"]
    ]

    if not bad_lengths.empty:
        examples = bad_lengths[
            ["pool_name", "total_length", "expected_total_length"]
        ].head(10)
        raise ValueError(
            "Mismatch between JSON total lengths and mapping TSV:\n"
            f"{examples.to_string(index=False)}"
        )


def validate_jsons(config: dict) -> None:

    json_dir = Path(config["json_validation_input"]["json_dir"])
    mapping_path = Path(config["json_validation_input"]["chain_mapping_tsv"])
    max_pool_size = int(config["validation"]["max_pool_size"])

    #json_files = sorted(json_dir.glob("*.json"))
    json_files = sorted(
    path for path in json_dir.glob("*.json")
    if not path.name.startswith("batch_")
)

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    rows = []

    for json_path in json_files:
        rows.append(validate_json_file(json_path, max_pool_size=max_pool_size))

    validation_df = pd.DataFrame(rows)

    validate_against_mapping(validation_df, mapping_path)

    output_path = json_dir.parent / "alphafold_server_json_validation_summary.tsv"
    validation_df.to_csv(output_path, sep="\t", index=False)

    print("Validation passed.")
    print(f"JSON files checked: {len(json_files)}")
    print(f"Summary written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_jsons(config)