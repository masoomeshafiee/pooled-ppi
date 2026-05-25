from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Protein:
    protein_id: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def extract_fasta_id(header: str) -> str:
    """
    Extract clean IDs from UniProt or simple FASTA headers.
    """
    header = header.strip().lstrip(">")

    gene_match = re.search(r"\bGN=([A-Za-z0-9_.-]+)", header)
    if gene_match:
        return gene_match.group(1)

    first_token = header.split()[0]

    if "|" in first_token:
        parts = first_token.split("|")
        if len(parts) >= 3:
            entry_name = parts[2]
            return entry_name.split("_")[0] if "_" in entry_name else entry_name

    return first_token


def validate_sequence(protein_id: str, sequence: str) -> str:
    """
    Validate sequence for AlphaFold Server.
    Only 20 standard amino acids allowed.
    """
    sequence = sequence.upper().replace(" ", "").replace("*", "")

    invalid = sorted(set(sequence) - VALID_AA)

    if invalid:
        raise ValueError(
            f"Protein '{protein_id}' contains invalid amino acid characters: {invalid}"
        )

    if not sequence:
        raise ValueError(f"Protein '{protein_id}' has empty sequence.")

    return sequence


def read_fasta(fasta_path: Path) -> dict[str, Protein]:
    proteins: dict[str, Protein] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    seq = validate_sequence(current_id, "".join(current_seq))
                    proteins[current_id] = Protein(current_id, seq)

                current_id = extract_fasta_id(line)
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        seq = validate_sequence(current_id, "".join(current_seq))
        proteins[current_id] = Protein(current_id, seq)

    if not proteins:
        raise ValueError(f"No proteins found in FASTA: {fasta_path}")

    return proteins


def make_pseudo_chain_ids(n: int) -> list[str]:
    """
    A-Z labels used only in mapping TSV.
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if n > len(alphabet):
        raise ValueError(
            f"Supports up to {len(alphabet)} proteins per pool. Requested {n}."
        )

    return list(alphabet[:n])


def parse_protein_ids_from_pool_row(row: pd.Series) -> list[str]:
    """
    Parse semicolon-separated protein IDs.
    """
    if "protein_ids" not in row:
        raise ValueError("pools.tsv must contain 'protein_ids' column.")

    raw = str(row["protein_ids"]).strip()

    if not raw:
        raise ValueError(f"Empty protein_ids row: {row.to_dict()}")

    return [pid.strip() for pid in raw.split(";") if pid.strip()]


def build_server_json(
    pool_name: str,
    protein_ids: list[str],
    proteins: dict[str, Protein],
    dialect: str,
    version: int,
    use_structure_template: bool,
    count: int,
) -> tuple[dict, list[dict]]:
    """
    Build one AlphaFold Server job.
    """
    pseudo_chain_ids = make_pseudo_chain_ids(len(protein_ids))

    sequences = []
    mapping_rows = []

    for entity_index, (pseudo_chain_id, protein_id) in enumerate(
        zip(pseudo_chain_ids, protein_ids),
        start=1,
    ):
        if protein_id not in proteins:
            raise ValueError(
                f"Protein ID '{protein_id}' from pools.tsv not found in FASTA."
            )

        protein = proteins[protein_id]

        sequences.append(
            {
                "proteinChain": {
                    "sequence": protein.sequence,
                    "count": count,
                    "useStructureTemplate": use_structure_template,
                }
            }
        )

        mapping_rows.append(
            {
                "pool_name": pool_name,
                "entity_index": entity_index,
                "pseudo_chain_id": pseudo_chain_id,
                "protein_id": protein_id,
                "count": count,
                "protein_length": protein.length,
                "total_residues_for_entity": protein.length * count,
            }
        )

    job = {
        "name": pool_name,
        "sequences": sequences,
        "dialect": dialect,
        "version": version,
    }

    return job, mapping_rows


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """
    Split list into chunks.
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def write_json(path: Path, payload, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"File exists and overwrite=false: {path}")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def json_builder(config: dict) -> None:


    fasta_path = Path(config["af3_json_build_input"]["fasta_path"])
    pools_tsv = Path(config["af3_json_build_input"]["pools_tsv"])
    json_dir = Path(config["af3_json_output"]["json_dir"])

    server_cfg = config.get("alphafold3", {})
    options_cfg = config.get("json_preparation_options", {})

    dialect = server_cfg.get("dialect", "alphafoldserver")
    version = int(server_cfg.get("version", 1))
    use_structure_template = bool(
        server_cfg.get("use_structure_template", True)
    )
    count = int(server_cfg.get("count", 1))

    overwrite = bool(options_cfg.get("overwrite", True))
    batch_size = int(options_cfg.get("batch_size", 100))

    json_dir.mkdir(parents=True, exist_ok=True)

    proteins = read_fasta(fasta_path)
    pools_df = pd.read_csv(pools_tsv, sep="\t")

    all_mapping_rows = []
    all_jobs = []

    for _, row in pools_df.iterrows():
        pool_id = str(row["pool_id"])
        protein_ids = parse_protein_ids_from_pool_row(row)

        job, mapping_rows = build_server_json(
            pool_name=pool_id,
            protein_ids=protein_ids,
            proteins=proteins,
            dialect=dialect,
            version=version,
            use_structure_template=use_structure_template,
            count=count,
        )

        # individual json file
        output_json = json_dir / f"{pool_id}.json"
        write_json(output_json, job, overwrite=overwrite)

        all_jobs.append(job)
        all_mapping_rows.extend(mapping_rows)

    # batch json files
    batches = chunk_list(all_jobs, batch_size)

    for batch_index, batch_jobs in enumerate(batches, start=1):
        batch_path = json_dir / f"batch_{batch_index:03d}.json"
        write_json(batch_path, batch_jobs, overwrite=overwrite)

    # mapping
    mapping_df = pd.DataFrame(all_mapping_rows)
    mapping_path = json_dir.parent / "pool_chain_mapping.tsv"
    mapping_df.to_csv(mapping_path, sep="\t", index=False)

    print("Done.")
    print(f"Proteins loaded: {len(proteins)}")
    print(f"Pools loaded: {len(pools_df)}")
    print(f"Individual JSON files written: {len(all_jobs)}")
    print(f"Batch JSON files written: {len(batches)}")
    print(f"JSON directory: {json_dir}")
    print(f"Chain mapping written: {mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    json_builder(config)