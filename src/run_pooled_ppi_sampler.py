#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class Protein:
    protein_id: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def extract_fasta_id(header: str) -> str:
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
                    proteins[current_id] = Protein(current_id, "".join(current_seq))

                current_id = extract_fasta_id(line)
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        proteins[current_id] = Protein(current_id, "".join(current_seq))

    if not proteins:
        raise ValueError(f"No proteins found in FASTA: {fasta_path}")

    return proteins


def write_lengths_table(proteins: dict[str, Protein], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for protein_id, protein in proteins.items():
            handle.write(f"{protein_id}\t{protein.length}\n")


def run_pooled_ppi_cli(
    proteins: dict[str, Protein],
    max_pool_size: int,
    pooled_ppi_command: str,
    max_pools: int | None,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        lengths_path = Path(tmpdir) / "protein_lengths.tsv"
        write_lengths_table(proteins, lengths_path)

        command = [
            pooled_ppi_command,
            "--max_pool_size",
            str(max_pool_size),
        ]

        if max_pools is not None:
            command.extend(["--max_pools", str(max_pools)])

        with lengths_path.open("r", encoding="utf-8") as stdin_handle:
            result = subprocess.run(
                command,
                stdin=stdin_handle,
                capture_output=True,
                text=True,
                check=True,
            )

    if result.stderr:
        print(result.stderr)

    return pd.read_csv(StringIO(result.stdout), sep="\t")


def build_output_table(
    pooled_ppi_df: pd.DataFrame,
    all_proteins: dict[str, Protein],
    fixed_bait_id: str | None,
    max_pool_size: int,
) -> pd.DataFrame:
    rows = []

    for _, row in pooled_ppi_df.iterrows():
        sampled_ids = row["pool_id"].split("_")

        if fixed_bait_id is None:
            protein_ids = sampled_ids
            prey_ids = []
            mode = "all_vs_all"
        else:
            prey_ids = [pid for pid in sampled_ids if pid != fixed_bait_id]
            protein_ids = [fixed_bait_id] + prey_ids
            mode = "all_vs_all_fixed_bait"

        total_length = sum(all_proteins[pid].length for pid in protein_ids)

        if total_length > max_pool_size:
            print(
                f"WARNING: skipping pool because total length exceeds max_pool_size: "
                f"{total_length} > {max_pool_size}"
            )
            continue

        rows.append(
            {
                "pool_id": f"pool_{len(rows) + 1:05d}",
                "mode": mode,
                "fixed_bait_id": fixed_bait_id or "",
                "protein_ids": ";".join(protein_ids),
                "prey_ids": ";".join(prey_ids),
                "n_proteins": len(protein_ids),
                "n_prey": len(prey_ids) if fixed_bait_id else "",
                "total_length": total_length,
                "original_pooled_ppi_pool_id": row["pool_id"],
                "original_pool_size": row.get("pool_size", ""),
            }
        )

    return pd.DataFrame(rows)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)

    if config["mode"] != "all_vs_all":
        raise ValueError("This wrapper currently supports mode: all_vs_all")

    fasta_path = Path(config["input"]["fasta_path"])
    fixed_bait_id = config["input"].get("fixed_bait_id")

    max_pool_size = int(config["pooling"].get("max_pool_size", 5000))
    max_pools = config["pooling"].get("max_pools")
    pooled_ppi_command = config["pooling"].get("pooled_ppi_command", "pooled-ppi-sample")

    output_path = Path(config["output"]["pools_tsv"])

    proteins = read_fasta(fasta_path)

    if fixed_bait_id:
        if fixed_bait_id not in proteins:
            raise ValueError(f"fixed_bait_id '{fixed_bait_id}' not found in FASTA.")

        bait_length = proteins[fixed_bait_id].length

        if bait_length >= max_pool_size:
            raise ValueError(
                f"Fixed bait length ({bait_length}) is >= max_pool_size ({max_pool_size})."
            )

        proteins_for_sampler = {
            pid: protein for pid, protein in proteins.items() if pid != fixed_bait_id
        }

        sampler_max_pool_size = max_pool_size - bait_length

    else:
        proteins_for_sampler = proteins
        sampler_max_pool_size = max_pool_size

    pooled_ppi_df = run_pooled_ppi_cli(
        proteins=proteins_for_sampler,
        max_pool_size=sampler_max_pool_size,
        pooled_ppi_command=pooled_ppi_command,
        max_pools=max_pools,
    )

    output_df = build_output_table(
        pooled_ppi_df=pooled_ppi_df,
        all_proteins=proteins,
        fixed_bait_id=fixed_bait_id,
        max_pool_size=max_pool_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, sep="\t", index=False)

    print("Done.")
    print(f"Mode: all_vs_all")
    print(f"Fixed bait: {fixed_bait_id or 'None'}")
    print(f"Proteins loaded: {len(proteins)}")
    print(f"Proteins passed to pooled-ppi: {len(proteins_for_sampler)}")
    print(f"Pools written: {len(output_df)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()