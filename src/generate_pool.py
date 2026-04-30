#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


Mode = Literal["bait_vs_all", "all_vs_all"]
Weighting = Literal["length", "target_x_prey_length", "none"]


@dataclass(frozen=True)
class Protein:
    protein_id: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class Pool:
    pool_id: str
    mode: str
    replicate: int
    protein_ids: list[str]
    target_id: str | None
    prey_ids: list[str]
    total_length: int


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

                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        proteins[current_id] = Protein(current_id, "".join(current_seq))

    if not proteins:
        raise ValueError(f"No proteins found in FASTA: {fasta_path}")

    return proteins


def filter_proteins(
    proteins: dict[str, Protein],
    max_protein_size: int | None,
) -> dict[str, Protein]:
    if max_protein_size is None:
        return proteins

    filtered = {
        protein_id: protein
        for protein_id, protein in proteins.items()
        if protein.length <= max_protein_size
    }

    removed = sorted(set(proteins) - set(filtered))
    if removed:
        print(f"WARNING: removed {len(removed)} proteins longer than {max_protein_size} aa")

    return filtered


def compute_bait_weights(
    prey_lengths: np.ndarray,
    target_length: int,
    weighting: Weighting,
) -> np.ndarray:
    if weighting == "length":
        return prey_lengths.astype(float)

    if weighting == "target_x_prey_length":
        return target_length * prey_lengths.astype(float)

    if weighting == "none":
        return np.ones_like(prey_lengths, dtype=float)

    raise ValueError(f"Unsupported weighting mode: {weighting}")


def generate_bait_vs_all_pools(
    proteins: dict[str, Protein],
    target_id: str,
    max_pool_size: int,
    n_replicates: int,
    seed: int,
    weighting: Weighting,
    shuffle_ties: bool,
) -> list[Pool]:
    if target_id not in proteins:
        raise ValueError(f"Target ID '{target_id}' not found in FASTA.")

    target = proteins[target_id]

    if target.length >= max_pool_size:
        raise ValueError(
            f"Target length ({target.length}) is >= max_pool_size ({max_pool_size})."
        )

    prey_ids = np.array([pid for pid in proteins if pid != target_id], dtype=object)
    prey_lengths = np.array([proteins[pid].length for pid in prey_ids], dtype=int)

    feasible_mask = target.length + prey_lengths <= max_pool_size

    if not np.all(feasible_mask):
        skipped = prey_ids[~feasible_mask]
        print(
            f"WARNING: skipping {len(skipped)} prey proteins because target + prey "
            f"exceeds max_pool_size."
        )

    prey_ids = prey_ids[feasible_mask]
    prey_lengths = prey_lengths[feasible_mask]

    if len(prey_ids) == 0:
        raise ValueError("No feasible prey proteins can fit with the target.")

    rng = random.Random(seed)

    coverage = np.zeros(len(prey_ids), dtype=int)
    weights = compute_bait_weights(
        prey_lengths=prey_lengths,
        target_length=target.length,
        weighting=weighting,
    )

    pools: list[Pool] = []

    while np.any(coverage < n_replicates):
        current_indices: list[int] = []
        current_length = target.length

        while True:
            remaining_capacity = max_pool_size - current_length

            candidate_mask = (
                (coverage < n_replicates)
                & (prey_lengths <= remaining_capacity)
            )

            if not np.any(candidate_mask):
                break

            candidate_indices = np.where(candidate_mask)[0]

            # Priority:
            # 1. proteins with fewer current target-prey observations
            # 2. higher weighted gain
            # 3. better fit into remaining capacity
            min_coverage = np.min(coverage[candidate_indices])
            candidate_indices = candidate_indices[
                coverage[candidate_indices] == min_coverage
            ]

            candidate_scores = weights[candidate_indices]

            max_score = np.max(candidate_scores)
            best_indices = candidate_indices[candidate_scores == max_score]

            if shuffle_ties and len(best_indices) > 1:
                chosen_idx = rng.choice(best_indices.tolist())
            else:
                chosen_idx = int(best_indices[0])

            current_indices.append(chosen_idx)
            current_length += int(prey_lengths[chosen_idx])

        if not current_indices:
            missing = prey_ids[coverage < n_replicates]
            raise RuntimeError(
                "Could not generate more valid pools. Missing prey examples: "
                + ", ".join(map(str, missing[:10]))
            )

        coverage[current_indices] += 1

        prey_in_pool = [str(prey_ids[i]) for i in current_indices]
        pool_protein_ids = [target_id] + prey_in_pool

        pool_number = len(pools) + 1
        replicate = int(np.max(coverage[current_indices]))

        pools.append(
            Pool(
                pool_id=f"pool_{pool_number:05d}",
                mode="bait_vs_all",
                replicate=replicate,
                protein_ids=pool_protein_ids,
                target_id=target_id,
                prey_ids=prey_in_pool,
                total_length=current_length,
            )
        )

    return pools


def write_pools_tsv(pools: list[Pool], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pool_id",
        "mode",
        "replicate",
        "target_id",
        "protein_ids",
        "prey_ids",
        "n_proteins",
        "n_prey",
        "total_length",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for pool in pools:
            writer.writerow(
                {
                    "pool_id": pool.pool_id,
                    "mode": pool.mode,
                    "replicate": pool.replicate,
                    "target_id": pool.target_id or "",
                    "protein_ids": ";".join(pool.protein_ids),
                    "prey_ids": ";".join(pool.prey_ids),
                    "n_proteins": len(pool.protein_ids),
                    "n_prey": len(pool.prey_ids),
                    "total_length": pool.total_length,
                }
            )


def write_bait_summary(
    pools: list[Pool],
    proteins: dict[str, Protein],
    target_id: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {
        pid: 0 for pid in proteins if pid != target_id
    }

    for pool in pools:
        for prey_id in pool.prey_ids:
            counts[prey_id] = counts.get(prey_id, 0) + 1

    fieldnames = [
        "target_id",
        "prey_id",
        "prey_length",
        "n_observations",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for prey_id, count in sorted(counts.items()):
            if prey_id not in proteins:
                continue

            writer.writerow(
                {
                    "target_id": target_id,
                    "prey_id": prey_id,
                    "prey_length": proteins[prey_id].length,
                    "n_observations": count,
                }
            )


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)

    mode: Mode = config["mode"]
    fasta_path = Path(config["input"]["fasta_path"])
    target_id = config["input"].get("target_id")

    pooling_cfg = config["pooling"]
    max_pool_size = int(pooling_cfg.get("max_pool_size", 5000))
    n_replicates = int(pooling_cfg.get("n_replicates", 1))
    seed = int(pooling_cfg.get("seed", 42))
    max_protein_size = pooling_cfg.get("max_protein_size")
    weighting: Weighting = pooling_cfg.get("weighting", "length")
    shuffle_ties = bool(pooling_cfg.get("shuffle_ties", True))

    output_cfg = config["output"]
    pools_tsv = Path(output_cfg["pools_tsv"])
    summary_tsv = Path(output_cfg.get("summary_tsv", pools_tsv.with_name("pool_summary.tsv")))

    proteins = read_fasta(fasta_path)
    proteins = filter_proteins(proteins, max_protein_size)

    if mode == "bait_vs_all":
        if target_id is None:
            raise ValueError("target_id is required for mode='bait_vs_all'.")

        pools = generate_bait_vs_all_pools(
            proteins=proteins,
            target_id=target_id,
            max_pool_size=max_pool_size,
            n_replicates=n_replicates,
            seed=seed,
            weighting=weighting,
            shuffle_ties=shuffle_ties,
        )

        write_pools_tsv(pools, pools_tsv)
        write_bait_summary(pools, proteins, target_id, summary_tsv)

    elif mode == "all_vs_all":
        raise NotImplementedError(
            "For all_vs_all, use the original pooled-ppi sampler for now. "
            "We can wrap it in this script later."
        )

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    print("Done.")
    print(f"Mode: {mode}")
    print(f"Proteins loaded: {len(proteins)}")
    print(f"Pools generated: {len(pools)}")
    print(f"Pools TSV: {pools_tsv}")
    print(f"Summary TSV: {summary_tsv}")


if __name__ == "__main__":
    main()