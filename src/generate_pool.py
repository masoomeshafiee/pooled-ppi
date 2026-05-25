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
import re


Mode = Literal["one_vs_all", "all_vs_all"]
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
    replicate: int | None
    protein_ids: list[str]
    target_id: str | None
    prey_ids: list[str]
    total_length: int





def extract_fasta_id(header: str) -> str:
    """
    Extract a clean protein/gene ID from a FASTA header.

    Supports UniProt-style headers, for example:
    >sp|P27636|RFA1_YEAST Replication factor A protein 1 OS=... GN=RFA1 ...

    Priority:
    1. GN= gene name if present
    2. UniProt entry name before _YEAST, e.g. RFA1_YEAST -> RFA1
    3. First token after >
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

            if "_" in entry_name:
                return entry_name.split("_")[0]

            return entry_name

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

def generate_bait_vs_all_pools_randomized(
    proteins: dict[str, Protein],
    target_id: str,
    max_pool_size: int,
    n_replicates: int,
    seed: int,
    weighting: Weighting,
    shuffle_ties: bool,
) -> list[Pool]:
    """ Core idea

    For each replicate:

    Start with all prey in unused_indices
    Build one pool at a time
    Among prey that still fit in remaining capacity, choose one randomly using weights
    Remove it from unused set
    Continue until pool is full
    Start next pool with remaining prey

    once all prey have been used in this replicate, move to the next replicate and reset unused set

    So weighting influences packing order, not total coverage.
    
    """
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

    weights = compute_bait_weights(
        prey_lengths=prey_lengths,
        target_length=target.length,
        weighting=weighting,
    )

    # Avoid zero probabilities
    weights = np.asarray(weights, dtype=float)
    weights = np.where(weights <= 0, 1.0, weights)

    rng = np.random.default_rng(seed)
    pools: list[Pool] = []

    for replicate in range(1, n_replicates + 1):
        unused_indices = set(range(len(prey_ids)))

        while unused_indices:
            current_indices: list[int] = []
            current_length = target.length

            while True:
                remaining_capacity = max_pool_size - current_length

                feasible_indices = np.array(
                    [
                        idx
                        for idx in unused_indices
                        if prey_lengths[idx] <= remaining_capacity
                    ],
                    dtype=int,
                )

                if len(feasible_indices) == 0:
                    break

                feasible_weights = weights[feasible_indices].astype(float)
                probabilities = feasible_weights / feasible_weights.sum()

                chosen_idx = int(
                    rng.choice(feasible_indices, p=probabilities)
                )

                current_indices.append(chosen_idx)
                unused_indices.remove(chosen_idx)
                current_length += int(prey_lengths[chosen_idx])

            if not current_indices:
                missing = [str(prey_ids[idx]) for idx in unused_indices]
                raise RuntimeError(
                    "Could not fit remaining prey proteins into pools. "
                    f"Examples: {missing[:10]}"
                )

            prey_in_pool = [str(prey_ids[idx]) for idx in current_indices]
            pool_protein_ids = [target_id] + prey_in_pool

            pool_number = len(pools) + 1

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
def generate_bait_vs_all_pools_probabilistic(
    proteins: dict[str, Protein],
    target_id: str,
    max_pool_size: int,
    n_replicates: int,
    seed: int,
    weighting: Weighting,
    shuffle_ties: bool,
) -> list[Pool]:
    """Core idea

    This version does not use replicate rounds.

    Instead:

    Keep generating pools until every prey has appeared n_replicates times.
    At each selection step:
    only proteins still below target coverage are eligible
    among them, proteins with lowest current coverage are prioritized
    then weighted random choice is used

    So weighting affects which under-covered prey gets selected next.


    this creates: balanced total observations
                + randomness
                + weighted preference within fairness constraint

    weighting affects WHICH under-covered proteins get chosen.
    """
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

    weights = compute_bait_weights(
        prey_lengths=prey_lengths,
        target_length=target.length,
        weighting=weighting,
    )

    weights = np.asarray(weights, dtype=float)
    weights = np.where(weights <= 0, 1.0, weights)

    rng = np.random.default_rng(seed)

    coverage = np.zeros(len(prey_ids), dtype=int)
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

            # Prevent adding same prey twice to the same pool
            if current_indices:
                candidate_mask[np.array(current_indices, dtype=int)] = False

            if not np.any(candidate_mask):
                break

            candidate_indices = np.where(candidate_mask)[0]

            # Prefer prey with the lowest current observation count.
            min_coverage = np.min(coverage[candidate_indices])
            candidate_indices = candidate_indices[
                coverage[candidate_indices] == min_coverage
            ]

            # Weighted random choice among lowest-coverage feasible candidates.
            candidate_weights = weights[candidate_indices]
            probabilities = candidate_weights / candidate_weights.sum()

            chosen_idx = int(rng.choice(candidate_indices, p=probabilities))

            current_indices.append(chosen_idx)
            current_length += int(prey_lengths[chosen_idx])

        if not current_indices:
            missing = [str(prey_ids[idx]) for idx in np.where(coverage < n_replicates)[0]]
            raise RuntimeError(
                "Could not generate more valid pools. Missing prey examples: "
                + ", ".join(missing[:10])
            )

        # Update global coverage only after the pool is finalized
        coverage[current_indices] += 1

        prey_in_pool = [str(prey_ids[idx]) for idx in current_indices]
        pool_protein_ids = [target_id] + prey_in_pool

        pool_number = len(pools) + 1

        pools.append(
            Pool(
                pool_id=f"pool_{pool_number:05d}",
                mode="bait_vs_all",
                replicate=None,  # no longer meaningful; we should remove/rename this later
                protein_ids=pool_protein_ids,
                target_id=target_id,
                prey_ids=prey_in_pool,
                total_length=current_length,
            )
        )

    return pools
def generate_bait_vs_all_pools_deterministic(
    proteins: dict[str, Protein],
    target_id: str,
    max_pool_size: int,
    n_replicates: int,
    seed: int,
    weighting: Weighting,
    shuffle_ties: bool,
) -> list[Pool]:
    
    """ Core idea
    In deterministic mode, weights define a greedy priority ranking; the highest-weight feasible prey is always selected next.
    Every prey appears exactly once.
    While building pool:
    look at proteins that still fit
    choose the highest-weight feasible protein
    repeat until full

    No random probabilities (except optional tie shuffle).

    in heree replicate doesn't really make sense since there's no randomness, so we can just ignore it for now and set it to None.

    maximum reproducibility
    structured packing
    minimal randomness
    """
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

    weights = compute_bait_weights(
        prey_lengths=prey_lengths,
        target_length=target.length,
        weighting=weighting,
    )

    rng = random.Random(seed)
    pools: list[Pool] = []

    for replicate in range(1, n_replicates + 1):
        unused_indices = set(range(len(prey_ids)))

        while unused_indices:
            current_indices: list[int] = []
            current_length = target.length

            while True:
                remaining_capacity = max_pool_size - current_length

                candidate_indices = [
                    idx
                    for idx in unused_indices
                    if prey_lengths[idx] <= remaining_capacity
                ]

                if not candidate_indices:
                    break

                candidate_scores = np.array(
                    [weights[idx] for idx in candidate_indices],
                    dtype=float,
                )

                max_score = np.max(candidate_scores)
                best_indices = [
                    idx
                    for idx, score in zip(candidate_indices, candidate_scores)
                    if score == max_score
                ]

                if shuffle_ties and len(best_indices) > 1:
                    chosen_idx = rng.choice(best_indices)
                else:
                    chosen_idx = best_indices[0]

                current_indices.append(chosen_idx)
                unused_indices.remove(chosen_idx)
                current_length += int(prey_lengths[chosen_idx])

            if not current_indices:
                missing = [str(prey_ids[idx]) for idx in unused_indices]
                raise RuntimeError(
                    "Could not fit remaining prey proteins into pools. "
                    f"Examples: {missing[:10]}"
                )

            prey_in_pool = [str(prey_ids[idx]) for idx in current_indices]
            pool_protein_ids = [target_id] + prey_in_pool

            pool_number = len(pools) + 1

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

def build_output_paths(
    output_cfg: dict,
    fasta_path: Path,
    mode: str,
    target_id: str | None,
    seed: int,
    n_replicates: int,
) -> tuple[Path, Path]:

    # default = create output folder beside fasta
    default_output_dir = fasta_path.parent / "pooled_ppi_results"

    output_dir = Path(output_cfg.get("output_dir", default_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_stem = fasta_path.stem

    if mode == "one_vs_all":
        prefix = (
            f"{fasta_stem}"
            f"__{mode}"
            f"__target-{target_id}"
            f"__rep-{n_replicates}"
            f"__seed-{seed}"
        )
    else:
        prefix = (
            f"{fasta_stem}"
            f"__{mode}"
            f"__rep-{n_replicates}"
            f"__seed-{seed}"
        )

    pools_tsv = output_dir / f"{prefix}__pools.tsv"
    summary_tsv = output_dir / f"{prefix}__summary.tsv"

    return pools_tsv, summary_tsv


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pooling(config: dict) -> None:
    

    mode: Mode = config["mode"]
    approach: str = config.get("approach", "probabilistic")
    fasta_path = Path(config["pooling_input"]["fasta_path"])
    target_id = config["pooling_input"].get("target_id")

    pooling_cfg = config["pooling"]
    max_pool_size = int(pooling_cfg.get("max_pool_size", 5000))
    n_replicates = int(pooling_cfg.get("n_replicates", 1))
    seed = int(pooling_cfg.get("seed", 42))
    max_protein_size = pooling_cfg.get("max_protein_size")
    weighting: Weighting = pooling_cfg.get("weighting", "length")
    shuffle_ties = bool(pooling_cfg.get("shuffle_ties", True))

    pooling_output_cfg = config.get("pooling_output")

    if pooling_output_cfg is not None:
        pools_tsv = Path(pooling_output_cfg["pools_tsv"])
        summary_tsv = Path(pooling_output_cfg["summary_tsv"])
    else:
        output_cfg = config.get("output", {})

        pools_tsv, summary_tsv = build_output_paths(
            output_cfg=output_cfg,
            fasta_path=fasta_path,
            mode=mode,
            target_id=target_id,
            seed=seed,
            n_replicates=n_replicates,
        )
    proteins = read_fasta(fasta_path)
    proteins = filter_proteins(proteins, max_protein_size)

    if mode == "one_vs_all":
        if target_id is None:
            raise ValueError("target_id is required for mode='one_vs_all'.")
        """
        Randomized

            Large proteins more likely early.

        Probabilistic

            Large proteins favored among under-covered proteins.

        Deterministic

            Large proteins always packed first.
            
        """

        if approach == "probabilistic":
            pools = generate_bait_vs_all_pools_probabilistic(
                proteins=proteins,
                target_id=target_id,
                max_pool_size=max_pool_size,
                n_replicates=n_replicates,
                seed=seed,
                weighting=weighting,
                shuffle_ties=shuffle_ties,
            )
        elif approach == "deterministic":
            pools = generate_bait_vs_all_pools_deterministic(
                proteins=proteins,
                target_id=target_id,
                max_pool_size=max_pool_size,
                n_replicates=n_replicates,
                seed=seed,
                weighting=weighting,
                shuffle_ties=shuffle_ties,
            )
        elif approach == "randomized":
            pools = generate_bait_vs_all_pools_randomized(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    pooling(config)