from pathlib import Path
import argparse
import yaml
from generate_pool import pooling
from write_af3_jsons import json_builder
from validate_af3_jsons import validate_jsons
from extract_af3_pair_scores import score_extractor
from size_correction import size_corrector



# -----------------------------
# helpers
# -----------------------------
def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
    

def build_paths(config: dict) -> dict:
    project_name = config["project"]["name"]
    base_output_dir = Path(config["project"]["output_dir"])
    project_dir = base_output_dir / project_name

    paths = {
        "project_dir": project_dir,
        "pools_tsv": project_dir / "pools.tsv",
        "pool_summary_tsv": project_dir / "pool_summary.tsv",
        "json_dir": project_dir / "alphafold3_jsons",
        "chain_mapping_tsv": project_dir / "pool_chain_mapping.tsv",
        "json_validation_summary_tsv": project_dir / "af3_json_validation_summary.tsv",
        "predictions_dir": Path(config["prediction"]["predictions_dir"]),
        "pair_scores_raw_tsv": project_dir / "pair_scores_raw.tsv",
        "prediction_summary_tsv": project_dir / "prediction_summary.tsv",
        "corrected_observations_tsv": project_dir / "pair_scores_size_corrected_observations.tsv",
        "corrected_aggregated_tsv": project_dir / "pair_scores_size_corrected_aggregated.tsv",
        "correction_model_tsv": project_dir / "size_correction_model.tsv",
    }

    return paths

def inject_paths(config: dict, paths: dict) -> dict:
    """
    Convert one clean pipeline config into the step-specific config format
    expected by your existing functions.
    """
    config = dict(config)

    # Step 1: pooling
    config["pooling_input"] = {
        "fasta_path": config["input"]["fasta_path"],
        "target_id": config["input"]["target_id"],
    }

    config["pooling_output"] = {
        "pools_tsv": str(paths["pools_tsv"]),
        "summary_tsv": str(paths["pool_summary_tsv"]),
    }

    # Step 2: AF3 JSON builder
    config["af3_json_build_input"] = {
        "fasta_path": config["input"]["fasta_path"],
        "pools_tsv": str(paths["pools_tsv"]),
    }

    config["af3_json_output"] = {
        "json_dir": str(paths["json_dir"]),
    }

    # Step 3: validation
    config["json_validation_input"] = {
        "json_dir": str(paths["json_dir"]),
        "chain_mapping_tsv": str(paths["chain_mapping_tsv"]),
    }

    config["json_validation_output"] = {
        "summary_tsv": str(paths["json_validation_summary_tsv"]),
    }

    # Step 4: extraction
    config["af3_result_input"] = {
        "predictions_dir": str(paths["predictions_dir"]),
        "chain_mapping_tsv": str(paths["chain_mapping_tsv"]),
    }

    config["af3_pair_scores_output"] = {
        "pair_scores_tsv": str(paths["pair_scores_raw_tsv"]),
        "pool_summary_tsv": str(paths["prediction_summary_tsv"]),
    }

    # Step 5: size correction
    config["size_correction_input"] = {
        "pair_scores_tsv": str(paths["pair_scores_raw_tsv"]),
    }

    config["size_correction_output"] = {
        "corrected_observations_tsv": str(paths["corrected_observations_tsv"]),
        "corrected_aggregated_tsv": str(paths["corrected_aggregated_tsv"]),
        "correction_model_tsv": str(paths["correction_model_tsv"]),
    }

    return config

def validate_stage_inputs(stage: str, config: dict, paths: dict) -> None:
    fasta_path = Path(config["input"]["fasta_path"])

    if stage in {"prepare", "all"}:
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    if stage == "process":
        if not paths["predictions_dir"].exists():
            raise FileNotFoundError(
                f"AF3 predictions directory not found: {paths['predictions_dir']}"
            )

        if not paths["chain_mapping_tsv"].exists():
            raise FileNotFoundError(
                "Missing chain mapping file. Run prepare stage first: "
                f"{paths['chain_mapping_tsv']}"
            )

def print_prepare_summary(paths: dict) -> None:
    print("\nPrepare stage complete.")
    print("Upload these JSON files to AlphaFold Server:")
    print(paths["json_dir"])
    print("\nAfter downloading AF3 results, put them here:")
    print(paths["predictions_dir"])
    print("\nThen run:")
    print("python main_workflow.py --config configs/pipeline.yaml")


def print_process_summary(paths: dict) -> None:
    print("\nProcess stage complete.")
    print("Final ranked interaction table:")
    print(paths["corrected_aggregated_tsv"])



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    stage = config["workflow"]["stage"]

    if stage not in {"prepare", "process", "all"}:
        raise ValueError(f"Invalid workflow stage: {stage}")

    paths = build_paths(config)
    paths["project_dir"].mkdir(parents=True, exist_ok=True)

    config = inject_paths(config, paths)
    validate_stage_inputs(stage, config, paths)

    if stage == "prepare":
        pooling(config)
        json_builder(config)
        validate_jsons(config)
        print_prepare_summary(paths)

    elif stage == "process":
        score_extractor(config)
        size_corrector(config)
        print_process_summary(paths)

    elif stage == "all":
        raise NotImplementedError(
            "The 'all' stage is not supported while using AlphaFold Server. "
            "Run 'prepare', then run AF3 manually/server-side, then run 'process'."
        )


if __name__ == "__main__":
    main()