from pathlib import Path
import json

from Bio import SeqIO
import re

import pandas as pd

INPUT_FASTA = Path("data/processed/candidate_proteins.fasta")

OUTPUT_DIR = Path("data/msa_jobs/test")

AF3_VERSION = 4
MODEL_SEEDS = [1]


# def get_protein_id(record):
#     """
#     Extract Caulobacter gene ID from UniProt FASTA header.

#     Example:
#     >tr|A0A0H3C2V8|... GN=CCNA_00145 ...
#     ->
#     CCNA_00145
#     """
#     match = re.search(r"\bGN=([A-Za-z0-9_.-]+)", record.description)

#     if match:
#         return match.group(1)

#     raise ValueError(
#         f"Could not find GN= gene name in FASTA header: {record.description}"
#     )

# def get_protein_id(record):
#     """
#     Use the FASTA record ID directly.

#     After filtering, candidate_proteins.fasta headers should be:
#     >CCNA_00145
#     """
#     return record.id

def get_protein_id(record):
    """
    Extract the unique UniProt ID (e.g., A0A0H3C8R6) from the FASTA header.
    Falls back to record.id if the UniProt pipe format is not found.
    """
    # Looks for text between pipes like |A0A0H3C8R6|
    match = re.search(r"\|([^|]+)\|", record.description)
    if match:
        return match.group(1)
        
    # Fallback if your headers were already cleaned up to just >CCNA_00145
    return record.id

def build_af3_json(protein_id: str, sequence: str) -> dict:
    return {
        "name": protein_id,
        "modelSeeds": MODEL_SEEDS,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                }
            }
        ],
        "dialect": "alphafold3",
        "version": AF3_VERSION,
    }


def main():
    

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_written = 0
    protein_ids = []

    for record in SeqIO.parse(INPUT_FASTA, "fasta"):
        
        print(f"DEBUG - ID: {record.id} | DESC: {record.description}")
        break # Stop after the first one just to check the format
        protein_id = get_protein_id(record)

        payload = build_af3_json(
            protein_id=protein_id,
            sequence=str(record.seq),
        )

        output_path = OUTPUT_DIR / f"{protein_id}.json"

        with open(output_path, "w") as f:
            try: 
                json.dump(payload, f, indent=2)
                
            except Exception as e:
                print(f"Error writing JSON to {output_path}: {e}")

        n_written += 1
        protein_ids.append(protein_id)

    # save the list of protein ids as well, for reference
        # Find duplicates in your list
    seen = set()
    duplicates = set()
    for pid in protein_ids:
        if pid in seen:
            duplicates.add(pid)
        seen.add(pid)
    
    print(f"Found {len(duplicates)} unique duplicate IDs: {duplicates}")

    with open(OUTPUT_DIR / "protein_ids.csv", "w") as f:
        try: 
            df = pd.DataFrame(protein_ids, columns=["protein_id"])
            df.to_csv(f, index=False)
            print(f"Protein IDs written: {len(protein_ids)} in the file {OUTPUT_DIR / 'protein_ids.csv'}")
        except Exception as e:
            print(f"Error writing protein IDs to CSV: {e}")
    

    print(f"JSON files written: {n_written}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()