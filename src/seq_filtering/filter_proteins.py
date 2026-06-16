from pathlib import Path
import re

import pandas as pd
from Bio import SeqIO


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ANNOTATION_FILE = RAW_DIR / "caulobacter_uniprot_annotations.tsv"
FASTA_FILE = RAW_DIR / "caulobacter_proteome.fasta"

ALL_OUTPUT = PROCESSED_DIR / "all_proteins_annotated.csv"
CANDIDATE_CSV = PROCESSED_DIR / "candidate_proteins.csv"
EXCLUDED_CSV = PROCESSED_DIR / "excluded_proteins_with_reasons.csv"
CANDIDATE_FASTA = PROCESSED_DIR / "candidate_proteins.fasta"


TEXT_COLUMNS = [
    "Entry",
    "Entry Name",
    "Gene Names",
    "Protein names",
    "Organism",
    "Subcellular location [CC]",
    "Transmembrane",
    "Gene Ontology (GO)",
    "Gene Ontology (cellular component)",
    "Gene Ontology (biological process)",
    "Gene Ontology (molecular function)",
    "Keywords",
]


def normalize_text(value) -> str:
    """Convert missing values to empty string and lowercase everything."""
    if pd.isna(value):
        return ""
    return str(value).lower()


def combined_annotation_text(row: pd.Series) -> str:
    """Combine all useful annotation fields into one searchable string."""
    values = []
    for col in TEXT_COLUMNS:
        if col in row:
            values.append(normalize_text(row[col]))
    return " ".join(values)


def get_gene_names(row: pd.Series) -> list[str]:
    """Extract gene names from UniProt gene-name field."""
    gene_text = normalize_text(row.get("Gene Names", ""))
    return gene_text.split()


def is_ribosomal(row: pd.Series, text: str) -> bool:
    genes = get_gene_names(row)

    ribosomal_gene_prefix = any(
        gene.startswith(("rps", "rpl", "rpm"))
        for gene in genes
    )

    ribosomal_terms = [
        "ribosomal",
        "ribosome",
        "30s ribosomal protein",
        "50s ribosomal protein",
        "small ribosomal subunit",
        "large ribosomal subunit",
    ]

    return ribosomal_gene_prefix or any(term in text for term in ribosomal_terms)


def is_membrane_bound(row: pd.Series, text: str) -> bool:
    transmembrane_text = normalize_text(row.get("Transmembrane", ""))

    membrane_terms = [
        "membrane",
        "transmembrane",
        "integral membrane",
        "cell membrane",
        "cytoplasmic membrane",
        "inner membrane",
    ]

    has_transmembrane_feature = bool(transmembrane_text.strip())

    return has_transmembrane_feature or any(term in text for term in membrane_terms)


def is_outer_membrane_or_cell_wall(row: pd.Series, text: str) -> bool:
    terms = [
        "outer membrane",
        "cell wall",
        "cell envelope",
        "peptidoglycan",
        "murein",
        "lipopolysaccharide",
        "lps biosynthesis",
        "udp-n-acetylmuramate",
        "muramoyl",
        "penicillin-binding protein",
        "glycosyltransferase involved in cell wall",
    ]

    return any(term in text for term in terms)


def is_periplasmic(row: pd.Series, text: str) -> bool:
    terms = [
        "periplasm",
        "periplasmic",
        "periplasmic space",
    ]

    return any(term in text for term in terms)


def get_exclusion_reasons(row: pd.Series) -> str:
    text = combined_annotation_text(row)
    reasons = []

    if is_ribosomal(row, text):
        reasons.append("ribosomal")

    if is_membrane_bound(row, text):
        reasons.append("membrane_bound")

    if is_outer_membrane_or_cell_wall(row, text):
        reasons.append("outer_membrane_or_cell_wall_synthesis")

    if is_periplasmic(row, text):
        reasons.append("periplasmic")

    return ";".join(sorted(set(reasons)))


def extract_uniprot_accession_from_fasta_id(record_id: str) -> str:
    """
    UniProt FASTA IDs usually look like:
    sp|B8H358|CTRA_CAUVN
    tr|A0A...|...
    
    This function extracts B8H358.
    """
    parts = record_id.split("|")
    if len(parts) >= 2:
        return parts[1]
    return record_id




def extract_gene_name_from_description(description: str) -> str:
    match = re.search(r"\bGN=([A-Za-z0-9_.-]+)", description)
    if not match:
        raise ValueError(f"Could not find GN= in FASTA header: {description}")
    return match.group(1)

def export_candidate_fasta(candidate_ids: set[str]) -> int:
    records_to_keep = []

    for record in SeqIO.parse(FASTA_FILE, "fasta"):
        accession = extract_uniprot_accession_from_fasta_id(record.id)

        if accession in candidate_ids:
            gene_id = extract_gene_name_from_description(record.description)

            record.id = gene_id
            record.name = gene_id
            record.description = gene_id

            records_to_keep.append(record)

    SeqIO.write(records_to_keep, CANDIDATE_FASTA, "fasta")
    return len(records_to_keep)


# def export_candidate_fasta(candidate_ids: set[str]) -> int:
#     records_to_keep = []

#     for record in SeqIO.parse(FASTA_FILE, "fasta"):
#         accession = extract_uniprot_accession_from_fasta_id(record.id)

#         if accession in candidate_ids:
#             records_to_keep.append(record)

#     SeqIO.write(records_to_keep, CANDIDATE_FASTA, "fasta")
#     return len(records_to_keep)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANNOTATION_FILE, sep="\t")

    if "Entry" not in df.columns:
        raise ValueError("Expected UniProt annotation table to contain an 'Entry' column.")

    df["exclusion_reason"] = df.apply(get_exclusion_reasons, axis=1)

    df["filter_status"] = df["exclusion_reason"].apply(
        lambda reason: "keep" if reason == "" else "exclude"
    )

    kept = df[df["filter_status"] == "keep"].copy()
    excluded = df[df["filter_status"] == "exclude"].copy()

    df.to_csv(ALL_OUTPUT, index=False)
    kept.to_csv(CANDIDATE_CSV, index=False)
    excluded.to_csv(EXCLUDED_CSV, index=False)

    candidate_ids = set(kept["Entry"].astype(str))
    fasta_count = export_candidate_fasta(candidate_ids)

    print("Filtering complete.")
    print(f"Total proteins: {len(df)}")
    print(f"Kept candidates: {len(kept)}")
    print(f"Excluded proteins: {len(excluded)}")
    print(f"FASTA records exported: {fasta_count}")

    print("\nOutput files:")
    print(f"- {ALL_OUTPUT}")
    print(f"- {CANDIDATE_CSV}")
    print(f"- {EXCLUDED_CSV}")
    print(f"- {CANDIDATE_FASTA}")


if __name__ == "__main__":
    main()