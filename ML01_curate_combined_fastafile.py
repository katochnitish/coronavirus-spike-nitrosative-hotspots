"""
Code developed by:
Nitish Katoch

Script title
------------
ML01: Curation, alignment, and Wuhan-coordinate standardization of combined
coronavirus spike-protein FASTA sequences.

Role in the machine-learning pipeline
-------------------------------------
This script is the first data-preparation and quality-control stage of the
NitroSpike machine-learning workflow. It does not train a machine-learning
model. Instead, it creates the curated and coordinate-standardized sequence
dataset required by ML02, ML03, ML04, and the subsequent validation scripts.

Scientific objective
--------------------
The script removes low-quality and redundant spike-protein sequences, performs
multiple-sequence alignment using MAFFT, evaluates alignment quality, and
converts the alignment into Wuhan-Hu-1 residue-coordinate space. This ensures
that residue positions used in downstream feature engineering correspond to a
single biologically interpretable reference coordinate system.

Input
-----
Required FASTA file:

    02_CuratedRawData/combined_all_raw_sequences_curated.fasta

The FASTA file must contain combined coronavirus spike-protein sequences and at
least one Wuhan-Hu-1 reference record. Wuhan is detected by searching record
identifiers and descriptions for:

    wuhan
    yp_009724390
    mn908947

Required external software
--------------------------
MAFFT is called from WSL through the Windows batch file:

    Tools/mafft/mafft.bat

The script uses ``wslpath`` to convert WSL paths to Windows paths and executes
MAFFT through ``cmd.exe`` with the ``--auto`` option.

Libraries used
--------------
Python standard library:

    pathlib.Path
        File and directory path management.

    hashlib
        MD5 hashing for exact duplicate sequence detection.

    subprocess
        Execution of ``wslpath``, ``cmd.exe``, and MAFFT.

    os
        Preparation of the MAFFT execution environment.

    csv
        Export of alignment quality-control and duplicate-cluster tables.

    collections.defaultdict
        Storage of representative sequences and their duplicate records.

Biopython:

    Bio.SeqIO
        Reading the original FASTA records.

    Bio.AlignIO
        Reading and processing the MAFFT alignment.

Processing workflow
-------------------
1. Verify that the required input FASTA exists.
2. Read all protein-sequence records.
3. Confirm that a Wuhan-Hu-1 reference is present.
4. Standardize sequences by converting them to uppercase and removing spaces
   and line-break characters.
5. Calculate the percentage of unknown ``X`` residues in each non-gap sequence.
6. Remove sequences containing more than 1 percent ``X`` residues.
7. Remove exact duplicate sequences using an MD5 hash of the non-gap sequence.
8. Save a temporary curated FASTA containing retained nonredundant sequences.
9. Perform multiple-sequence alignment using MAFFT.
10. Calculate per-sequence alignment quality-control measurements.
11. Identify the Wuhan-Hu-1 record in the aligned FASTA.
12. Retain only alignment columns in which Wuhan contains a real amino acid.
13. Apply the same retained columns to every aligned sequence.
14. Place Wuhan first in the coordinate-standardized output.
15. Remove the temporary FASTA after successful completion.

Quality filtering calculation
-----------------------------
Unknown-residue percentage is calculated as:

    X percentage = 100 * X count / non-gap sequence length

where:

    X count
        Number of ``X`` residues after removing gap characters.

    non-gap sequence length
        Sequence length after removing ``-`` characters.

A sequence is removed when:

    X percentage > MAX_X_PERCENT

The default threshold is:

    MAX_X_PERCENT = 1.0

An empty non-gap sequence is assigned an X percentage of 100 percent.

Duplicate detection
-------------------
Each cleaned sequence is converted into a non-gap sequence and hashed:

    sequence hash = MD5(cleaned sequence with gaps removed)

Sequences with identical hashes are treated as exact duplicates. The first
record is retained as the representative, and subsequent identical records are
reported as duplicates.

Multiple-sequence alignment method
----------------------------------
MAFFT is run using:

    mafft --auto

The ``--auto`` option allows MAFFT to select an alignment strategy based on the
number and length of input sequences. The aligned FASTA is written directly
from MAFFT standard output.

Alignment quality-control calculations
--------------------------------------
For every aligned sequence, the script calculates:

    Aligned_Length
        Total aligned sequence length, including gaps.

    NonGap_Length
        Number of residues after removing alignment gaps.

    Gap_Count
        Number of ``-`` characters.

    Gap_Percent
        100 * Gap_Count / Aligned_Length

    X_Count
        Number of ``X`` residues in the non-gap sequence.

    X_Percent_NonGap
        100 * X_Count / NonGap_Length

    Nonstandard_AA_Count
        Number of characters not included in ``VALID_AA``.

Wuhan-coordinate transformation
--------------------------------
MAFFT may insert gaps into the Wuhan reference. Downstream residue-level
analysis requires coordinates to match Wuhan-Hu-1 numbering. The script
therefore retains only columns where:

    Wuhan amino acid != "-"

The retained column indices are applied to all sequences. Consequently:

    Insertions relative to Wuhan are removed.
    Substitutions at Wuhan positions are retained.
    Deletions relative to Wuhan remain represented as gaps.
    Every output column maps to a Wuhan-Hu-1 residue position.

Outputs
-------
All outputs are written to ``02_CuratedRawData``:

1. ``combined_curated_aligned.fasta``
   MAFFT alignment of the filtered and nonredundant sequences.

2. ``combined_curated_wuhan_coordinate.fasta``
   Alignment transformed into Wuhan-Hu-1 residue-coordinate space.

3. ``removed_sequences_report.txt``
   Report of records removed because of excessive X residues or duplication.

4. ``alignment_qc_report.csv``
   Per-record alignment quality-control measurements.

5. ``duplicate_cluster_summary.csv``
   Mapping of retained representative records to removed duplicates.

Temporary file
--------------
The following intermediate file is created and deleted automatically:

    _temporary_curated_noHighX_noDuplicates.fasta

Validation and failure checks
-----------------------------
The script raises an error when:

    The input FASTA is missing.
    No sequence records are present.
    No Wuhan-Hu-1 reference can be identified.
    All sequences are removed during quality filtering.
    The Wuhan reference is removed during filtering.
    MAFFT is missing or returns an execution error.
    The aligned FASTA does not contain a Wuhan reference.
    The Wuhan reference contains no non-gap residues.

Publication note
----------------
This script performs deterministic sequence curation, alignment, and coordinate
standardization. It does not provide machine-learning training, prediction, or
performance validation. Its outputs define the sequence cohort and coordinate
framework used for downstream feature extraction and NSI modelling.

Pipeline order
--------------
ML01 -> ML02 -> ML03 -> ML04 -> ML05 -> validation and plotting scripts
"""

from pathlib import Path
import hashlib
import subprocess
import os
import csv
from collections import defaultdict

from Bio import SeqIO, AlignIO


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CURATED_DIR = BASE_DIR / "02_CuratedRawData"
CURATED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FASTA = CURATED_DIR / "combined_all_raw_sequences_curated.fasta"

TEMP_CURATED_FASTA = CURATED_DIR / "_temporary_curated_noHighX_noDuplicates.fasta"

OUTPUT_ALIGNED_FASTA = CURATED_DIR / "combined_curated_aligned.fasta"
OUTPUT_WUHAN_COORDINATE_FASTA = CURATED_DIR / "combined_curated_wuhan_coordinate.fasta"

OUTPUT_REMOVED_REPORT = CURATED_DIR / "removed_sequences_report.txt"
OUTPUT_ALIGNMENT_QC = CURATED_DIR / "alignment_qc_report.csv"
OUTPUT_DUPLICATE_CLUSTER_SUMMARY = CURATED_DIR / "duplicate_cluster_summary.csv"

# Correct MAFFT .bat path
MAFFT_BAT = BASE_DIR / "Tools" / "mafft" / "mafft.bat"


# =========================================================
# Curation settings
# =========================================================

MAX_X_PERCENT = 1.0
FASTA_LINE_WIDTH = 80

# Your reference identifier. The script will search record ID and description.
WUHAN_KEYWORDS = ["wuhan", "yp_009724390", "mn908947"]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBZJUO-")


# =========================================================
# Helper functions
# =========================================================

def clean_sequence(sequence):
    # Standardize sequence text before QC, hashing, and alignment.
    sequence = str(sequence).upper()
    sequence = sequence.replace(" ", "")
    sequence = sequence.replace("\n", "")
    sequence = sequence.replace("\r", "")
    return sequence


def sequence_hash(sequence):
    # Generate a reproducible identifier for exact non-gap sequence content.
    sequence = clean_sequence(sequence)
    sequence = sequence.replace("-", "")
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


def x_percentage(sequence):
    # Quantify ambiguous X residues relative to the non-gap sequence length.
    sequence = clean_sequence(sequence)
    sequence_no_gaps = sequence.replace("-", "")

    if len(sequence_no_gaps) == 0:
        return 100.0

    x_count = sequence_no_gaps.count("X")
    return 100.0 * x_count / len(sequence_no_gaps)


def is_wuhan_record(record):
    text = f"{record.id} {record.description}".lower()
    return any(keyword in text for keyword in WUHAN_KEYWORDS)


def write_fasta(records, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        for record_id, sequence in records:
            f.write(f">{record_id}\n")
            for i in range(0, len(sequence), FASTA_LINE_WIDTH):
                f.write(sequence[i:i + FASTA_LINE_WIDTH] + "\n")


def wsl_to_windows_path(path):
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to convert WSL path to Windows path.\n\n"
            f"Path: {path}\n"
            f"Error: {result.stderr}"
        )

    return result.stdout.strip()


def find_mafft_bat():
    if MAFFT_BAT.exists():
        print("\nMAFFT .bat found:")
        print(f"  {MAFFT_BAT}")
        return MAFFT_BAT

    raise FileNotFoundError(
        "MAFFT .bat file was not found.\n\n"
        f"Expected path:\n{MAFFT_BAT}\n\n"
        "Please check your MAFFT installation path."
    )


def run_mafft_bat_from_wsl(input_fasta, output_fasta):
    mafft_bat = find_mafft_bat()

    mafft_bat_win = wsl_to_windows_path(mafft_bat)
    input_fasta_win = wsl_to_windows_path(input_fasta)

    command = [
        "cmd.exe",
        "/c",
        mafft_bat_win,
        "--auto",
        input_fasta_win,
    ]

    print("\nRunning MAFFT through Windows .bat from WSL:")
    print("  " + " ".join(command))

    clean_env = dict(os.environ)
    clean_env.pop("MAFFT_BINARIES", None)

    with output_fasta.open("w", encoding="utf-8") as out_f:
        result = subprocess.run(
            command,
            stdout=out_f,
            stderr=subprocess.PIPE,
            text=True,
            env=clean_env,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "MAFFT failed.\n\n"
            f"Command used:\n{' '.join(command)}\n\n"
            f"Error message:\n{result.stderr}"
        )

    print("\nMAFFT alignment completed successfully.")
    print(f"Final aligned FASTA saved:\n  {output_fasta}")


def save_removed_sequences_report(
    output_report,
    input_fasta,
    output_aligned_fasta,
    output_wuhan_coordinate_fasta,
    total_records,
    kept_records,
    removed_high_x_records,
    removed_duplicate_records,
):
    with output_report.open("w", encoding="utf-8") as f:
        f.write("Sequence curation report\n")
        f.write("========================\n\n")

        f.write(f"Input FASTA:\n{input_fasta}\n\n")
        f.write(f"Final aligned FASTA:\n{output_aligned_fasta}\n\n")
        f.write(f"Final Wuhan-coordinate FASTA:\n{output_wuhan_coordinate_fasta}\n\n")

        f.write("Curation rules\n")
        f.write("--------------\n")
        f.write(f"1. Excluded sequences with > {MAX_X_PERCENT:.1f}% X residues\n")
        f.write("2. Removed exact duplicate sequences after removing gap characters\n")
        f.write("3. Multiple sequence alignment was performed using MAFFT\n")
        f.write("4. Wuhan-coordinate FASTA was generated by retaining only columns where Wuhan has a real amino acid\n\n")

        f.write("Summary\n")
        f.write("-------\n")
        f.write(f"Total input records: {total_records}\n")
        f.write(f"Removed due to > {MAX_X_PERCENT:.1f}% X residues: {len(removed_high_x_records)}\n")
        f.write(f"Removed exact duplicates: {len(removed_duplicate_records)}\n")
        f.write(f"Final curated records used for alignment: {kept_records}\n\n")

        f.write("Sequences removed due to > 1% X residues\n")
        f.write("----------------------------------------\n")

        if removed_high_x_records:
            f.write("Record_ID\tX_percent\tSequence_length\tOriginal_description\n")
            for item in removed_high_x_records:
                f.write(
                    f"{item['record_id']}\t"
                    f"{item['x_percent']:.4f}\t"
                    f"{item['sequence_length']}\t"
                    f"{item['description']}\n"
                )
        else:
            f.write("None\n")

        f.write("\nSequences removed as exact duplicates\n")
        f.write("-------------------------------------\n")

        if removed_duplicate_records:
            f.write("Duplicate_Record_ID\tDuplicate_of_Record_ID\tSequence_length\tOriginal_description\n")
            for item in removed_duplicate_records:
                f.write(
                    f"{item['record_id']}\t"
                    f"{item['duplicate_of']}\t"
                    f"{item['sequence_length']}\t"
                    f"{item['description']}\n"
                )
        else:
            f.write("None\n")


def save_duplicate_cluster_summary(output_csv, duplicate_clusters):
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Representative_Record_ID", "Duplicate_Count", "Duplicate_Record_IDs"],
        )
        writer.writeheader()

        for representative, duplicates in duplicate_clusters.items():
            writer.writerow({
                "Representative_Record_ID": representative,
                "Duplicate_Count": len(duplicates),
                "Duplicate_Record_IDs": ";".join(duplicates),
            })


def save_alignment_qc_report(alignment_path, output_csv):
    # Export per-sequence alignment completeness and ambiguity measurements.
    alignment = AlignIO.read(str(alignment_path), "fasta")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Variant_ID",
                "Aligned_Length",
                "NonGap_Length",
                "Gap_Count",
                "Gap_Percent",
                "X_Count",
                "X_Percent_NonGap",
                "Nonstandard_AA_Count",
            ],
        )
        writer.writeheader()

        for record in alignment:
            seq = clean_sequence(record.seq)
            aligned_len = len(seq)
            gap_count = seq.count("-")
            non_gap_seq = seq.replace("-", "")
            non_gap_len = len(non_gap_seq)
            x_count = non_gap_seq.count("X")
            nonstandard_count = sum(1 for aa in seq if aa not in VALID_AA)

            writer.writerow({
                "Variant_ID": record.id,
                "Aligned_Length": aligned_len,
                "NonGap_Length": non_gap_len,
                "Gap_Count": gap_count,
                "Gap_Percent": 100.0 * gap_count / aligned_len if aligned_len else 0.0,
                "X_Count": x_count,
                "X_Percent_NonGap": 100.0 * x_count / non_gap_len if non_gap_len else 0.0,
                "Nonstandard_AA_Count": nonstandard_count,
            })


def make_wuhan_coordinate_fasta(alignment_path, output_path):
    # Convert MAFFT columns into a coordinate system defined by Wuhan-Hu-1.
    alignment = AlignIO.read(str(alignment_path), "fasta")
    records = list(alignment)

    wuhan_records = [record for record in records if is_wuhan_record(record)]

    if len(wuhan_records) == 0:
        raise ValueError(
            "No Wuhan reference was found in the aligned FASTA. "
            "Please check WUHAN_KEYWORDS or FASTA record names."
        )

    if len(wuhan_records) > 1:
        print("\nWarning: more than one Wuhan-like record was found. Using the first match:")
        for record in wuhan_records:
            print(f"  {record.id}")

    wuhan_record = wuhan_records[0]
    wuhan_seq = clean_sequence(wuhan_record.seq)

    # Keep only alignment columns where Wuhan has a real residue.
    # This converts the MAFFT alignment into Wuhan residue-coordinate space.
    keep_indices = [i for i, aa in enumerate(wuhan_seq) if aa != "-"]

    if not keep_indices:
        raise ValueError("Wuhan reference contains no non-gap amino acids after alignment.")

    wuhan_coordinate_records = []

    # Put Wuhan first to protect script 101.
    ordered_records = [wuhan_record] + [record for record in records if record.id != wuhan_record.id]

    for record in ordered_records:
        seq = clean_sequence(record.seq)
        new_seq = "".join(seq[i] for i in keep_indices)
        wuhan_coordinate_records.append((record.id, new_seq))

    write_fasta(wuhan_coordinate_records, output_path)

    print("\nWuhan-coordinate FASTA generated.")
    print(f"  Wuhan reference: {wuhan_record.id}")
    print(f"  Original aligned length: {len(wuhan_seq)}")
    print(f"  Wuhan-coordinate length: {len(keep_indices)}")
    print(f"  Saved: {output_path}")

    return wuhan_record.id, len(wuhan_seq), len(keep_indices)


# =========================================================
# Main workflow
# =========================================================

def main():
    if not INPUT_FASTA.exists():
        raise FileNotFoundError(
            "Input FASTA file not found:\n"
            f"{INPUT_FASTA}\n\n"
            "Expected file:\n"
            "02_CuratedRawData/combined_all_raw_sequences_curated.fasta"
        )

    records = list(SeqIO.parse(str(INPUT_FASTA), "fasta"))

    if not records:
        raise ValueError(f"No sequences found in:\n{INPUT_FASTA}")

    print("\nLoaded input FASTA:")
    print(f"  {INPUT_FASTA}")
    print(f"  Total input records: {len(records)}")

    wuhan_input_records = [record for record in records if is_wuhan_record(record)]
    if not wuhan_input_records:
        raise ValueError(
            "No Wuhan reference was detected in the input FASTA. "
            "Please confirm that the reference record contains Wuhan, YP_009724390, or MN908947."
        )

    if not is_wuhan_record(records[0]):
        print("\nWarning: Wuhan is not the first input record.")
        print(f"  First input record: {records[0].id}")
        print(f"  Wuhan detected: {wuhan_input_records[0].id}")
        print("  The Wuhan-coordinate output will automatically place Wuhan first.")

    curated_records = []
    seen_hashes = {}
    duplicate_clusters = defaultdict(list)

    removed_high_x_records = []
    removed_duplicate_records = []

    for record in records:
        record_id = str(record.id)
        description = str(record.description)
        sequence = clean_sequence(record.seq)
        sequence_no_gaps = sequence.replace("-", "")

        # Calculate ambiguity before deciding whether the sequence is retained.
        x_pct = x_percentage(sequence)

        if x_pct > MAX_X_PERCENT:
            removed_high_x_records.append({
                "record_id": record_id,
                "description": description,
                "x_percent": x_pct,
                "sequence_length": len(sequence_no_gaps),
            })
            continue

        # Detect exact duplicate biological sequences independently of gap symbols.
        seq_hash = sequence_hash(sequence)

        if seq_hash in seen_hashes:
            representative = seen_hashes[seq_hash]
            removed_duplicate_records.append({
                "record_id": record_id,
                "description": description,
                "duplicate_of": representative,
                "sequence_length": len(sequence_no_gaps),
            })
            duplicate_clusters[representative].append(record_id)
            continue

        seen_hashes[seq_hash] = record_id
        duplicate_clusters[record_id] = []
        curated_records.append((record_id, sequence))

    if not curated_records:
        raise ValueError(
            "No sequences remained after filtering. "
            "Please check the input FASTA or the X residue threshold."
        )

    if not any(any(keyword in record_id.lower() for keyword in WUHAN_KEYWORDS) for record_id, _ in curated_records):
        raise ValueError(
            "The Wuhan reference was removed during filtering. "
            "Please inspect X percentage and duplicate filtering."
        )

    write_fasta(curated_records, TEMP_CURATED_FASTA)

    save_removed_sequences_report(
        output_report=OUTPUT_REMOVED_REPORT,
        input_fasta=INPUT_FASTA,
        output_aligned_fasta=OUTPUT_ALIGNED_FASTA,
        output_wuhan_coordinate_fasta=OUTPUT_WUHAN_COORDINATE_FASTA,
        total_records=len(records),
        kept_records=len(curated_records),
        removed_high_x_records=removed_high_x_records,
        removed_duplicate_records=removed_duplicate_records,
    )

    save_duplicate_cluster_summary(
        output_csv=OUTPUT_DUPLICATE_CLUSTER_SUMMARY,
        duplicate_clusters=duplicate_clusters,
    )

    print("\nCuration completed.")
    print(f"  Total input records: {len(records)}")
    print(f"  Removed due to > {MAX_X_PERCENT:.1f}% X residues: {len(removed_high_x_records)}")
    print(f"  Removed exact duplicates: {len(removed_duplicate_records)}")
    print(f"  Final curated records used for alignment: {len(curated_records)}")

    print("\nRemoved sequence report saved:")
    print(f"  {OUTPUT_REMOVED_REPORT}")

    print("\nDuplicate cluster summary saved:")
    print(f"  {OUTPUT_DUPLICATE_CLUSTER_SUMMARY}")

    # Align the filtered and nonredundant sequence cohort using MAFFT.
    run_mafft_bat_from_wsl(TEMP_CURATED_FASTA, OUTPUT_ALIGNED_FASTA)

    # Quantify alignment gaps, unknown residues, and nonstandard characters.
    save_alignment_qc_report(OUTPUT_ALIGNED_FASTA, OUTPUT_ALIGNMENT_QC)
    print("\nAlignment QC report saved:")
    print(f"  {OUTPUT_ALIGNMENT_QC}")

    # Standardize all sequences to Wuhan-Hu-1 residue numbering.
    make_wuhan_coordinate_fasta(OUTPUT_ALIGNED_FASTA, OUTPUT_WUHAN_COORDINATE_FASTA)

    if TEMP_CURATED_FASTA.exists():
        TEMP_CURATED_FASTA.unlink()
        print("\nTemporary curated FASTA removed.")

    print("\nFinal files saved:")
    print(f"  Aligned FASTA: {OUTPUT_ALIGNED_FASTA}")
    print(f"  Wuhan-coordinate FASTA: {OUTPUT_WUHAN_COORDINATE_FASTA}")
    print(f"  Removed sequence report: {OUTPUT_REMOVED_REPORT}")
    print(f"  Alignment QC report: {OUTPUT_ALIGNMENT_QC}")
    print(f"  Duplicate cluster summary: {OUTPUT_DUPLICATE_CLUSTER_SUMMARY}")


if __name__ == "__main__":
    main()
