"""
Combine multiple raw FASTA files into one curated FASTA file.

Purpose
-------
This script recursively searches the ``01_RawData`` directory for supported
FASTA files, reads all sequence records, creates standardized FASTA identifiers,
resolves repeated identifiers, and writes all sequences into one combined FASTA
file.

Inputs
------
1. Input directory:
       01_RawData/

   This directory must be located in the same folder as this script. It may
   contain FASTA files directly or within subdirectories.

2. Supported FASTA extensions:
       .fasta, .fa, .faa, .fna, .fas

3. Folder names:
   The immediate parent folder of each FASTA file is used to identify the
   corresponding virus or SARS-CoV-2 variant. Known folder names are converted
   using ``DISPLAY_NAME_MAP``.

FASTA processing
----------------
For every FASTA record, the script:

1. Determines the virus or variant name from the source folder.
2. Extracts an accession number from the FASTA identifier or description.
3. Cleans the sequence by converting it to uppercase and removing whitespace.
4. Constructs a curated identifier using the format:

       Virus_or_variant|Accession

5. Appends a numerical suffix when the same curated identifier occurs more than
   once, for example:

       Alpha|YP_009724390.1
       Alpha|YP_009724390.1_2

6. Writes each sequence using lines of 80 characters.

Output
------
The script creates the following file:

    02_CuratedRawData/combined_all_raw_sequences_curated.fasta

The output directory is created automatically when it does not already exist.

Dependencies
------------
- Python 3
- Biopython

Install Biopython using:

    pip install biopython

Notes
-----
- Unreadable FASTA files are reported as warnings and skipped.
- The script combines records but does not remove duplicate sequences.
- Existing output files with the same name are overwritten.
"""

from pathlib import Path
import re
from Bio import SeqIO


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "01_RawData"

OUTPUT_DIR = BASE_DIR / "02_CuratedRawData"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FASTA = OUTPUT_DIR / "combined_all_raw_sequences_curated.fasta"


# =========================================================
# FASTA extensions
# =========================================================

FASTA_EXTENSIONS = [".fasta", ".fa", ".faa", ".fna", ".fas"]


# =========================================================
# Folder name mapping
# =========================================================

DISPLAY_NAME_MAP = {
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Delta": "Delta",
    "Omicron_BA1": "Omicron_BA1",
    "Omicron_BA2": "Omicron_BA2",
    "Omicron_XBB": "Omicron_XBB",
    "SARS_CoV_2": "SARS_CoV_2",
    "SARS_CoV": "SARS_CoV",
    "MERS_CoV": "MERS_CoV",
    "Wuhan_RefSeq": "Wuhan",
    "Bat_CoV": "Bat_CoV",
    "H229E": "HCoV_229E",
    "HKU1": "HCoV_HKU1",
    "OC43": "HCoV_OC43",
    "NL63": "HCoV_NL63",
}


# =========================================================
# Helper functions
# =========================================================

def clean_for_fasta_id(text):
    """Convert text into a safe string for use within a FASTA identifier.

    Parameters
    ----------
    text : object
        Text to be cleaned.

    Returns
    -------
    str
        Cleaned text with spaces and selected punctuation replaced by
        underscores. Repeated underscores and leading or trailing underscores
        are removed.
    """
    text = str(text).strip()

    for ch in [" ", "/", "\\", "|", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def get_virus_name_from_folder(folder_name):
    """Return the standardized virus or variant name for a source folder.

    Parameters
    ----------
    folder_name : str
        Name of the folder containing the FASTA file.

    Returns
    -------
    str
        Mapped display name when the folder is listed in ``DISPLAY_NAME_MAP``.
        Otherwise, a cleaned version of the original folder name is returned.
    """
    return DISPLAY_NAME_MAP.get(folder_name, clean_for_fasta_id(folder_name))


def get_accession(record):
    """Extract an accession-like identifier from a Biopython FASTA record.

    The function first checks pipe-delimited FASTA identifiers. If no suitable
    value is found, it searches the identifier and description using common
    accession-number patterns. The cleaned original record identifier is used
    as a fallback.

    Parameters
    ----------
    record : Bio.SeqRecord.SeqRecord
        FASTA sequence record.

    Returns
    -------
    str
        Extracted accession number or cleaned record identifier.
    """
    raw_id = str(record.id)

    if "|" in raw_id:
        parts = raw_id.split("|")

        for part in reversed(parts):
            part = part.strip()

            if part:
                return part

    text = f"{record.id} {record.description}"

    accession_patterns = [
        r"YP_\d+\.\d+",
        r"NP_\d+\.\d+",
        r"NC_\d+\.\d+",
        r"[A-Z]{2,4}_?\d{5,}\.?\d*",
        r"[A-Z]{3}\d{5,}\.?\d*",
    ]

    for pattern in accession_patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return clean_for_fasta_id(raw_id)


def clean_sequence(sequence):
    """Standardize a biological sequence before writing it to the output file.

    Parameters
    ----------
    sequence : object
        Sequence object or sequence-like text.

    Returns
    -------
    str
        Uppercase sequence with spaces and newline characters removed.
    """
    sequence = str(sequence).upper()
    sequence = sequence.replace(" ", "")
    sequence = sequence.replace("\n", "")
    sequence = sequence.replace("\r", "")
    return sequence


def get_fasta_files(raw_dir):
    """Recursively locate all supported FASTA files in a directory.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Root directory containing the raw FASTA files.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of FASTA file paths.
    """
    fasta_files = []

    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FASTA_EXTENSIONS:
            fasta_files.append(path)

    return sorted(fasta_files)


def read_fasta_records(fasta_file):
    """Read all FASTA records from one file.

    Parameters
    ----------
    fasta_file : pathlib.Path
        Path to a FASTA file.

    Returns
    -------
    list[Bio.SeqRecord.SeqRecord]
        Parsed FASTA records. An empty list is returned when the file cannot be
        read.
    """
    try:
        return list(SeqIO.parse(str(fasta_file), "fasta"))
    except Exception as e:
        print(f"\nWarning. Could not read file: {fasta_file}")
        print(f"  Error: {e}")
        return []


def write_combined_fasta(rows, output_fasta):
    """Write curated sequence records into one FASTA file.

    Parameters
    ----------
    rows : list[dict]
        Sequence records containing at least ``Curated FASTA ID`` and
        ``Sequence`` entries.

    output_fasta : pathlib.Path
        Destination path for the combined FASTA file.

    Returns
    -------
    None
        The function writes the FASTA file directly to disk.

    Notes
    -----
    Sequences are wrapped at 80 characters per line.
    """
    with output_fasta.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f">{row['Curated FASTA ID']}\n")

            sequence = row["Sequence"]

            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i + 80] + "\n")


# =========================================================
# Main
# =========================================================

def main():
    """Run the complete FASTA discovery, curation, and combination workflow.

    The function validates the input directory, discovers FASTA files, reads
    each record, creates unique curated identifiers, combines all records, and
    writes the final FASTA file.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist or contains no supported FASTA
        files.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found:\n{RAW_DIR}")

    fasta_files = get_fasta_files(RAW_DIR)

    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found inside:\n{RAW_DIR}")

    print("\nFound FASTA files:")
    for fasta_file in fasta_files:
        print(f"  {fasta_file.relative_to(BASE_DIR)}")

    combined_rows = []
    used_ids = {}

    total_records = 0

    for fasta_file in fasta_files:
        source_folder = fasta_file.parent.name
        source_file = fasta_file.name
        virus_name = get_virus_name_from_folder(source_folder)

        records = read_fasta_records(fasta_file)

        print(f"\nReading: {fasta_file.relative_to(BASE_DIR)}")
        print(f"  Virus or variant: {virus_name}")
        print(f"  Records: {len(records)}")

        for record in records:
            total_records += 1

            # Extract a stable accession and standardize the sequence content.
            accession = get_accession(record)
            sequence = clean_sequence(record.seq)

            # Build the preferred curated FASTA identifier.
            base_id = (
                f"{clean_for_fasta_id(virus_name)}|"
                f"{clean_for_fasta_id(accession)}"
            )

            # Ensure that every FASTA header remains unique in the output.
            if base_id not in used_ids:
                used_ids[base_id] = 1
                curated_id = base_id
            else:
                used_ids[base_id] += 1
                curated_id = f"{base_id}_{used_ids[base_id]}"

            combined_rows.append({
                "Virus or variant": virus_name,
                "Source folder": source_folder,
                "Source file": source_file,
                "Original FASTA ID": record.id,
                "Original description": record.description,
                "Accession": accession,
                "Curated FASTA ID": curated_id,
                "Sequence": sequence,
            })

    # Write all curated records into a single combined FASTA file.
    write_combined_fasta(combined_rows, OUTPUT_FASTA)

    print("\nCombined FASTA created successfully:")
    print(f"  {OUTPUT_FASTA}")

    print(f"\nTotal sequences written: {total_records}")


if __name__ == "__main__":
    main()