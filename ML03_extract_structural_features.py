"""
Code developed by:
Nitish Katoch

Script title
------------
ML03: Structural feature extraction for nitrosative susceptibility modelling

Purpose
-------
This script converts the residue-level target matrix (ML02 output) into a
high-dimensional structural and physicochemical feature matrix. It integrates
sequence, structural biology, and biophysical context into quantitative features
used for downstream NSI scoring and machine learning.

This is the most critical feature-engineering stage in the pipeline.

Scientific objective
--------------------
To characterize the microenvironment of nitrosative target residues (Cys/Tyr)
using structural, electrostatic, evolutionary, and spatial descriptors.

These features approximate susceptibility of residues to reactive nitrogen species.

Input
-----
09_AI_NitroSpike/extracted_nitro_targets.csv

This contains:
    Rows    = Variants
    Columns = Target residues (C15, Y28, etc.)

Additional input:
    PDB structures automatically downloaded from RCSB

Libraries used
--------------
numpy, pandas
    Numerical and tabular feature computation

Bio.PDB
    Structural parsing and atomic coordinate extraction

matplotlib, seaborn
    Visualization of feature relationships

urllib
    Automated PDB download

Core feature categories
-----------------------
1. Structural accessibility
    - True SASA (if available)
    - Relative SASA
    - Packing density proxy

2. Electrostatic environment
    - Charge balance within 8 Å
    - Basic vs acidic residue counts

3. Local microenvironment composition
    - Hydrophobic, polar, aromatic, sulfur residue counts

4. Spatial structural context
    - Residue depth proxy
    - Distance to ACE2 interface
    - Distance to glycan shielding

5. Chemical properties
    - Hydropathy (Kyte-Doolittle scale)
    - Formal charge

6. Evolutionary features
    - Conservation frequency
    - Mutation frequency
    - Shannon entropy

7. Disulfide bonding
    - Presence of disulfide bonds
    - Distance to nearest cysteine

Key calculations
----------------
Packing density proxy:
    100 / (neighbor atom count within 3.5 Å + 1)

Electrostatic balance:
    (# basic residues - # acidic residues) within 8 Å

Relative SASA:
    observed SASA / maximum SASA (Tien et al.)

Shannon entropy:
    H = - Σ p log2(p)

Residue depth proxy:
    Distance from residue centroid to structure centroid

Interface proximity:
    Minimum distance to ACE2-contact residues

Glycan shielding:
    Minimum distance to glycosylation sites

Disulfide detection:
    SG-SG distance <= 2.3 Å

Outputs
-------
1. structural_ai_feature_matrix.csv
   Main ML-ready dataset

2. structural_feature_missing_residues.csv
   Missing structural mappings

3. structural_feature_summary_by_variant.csv
   Aggregated per-variant statistics

4. structural_feature_correlations.png
   Feature correlation heatmap

5. variant_microenvironment_comparison.png
   Feature distribution scatter plot

Machine learning relevance
-------------------------
This script defines the complete feature space used in ML04 and ML05.

Important:
    These are engineered proxy features, not experimental measurements.

Pipeline position
-----------------
ML01 -> ML02 -> ML03 -> ML04 -> ML05
"""

"""
102_extract_structural_features_updated.py

Purpose
-------
Build a reviewer-ready feature matrix for nitrosative susceptibility analysis of
Wuhan-coordinate Cys/Tyr target residues in coronavirus spike proteins.

Expected input
--------------
09_AI_NitroSpike/extracted_nitro_targets.csv

This file is expected to be generated from a Wuhan-coordinate target extraction
script. Columns should include:
    Variant_ID, C15, Y28, ...
where residue columns encode the Wuhan reference residue type and Wuhan residue
number.

Main outputs
------------
09_AI_NitroSpike/structural_ai_feature_matrix.csv
09_AI_NitroSpike/structural_feature_missing_residues.csv
09_AI_NitroSpike/structural_feature_summary_by_variant.csv
09_AI_NitroSpike/structural_feature_correlations.png
09_AI_NitroSpike/variant_microenvironment_comparison.png

Notes
-----
This script intentionally records missing structural coverage instead of silently
removing sites. Reviewers usually want to know which residues were not resolved
in the structural template.

Optional dependency
-------------------
If Biopython ShrakeRupley is available, the script computes real residue SASA.
If it is unavailable, SASA fields are written as NaN and the packing-density
proxy remains available.
"""

from pathlib import Path
import math
import urllib.request
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Bio.PDB import PDBParser
try:
    from Bio.PDB.SASA import ShrakeRupley
    HAS_SHRAKE_RUPLEY = True
except Exception:
    ShrakeRupley = None
    HAS_SHRAKE_RUPLEY = False

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. PATHS AND CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "09_AI_NitroSpike"
PDB_CACHE_DIR = BASE_DIR / "04_PDB_Structures"
PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "extracted_nitro_targets.csv"
OUTPUT_FEATURES_CSV = DATA_DIR / "structural_ai_feature_matrix.csv"
OUTPUT_MISSING_CSV = DATA_DIR / "structural_feature_missing_residues.csv"
OUTPUT_VARIANT_SUMMARY_CSV = DATA_DIR / "structural_feature_summary_by_variant.csv"

PLOT_CORRELATION = DATA_DIR / "structural_feature_correlations.png"
PLOT_VARIANT_DIST = DATA_DIR / "variant_microenvironment_comparison.png"

# Structural templates. Add more if you have variant-specific or homologous structures.
# For distant coronaviruses, do not blindly map to Wuhan unless this is explicitly intended.
VARIANT_PDB_MAPPING = {
    "Wuhan": "6VXX",
    "Alpha": "7NXA",
    "Beta": "7V7A",
    "Delta": "7V7Q",
    "Omicron_BA2": "7XIX",
    "BA2": "7XIX",
    "BA.2": "7XIX",
}

DEFAULT_PDB = "6VXX"
ALLOW_WUHAN_FALLBACK = True

# Distance cutoffs in Angstroms.
PACKING_RADIUS = 3.5
ELECTROSTATIC_RADIUS = 8.0
LOCAL_WINDOW_RADIUS = 8.0
DISULFIDE_SG_DISTANCE = 2.3
GLYCAN_DISTANCE_RADIUS = 10.0
ACE2_DISTANCE_RADIUS = 8.0

TARGET_AMINO_ACIDS = {"C", "Y"}
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_TO_AA3 = {v: k for k, v in AA3_TO_AA1.items()}

BASIC_RESIDUES = {"ARG", "LYS", "HIS"}
ACIDIC_RESIDUES = {"ASP", "GLU"}
POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS", "TRP"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
SULFUR_RESIDUES = {"CYS", "MET"}

# Approximate maximum residue SASA values in Angstrom squared.
# Values are commonly used normalization constants for relative SASA estimates.
MAX_ASA_TIEN = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# Spike domain coordinates in Wuhan numbering.
SPIKE_DOMAINS = [
    (1, 13, "Signal_Peptide"),
    (14, 305, "NTD"),
    (306, 319, "RBD_Linker"),
    (320, 541, "RBD"),
    (437, 508, "RBM"),
    (542, 685, "S1_C_terminal"),
    (681, 685, "Furin_Cleavage_Region"),
    (686, 815, "S2_N_terminal"),
    (816, 833, "Fusion_Peptide"),
    (834, 911, "S2_Region"),
    (912, 984, "HR1"),
    (985, 1162, "Central_Helix_Connector"),
    (1163, 1213, "HR2"),
    (1214, 1237, "Transmembrane"),
    (1238, 1273, "Cytoplasmic_Tail"),
]

# Approximate ACE2-contacting Wuhan RBD residues reported across structural studies.
# Used as a distance-to-interface feature, not as a definitive binding annotation.
ACE2_INTERFACE_POSITIONS = {
    417, 446, 449, 453, 455, 456, 475, 486, 487, 489, 493, 496, 498, 500, 501, 502, 505,
}

# N-linked glycosylation sequons often reported for SARS-CoV-2 spike.
# Used as an approximate shielding-distance feature.
GLYCAN_SITE_POSITIONS = {
    17, 61, 74, 122, 149, 165, 234, 282, 331, 343, 603, 616, 657, 709, 717, 801, 1074, 1098, 1134, 1158, 1173, 1194,
}

KYTOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

CHARGE_AA1 = {
    "K": 1, "R": 1, "H": 0.5, "D": -1, "E": -1,
}

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================
def download_pdb(pdb_id: str) -> Path:
    local_path = PDB_CACHE_DIR / f"{pdb_id.lower()}.pdb"
    if not local_path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        print(f"  Downloading PDB {pdb_id.upper()}...")
        urllib.request.urlretrieve(url, local_path)
    return local_path


def select_pdb_for_variant(variant_id: str):
    variant_lower = str(variant_id).lower()
    for key, pdb_id in VARIANT_PDB_MAPPING.items():
        if key.lower() in variant_lower:
            return pdb_id, False
    if ALLOW_WUHAN_FALLBACK:
        return DEFAULT_PDB, True
    return None, False


def parse_residue_column(column_name: str):
    ref_aa = column_name[0]
    try:
        ref_pos = int(column_name[1:])
    except ValueError:
        return None, None
    if ref_aa not in TARGET_AMINO_ACIDS:
        return None, None
    return ref_aa, ref_pos


def get_domain(position: int) -> str:
    # RBM is nested within RBD, so prioritize RBM.
    if 437 <= position <= 508:
        return "RBM"
    for start, end, name in SPIKE_DOMAINS:
        if start <= position <= end:
            return name
    return "Unknown"


def classify_target_status(reference_aa: str, observed_aa: str) -> str:
    observed_aa = str(observed_aa)
    if observed_aa == "-":
        return "Deleted_or_Gap"
    if observed_aa == reference_aa and observed_aa in TARGET_AMINO_ACIDS:
        return "Conserved_Target"
    if reference_aa == "C" and observed_aa == "Y":
        return "Target_Type_Switched_C_to_Y"
    if reference_aa == "Y" and observed_aa == "C":
        return "Target_Type_Switched_Y_to_C"
    if reference_aa in TARGET_AMINO_ACIDS and observed_aa not in TARGET_AMINO_ACIDS:
        return "Lost_Target"
    return "Other"


def get_reactive_atom(residue):
    resname = residue.get_resname()
    if resname == "CYS" and "SG" in residue:
        return residue["SG"]
    if resname == "TYR" and "OH" in residue:
        return residue["OH"]
    if "CA" in residue:
        return residue["CA"]
    atoms = list(residue.get_atoms())
    return atoms[0] if atoms else None


def residue_centroid(residue):
    coords = [atom.get_coord() for atom in residue.get_atoms()]
    if not coords:
        return None
    return np.mean(np.vstack(coords), axis=0)


def atom_distance(coord_a, coord_b) -> float:
    return float(np.linalg.norm(coord_a - coord_b))


def shannon_entropy(values) -> float:
    cleaned = [v for v in values if pd.notna(v) and v != "-"]
    if not cleaned:
        return np.nan
    counts = Counter(cleaned)
    probs = np.array(list(counts.values()), dtype=float) / len(cleaned)
    return float(-np.sum(probs * np.log2(probs)))


def hydropathy_value(aa: str) -> float:
    return float(KYTOOLITTLE.get(str(aa), np.nan))


def charge_value(aa: str) -> float:
    return float(CHARGE_AA1.get(str(aa), 0.0))


def build_position_conservation_table(df_seq: pd.DataFrame) -> pd.DataFrame:
    records = []
    residue_cols = [c for c in df_seq.columns if c != "Variant_ID"]

    for col in residue_cols:
        ref_aa, ref_pos = parse_residue_column(col)
        if ref_pos is None:
            continue
        observed = df_seq[col].astype(str).tolist()
        valid = [aa for aa in observed if aa != "-"]
        n_total = len(observed)
        n_valid = len(valid)
        counts = Counter(valid)
        records.append({
            "Residue_Label": col,
            "Reference_AA": ref_aa,
            "Sequence_Position": ref_pos,
            "Conservation_Frequency": counts.get(ref_aa, 0) / n_total if n_total else np.nan,
            "Cys_Frequency": counts.get("C", 0) / n_total if n_total else np.nan,
            "Tyr_Frequency": counts.get("Y", 0) / n_total if n_total else np.nan,
            "Target_Retention_Frequency": (counts.get("C", 0) + counts.get("Y", 0)) / n_total if n_total else np.nan,
            "Gap_Frequency": observed.count("-") / n_total if n_total else np.nan,
            "Mutation_Frequency": sum(1 for aa in observed if aa != ref_aa) / n_total if n_total else np.nan,
            "Dominant_AA": counts.most_common(1)[0][0] if counts else np.nan,
            "Shannon_Entropy": shannon_entropy(observed),
            "Valid_Residue_Count": n_valid,
            "Total_Variant_Count": n_total,
        })
    return pd.DataFrame(records)


def compute_sasa(structure):
    if not HAS_SHRAKE_RUPLEY:
        return
    sr = ShrakeRupley(n_points=100)
    sr.compute(structure, level="R")


def collect_structure_objects(structure):
    residues = []
    atoms = []
    residue_by_position = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag, resseq, insertion = residue.get_id()
                if hetero_flag.strip():
                    continue
                if residue.get_resname() not in AA3_TO_AA1:
                    continue
                residues.append(residue)
                atoms.extend(list(residue.get_atoms()))
                residue_by_position.setdefault(resseq, []).append((model.id, chain.id, residue))

    return residues, atoms, residue_by_position


def find_best_residue(residue_by_position, ref_pos: int, observed_aa: str):
    candidates = residue_by_position.get(ref_pos, [])
    if not candidates:
        return None, None, None, "Missing_Position"

    expected_resname = AA1_TO_AA3.get(observed_aa)
    if expected_resname:
        for model_id, chain_id, residue in candidates:
            if residue.get_resname() == expected_resname:
                return model_id, chain_id, residue, "Found_Observed_AA"

    # If the observed residue is not present in the template, retain the structural
    # coordinate at this Wuhan position but mark it as template mismatch.
    model_id, chain_id, residue = candidates[0]
    return model_id, chain_id, residue, "Found_Template_Mismatch"


# Compute structural microenvironment around reactive residue
def calculate_microenvironment_features(residue, all_residues, all_atoms):
    target_atom = get_reactive_atom(residue)
    if target_atom is None:
        return {}

    target_coord = target_atom.get_coord()
    residue_coord = residue_centroid(residue)
    resname = residue.get_resname()

    packing_neighbor_atom_count = 0
    basic_count = 0
    acidic_count = 0
    polar_count = 0
    hydrophobic_count = 0
    aromatic_count = 0
    sulfur_count = 0
    total_residue_count = 0
    min_neighbor_distance = np.inf
    local_ca_distances = []

    for other in all_residues:
        if other == residue:
            continue
        if "CA" not in other:
            continue
        other_resname = other.get_resname()
        d = atom_distance(other["CA"].get_coord(), target_coord)
        if d < min_neighbor_distance:
            min_neighbor_distance = d
        if d <= LOCAL_WINDOW_RADIUS:
            total_residue_count += 1
            local_ca_distances.append(d)
            if other_resname in BASIC_RESIDUES:
                basic_count += 1
            if other_resname in ACIDIC_RESIDUES:
                acidic_count += 1
            if other_resname in POLAR_RESIDUES:
                polar_count += 1
            if other_resname in HYDROPHOBIC_RESIDUES:
                hydrophobic_count += 1
            if other_resname in AROMATIC_RESIDUES:
                aromatic_count += 1
            if other_resname in SULFUR_RESIDUES:
                sulfur_count += 1

    for atom in all_atoms:
        if atom.get_parent() == residue:
            continue
        if atom_distance(atom.get_coord(), target_coord) <= PACKING_RADIUS:
            packing_neighbor_atom_count += 1

    packing_density_exposure_proxy = 100.0 / (packing_neighbor_atom_count + 1)
    electrostatic_balance = basic_count - acidic_count

    # True residue SASA from Shrake-Rupley if available.
    residue_sasa = getattr(residue, "sasa", np.nan)
    max_asa = MAX_ASA_TIEN.get(resname, np.nan)
    relative_sasa = residue_sasa / max_asa if pd.notna(residue_sasa) and pd.notna(max_asa) and max_asa > 0 else np.nan

    # Approximate residue depth proxy: distance from residue centroid to the structure centroid.
    # Larger values usually indicate more peripheral residues. This is not MSMS residue depth.
    structure_centroid = np.mean(np.vstack([a.get_coord() for a in all_atoms]), axis=0)
    residue_depth_proxy = atom_distance(residue_coord, structure_centroid) if residue_coord is not None else np.nan

    avg_local_ca_distance = float(np.mean(local_ca_distances)) if local_ca_distances else np.nan

    return {
        "Reactive_Atom_Name": target_atom.get_name(),
        "Packing_Neighbor_Atom_Count_3p5A": packing_neighbor_atom_count,
        "Packing_Density_Exposure_Proxy": packing_density_exposure_proxy,
        "True_SASA_A2": residue_sasa,
        "Relative_SASA": relative_sasa,
        "Electrostatic_Balance_8A": electrostatic_balance,
        "Basic_Count_8A": basic_count,
        "Acidic_Count_8A": acidic_count,
        "Polar_Count_8A": polar_count,
        "Hydrophobic_Count_8A": hydrophobic_count,
        "Aromatic_Count_8A": aromatic_count,
        "Sulfur_Count_8A": sulfur_count,
        "Total_Residue_Count_8A": total_residue_count,
        "Min_Neighbor_CA_Distance_A": min_neighbor_distance if np.isfinite(min_neighbor_distance) else np.nan,
        "Mean_Local_CA_Distance_8A": avg_local_ca_distance,
        "Residue_Depth_Proxy_A": residue_depth_proxy,
    }


# Identify disulfide bonding state of cysteine residues
def compute_disulfide_status(residue, all_residues):
    if residue.get_resname() != "CYS" or "SG" not in residue:
        return 0, np.nan, np.nan
    sg_coord = residue["SG"].get_coord()
    min_sg_dist = np.inf
    partner_pos = np.nan
    for other in all_residues:
        if other == residue:
            continue
        if other.get_resname() == "CYS" and "SG" in other:
            d = atom_distance(sg_coord, other["SG"].get_coord())
            if d < min_sg_dist:
                min_sg_dist = d
                partner_pos = other.get_id()[1]
    is_bonded = int(np.isfinite(min_sg_dist) and min_sg_dist <= DISULFIDE_SG_DISTANCE)
    return is_bonded, min_sg_dist if np.isfinite(min_sg_dist) else np.nan, partner_pos


# Compute spatial proximity to functional regions (ACE2, glycan)
def min_distance_to_positions(residue, residue_by_position, positions):
    target_atom = get_reactive_atom(residue)
    if target_atom is None:
        return np.nan, np.nan
    target_coord = target_atom.get_coord()
    min_dist = np.inf
    nearest_pos = np.nan
    for pos in positions:
        candidates = residue_by_position.get(pos, [])
        for _, _, other_residue in candidates:
            other_coord = residue_centroid(other_residue)
            if other_coord is None:
                continue
            d = atom_distance(target_coord, other_coord)
            if d < min_dist:
                min_dist = d
                nearest_pos = pos
    return min_dist if np.isfinite(min_dist) else np.nan, nearest_pos


def aa_window_features(df_seq, variant_id, residue_col, ref_pos, window=5):
    # Uses neighboring target columns only if present in the extracted target matrix.
    # Full-sequence local windows should ideally be computed in script 101 from the FASTA.
    return {
        "Observed_AA_Hydropathy": hydropathy_value(df_seq.loc[df_seq["Variant_ID"] == variant_id, residue_col].iloc[0]),
        "Observed_AA_Formal_Charge": charge_value(df_seq.loc[df_seq["Variant_ID"] == variant_id, residue_col].iloc[0]),
    }

# ==============================================================================
# 3. MAIN WORKFLOW
# ==============================================================================
def main():
    print("\n[1/5] Loading Wuhan-coordinate nitrosative target matrix...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing target matrix: {INPUT_CSV}")

    df_seq = pd.read_csv(INPUT_CSV)
    if "Variant_ID" not in df_seq.columns:
        raise ValueError("Input CSV must contain a Variant_ID column.")

    print(f"  Loaded {len(df_seq)} variants and {len(df_seq.columns) - 1} target columns.")

    print("\n[2/5] Computing evolutionary conservation features from target matrix...")
    conservation_df = build_position_conservation_table(df_seq)
    conservation_lookup = conservation_df.set_index("Residue_Label").to_dict(orient="index")

    parser = PDBParser(QUIET=True)
    structure_cache = {}
    structural_records = []
    missing_records = []

    residue_cols = [c for c in df_seq.columns if c != "Variant_ID"]

    print("\n[3/5] Extracting structural and microenvironment features...")
    # Iterate over each variant to extract residue-level features
    for _, row in df_seq.iterrows():
        variant_id = str(row["Variant_ID"])
        pdb_id, used_fallback = select_pdb_for_variant(variant_id)

        if pdb_id is None:
            print(f"  No PDB mapping for {variant_id}; skipping.")
            continue

        if pdb_id not in structure_cache:
            try:
                pdb_file = download_pdb(pdb_id)
                structure = parser.get_structure(pdb_id, str(pdb_file))
                compute_sasa(structure)
                all_residues, all_atoms, residue_by_position = collect_structure_objects(structure)
                structure_cache[pdb_id] = {
                    "structure": structure,
                    "all_residues": all_residues,
                    "all_atoms": all_atoms,
                    "residue_by_position": residue_by_position,
                }
            except Exception as exc:
                print(f"  Failed to load {pdb_id} for {variant_id}: {exc}")
                continue

        cached = structure_cache[pdb_id]
        all_residues = cached["all_residues"]
        all_atoms = cached["all_atoms"]
        residue_by_position = cached["residue_by_position"]

        for col in residue_cols:
            ref_aa, ref_pos = parse_residue_column(col)
            if ref_pos is None:
                continue

            observed_aa = str(row[col])
            target_status = classify_target_status(ref_aa, observed_aa)
            domain = get_domain(ref_pos)
            conservation_features = conservation_lookup.get(col, {})

            base_record = {
                "Variant_ID": variant_id,
                "PDB_Template": pdb_id,
                "Used_Wuhan_Fallback_Template": int(used_fallback),
                "Residue_Label": col,
                "Reference_AA": ref_aa,
                "Observed_AA": observed_aa,
                "Residue_Type": "Cysteine" if ref_aa == "C" else "Tyrosine",
                "Sequence_Position": ref_pos,
                "Domain": domain,
                "Is_RBD": int(320 <= ref_pos <= 541),
                "Is_RBM": int(437 <= ref_pos <= 508),
                "Is_S1": int(1 <= ref_pos <= 685),
                "Is_S2": int(686 <= ref_pos <= 1273),
                "Target_Status": target_status,
                "Is_Conserved_Target": int(target_status == "Conserved_Target"),
                "Is_Lost_Target": int(target_status == "Lost_Target"),
                "Is_Target_Type_Switched": int(target_status.startswith("Target_Type_Switched")),
                "Observed_AA_Hydropathy": hydropathy_value(observed_aa),
                "Observed_AA_Formal_Charge": charge_value(observed_aa),
            }
            base_record.update(conservation_features)

            if observed_aa == "-":
                missing_records.append({
                    **base_record,
                    "Residue_Found": 0,
                    "Structure_Coverage_Status": "Deleted_or_Gap_in_Sequence",
                })
                continue

            model_id, chain_id, residue, coverage_status = find_best_residue(
                residue_by_position=residue_by_position,
                ref_pos=ref_pos,
                observed_aa=observed_aa,
            )

            if residue is None:
                missing_records.append({
                    **base_record,
                    "Residue_Found": 0,
                    "Model_ID": np.nan,
                    "Chain_ID": np.nan,
                    "PDB_Residue_Name": np.nan,
                    "Structure_Coverage_Status": coverage_status,
                })
                continue

            pdb_resname = residue.get_resname()
            pdb_aa = AA3_TO_AA1.get(pdb_resname, "X")

            # Extract local physicochemical environment features
            micro_features = calculate_microenvironment_features(residue, all_residues, all_atoms)
            disulfide_bonded, nearest_cys_sg_dist, disulfide_partner_pos = compute_disulfide_status(residue, all_residues)
            ace2_dist, nearest_ace2_pos = min_distance_to_positions(residue, residue_by_position, ACE2_INTERFACE_POSITIONS)
            glycan_dist, nearest_glycan_pos = min_distance_to_positions(residue, residue_by_position, GLYCAN_SITE_POSITIONS)

            full_record = {
                **base_record,
                "Residue_Found": 1,
                "Model_ID": model_id,
                "Chain_ID": chain_id,
                "PDB_Residue_Name": pdb_resname,
                "PDB_AA": pdb_aa,
                "Structure_Coverage_Status": coverage_status,
                "PDB_Matches_Observed_AA": int(pdb_aa == observed_aa),
                "Disulfide_Bonded": disulfide_bonded,
                "Nearest_Cys_SG_Distance_A": nearest_cys_sg_dist,
                "Disulfide_Partner_Position": disulfide_partner_pos,
                "Nearest_ACE2_Interface_Distance_A": ace2_dist,
                "Nearest_ACE2_Interface_Position": nearest_ace2_pos,
                "Within_ACE2_Interface_8A": int(pd.notna(ace2_dist) and ace2_dist <= ACE2_DISTANCE_RADIUS),
                "Nearest_Glycan_Sequon_Distance_A": glycan_dist,
                "Nearest_Glycan_Sequon_Position": nearest_glycan_pos,
                "Within_Glycan_Shield_10A": int(pd.notna(glycan_dist) and glycan_dist <= GLYCAN_DISTANCE_RADIUS),
            }
            full_record.update(micro_features)
            structural_records.append(full_record)

    # Final structured ML feature matrix
    df_features = pd.DataFrame(structural_records)
    df_missing = pd.DataFrame(missing_records)

    print("\n[4/5] Saving feature matrices and QC reports...")
    df_features.to_csv(OUTPUT_FEATURES_CSV, index=False)
    df_missing.to_csv(OUTPUT_MISSING_CSV, index=False)

    if not df_features.empty:
        variant_summary = (
            df_features.groupby("Variant_ID")
            .agg(
                N_Featured_Sites=("Residue_Label", "count"),
                N_Fallback_Template=("Used_Wuhan_Fallback_Template", "sum"),
                Mean_Relative_SASA=("Relative_SASA", "mean"),
                Mean_Exposure_Proxy=("Packing_Density_Exposure_Proxy", "mean"),
                Mean_Electrostatic_Balance=("Electrostatic_Balance_8A", "mean"),
                Mean_Entropy=("Shannon_Entropy", "mean"),
                N_ACE2_Proximal=("Within_ACE2_Interface_8A", "sum"),
                N_Glycan_Proximal=("Within_Glycan_Shield_10A", "sum"),
            )
            .reset_index()
        )
    else:
        variant_summary = pd.DataFrame()
    variant_summary.to_csv(OUTPUT_VARIANT_SUMMARY_CSV, index=False)

    print(f"  Feature matrix saved: {OUTPUT_FEATURES_CSV}")
    print(f"  Missing residue report saved: {OUTPUT_MISSING_CSV}")
    print(f"  Variant summary saved: {OUTPUT_VARIANT_SUMMARY_CSV}")
    print(f"  Extracted feature rows: {len(df_features)}")
    print(f"  Missing or skipped rows: {len(df_missing)}")

    print("\n[5/5] Rendering diagnostic plots...")
    if df_features.empty:
        print("  No structural features were extracted. Please check PDB mapping and residue numbering.")
        return

    sns.set_theme(style="whitegrid")

    numeric_candidates = [
        "Sequence_Position",
        "Relative_SASA",
        "True_SASA_A2",
        "Packing_Density_Exposure_Proxy",
        "Packing_Neighbor_Atom_Count_3p5A",
        "Electrostatic_Balance_8A",
        "Basic_Count_8A",
        "Acidic_Count_8A",
        "Polar_Count_8A",
        "Hydrophobic_Count_8A",
        "Aromatic_Count_8A",
        "Sulfur_Count_8A",
        "Residue_Depth_Proxy_A",
        "Nearest_ACE2_Interface_Distance_A",
        "Nearest_Glycan_Sequon_Distance_A",
        "Conservation_Frequency",
        "Mutation_Frequency",
        "Shannon_Entropy",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df_features.columns]
    corr_matrix = df_features[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    # Plot correlation between engineered features
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=False)
    plt.title("Structural, Evolutionary, and Interface Feature Correlation Matrix", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(PLOT_CORRELATION, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    y_col = "Relative_SASA" if "Relative_SASA" in df_features.columns and df_features["Relative_SASA"].notna().any() else "Packing_Density_Exposure_Proxy"
    # Plot feature space distribution of residues
    sns.scatterplot(
        data=df_features,
        x="Electrostatic_Balance_8A",
        y=y_col,
        hue="Residue_Type",
        style="Domain",
        size="Shannon_Entropy" if "Shannon_Entropy" in df_features.columns else None,
        sizes=(40, 180),
        edgecolor="black",
        alpha=0.85,
    )
    plt.title("Nitrosative Target Microenvironment Feature Space", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Electrostatic Balance Within 8 Angstroms")
    plt.ylabel(y_col.replace("_", " "))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Feature Metadata")
    plt.tight_layout()
    plt.savefig(PLOT_VARIANT_DIST, dpi=300)
    plt.close()

    print(f"  Correlation plot saved: {PLOT_CORRELATION}")
    print(f"  Feature-space plot saved: {PLOT_VARIANT_DIST}")
    print("\n--- STRUCTURAL FEATURE EXTRACTION COMPLETE ---\n")


if __name__ == "__main__":
    main()
