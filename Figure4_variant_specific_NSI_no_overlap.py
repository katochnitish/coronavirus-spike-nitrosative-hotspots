"""
Figure 4: Variant-specific NSI analysis with non-overlapping figure layout.

Code developed by
-----------------
Nitish Katoch

Purpose
-------
This script computes position-wise Nitrosative Susceptibility Index profiles
for Wuhan, Alpha, Beta, Delta, and Omicron BA.2 spike proteins. It calculates
variant-specific changes relative to Wuhan, summarizes values across functional
spike domains, identifies residue hotspots, and creates individual panels plus
a combined Figure 4 arranged to minimize overlap between plot elements.

Input
-----
02_CuratedRawData/combined_curated_aligned.fasta

The input must be a multiple-sequence FASTA alignment with recognizable group
names in the headers. Wuhan-Hu-1 is used as the positional reference, and
alignment columns containing gaps in the Wuhan sequence are removed. Analysis
is restricted to the 1,273-residue spike protein.

Main calculations
-----------------
1. Reads, classifies, and groups aligned spike sequences.
2. Derives a consensus amino acid at every position for each group.
3. Assigns each position to a spike domain and ACE2-interface category.
4. Calculates normalized within-group Shannon entropy.
5. Calculates mutation frequency relative to Wuhan.
6. Assigns amino acid reactivity, approximate exposure, domain, and
   ACE2-interface scores.
7. Calculates raw NSI using the weighted model:

   Raw NSI =
       0.32 x residue reactivity
     + 0.22 x approximate exposure
     + 0.14 x domain weight
     + 0.10 x within-group entropy
     + 0.10 x mutation frequency from Wuhan
     + 0.12 x ACE2-interface score

8. Min-max normalizes raw NSI and applies seven-residue smoothing.
9. Calculates position-wise delta NSI relative to Wuhan.
10. Computes domain-level mean, median, standard deviation, and delta NSI
    statistics.
11. Selects and ranks hotspots using NSI magnitude, positive delta NSI,
    mutation frequency, entropy, residue reactivity, and interface relevance.

Plots generated
---------------
Figure 4A, line plot:
    Smoothed position-wise NSI curves for Wuhan and all variants.

Figure 4B, heatmap:
    Delta NSI relative to Wuhan across spike positions for Alpha, Beta, Delta,
    and Omicron BA.2.

Figure 4C, violin plots:
    Domain-wise NSI distributions for all analyzed groups.

Combined Figure 4:
    A non-overlapping multi-panel arrangement of the three plot types.

Image outputs
-------------
03_Results/Figure4A_variant_specific_NSI_curves.png
03_Results/Figure4B_delta_NSI_heatmap_vs_Wuhan.png
03_Results/Figure4C_domainwise_variant_NSI_violin.png
03_Results/Figure4_variant_specific_NSI_combined.png

Table outputs
-------------
03_Results/Figure4_positionwise_variant_NSI.csv
03_Results/Figure4_delta_NSI_vs_Wuhan.csv
03_Results/Figure4_domainwise_variant_NSI_summary.csv
03_Results/Figure4_variant_NSI_hotspot_amino_acid_table.csv

Dependencies
------------
Python 3, NumPy, and Matplotlib.

Notes
-----
The NSI model is a weighted computational framework implemented directly in the
script. The exposure and domain scores are biologically informed approximations
and should not be interpreted as direct structural measurements.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =========================================================
# Variant-specific Nitrosative Susceptibility Index analysis
# Focused SARS-CoV-2 variant comparison
# Wuhan, Alpha, Beta, Delta, Omicron BA.2
# =========================================================
#
# Input:
#   02_CuratedRawData/combined_curated_aligned.fasta
#
# Outputs:
#   03_Results/Figure4A_variant_specific_NSI_curves.png
#   03_Results/Figure4B_delta_NSI_heatmap_vs_Wuhan.png
#   03_Results/Figure4C_domainwise_variant_NSI_violin.png
#   03_Results/Figure4_variant_specific_NSI_combined.png
#
#   03_Results/Figure4_positionwise_variant_NSI.csv
#   03_Results/Figure4_delta_NSI_vs_Wuhan.csv
#   03_Results/Figure4_domainwise_variant_NSI_summary.csv
#   03_Results/Figure4_variant_NSI_hotspot_amino_acid_table.csv
#
# Notes:
#   1. The script computes variant-specific NSI profiles.
#   2. ΔNSI is computed relative to Wuhan-Hu-1 at each amino acid position.
#   3. Domain-wise NSI distributions are shown using violin plots.
#   4. A residue-level hotspot table is exported for all variants and Wuhan.
#   5. Only PNG and CSV files are saved.
#
# =========================================================


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ALIGNMENT_FILE = BASE_DIR / "02_CuratedRawData" / "combined_curated_aligned.fasta"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CURVES_PNG = OUTPUT_DIR / "Figure4A_variant_specific_NSI_curves.png"
OUTPUT_HEATMAP_PNG = OUTPUT_DIR / "Figure4B_delta_NSI_heatmap_vs_Wuhan.png"
OUTPUT_VIOLIN_PNG = OUTPUT_DIR / "Figure4C_domainwise_variant_NSI_violin.png"
OUTPUT_COMBINED_PNG = OUTPUT_DIR / "Figure4_variant_specific_NSI_combined.png"

OUTPUT_VARIANT_NSI_CSV = OUTPUT_DIR / "Figure4_positionwise_variant_NSI.csv"
OUTPUT_DELTA_CSV = OUTPUT_DIR / "Figure4_delta_NSI_vs_Wuhan.csv"
OUTPUT_DOMAIN_SUMMARY_CSV = OUTPUT_DIR / "Figure4_domainwise_variant_NSI_summary.csv"
OUTPUT_HOTSPOT_TABLE_CSV = OUTPUT_DIR / "Figure4_variant_NSI_hotspot_amino_acid_table.csv"


# =========================================================
# Main settings
# =========================================================

SPIKE_LENGTH = 1273
SMOOTHING_WINDOW = 7

INVALID_AA = {"-", "?", ".", "X", "B", "Z", "J", "U", "O", "*"}

GROUP_ORDER = [
    "Wuhan",
    "Alpha",
    "Beta",
    "Delta",
    "Omicron_BA2",
]

DISPLAY_NAMES = {
    "Wuhan": "Wuhan",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Delta": "Delta",
    "Omicron_BA2": "Omicron BA.2",
}

CURVE_GROUPS = GROUP_ORDER
HEATMAP_GROUPS = ["Alpha", "Beta", "Delta", "Omicron_BA2"]
VIOLIN_GROUPS = GROUP_ORDER

HOTSPOT_TOP_N_PER_VARIANT = 40
HOTSPOT_MIN_NSI = 0.60
HOTSPOT_MIN_DELTA_NSI = 0.035


# =========================================================
# Spike domain definitions
# =========================================================

DOMAINS = [
    ("SP", 1, 13, "#BDBDBD"),
    ("NTD", 14, 305, "#F2D27A"),
    ("RBD", 319, 437, "#A6CEE3"),
    ("RBM", 438, 506, "#1B9E77"),
    ("RBD_core", 507, 541, "#A6CEE3"),
    ("S1/S2", 675, 692, "#4F4F4F"),
    ("FP", 816, 833, "#B8D78A"),
    ("HR1", 912, 984, "#F0A33B"),
    ("HR2", 1163, 1213, "#F4C15D"),
    ("TM/CT", 1214, 1273, "#D9D9D9"),
]

DOMAIN_ORDER = [
    "NTD",
    "RBD",
    "RBM",
    "RBD_core",
    "S1/S2",
    "FP",
    "HR1",
    "HR2",
    "TM/CT",
    "S1_other",
    "S2_other",
]

VIOLIN_DOMAINS = [
    "NTD",
    "RBD",
    "RBM",
    "S1/S2",
    "FP",
    "HR1",
    "HR2",
    "TM/CT",
]

DOMAIN_DISPLAY = {
    "SP": "SP",
    "NTD": "NTD",
    "RBD": "RBD",
    "RBM": "RBM",
    "RBD_core": "RBD core",
    "S1/S2": "S1/S2",
    "FP": "FP",
    "HR1": "HR1",
    "HR2": "HR2",
    "TM/CT": "TM/CT",
    "S1_other": "S1 other",
    "S2_other": "S2 other",
    "Outside": "Outside",
}


# =========================================================
# NSI model
# =========================================================

REACTIVITY_SCORE = {
    "C": 1.00,
    "Y": 0.90,
    "M": 0.75,
    "W": 0.70,
    "H": 0.60,
    "K": 0.25,
    "R": 0.25,
}

DOMAIN_WEIGHT = {
    "SP": 0.25,
    "NTD": 0.65,
    "RBD": 0.85,
    "RBM": 1.00,
    "RBD_core": 0.85,
    "S1/S2": 1.00,
    "FP": 0.85,
    "HR1": 0.70,
    "HR2": 0.70,
    "TM/CT": 0.75,
    "S1_other": 0.50,
    "S2_other": 0.55,
    "Outside": 0.00,
}

DOMAIN_EXPOSURE_WEIGHT = {
    "SP": 0.60,
    "NTD": 0.75,
    "RBD": 0.85,
    "RBM": 0.95,
    "RBD_core": 0.80,
    "S1/S2": 1.00,
    "FP": 0.70,
    "HR1": 0.55,
    "HR2": 0.55,
    "TM/CT": 0.45,
    "S1_other": 0.65,
    "S2_other": 0.55,
    "Outside": 0.00,
}

ACE2_DIRECT_CONTACT_POSITIONS = {
    417, 449, 453, 455, 456, 475, 486, 487, 489, 493, 498, 501, 505
}

ACE2_NEAR_INTERFACE_POSITIONS = {
    403, 405, 408, 416, 420, 421, 439, 440, 441, 442, 443, 444, 445,
    446, 447, 448, 450, 451, 452, 454, 457, 458, 459, 460, 472, 473,
    474, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 488, 490,
    491, 492, 494, 495, 496, 497, 499, 500, 502, 503, 504, 506,
}

KEY_POSITIONS = {
    417: "K417",
    449: "Y449",
    453: "Y453",
    486: "F486",
    489: "Y489",
    493: "Q493",
    498: "Q498",
    501: "N501",
    505: "Y505",
    681: "P681",
    816: "FP",
    912: "HR1",
    1163: "HR2",
    1236: "TM/CT",
}

W_REACTIVITY = 0.32
W_EXPOSURE = 0.22
W_DOMAIN = 0.14
W_WITHIN_GROUP_ENTROPY = 0.10
W_MUTATION_FROM_WUHAN = 0.10
W_ACE2 = 0.12


# =========================================================
# Figure style
# =========================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GROUP_COLORS = {
    "Wuhan": "#111111",
    "Alpha": "#4C78A8",
    "Beta": "#F58518",
    "Delta": "#E45756",
    "Omicron_BA2": "#7F3C8D",
}


# =========================================================
# FASTA utilities
# =========================================================

def read_fasta(path):
    records = []
    header = None
    seq_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines).upper()))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

    if header is not None:
        records.append((header, "".join(seq_lines).upper()))

    return records


def normalize_text(text):
    text = str(text).lower()

    for ch in ["|", "-", ".", " ", "/", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def classify_record(header):
    text = normalize_text(header)

    if (
        "wuhan" in text
        or "hu_1" in text
        or "hu1" in text
        or "refseq" in text
        or "yp_009724390" in text
        or "nc_045512" in text
    ):
        return "Wuhan"

    if "alpha" in text or "b_1_1_7" in text or "b117" in text or "501y_v1" in text:
        return "Alpha"

    if "beta" in text or "b_1_351" in text or "b1351" in text or "501y_v2" in text:
        return "Beta"

    if "delta" in text or "b_1_617_2" in text or "b16172" in text:
        return "Delta"

    if (
        "omicron_ba2" in text
        or "omicron_ba_2" in text
        or "ba2" in text
        or "ba_2" in text
        or "b_1_1_529_ba_2" in text
    ):
        return "Omicron_BA2"

    return None


def find_wuhan_reference(records):
    for header, seq in records:
        if classify_record(header) == "Wuhan":
            return header, seq

    raise ValueError(
        "Wuhan reference was not found. Please check the FASTA header for Wuhan, Hu-1, RefSeq, YP_009724390, or NC_045512."
    )


def group_sequences(records):
    grouped = defaultdict(list)
    unclassified = []

    for header, seq in records:
        group = classify_record(header)

        if group is None:
            unclassified.append(header)
        else:
            grouped[group].append((header, seq))

    return grouped, unclassified


def remove_reference_gap_columns(reference_seq, sequences_by_group):
    keep_indices = [i for i, aa in enumerate(reference_seq) if aa != "-"]

    cleaned = {}

    for group, seqs in sequences_by_group.items():
        cleaned[group] = []

        for seq in seqs:
            cleaned_seq = "".join(seq[i] if i < len(seq) else "-" for i in keep_indices)
            cleaned[group].append(cleaned_seq[:SPIKE_LENGTH])

    return cleaned


# =========================================================
# NSI calculation
# =========================================================

def assign_domain(position):
    for name, start, end, _ in DOMAINS:
        if start <= position <= end:
            return name

    if 1 <= position <= 685:
        return "S1_other"

    if 686 <= position <= SPIKE_LENGTH:
        return "S2_other"

    return "Outside"


def ace2_relation(position):
    if position in ACE2_DIRECT_CONTACT_POSITIONS:
        return "Direct contact"

    if position in ACE2_NEAR_INTERFACE_POSITIONS:
        return "Near interface"

    return "No"


def shannon_entropy(residues):
    residues = [aa for aa in residues if aa not in INVALID_AA]

    if not residues:
        return 0.0

    counts = Counter(residues)
    total = sum(counts.values())

    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def rolling_mean(values, window):
    values = np.array(values, dtype=float)

    if window <= 1:
        return values

    output = np.zeros_like(values, dtype=float)
    half = window // 2

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        output[i] = np.nanmean(values[start:end])

    return output


def consensus_aa(residues, fallback="-"):
    residues = [aa for aa in residues if aa not in INVALID_AA]

    if not residues:
        return fallback

    counts = Counter(residues)
    return counts.most_common(1)[0][0]


def compute_group_entropy_profile(seqs):
    entropy = []

    for idx in range(SPIKE_LENGTH):
        residues = []

        for seq in seqs:
            if idx < len(seq):
                aa = seq[idx]
                if aa not in INVALID_AA:
                    residues.append(aa)

        entropy.append(shannon_entropy(residues))

    entropy = np.array(entropy, dtype=float)

    max_entropy = np.nanmax(entropy) if len(entropy) else 0.0

    if max_entropy > 0:
        entropy_norm = entropy / max_entropy
    else:
        entropy_norm = np.zeros_like(entropy)

    return rolling_mean(entropy_norm, SMOOTHING_WINDOW)


def approximate_exposure_score(position, aa, domain):
    base = DOMAIN_EXPOSURE_WEIGHT.get(domain, 0.50)

    if position in ACE2_DIRECT_CONTACT_POSITIONS:
        base = max(base, 1.00)
    elif position in ACE2_NEAR_INTERFACE_POSITIONS:
        base = max(base, 0.85)

    if 675 <= position <= 692:
        base = max(base, 1.00)

    if 816 <= position <= 833:
        base = max(base, 0.70)

    if 1235 <= position <= 1273 and aa == "C":
        base = max(base, 0.70)

    return min(max(base, 0.0), 1.0)


def ace2_interface_score(position):
    if position in ACE2_DIRECT_CONTACT_POSITIONS:
        return 1.00

    if position in ACE2_NEAR_INTERFACE_POSITIONS:
        return 0.60

    return 0.00


def compute_variant_specific_nsi(cleaned_sequences):
    reference = cleaned_sequences["Wuhan"][0][:SPIKE_LENGTH]

    groups_available = [
        group for group in GROUP_ORDER
        if group in cleaned_sequences and len(cleaned_sequences[group]) > 0
    ]

    raw_rows = []

    for group in groups_available:
        seqs = cleaned_sequences[group]
        group_entropy_smooth = compute_group_entropy_profile(seqs)

        for idx in range(SPIKE_LENGTH):
            position = idx + 1
            ref_aa = reference[idx] if idx < len(reference) else "-"
            domain = assign_domain(position)

            residues = []

            for seq in seqs:
                if idx < len(seq):
                    aa = seq[idx]
                    if aa not in INVALID_AA:
                        residues.append(aa)

            aa = consensus_aa(residues, fallback=ref_aa)

            if aa in INVALID_AA:
                continue

            total_valid = len(residues)

            if total_valid > 0 and ref_aa not in INVALID_AA:
                mutation_count = sum(1 for residue in residues if residue != ref_aa)
                mutation_frequency_from_wuhan = mutation_count / total_valid
            else:
                mutation_frequency_from_wuhan = 0.0

            reactivity = REACTIVITY_SCORE.get(aa, 0.0)
            exposure = approximate_exposure_score(position, aa, domain)
            domain_score = DOMAIN_WEIGHT.get(domain, 0.0)
            entropy_score = group_entropy_smooth[idx]
            mutation_score = mutation_frequency_from_wuhan
            ace2_score = ace2_interface_score(position)

            raw_nsi = (
                W_REACTIVITY * reactivity
                + W_EXPOSURE * exposure
                + W_DOMAIN * domain_score
                + W_WITHIN_GROUP_ENTROPY * entropy_score
                + W_MUTATION_FROM_WUHAN * mutation_score
                + W_ACE2 * ace2_score
            )

            raw_rows.append(
                {
                    "Group": group,
                    "Variant": DISPLAY_NAMES[group],
                    "Position": position,
                    "Reference_AA": ref_aa,
                    "Variant_consensus_AA": aa,
                    "Residue_change": f"{ref_aa}{position}{aa}" if ref_aa != aa else f"{ref_aa}{position}",
                    "Domain": domain,
                    "Domain_display": DOMAIN_DISPLAY.get(domain, domain),
                    "Reactive_residue": "Yes" if aa in {"C", "Y", "M", "W", "H"} else "No",
                    "ACE2_relation": ace2_relation(position),
                    "Within_group_entropy": float(entropy_score),
                    "Mutation_frequency_from_Wuhan": float(mutation_frequency_from_wuhan),
                    "Residue_reactivity_score": float(reactivity),
                    "Approximate_exposure_score": float(exposure),
                    "Domain_weight_score": float(domain_score),
                    "ACE2_interface_score": float(ace2_score),
                    "Raw_NSI": float(raw_nsi),
                }
            )

    all_raw_scores = np.array([row["Raw_NSI"] for row in raw_rows], dtype=float)
    min_raw = float(np.nanmin(all_raw_scores))
    max_raw = float(np.nanmax(all_raw_scores))

    for row in raw_rows:
        if max_raw > min_raw:
            row["NSI"] = (row["Raw_NSI"] - min_raw) / (max_raw - min_raw)
        else:
            row["NSI"] = 0.0

    rows_by_group = defaultdict(list)

    for row in raw_rows:
        rows_by_group[row["Group"]].append(row)

    for group, group_rows in rows_by_group.items():
        group_rows = sorted(group_rows, key=lambda r: r["Position"])

        nsi_values = np.array([row["NSI"] for row in group_rows], dtype=float)
        smoothed_values = rolling_mean(nsi_values, SMOOTHING_WINDOW)

        for row, smoothed_value in zip(group_rows, smoothed_values):
            row["Smoothed_NSI"] = float(smoothed_value)

    return raw_rows


def compute_delta_rows(variant_rows):
    by_group_position = {}

    for row in variant_rows:
        by_group_position[(row["Group"], row["Position"])] = row

    delta_rows = []

    for row in variant_rows:
        group = row["Group"]
        position = row["Position"]

        if group == "Wuhan":
            delta_nsi = 0.0
        else:
            wuhan_row = by_group_position.get(("Wuhan", position))

            if wuhan_row is None:
                continue

            delta_nsi = row["Smoothed_NSI"] - wuhan_row["Smoothed_NSI"]

        out = row.copy()
        out["Delta_NSI_vs_Wuhan"] = float(delta_nsi)
        delta_rows.append(out)

    return delta_rows


def summarize_domainwise(delta_rows):
    summary_rows = []

    for group in GROUP_ORDER:
        for domain in DOMAIN_ORDER:
            domain_rows = [
                row for row in delta_rows
                if row["Group"] == group and row["Domain"] == domain
            ]

            if not domain_rows:
                continue

            values = np.array([row["Smoothed_NSI"] for row in domain_rows], dtype=float)
            delta_values = np.array([row["Delta_NSI_vs_Wuhan"] for row in domain_rows], dtype=float)

            summary_rows.append(
                {
                    "Group": group,
                    "Variant": DISPLAY_NAMES[group],
                    "Domain": domain,
                    "Domain_display": DOMAIN_DISPLAY.get(domain, domain),
                    "N_positions": int(len(values)),
                    "Mean_NSI": float(np.mean(values)),
                    "Median_NSI": float(np.median(values)),
                    "SD_NSI": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "Mean_delta_NSI_vs_Wuhan": float(np.mean(delta_values)),
                    "Median_delta_NSI_vs_Wuhan": float(np.median(delta_values)),
                }
            )

    return summary_rows


def build_hotspot_table(delta_rows):
    hotspot_rows = []

    for group in GROUP_ORDER:
        group_rows = [
            row for row in delta_rows
            if row["Group"] == group and row["Domain"] in DOMAIN_ORDER
        ]

        selected = []

        for row in group_rows:
            is_key_position = row["Position"] in KEY_POSITIONS
            is_high_nsi = row["Smoothed_NSI"] >= HOTSPOT_MIN_NSI
            is_delta_hotspot = (
                group != "Wuhan"
                and row["Delta_NSI_vs_Wuhan"] >= HOTSPOT_MIN_DELTA_NSI
            )
            is_reactive_interface = (
                row["Reactive_residue"] == "Yes"
                and row["ACE2_relation"] != "No"
            )

            if is_key_position or is_high_nsi or is_delta_hotspot or is_reactive_interface:
                selected.append(row)

        selected = sorted(
            selected,
            key=lambda r: (
                r["Smoothed_NSI"],
                r["Delta_NSI_vs_Wuhan"],
                r["Mutation_frequency_from_Wuhan"],
                r["Within_group_entropy"],
            ),
            reverse=True,
        )

        selected = selected[:HOTSPOT_TOP_N_PER_VARIANT]

        for rank, row in enumerate(selected, start=1):
            out = {
                "Variant": row["Variant"],
                "Group": row["Group"],
                "Rank_within_variant": rank,
                "Position": row["Position"],
                "Reference_AA": row["Reference_AA"],
                "Variant_consensus_AA": row["Variant_consensus_AA"],
                "Residue_change": row["Residue_change"],
                "Domain": row["Domain"],
                "Domain_display": row["Domain_display"],
                "Reactive_residue": row["Reactive_residue"],
                "ACE2_relation": row["ACE2_relation"],
                "Smoothed_NSI": row["Smoothed_NSI"],
                "Delta_NSI_vs_Wuhan": row["Delta_NSI_vs_Wuhan"],
                "Mutation_frequency_from_Wuhan": row["Mutation_frequency_from_Wuhan"],
                "Within_group_entropy": row["Within_group_entropy"],
                "Residue_reactivity_score": row["Residue_reactivity_score"],
                "Approximate_exposure_score": row["Approximate_exposure_score"],
                "Domain_weight_score": row["Domain_weight_score"],
                "ACE2_interface_score": row["ACE2_interface_score"],
            }
            hotspot_rows.append(out)

    return hotspot_rows


# =========================================================
# CSV output
# =========================================================

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = {}

            for key in fieldnames:
                value = row.get(key, "")

                if isinstance(value, float):
                    value = f"{value:.6f}"

                out[key] = value

            writer.writerow(out)


def save_variant_nsi_csv(rows):
    fieldnames = [
        "Group",
        "Variant",
        "Position",
        "Reference_AA",
        "Variant_consensus_AA",
        "Residue_change",
        "Domain",
        "Domain_display",
        "Reactive_residue",
        "ACE2_relation",
        "Within_group_entropy",
        "Mutation_frequency_from_Wuhan",
        "Residue_reactivity_score",
        "Approximate_exposure_score",
        "Domain_weight_score",
        "ACE2_interface_score",
        "Raw_NSI",
        "NSI",
        "Smoothed_NSI",
    ]

    rows_sorted = sorted(rows, key=lambda r: (GROUP_ORDER.index(r["Group"]), r["Position"]))
    write_csv(OUTPUT_VARIANT_NSI_CSV, rows_sorted, fieldnames)


def save_delta_csv(rows):
    fieldnames = [
        "Group",
        "Variant",
        "Position",
        "Reference_AA",
        "Variant_consensus_AA",
        "Residue_change",
        "Domain",
        "Domain_display",
        "Reactive_residue",
        "ACE2_relation",
        "Smoothed_NSI",
        "Delta_NSI_vs_Wuhan",
        "Mutation_frequency_from_Wuhan",
        "Within_group_entropy",
    ]

    rows_sorted = sorted(rows, key=lambda r: (GROUP_ORDER.index(r["Group"]), r["Position"]))
    write_csv(OUTPUT_DELTA_CSV, rows_sorted, fieldnames)


def save_domain_summary_csv(summary_rows):
    fieldnames = [
        "Group",
        "Variant",
        "Domain",
        "Domain_display",
        "N_positions",
        "Mean_NSI",
        "Median_NSI",
        "SD_NSI",
        "Mean_delta_NSI_vs_Wuhan",
        "Median_delta_NSI_vs_Wuhan",
    ]

    write_csv(OUTPUT_DOMAIN_SUMMARY_CSV, summary_rows, fieldnames)


def save_hotspot_table_csv(hotspot_rows):
    fieldnames = [
        "Variant",
        "Group",
        "Rank_within_variant",
        "Position",
        "Reference_AA",
        "Variant_consensus_AA",
        "Residue_change",
        "Domain",
        "Domain_display",
        "Reactive_residue",
        "ACE2_relation",
        "Smoothed_NSI",
        "Delta_NSI_vs_Wuhan",
        "Mutation_frequency_from_Wuhan",
        "Within_group_entropy",
        "Residue_reactivity_score",
        "Approximate_exposure_score",
        "Domain_weight_score",
        "ACE2_interface_score",
    ]

    write_csv(OUTPUT_HOTSPOT_TABLE_CSV, hotspot_rows, fieldnames)


# =========================================================
# Plot helpers
# =========================================================

def add_domain_backgrounds(ax, alpha=0.065):
    for name, start, end, color in DOMAINS:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)


def draw_domain_track(ax, y=-0.165, height=0.045, fontsize=10):
    ax.plot(
        [1, SPIKE_LENGTH],
        [y, y],
        transform=ax.get_xaxis_transform(),
        color="#666666",
        lw=0.9,
        alpha=0.50,
        clip_on=False,
    )

    for name, start, end, color in DOMAINS:
        ax.add_patch(
            Rectangle(
                (start, y - height / 2),
                end - start + 1,
                height,
                transform=ax.get_xaxis_transform(),
                facecolor=color,
                edgecolor="none",
                alpha=0.95,
                clip_on=False,
                zorder=3,
            )
        )

        label = DOMAIN_DISPLAY.get(name, name)

        if end - start >= 35:
            ax.text(
                (start + end) / 2,
                y - 0.047,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=fontsize,
                fontweight="bold",
                color="#222222",
                clip_on=False,
            )


def get_group_curve(rows, group):
    group_rows = sorted(
        [row for row in rows if row["Group"] == group],
        key=lambda r: r["Position"],
    )

    positions = np.array([row["Position"] for row in group_rows], dtype=int)
    values = np.array([row["Smoothed_NSI"] for row in group_rows], dtype=float)

    return positions, values


def get_delta_matrix(delta_rows):
    matrix = []
    labels = []

    for group in HEATMAP_GROUPS:
        group_rows = sorted(
            [row for row in delta_rows if row["Group"] == group],
            key=lambda r: r["Position"],
        )

        if not group_rows:
            continue

        values = np.full(SPIKE_LENGTH, np.nan, dtype=float)

        for row in group_rows:
            position = row["Position"]
            if 1 <= position <= SPIKE_LENGTH:
                values[position - 1] = row["Delta_NSI_vs_Wuhan"]

        matrix.append(values)
        labels.append(DISPLAY_NAMES[group])

    return np.array(matrix, dtype=float), labels


# =========================================================
# Panel A: variant-specific NSI curves
# =========================================================

def plot_variant_curves(rows, ax=None, save=True):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(16.8, 6.6), facecolor="white")
    else:
        fig = ax.figure

    ax.set_facecolor("white")
    add_domain_backgrounds(ax, alpha=0.055)

    for group in CURVE_GROUPS:
        positions, values = get_group_curve(rows, group)

        if len(positions) == 0:
            continue

        if group == "Wuhan":
            lw = 3.4
            alpha = 0.98
            zorder = 8
        else:
            lw = 2.5
            alpha = 0.90
            zorder = 5

        ax.plot(
            positions,
            values,
            color=GROUP_COLORS[group],
            lw=lw,
            alpha=alpha,
            label=DISPLAY_NAMES[group],
            zorder=zorder,
        )

    # Staggered residue labels.
    # This keeps the original figure size but avoids label collision
    # around the dense RBD/RBM region.
    label_positions = [449, 453, 489, 505, 681]
    label_y_positions = [1.030, 1.15, 1.030, 1.115, 1.030]
    for pos, label_y in zip(label_positions, label_y_positions):
        label = KEY_POSITIONS[pos]

        ax.axvline(
            pos,
            color="#777777",
            linestyle=":",
            lw=0.8,
            alpha=0.50,
            zorder=1,
        )

        ax.text(
            pos,
            label_y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            rotation=90,
            color="#333333",
            clip_on=False,
        )

    draw_domain_track(ax, y=-0.18, height=0.045, fontsize=10)

    ax.set_xlim(1, SPIKE_LENGTH)
    ax.set_ylim(0.0, 1.08)

    ax.set_xticks([1, 100, 200, 300, 400, 500, 600, 681, 800, 900, 1000, 1100, 1200, 1273])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    ax.tick_params(axis="x", labelsize=14, length=5)
    ax.tick_params(axis="y", labelsize=14, length=5)

    # ax.set_xlabel("Wuhan-Hu-1 spike residue position", fontsize=17, fontweight="bold", labelpad=26)
    ax.set_ylabel("Variant-specific NSI", fontsize=17, fontweight="bold")

    ax.grid(axis="y", color="#E5E5E5", lw=0.8, alpha=0.85)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=12,
        handlelength=2.4,
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(0.8)

    if own_fig and save:
        plt.tight_layout(rect=[0.035, 0.10, 0.995, 0.90])
        fig.savefig(OUTPUT_CURVES_PNG, dpi=900, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    return ax


# =========================================================
# Panel B: ΔNSI heatmap relative to Wuhan
# =========================================================

def plot_delta_heatmap(delta_rows, ax=None, save=True):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(16.8, 5.8), facecolor="white")
    else:
        fig = ax.figure

    matrix, labels = get_delta_matrix(delta_rows)

    abs_max = np.nanpercentile(np.abs(matrix), 99)

    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = 0.05

    abs_max = max(abs_max, 0.05)

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-abs_max,
        vmax=abs_max,
        extent=[1, SPIKE_LENGTH, len(labels) - 0.5, -0.5],
    )

    for name, start, end, color in DOMAINS:
        ax.axvline(start, color="white", lw=0.8, alpha=0.60)
        ax.axvline(end, color="white", lw=0.8, alpha=0.60)

        if end - start >= 35:
            ax.text(
                (start + end) / 2,
                len(labels) + 0.12,
                DOMAIN_DISPLAY.get(name, name),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#222222",
                clip_on=False,
            )

    # Staggered residue labels above the heatmap.
    # This keeps the original heatmap size but separates nearby labels.
    label_positions = [449, 453, 489, 505, 681]
    label_y_positions = [-0.82, -1.30, -0.82, -1.18, -0.82]
    for pos, label_y in zip(label_positions, label_y_positions):
        label = KEY_POSITIONS[pos]

        ax.axvline(
            pos,
            color="#222222",
            linestyle=":",
            lw=0.75,
            alpha=0.65,
        )

        ax.text(
            pos,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            rotation=90,
            color="#222222",
            clip_on=False,
        )

    ax.set_xlim(1, SPIKE_LENGTH)
    ax.set_ylim(len(labels) - 0.5, -0.5)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=14, fontweight="bold")

    ax.set_xticks([1, 100, 200, 300, 400, 500, 600, 681, 800, 900, 1000, 1100, 1200, 1273])
    ax.tick_params(axis="x", labelsize=14, length=5)
    ax.tick_params(axis="y", labelsize=14, length=0)

    # ax.set_xlabel("Wuhan-Hu-1 spike residue position", fontsize=17, fontweight="bold", labelpad=32)
    ax.set_ylabel("Variant", fontsize=17, fontweight="bold")

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_linewidth(1.0)

    colorbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.026)
    colorbar.set_label("ΔNSI relative to Wuhan", fontsize=14, fontweight="bold")
    colorbar.ax.tick_params(labelsize=12)

    if own_fig and save:
        plt.tight_layout(rect=[0.04, 0.15, 0.96, 0.82])
        fig.savefig(OUTPUT_HEATMAP_PNG, dpi=900, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    return ax


# =========================================================
# Panel C: domain-wise variant NSI violin plot
# =========================================================

def plot_domainwise_violin(delta_rows, ax=None, save=True):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(17.2, 7.4), facecolor="white")
    else:
        fig = ax.figure

    data = []
    positions = []
    colors = []
    domain_tick_positions = []
    domain_tick_labels = []
    group_centers = []

    gap = 1.35
    group_width = len(VIOLIN_DOMAINS) + gap

    for gi, group in enumerate(VIOLIN_GROUPS):
        base = gi * group_width
        local_positions = []

        for di, domain in enumerate(VIOLIN_DOMAINS):
            values = [
                row["Smoothed_NSI"]
                for row in delta_rows
                if row["Group"] == group and row["Domain"] == domain
            ]

            if len(values) == 0:
                values = [np.nan, np.nan]
            elif len(values) == 1:
                values = values + values

            current_position = base + di + 1
            local_positions.append(current_position)

            data.append(values)
            positions.append(current_position)
            colors.append(GROUP_COLORS[group])

            domain_tick_positions.append(current_position)
            domain_tick_labels.append(DOMAIN_DISPLAY[domain])

        group_centers.append((np.mean(local_positions), DISPLAY_NAMES[group]))

        if gi < len(VIOLIN_GROUPS) - 1:
            ax.axvline(
                base + len(VIOLIN_DOMAINS) + 0.68,
                color="#D0D0D0",
                lw=0.9,
                alpha=0.90,
                zorder=0,
            )

    violin = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.72)
        body.set_linewidth(0.8)

    violin["cmedians"].set_color("#111111")
    violin["cmedians"].set_linewidth(1.5)

    for pos, values in zip(positions, data):
        values = np.array(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        median_value = np.nanmedian(values)

        ax.scatter(
            [pos],
            [median_value],
            s=22,
            color="white",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=5,
        )

    for center, label in group_centers:
        ax.text(
            center,
            1.035,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#222222",
            clip_on=False,
        )

    ax.set_xlim(0.3, max(positions) + 0.8)
    ax.set_ylim(0.0, 1.03)

    ax.set_xticks(domain_tick_positions)
    ax.set_xticklabels(
        domain_tick_labels,
        rotation=90,
        fontsize=12,
        fontweight="bold",
    )

    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=14, length=5)
    ax.tick_params(axis="x", length=4)

    ax.set_ylabel("Variant-specific NSI", fontsize=17, fontweight="bold")
    # ax.set_xlabel("Spike domain grouped by variant", fontsize=17, fontweight="bold", labelpad=16)

    ax.grid(axis="y", color="#E5E5E5", lw=0.8, alpha=0.85)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if own_fig and save:
        plt.tight_layout(rect=[0.035, 0.20, 0.995, 0.90])
        fig.savefig(OUTPUT_VIOLIN_PNG, dpi=900, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    return ax


# =========================================================
# Combined manuscript figure
# =========================================================

def plot_combined_figure(variant_rows, delta_rows):
    fig = plt.figure(figsize=(18.5, 19.5), facecolor="white")

    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.15, 0.92, 1.25],
        hspace=0.56,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    plot_variant_curves(variant_rows, ax=ax1, save=False)
    plot_delta_heatmap(delta_rows, ax=ax2, save=False)
    plot_domainwise_violin(delta_rows, ax=ax3, save=False)

    panel_labels = [
        (ax1, "A"),
        (ax2, "B"),
        (ax3, "C"),
    ]

    for ax, label in panel_labels:
        ax.text(
            -0.055,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=25,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.savefig(
        OUTPUT_COMBINED_PNG,
        dpi=900,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():
    if not ALIGNMENT_FILE.exists():
        raise FileNotFoundError(f"Alignment file was not found:\n{ALIGNMENT_FILE}")

    records = read_fasta(ALIGNMENT_FILE)

    print("\nLoaded aligned FASTA:")
    print(f"  {ALIGNMENT_FILE}")
    print(f"  Records: {len(records)}")

    wuhan_header, wuhan_seq = find_wuhan_reference(records)

    print("\nWuhan reference detected:")
    print(f"  {wuhan_header}")

    grouped, unclassified = group_sequences(records)

    print("\nFocused group summary:")
    for group in GROUP_ORDER:
        print(f"  {DISPLAY_NAMES[group]}: n={len(grouped.get(group, []))}")

    if unclassified:
        print(f"\nRecords ignored because they do not belong to the focused groups: {len(unclassified)}")
        print("First 20 ignored headers:")
        for header in unclassified[:20]:
            print(f"  {header}")

    sequences_by_group = {"Wuhan": [wuhan_seq]}

    for group in GROUP_ORDER:
        if group == "Wuhan":
            continue

        records_for_group = grouped.get(group, [])

        if not records_for_group:
            print(f"\nWarning: {DISPLAY_NAMES[group]} was not found and will be skipped.")
            continue

        sequences_by_group[group] = [seq for _, seq in records_for_group]

    cleaned_sequences = remove_reference_gap_columns(
        reference_seq=sequences_by_group["Wuhan"][0],
        sequences_by_group=sequences_by_group,
    )

    print("\nComputing variant-specific NSI profiles...")
    variant_rows = compute_variant_specific_nsi(cleaned_sequences)

    print("Computing ΔNSI relative to Wuhan...")
    delta_rows = compute_delta_rows(variant_rows)

    print("Computing domain-wise summary...")
    domain_summary_rows = summarize_domainwise(delta_rows)

    print("Building variant-wise amino acid NSI hotspot table...")
    hotspot_rows = build_hotspot_table(delta_rows)

    print("Saving CSV files...")
    save_variant_nsi_csv(variant_rows)
    save_delta_csv(delta_rows)
    save_domain_summary_csv(domain_summary_rows)
    save_hotspot_table_csv(hotspot_rows)

    print("Generating manuscript-quality PNG figures...")
    plot_variant_curves(variant_rows)
    plot_delta_heatmap(delta_rows)
    plot_domainwise_violin(delta_rows)
    plot_combined_figure(variant_rows, delta_rows)

    print("\nSaved PNG outputs:")
    print(f"  {OUTPUT_CURVES_PNG}")
    print(f"  {OUTPUT_HEATMAP_PNG}")
    print(f"  {OUTPUT_VIOLIN_PNG}")
    print(f"  {OUTPUT_COMBINED_PNG}")

    print("\nSaved CSV outputs:")
    print(f"  {OUTPUT_VARIANT_NSI_CSV}")
    print(f"  {OUTPUT_DELTA_CSV}")
    print(f"  {OUTPUT_DOMAIN_SUMMARY_CSV}")
    print(f"  {OUTPUT_HOTSPOT_TABLE_CSV}")

    print("\nTop hotspot amino acid positions by variant:")
    for group in GROUP_ORDER:
        rows = [row for row in hotspot_rows if row["Group"] == group][:10]

        print(f"\n{DISPLAY_NAMES[group]}:")

        for row in rows:
            print(
                f"  Rank {row['Rank_within_variant']:02d}: "
                f"{row['Residue_change']} | "
                f"{row['Domain_display']} | "
                f"NSI={row['Smoothed_NSI']:.3f} | "
                f"ΔNSI={row['Delta_NSI_vs_Wuhan']:.3f} | "
                f"ACE2={row['ACE2_relation']}"
            )

    print("\nTop domain-level increases in mean ΔNSI relative to Wuhan:")
    non_wuhan_summary = [
        row for row in domain_summary_rows
        if row["Group"] != "Wuhan"
    ]

    top_rows = sorted(
        non_wuhan_summary,
        key=lambda r: r["Mean_delta_NSI_vs_Wuhan"],
        reverse=True,
    )[:15]

    for row in top_rows:
        print(
            f"  {row['Variant']} | {row['Domain_display']} | "
            f"mean ΔNSI={row['Mean_delta_NSI_vs_Wuhan']:.4f} | "
            f"mean NSI={row['Mean_NSI']:.4f} | "
            f"n={row['N_positions']}"
        )


if __name__ == "__main__":
    main()