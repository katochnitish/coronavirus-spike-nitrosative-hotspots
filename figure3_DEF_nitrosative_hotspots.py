# ==============================================================================
# Script Name: figure3_DEF_nitrosative_hotspots.py
# Author: Code made by Nitish Katoch
#
# Description:
#   Fuses site-specific evolutionary dynamics (entropy and frequency) with rule-based 
#   biochemical indicators to localize vulnerable molecular targets on the 
#   SARS-CoV-2 spike protein.
#
# Plots Made:
#   - Panel D: Position-wise line chart tracking Nitrosative Susceptibility Index (NSI).
#   - Panel E: Three-dimensional overlap bubble plot matching mutation frequency, 
#              Shannon entropy, and susceptibility across structural groups.
#   - Panel F: Descriptive topological schematic mapping hot interface locations.
#
# Inputs:
#   - 02_CuratedRawData/combined_curated_aligned.fasta
#
# Calculations:
#   - Compiles positional amino acid substitution rates and Shannon entropy values.
#   - Evaluates weighted multi-metric cross-overlap scoring structures.
#   - Identifies high-priority hotspots and automatically builds automated scripts 
#     to project structural mappings onto 3D coordinate spaces (PDB: 6M0J).
#
# Outputs:
#   - 03_Results/Figure3D_nitrosative_susceptibility_index.{png, pdf}
#   - 03_Results/Figure3E_mutation_entropy_nitrosative_overlap_bubbleplot.{png, pdf}
#   - 03_Results/Figure3F_RBD_ACE2_hotspot_schematic.{png, pdf}
#   - 03_Results/Figure3D_E_positionwise_metrics.csv
#   - 03_Results/Figure3F_top_structural_hotspots.csv
#   - 03_Results/Figure3F_pymol_hotspot_mapping_6M0J.pml
# ==============================================================================
from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
ALIGNMENT_FILE = BASE_DIR / "02_CuratedRawData" / "combined_curated_aligned.fasta"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PANEL_D_PNG = OUTPUT_DIR / "Figure3D_nitrosative_susceptibility_index.png"
OUTPUT_PANEL_D_PDF = OUTPUT_DIR / "Figure3D_nitrosative_susceptibility_index.pdf"

OUTPUT_PANEL_E_PNG = OUTPUT_DIR / "Figure3E_mutation_entropy_nitrosative_overlap_bubbleplot.png"
OUTPUT_PANEL_E_PDF = OUTPUT_DIR / "Figure3E_mutation_entropy_nitrosative_overlap_bubbleplot.pdf"

OUTPUT_PANEL_F_PNG = OUTPUT_DIR / "Figure3F_RBD_ACE2_hotspot_schematic.png"
OUTPUT_PANEL_F_PDF = OUTPUT_DIR / "Figure3F_RBD_ACE2_hotspot_schematic.pdf"

OUTPUT_METRICS_CSV = OUTPUT_DIR / "Figure3D_E_positionwise_metrics.csv"
OUTPUT_STRUCTURAL_CSV = OUTPUT_DIR / "Figure3F_top_structural_hotspots.csv"
OUTPUT_PYMOL_SCRIPT = OUTPUT_DIR / "Figure3F_pymol_hotspot_mapping_6M0J.pml"

GROUP_ORDER = [
    "Wuhan",
    "Alpha",
    "Beta",
    "Delta",
    "Omicron_BA1",
    "Omicron_BA2",
    "SARS_CoV_2",
    "SARS_CoV",
    "MERS_CoV",
    "HCoV_HKU1",
    "HCoV_OC43",
    "HCoV_229E",
    "HCoV_NL63",
    "Bat_CoV",
]

DISPLAY_NAMES = {
    "Wuhan": "Wuhan",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Delta": "Delta",
    "Omicron_BA1": "Omicron BA.1",
    "Omicron_BA2": "Omicron BA.2",
    "SARS_CoV_2": "SARS-CoV-2",
    "SARS_CoV": "SARS-CoV",
    "MERS_CoV": "MERS-CoV",
    "HCoV_HKU1": "HCoV-HKU1",
    "HCoV_OC43": "HCoV-OC43",
    "HCoV_229E": "HCoV-229E",
    "HCoV_NL63": "HCoV-NL63",
    "Bat_CoV": "Bat-CoV",
}

SPIKE_LENGTH = 1273
SMOOTHING_WINDOW = 5
INVALID_AA = {"-", "?", ".", "X", "B", "Z", "J", "U", "O", "*"}

DOMAINS = [
    ("SP", 1, 13, "#BDBDBD"),
    ("NTD", 14, 305, "#F2D27A"),
    ("RBD/RBM", 319, 541, "#9CC586"),
    ("S1/S2", 675, 692, "#4F4F4F"),
    ("FP", 816, 833, "#B8D78A"),
    ("HR1", 912, 984, "#F0A33B"),
    ("HR2", 1163, 1213, "#F4C15D"),
    ("TM/CT", 1214, 1273, "#D9D9D9"),
]

DOMAIN_ORDER_FOR_PANEL_E = [
    "NTD",
    "RBD/RBM",
    "S1/S2",
    "FP",
    "HR1",
    "HR2",
    "TM/CT",
    "S1_other",
    "S2_other",
]
DOMAIN_Y = {name: i for i, name in enumerate(DOMAIN_ORDER_FOR_PANEL_E)}

FUNCTIONAL_REGIONS = [
    ("NTD antigenic\nregion", 14, 305, "#8A6A00"),
    ("RBD/RBM", 437, 505, "#1B7837"),
    ("S1/S2\ncleavage", 675, 692, "#222222"),
    ("Fusion\npeptide", 816, 833, "#558B2F"),
    ("HR1/HR2", 912, 1213, "#B36B00"),
    ("TM/CT", 1214, 1273, "#444444"),
]

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
    "RBD/RBM": 1.00,
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
    "RBD/RBM": 0.90,
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
    491, 492, 494, 495, 496, 497, 499, 500, 502, 503, 504, 506
}

KEY_NITROSATIVE_RESIDUES = {
    449: "Y449",
    453: "Y453",
    489: "Y489",
    505: "Y505",
    681: "P681",
    816: "FP",
    912: "HR1",
    1163: "HR2",
    1236: "TM/CT",
}

W_REACTIVITY = 0.35
W_EXPOSURE = 0.25
W_DOMAIN = 0.15
W_ENTROPY = 0.10
W_ACE2 = 0.15


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
        "omicron_ba1" in text
        or "omicron_ba_1" in text
        or "ba1" in text
        or "ba_1" in text
        or "b_1_1_529_ba_1" in text
    ):
        return "Omicron_BA1"

    if (
        "omicron_ba2" in text
        or "omicron_ba_2" in text
        or "ba2" in text
        or "ba_2" in text
        or "b_1_1_529_ba_2" in text
    ):
        return "Omicron_BA2"

    if (
        "sars_cov_2" in text
        or "sars_cov2" in text
        or "severe_acute_respiratory_syndrome_coronavirus_2" in text
    ) and "wuhan" not in text:
        return "SARS_CoV_2"

    if (
        "sars_cov" in text
        or "sars_cov_1" in text
        or "sarscoronavirus" in text
        or "sars_coronavirus" in text
    ) and "sars_cov_2" not in text and "sars_cov2" not in text:
        return "SARS_CoV"

    if "mers" in text or "middle_east_respiratory" in text:
        return "MERS_CoV"

    if "hku1" in text:
        return "HCoV_HKU1"

    if "oc43" in text:
        return "HCoV_OC43"

    if "229e" in text:
        return "HCoV_229E"

    if "nl63" in text:
        return "HCoV_NL63"

    if "bat" in text or "bat_cov" in text or "bat_coronavirus" in text:
        return "Bat_CoV"

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
            continue
        grouped[group].append((header, seq))
    return grouped, unclassified


def remove_reference_gap_columns(reference_seq, sequences_by_group):
    keep_indices = [i for i, aa in enumerate(reference_seq) if aa != "-"]
    cleaned = {}
    for group, seqs in sequences_by_group.items():
        cleaned[group] = []
        for seq in seqs:
            cleaned_seq = "".join(seq[i] if i < len(seq) else "-" for i in keep_indices)
            cleaned[group].append(cleaned_seq)
    return cleaned


def assign_domain(position):
    for name, start, end, _ in DOMAINS:
        if start <= position <= end:
            return name
    if 1 <= position <= 685:
        return "S1_other"
    if 686 <= position <= SPIKE_LENGTH:
        return "S2_other"
    return "Outside"


def shannon_entropy(residues):
    residues = [aa for aa in residues if aa not in INVALID_AA]
    if not residues:
        return np.nan
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


def calculate_entropy_and_mutation(cleaned_sequences):
    reference = cleaned_sequences["Wuhan"][0]
    comparison_groups = [
        group for group in GROUP_ORDER
        if group != "Wuhan" and group in cleaned_sequences and cleaned_sequences[group]
    ]

    raw_entropy = []
    mutation_frequency = []
    dominant_variant_aa = []

    for idx in range(min(len(reference), SPIKE_LENGTH)):
        ref_aa = reference[idx]
        residues = []
        mutated = 0
        total = 0

        for group in comparison_groups:
            for seq in cleaned_sequences[group]:
                if idx >= len(seq):
                    continue
                aa = seq[idx]
                if aa in INVALID_AA:
                    continue
                residues.append(aa)
                total += 1
                if ref_aa not in INVALID_AA and aa != ref_aa:
                    mutated += 1

        raw_entropy.append(shannon_entropy(residues))
        mutation_frequency.append(mutated / total if total > 0 else 0.0)
        counts = Counter([aa for aa in residues if aa not in INVALID_AA and aa != ref_aa])
        dominant_variant_aa.append(counts.most_common(1)[0][0] if counts else "")

    raw_entropy = np.array(raw_entropy, dtype=float)
    raw_entropy_clean = np.nan_to_num(raw_entropy, nan=0.0)
    max_entropy = float(np.max(raw_entropy_clean)) if len(raw_entropy_clean) else 0.0
    if max_entropy == 0.0:
        entropy_norm = np.zeros_like(raw_entropy_clean)
    else:
        entropy_norm = raw_entropy_clean / max_entropy
    entropy_smooth = rolling_mean(entropy_norm, SMOOTHING_WINDOW)
    mutation_frequency = np.array(mutation_frequency, dtype=float)
    return raw_entropy, entropy_norm, entropy_smooth, mutation_frequency, dominant_variant_aa


def calculate_metrics(cleaned_sequences):
    reference = cleaned_sequences["Wuhan"][0]
    raw_entropy, entropy_norm, entropy_smooth, mutation_frequency, dominant_variant_aa = calculate_entropy_and_mutation(cleaned_sequences)
    rows = []

    for idx in range(min(len(reference), SPIKE_LENGTH)):
        position = idx + 1
        aa = reference[idx]
        if aa in INVALID_AA:
            continue

        domain = assign_domain(position)
        reactivity = REACTIVITY_SCORE.get(aa, 0.00)
        exposure = approximate_exposure_score(position, aa, domain)
        domain_score = DOMAIN_WEIGHT.get(domain, 0.00)
        ace2_score = ace2_interface_score(position)

        raw_nsi = (
            W_REACTIVITY * reactivity
            + W_EXPOSURE * exposure
            + W_DOMAIN * domain_score
            + W_ENTROPY * entropy_smooth[idx]
            + W_ACE2 * ace2_score
        )

        rows.append(
            {
                "Position": position,
                "Reference_AA": aa,
                "Dominant_variant_AA": dominant_variant_aa[idx],
                "Domain": domain,
                "Reactive_residue": "Yes" if aa in {"C", "Y", "M", "W", "H"} else "No",
                "ACE2_relation": (
                    "Direct contact"
                    if position in ACE2_DIRECT_CONTACT_POSITIONS
                    else "Near interface"
                    if position in ACE2_NEAR_INTERFACE_POSITIONS
                    else "No"
                ),
                "Residue_reactivity_score": reactivity,
                "Approximate_exposure_score": exposure,
                "Domain_weight_score": domain_score,
                "Raw_entropy": raw_entropy[idx],
                "Normalized_entropy": entropy_norm[idx],
                "Smoothed_normalized_entropy": entropy_smooth[idx],
                "Mutation_frequency": mutation_frequency[idx],
                "ACE2_interface_score": ace2_score,
                "Raw_NSI": raw_nsi,
            }
        )

    raw_scores = np.array([row["Raw_NSI"] for row in rows], dtype=float)
    min_score = np.nanmin(raw_scores)
    max_score = np.nanmax(raw_scores)
    if max_score == min_score:
        nsi = np.zeros_like(raw_scores)
    else:
        nsi = (raw_scores - min_score) / (max_score - min_score)

    nsi_smooth = rolling_mean(nsi, SMOOTHING_WINDOW)

    for row, score, smooth_score in zip(rows, nsi, nsi_smooth):
        row["Nitrosative_susceptibility_index"] = score
        row["Smoothed_NSI"] = smooth_score
        row["Overlap_score"] = (
            0.40 * row["Smoothed_NSI"]
            + 0.35 * row["Mutation_frequency"]
            + 0.25 * row["Smoothed_normalized_entropy"]
        )

    return rows


def save_metrics_table(rows):
    fieldnames = [
        "Position",
        "Reference_AA",
        "Dominant_variant_AA",
        "Domain",
        "Reactive_residue",
        "ACE2_relation",
        "Residue_reactivity_score",
        "Approximate_exposure_score",
        "Domain_weight_score",
        "Raw_entropy",
        "Normalized_entropy",
        "Smoothed_normalized_entropy",
        "Mutation_frequency",
        "ACE2_interface_score",
        "Raw_NSI",
        "Nitrosative_susceptibility_index",
        "Smoothed_NSI",
        "Overlap_score",
    ]

    with open(OUTPUT_METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key in fieldnames:
                if isinstance(out[key], float):
                    out[key] = "NA" if np.isnan(out[key]) else f"{out[key]:.6f}"
            writer.writerow(out)


def add_domain_backgrounds(ax):
    for name, start, end, color in DOMAINS:
        alpha = 0.08
        if name in {"RBD/RBM", "S1/S2", "FP"}:
            alpha = 0.12
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)


def draw_domain_track(ax):
    y = -0.105
    height = 0.035

    ax.plot(
        [1, SPIKE_LENGTH],
        [y, y],
        transform=ax.get_xaxis_transform(),
        color="#666666",
        lw=0.8,
        alpha=0.45,
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
                alpha=0.90,
                clip_on=False,
                zorder=3,
            )
        )
        if end - start > 35:
            ax.text(
                (start + end) / 2,
                y - 0.035,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.0,
                fontweight="bold",
                color="#222222",
                clip_on=False,
            )


def add_functional_guides(ax):
    for label, start, end, color in FUNCTIONAL_REGIONS:
        mid = (start + end) / 2
        ax.plot(
            [start, end],
            [1.02, 1.02],
            transform=ax.get_xaxis_transform(),
            color=color,
            lw=2.0,
            solid_capstyle="butt",
            clip_on=False,
        )
        ax.text(
            mid,
            1.045,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
            color=color,
            linespacing=1.0,
            clip_on=False,
        )


def plot_panel_d(rows):
    positions = np.array([row["Position"] for row in rows], dtype=int)
    nsi = np.array([row["Nitrosative_susceptibility_index"] for row in rows], dtype=float)
    nsi_smooth = np.array([row["Smoothed_NSI"] for row in rows], dtype=float)
    entropy = np.array([row["Smoothed_normalized_entropy"] for row in rows], dtype=float)
    aa_list = [row["Reference_AA"] for row in rows]

    fig, ax = plt.subplots(figsize=(15.8, 5.8), facecolor="white")
    ax.set_facecolor("white")
    add_domain_backgrounds(ax)

    heat_y_min = -0.075
    heat_y_max = -0.025
    for pos, score in zip(positions, nsi):
        ax.add_patch(
            Rectangle(
                (pos - 0.5, heat_y_min),
                1.0,
                heat_y_max - heat_y_min,
                transform=ax.get_xaxis_transform(),
                facecolor=plt.cm.magma(score),
                edgecolor="none",
                clip_on=False,
                zorder=2,
            )
        )

    ax.plot(positions, nsi, color="#C51B7D", lw=0.95, alpha=0.45, label="Raw NSI", zorder=3)
    ax.plot(positions, nsi_smooth, color="#7A0177", lw=1.85, label=f"Smoothed NSI, window={SMOOTHING_WINDOW}", zorder=4)
    ax.fill_between(positions, nsi_smooth, 0, color="#C51B7D", alpha=0.15, zorder=2)
    ax.plot(positions, entropy, color="#2C5AA0", lw=1.0, alpha=0.40, label="Normalized entropy", zorder=3)

    reactive_positions = []
    reactive_scores = []
    for pos, aa, score in zip(positions, aa_list, nsi_smooth):
        if aa in {"C", "Y", "M", "W", "H"} and score >= 0.55:
            reactive_positions.append(pos)
            reactive_scores.append(score)

    ax.scatter(
        reactive_positions,
        reactive_scores,
        s=16,
        color="#D73027",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.95,
        zorder=5,
        label="High-score reactive residues",
    )

    for pos, label in KEY_NITROSATIVE_RESIDUES.items():
        idx = np.where(positions == pos)[0]
        if len(idx) == 0:
            continue
        y = nsi_smooth[idx[0]]
        ax.axvline(pos, color="#808080", linestyle=":", lw=0.65, alpha=0.65, zorder=1)
        ax.text(pos, min(1.02, y + 0.10), label, ha="center", va="bottom", fontsize=8.0, rotation=90, fontweight="bold", color="#444444", zorder=6)

    for pos in [449, 453, 489, 505]:
        idx = np.where(positions == pos)[0]
        if len(idx) == 0:
            continue
        ax.scatter([pos], [nsi_smooth[idx[0]]], s=42, color="#1B7837", edgecolor="white", linewidth=0.7, zorder=7)

    add_functional_guides(ax)
    draw_domain_track(ax)

    ax.set_xlim(1, SPIKE_LENGTH)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([1, 100, 200, 300, 400, 500, 600, 681, 800, 900, 1000, 1100, 1200, 1273])
    ax.tick_params(axis="x", labelsize=9.0)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.set_xlabel("Wuhan-Hu-1 spike residue position", fontsize=12, fontweight="bold", labelpad=18)
    ax.set_ylabel("Nitrosative\nsusceptibility index", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", color="#E0E0E0", lw=0.7, alpha=0.8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.legend(loc="upper right", frameon=True, fontsize=8.7, handlelength=2.0)

    plt.tight_layout(rect=[0.035, 0.08, 0.995, 0.98])
    fig.savefig(OUTPUT_PANEL_D_PNG, dpi=900, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_PANEL_D_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def add_vertical_jitter(y_values, positions, jitter_strength=0.16):
    y_jittered = []
    for y, x in zip(y_values, positions):
        offset = ((x % 7) - 3) / 3.0
        y_jittered.append(y + jitter_strength * offset)
    return np.array(y_jittered, dtype=float)


def get_label_y_below(point_y, position, base_offset=0.32):
    """
    Places residue labels below bubbles.
    A small deterministic offset prevents neighboring labels from overlapping.
    """
    extra_offset = 0.08 * (position % 3)
    return point_y + base_offset + extra_offset


def plot_panel_e(rows):
    plot_rows = [
        row for row in rows
        if row["Domain"] in DOMAIN_Y and (
            row["Mutation_frequency"] >= 0.04
            or row["Smoothed_normalized_entropy"] >= 0.25
            or row["Smoothed_NSI"] >= 0.58
            or row["Position"] in [449, 453, 489, 505, 681]
        )
    ]

    plot_rows = sorted(plot_rows, key=lambda r: r["Overlap_score"])

    positions = np.array([row["Position"] for row in plot_rows], dtype=int)
    y_base = np.array([DOMAIN_Y[row["Domain"]] for row in plot_rows], dtype=float)
    y_values = add_vertical_jitter(y_base, positions, jitter_strength=0.16)

    nsi = np.array([row["Smoothed_NSI"] for row in plot_rows], dtype=float)
    mutation_frequency = np.array([row["Mutation_frequency"] for row in plot_rows], dtype=float)
    entropy = np.array([row["Smoothed_normalized_entropy"] for row in plot_rows], dtype=float)

    sizes = 20 + 280 * mutation_frequency
    sizes = np.clip(sizes, 18, 220)

    linewidths = 0.5 + 1.6 * entropy
    linewidths = np.clip(linewidths, 0.5, 2.0)

    fig, ax = plt.subplots(figsize=(15.8, 5.8), facecolor="white")
    ax.set_facecolor("white")

    for name, start, end, color in DOMAINS:
        ax.axvspan(start, end, color=color, alpha=0.045, lw=0)

    for y in range(len(DOMAIN_ORDER_FOR_PANEL_E)):
        ax.axhline(y, color="#E8E8E8", lw=0.7, zorder=0)

    sc = ax.scatter(
        positions,
        y_values,
        s=sizes,
        c=nsi,
        cmap="magma",
        vmin=0,
        vmax=1,
        edgecolor="#1A1A1A",
        linewidth=linewidths,
        alpha=0.78,
        zorder=4,
    )

    top_label_rows = sorted(
        plot_rows,
        key=lambda r: (r["Overlap_score"], r["Smoothed_NSI"], r["Mutation_frequency"]),
        reverse=True,
    )[:14]

    labeled_positions = set()
    for row in top_label_rows:
        pos = row["Position"]
        if pos in labeled_positions:
            continue
        idx = np.where(positions == pos)[0]
        if len(idx) == 0:
            continue
        point_y = y_values[idx[0]]
        label_y = get_label_y_below(point_y, pos, base_offset=0.34)
        label = f"{row['Reference_AA']}{row['Position']}"
        ax.text(
            pos,
            label_y,
            label,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color="#222222",
            zorder=7,
            clip_on=False,
        )
        labeled_positions.add(pos)

    for pos in [449, 453, 489, 505, 681]:
        ax.axvline(pos, color="#777777", linestyle=":", lw=0.7, alpha=0.5, zorder=1)

    ax.set_xlim(1, SPIKE_LENGTH)
    ax.set_ylim(-0.4, len(DOMAIN_ORDER_FOR_PANEL_E) - 0.25)
    ax.set_yticks(range(len(DOMAIN_ORDER_FOR_PANEL_E)))
    ax.set_yticklabels(DOMAIN_ORDER_FOR_PANEL_E, fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xticks([1, 100, 200, 300, 400, 500, 600, 681, 800, 900, 1000, 1100, 1200, 1273])
    ax.tick_params(axis="x", labelsize=9.0)
    ax.set_xlabel("Wuhan-Hu-1 spike residue position", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spike domain", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", lw=0.7, alpha=0.7)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    colorbar = fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.026)
    colorbar.set_label("Smoothed NSI", fontsize=10.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=8.5)

    size_handles = [
        plt.scatter([], [], s=s, facecolor="#BDBDBD", edgecolor="#1A1A1A", linewidth=0.7)
        for s in [35, 90, 160]
    ]
    entropy_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#BDBDBD", markeredgecolor="#1A1A1A", markeredgewidth=lw, markersize=7.0)
        for lw in [0.6, 1.2, 1.8]
    ]

    legend1 = ax.legend(
        size_handles,
        ["Low", "Medium", "High"],
        title="Mutation\nfrequency",
        loc="upper left",
        bbox_to_anchor=(1.055, 0.78),
        frameon=True,
        fontsize=8.5,
        title_fontsize=9,
        borderpad=0.8,
    )
    ax.add_artist(legend1)

    ax.legend(
        entropy_handles,
        ["Low", "Medium", "High"],
        title="Entropy\nborder",
        loc="upper left",
        bbox_to_anchor=(1.055, 0.42),
        frameon=True,
        fontsize=8.5,
        title_fontsize=9,
        borderpad=0.8,
    )


    plt.tight_layout(rect=[0.035, 0.04, 0.88, 0.98])
    fig.savefig(OUTPUT_PANEL_E_PNG, dpi=900, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_PANEL_E_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_panel_f_schematic(rows):
    rbd_rows = [
        row for row in rows
        if 319 <= row["Position"] <= 541 and (
            row["Smoothed_NSI"] >= 0.50
            or row["Mutation_frequency"] >= 0.02
            or row["ACE2_relation"] != "No"
            or row["Position"] in [449, 453, 489, 505]
        )
    ]

    rbd_rows = sorted(rbd_rows, key=lambda r: r["Overlap_score"])

    positions = np.array([row["Position"] for row in rbd_rows], dtype=int)
    nsi = np.array([row["Smoothed_NSI"] for row in rbd_rows], dtype=float)
    mutation_frequency = np.array([row["Mutation_frequency"] for row in rbd_rows], dtype=float)
    entropy = np.array([row["Smoothed_normalized_entropy"] for row in rbd_rows], dtype=float)

    y_values = []
    for row in rbd_rows:
        if row["ACE2_relation"] == "Direct contact":
            y_values.append(0.0)
        elif row["ACE2_relation"] == "Near interface":
            y_values.append(1.0)
        else:
            y_values.append(2.0)
    y_values = np.array(y_values, dtype=float)
    y_values = add_vertical_jitter(y_values, positions, jitter_strength=0.11)

    sizes = 35 + 360 * mutation_frequency
    sizes = np.clip(sizes, 28, 260)

    linewidths = 0.6 + 1.8 * entropy
    linewidths = np.clip(linewidths, 0.6, 2.2)

    fig, ax = plt.subplots(figsize=(11.5, 4.8), facecolor="white")
    ax.set_facecolor("white")

    ax.axvspan(319, 437, color="#9CC586", alpha=0.10, lw=0)
    ax.axvspan(438, 506, color="#1B7837", alpha=0.12, lw=0)
    ax.axvspan(507, 541, color="#9CC586", alpha=0.10, lw=0)

    ax.text(378, -0.55, "RBD core", ha="center", va="center", fontsize=9, fontweight="bold", color="#3B6B35")
    ax.text(472, -0.55, "RBM / ACE2-contact surface", ha="center", va="center", fontsize=9, fontweight="bold", color="#1B7837")
    ax.text(524, -0.55, "RBD core", ha="center", va="center", fontsize=9, fontweight="bold", color="#3B6B35")

    sc = ax.scatter(
        positions,
        y_values,
        s=sizes,
        c=nsi,
        cmap="magma",
        vmin=0,
        vmax=1,
        edgecolor="#1A1A1A",
        linewidth=linewidths,
        alpha=0.86,
        zorder=4,
    )

    key_positions = [417, 449, 453, 486, 489, 493, 498, 501, 505]
    for pos in key_positions:
        ax.axvline(pos, color="#777777", linestyle=":", lw=0.7, alpha=0.45, zorder=1)

    label_rows = [row for row in rbd_rows if row["Position"] in [449, 453, 489, 505, 417, 486, 493, 498, 501]]
    extra_top_rows = sorted(rbd_rows, key=lambda r: (r["Overlap_score"], r["Smoothed_NSI"]), reverse=True)[:10]
    label_rows = label_rows + extra_top_rows

    labeled = set()
    for row in label_rows:
        pos = row["Position"]
        if pos in labeled:
            continue
        idx = np.where(positions == pos)[0]
        if len(idx) == 0:
            continue
        point_y = y_values[idx[0]]
        label_y = get_label_y_below(point_y, pos, base_offset=0.30)
        label = f"{row['Reference_AA']}{row['Position']}"
        ax.text(
            pos,
            label_y,
            label,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color="#222222",
            zorder=7,
            clip_on=False,
        )
        labeled.add(pos)

    ax.set_xlim(319, 541)
    ax.set_ylim(2.90, -0.85)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Direct ACE2 contact", "Near ACE2 interface", "Other RBD/RBM sites"], fontsize=9.5, fontweight="bold")
    ax.set_xticks([319, 350, 400, 437, 449, 453, 489, 505, 541])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Wuhan-Hu-1 spike residue position within RBD", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Structural relation to ACE2", fontsize=11.5, fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", lw=0.7, alpha=0.8)
    ax.grid(axis="y", color="#EAEAEA", lw=0.7, alpha=0.8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    colorbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.035)
    colorbar.set_label("Smoothed NSI", fontsize=10.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=8.5)


    plt.tight_layout(rect=[0.04, 0.06, 0.94, 0.98])
    fig.savefig(OUTPUT_PANEL_F_PNG, dpi=900, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_PANEL_F_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_structural_hotspots(rows, top_n=30):
    rbd_rows = [
        row for row in rows
        if 319 <= row["Position"] <= 541 and (
            row["Smoothed_NSI"] >= 0.55
            or row["Mutation_frequency"] >= 0.02
            or row["ACE2_relation"] != "No"
        )
    ]
    rbd_rows = sorted(
        rbd_rows,
        key=lambda row: (
            row["Overlap_score"],
            row["Smoothed_NSI"],
            row["Mutation_frequency"],
            row["Smoothed_normalized_entropy"],
        ),
        reverse=True,
    )[:top_n]

    fieldnames = [
        "Position",
        "Reference_AA",
        "Dominant_variant_AA",
        "Domain",
        "ACE2_relation",
        "Mutation_frequency",
        "Smoothed_normalized_entropy",
        "Smoothed_NSI",
        "Overlap_score",
    ]

    with open(OUTPUT_STRUCTURAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rbd_rows:
            writer.writerow(
                {
                    "Position": row["Position"],
                    "Reference_AA": row["Reference_AA"],
                    "Dominant_variant_AA": row["Dominant_variant_AA"],
                    "Domain": row["Domain"],
                    "ACE2_relation": row["ACE2_relation"],
                    "Mutation_frequency": f"{row['Mutation_frequency']:.6f}",
                    "Smoothed_normalized_entropy": f"{row['Smoothed_normalized_entropy']:.6f}",
                    "Smoothed_NSI": f"{row['Smoothed_NSI']:.6f}",
                    "Overlap_score": f"{row['Overlap_score']:.6f}",
                }
            )

    hotspot_positions = [row["Position"] for row in rbd_rows]
    ace2_positions = sorted(list(ACE2_DIRECT_CONTACT_POSITIONS))
    selected_overlap_positions = [
        row["Position"] for row in rbd_rows
        if row["ACE2_relation"] != "No" and row["Smoothed_NSI"] >= 0.55
    ]

    hotspot_selection = "+".join(str(pos) for pos in hotspot_positions) if hotspot_positions else "449+453+489+505"
    ace2_selection = "+".join(str(pos) for pos in ace2_positions)
    overlap_selection = "+".join(str(pos) for pos in selected_overlap_positions) if selected_overlap_positions else "449+453+489+505"

    pml = f"""
fetch 6M0J, async=0
remove solvent
hide everything
show cartoon, all

color gray85, all
color lightblue, chain A
color palegreen, chain E

select rbd, chain E
select ace2, chain A

select ace2_contact_rbd, chain E and resi {ace2_selection}
show sticks, ace2_contact_rbd
color marine, ace2_contact_rbd

select nsi_hotspots, chain E and resi {hotspot_selection}
show spheres, nsi_hotspots
set sphere_scale, 0.42, nsi_hotspots
color magenta, nsi_hotspots

select overlap_hotspots, chain E and resi {overlap_selection}
show spheres, overlap_hotspots
set sphere_scale, 0.62, overlap_hotspots
color red, overlap_hotspots

select rbm_key_sites, chain E and resi 449+453+489+505
show sticks, rbm_key_sites
color yelloworange, rbm_key_sites

label rbm_key_sites and name CA, resn + resi
set label_size, 18
set label_color, black
set ray_opaque_background, off
set cartoon_transparency, 0.10
bg_color white
orient rbd

png Figure3F_RBD_ACE2_nitrosative_hotspots_6M0J.png, dpi=600, ray=1
"""

    with open(OUTPUT_PYMOL_SCRIPT, "w", encoding="utf-8") as f:
        f.write(pml.strip() + "\n")


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
    print("\nGroup summary:")
    for group in GROUP_ORDER:
        print(f"  {DISPLAY_NAMES[group]}: n={len(grouped.get(group, []))}")

    if unclassified:
        print(f"\nUnclassified records: {len(unclassified)}")
        print("First 25 unclassified headers:")
        for header in unclassified[:25]:
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

    rows = calculate_metrics(cleaned_sequences)
    save_metrics_table(rows)
    plot_panel_d(rows)
    plot_panel_e(rows)
    plot_panel_f_schematic(rows)
    write_structural_hotspots(rows, top_n=30)

    print("\nSaved outputs:")
    print(f"  {OUTPUT_PANEL_D_PNG}")
    print(f"  {OUTPUT_PANEL_D_PDF}")
    print(f"  {OUTPUT_PANEL_E_PNG}")
    print(f"  {OUTPUT_PANEL_E_PDF}")
    print(f"  {OUTPUT_PANEL_F_PNG}")
    print(f"  {OUTPUT_PANEL_F_PDF}")
    print(f"  {OUTPUT_METRICS_CSV}")
    print(f"  {OUTPUT_STRUCTURAL_CSV}")
    print(f"  {OUTPUT_PYMOL_SCRIPT}")

    top_rows = sorted(rows, key=lambda r: r["Overlap_score"], reverse=True)[:30]
    print("\nTop 30 positions by overlap score:")
    for row in top_rows:
        print(
            f"  {row['Reference_AA']}{row['Position']} | {row['Domain']} | "
            f"overlap={row['Overlap_score']:.3f} | "
            f"NSI={row['Smoothed_NSI']:.3f} | "
            f"mutation={row['Mutation_frequency']:.3f} | "
            f"entropy={row['Smoothed_normalized_entropy']:.3f} | "
            f"ACE2={row['ACE2_relation']}"
        )


if __name__ == "__main__":
    main()
