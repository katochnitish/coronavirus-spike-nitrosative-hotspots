# ==============================================================================
# Script Name: figure3B_mutation_frequency_entropy_across_spike.py
# Author: Code made by Nitish Katoch
#
# Description:
#   Performs a position-wise evolutionary diversity and variability screening 
#   across the full linear length of the SARS-CoV-2 Spike glycoprotein.
#
# Plots Made:
#   - Single-panel publication layout displaying localized site-by-site 
#     mutation frequency percentages and Shannon entropy scores mapped directly 
#     underneath structural domain boundaries.
#
# Inputs:
#   - 02_CuratedRawData/combined_curated_aligned.fasta
#
# Calculations:
#   - Extracts amino acid frequency matrices per individual alignment column.
#   - Computes Shannon evolutionary entropy: H = -Σ (p_i * log2(p_i)) at each position.
#   - Tracks percentage-wise deviations from ancestral sequences across individual sites.
#
# Outputs:
#   - 03_Results/Figure3B_single_panel_mutation_frequency_entropy_across_spike.png
#   - 03_Results/Figure3B_positionwise_mutation_frequency_entropy.csv
# ==============================================================================
from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# =========================================================
# Figure 3B
# Single-panel plot:
# Mutation frequency and Shannon entropy across SARS-CoV-2 spike
# =========================================================


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

ALIGNMENT_FILE = BASE_DIR / "02_CuratedRawData" / "combined_curated_aligned.fasta"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "Figure3B_single_panel_mutation_frequency_entropy_across_spike.png"
OUTPUT_CSV = OUTPUT_DIR / "Figure3B_positionwise_mutation_frequency_entropy.csv"


# =========================================================
# Selected SARS-CoV-2 variant groups
# =========================================================

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


# =========================================================
# Wuhan-Hu-1 spike domains
# =========================================================

SPIKE_LENGTH = 1273

DOMAINS = [
    ("SP", 1, 13, "#BDBDBD"),
    ("NTD", 14, 305, "#D6A84F"),
    ("RBD/RBM", 319, 541, "#5F8D3A"),
    ("S1/S2", 681, 686, "#303030"),
    ("FP", 816, 833, "#7F2F12"),
    ("HR1", 912, 984, "#B36B00"),
    ("HR2", 1163, 1213, "#7B4F00"),
    ("TM", 1214, 1237, "#A14A36"),
    ("CT", 1238, 1273, "#7A7A7A"),
]

FUNCTIONAL_REGIONS = [
    ("NTD antigenic supersite", 14, 305, "#8A6A00"),
    ("RBD/RBM", 437, 505, "#1B7837"),
    ("S1/S2", 681, 686, "#222222"),
    ("FP", 816, 833, "#7F2F12"),
    ("HR1/HR2", 912, 1213, "#B36B00"),
]


# =========================================================
# FASTA reader
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
                seq_lines.append(line.strip())

    if header is not None:
        records.append((header, "".join(seq_lines).upper()))

    return records


# =========================================================
# Classification
# =========================================================

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

    if (
        "alpha" in text
        or "b_1_1_7" in text
        or "b117" in text
        or "20i" in text
        or "501y_v1" in text
    ):
        return "Alpha"

    if (
        "beta" in text
        or "b_1_351" in text
        or "b1351" in text
        or "20h" in text
        or "501y_v2" in text
    ):
        return "Beta"

    if (
        "delta" in text
        or "b_1_617_2" in text
        or "b16172" in text
        or "21a" in text
        or "21i" in text
        or "21j" in text
    ):
        return "Delta"

    if (
        "omicron_ba2" in text
        or "omicron_ba_2" in text
        or "ba2" in text
        or "ba_2" in text
        or "b_1_1_529_ba_2" in text
        or "21l" in text
    ):
        return "Omicron_BA2"

    return None


def find_wuhan_reference(records):
    for header, seq in records:
        if classify_record(header) == "Wuhan":
            return header, seq

    raise ValueError(
        "Wuhan reference was not found. Please check whether the FASTA header contains "
        "Wuhan, Hu-1, RefSeq, YP_009724390, or NC_045512."
    )


def group_sequences(records):
    grouped = defaultdict(list)
    unclassified = []

    for header, seq in records:
        group = classify_record(header)

        if group is None:
            unclassified.append(header)
            continue

        if group in GROUP_ORDER:
            grouped[group].append((header, seq))
        else:
            unclassified.append(header)

    return grouped, unclassified


# =========================================================
# Cleaning
# =========================================================

def remove_reference_gap_columns(reference_seq, sequences_by_group):
    keep_indices = [i for i, aa in enumerate(reference_seq) if aa != "-"]

    cleaned = {}

    for group, seqs in sequences_by_group.items():
        cleaned[group] = []

        for seq in seqs:
            cleaned_seq = "".join(
                seq[i] if i < len(seq) else "-"
                for i in keep_indices
            )
            cleaned[group].append(cleaned_seq)

    return cleaned


# =========================================================
# Position-wise metrics
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


def shannon_entropy(residues):
    if not residues:
        return np.nan

    counts = Counter(residues)
    total = sum(counts.values())

    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def calculate_positionwise_metrics(cleaned_sequences):
    INCLUDE_WUHAN_IN_ENTROPY = False

    reference = cleaned_sequences["Wuhan"][0]

    rows = []

    available_variant_groups = [
        group for group in GROUP_ORDER
        if group != "Wuhan" and group in cleaned_sequences and cleaned_sequences[group]
    ]

    max_len = min(len(reference), SPIKE_LENGTH)

    for idx in range(max_len):
        pos = idx + 1
        ref_aa = reference[idx]

        if ref_aa in ["-", "?", ".", "X", "B", "Z", "J", "U", "O", "*"]:
            continue

        entropy_residues = []
        mismatch_count = 0
        deletion_count = 0
        valid_count = 0

        if INCLUDE_WUHAN_IN_ENTROPY:
            entropy_residues.append(ref_aa)

        for group in available_variant_groups:
            for seq in cleaned_sequences[group]:
                if idx >= len(seq):
                    continue

                aa = seq[idx].upper()

                if aa in ["?", ".", "X", "B", "Z", "J", "U", "O", "*"]:
                    continue

                valid_count += 1

                if aa == "-":
                    deletion_count += 1
                    mismatch_count += 1
                    continue

                entropy_residues.append(aa)

                if aa != ref_aa:
                    mismatch_count += 1

        if valid_count == 0:
            mut_freq = np.nan
        else:
            mut_freq = 100.0 * mismatch_count / valid_count

        ent = shannon_entropy(entropy_residues)

        rows.append(
            {
                "Position": pos,
                "Reference AA": ref_aa,
                "Domain": assign_domain(pos),
                "Valid variant residues": valid_count,
                "Mismatched residues": mismatch_count,
                "Deletion count": deletion_count,
                "Mutation frequency (%)": mut_freq,
                "Shannon entropy": ent,
            }
        )

    return rows


def save_metrics_table(rows):
    fieldnames = [
        "Position",
        "Reference AA",
        "Domain",
        "Valid variant residues",
        "Mismatched residues",
        "Deletion count",
        "Mutation frequency (%)",
        "Shannon entropy",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = row.copy()

            if np.isnan(out["Mutation frequency (%)"]):
                out["Mutation frequency (%)"] = "NA"
            else:
                out["Mutation frequency (%)"] = f"{out['Mutation frequency (%)']:.6f}"

            if np.isnan(out["Shannon entropy"]):
                out["Shannon entropy"] = "NA"
            else:
                out["Shannon entropy"] = f"{out['Shannon entropy']:.6f}"

            writer.writerow(out)


# =========================================================
# Plotting
# =========================================================

def add_domain_backgrounds(ax):
    for name, start, end, color in DOMAINS:
        if name in ["SP", "TM", "CT"]:
            alpha = 0.050
        else:
            alpha = 0.080

        ax.axvspan(
            start,
            end,
            color=color,
            alpha=alpha,
            lw=0,
            zorder=0,
        )


def draw_domain_track(ax):
    y = -0.125
    height = 0.028

    ax.plot(
        [1, SPIKE_LENGTH],
        [y, y],
        color="#606060",
        lw=0.85,
        alpha=0.55,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=10,
    )

    for name, start, end, color in DOMAINS:
        ax.add_patch(
            Rectangle(
                (start, y - height / 2),
                end - start + 1,
                height,
                facecolor=color,
                edgecolor="none",
                alpha=0.95,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                zorder=11,
            )
        )

        width = end - start + 1

        if width > 35:
            ax.text(
                (start + end) / 2,
                y - 0.040,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=11.0,
                fontweight="bold",
                color="#222222",
                clip_on=False,
                zorder=12,
            )


def add_functional_guides(ax):
    for label, start, end, color in FUNCTIONAL_REGIONS:
        mid = (start + end) / 2

        ax.plot(
            [start, end],
            [1.035, 1.035],
            transform=ax.get_xaxis_transform(),
            color=color,
            lw=2.5,
            solid_capstyle="butt",
            clip_on=False,
            zorder=13,
        )

        ax.text(
            mid,
            1.064,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color=color,
            clip_on=False,
            zorder=14,
        )


def rolling_mean(values, window=21):
    values = np.asarray(values, dtype=float)

    if window <= 1:
        return values

    valid = np.isfinite(values)
    y = np.where(valid, values, 0.0)

    kernel = np.ones(window, dtype=float)

    numerator = np.convolve(y, kernel, mode="same")
    denominator = np.convolve(valid.astype(float), kernel, mode="same")

    smoothed = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )

    return smoothed


def plot_single_panel_figure(rows):
    positions = np.array([row["Position"] for row in rows], dtype=int)
    mutation_frequency = np.array([row["Mutation frequency (%)"] for row in rows], dtype=float)
    entropy = np.array([row["Shannon entropy"] for row in rows], dtype=float)

    mut_smooth = rolling_mean(mutation_frequency, window=21)
    ent_smooth = rolling_mean(entropy, window=21)

    fig, ax1 = plt.subplots(figsize=(18.0, 7.2), facecolor="white")
    ax1.set_facecolor("white")

    ax2 = ax1.twinx()

    add_domain_backgrounds(ax1)

    # Mutation frequency raw line
    ax1.plot(
        positions,
        mutation_frequency,
        color="#E0A32A",
        lw=0.75,
        alpha=0.22,
        zorder=3,
    )

    # Mutation frequency smoothed line
    ax1.plot(
        positions,
        mut_smooth,
        color="#B2182B",
        lw=2.95,
        alpha=0.98,
        zorder=5,
    )

    ax1.fill_between(
        positions,
        mut_smooth,
        0,
        color="#B2182B",
        alpha=0.075,
        zorder=2,
    )

    # Shannon entropy raw line
    ax2.plot(
        positions,
        entropy,
        color="#8DBBE8",
        lw=0.75,
        alpha=0.24,
        zorder=3,
    )

    # Shannon entropy smoothed line
    ax2.plot(
        positions,
        ent_smooth,
        color="#2166AC",
        lw=2.95,
        alpha=0.98,
        zorder=6,
    )

    ax2.fill_between(
        positions,
        ent_smooth,
        0,
        color="#2166AC",
        alpha=0.075,
        zorder=1,
    )

    # Fixed publication-scale axes
    ax1.set_xlim(1, SPIKE_LENGTH)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 1.5)

    # Axis labels
    ax1.set_ylabel(
        "Mutation frequency (%)",
        fontsize=17.0,
        fontweight="bold",
        color="#B2182B",
        labelpad=10,
    )

    ax2.set_ylabel(
        "Shannon entropy",
        fontsize=17.0,
        fontweight="bold",
        color="#2166AC",
        labelpad=10,
    )

    ax1.tick_params(
        axis="y",
        labelcolor="#B2182B",
        labelsize=15.0,
        width=1.2,
        length=5,
    )

    ax2.tick_params(
        axis="y",
        labelcolor="#2166AC",
        labelsize=15.0,
        width=1.2,
        length=5,
    )

    ax1.tick_params(
        axis="x",
        labelsize=15.0,
        width=1.2,
        length=5,
    )

    ax1.set_xticks(
        [1, 100, 200, 300, 400, 500, 600, 681, 800, 900, 1000, 1100, 1200, 1273]
    )

    # Key residue guide lines
    key_positions = [142, 144, 417, 452, 478, 484, 501, 614, 681]

    for p in key_positions:
        ax1.axvline(
            p,
            color="#A0A0A0",
            lw=0.65,
            linestyle=":",
            alpha=0.55,
            zorder=1,
        )

    key_labels = {
        417: "K417",
        452: "L452",
        478: "T478",
        484: "E484",
        501: "N501",
        614: "D614",
        681: "P681",
    }

    y_label = 96

    for pos, label in key_labels.items():
        ax1.text(
            pos,
            y_label,
            label,
            ha="center",
            va="top",
            fontsize=10.5,
            rotation=90,
            color="#505050",
            zorder=20,
        )

    add_functional_guides(ax1)
    draw_domain_track(ax1)

    # Styling
    ax1.grid(axis="y", color="#E0E0E0", lw=0.85, alpha=0.85)

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)

    ax1.spines["left"].set_color("#B2182B")
    ax1.spines["left"].set_linewidth(1.3)

    ax2.spines["right"].set_color("#2166AC")
    ax2.spines["right"].set_linewidth(1.3)

    ax1.spines["bottom"].set_linewidth(1.2)

    # Legend only, no title or panel label
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#B2182B",
            lw=3.2,
            label="Mutation frequency",
        ),
        Line2D(
            [0],
            [0],
            color="#2166AC",
            lw=3.2,
            label="Shannon entropy",
        ),
    ]

    ax1.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        frameon=True,
        fontsize=15.0,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0.035, 0.10, 0.975, 0.90])

    fig.savefig(
        OUTPUT_PNG,
        dpi=900,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print("\nSaved single-panel Figure 3B:")
    print(f"  {OUTPUT_PNG}")


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

    rows = calculate_positionwise_metrics(cleaned_sequences)

    save_metrics_table(rows)

    print("\nSaved position-wise metrics table:")
    print(f"  {OUTPUT_CSV}")

    plot_single_panel_figure(rows)

    mut_freq_values = [
        row["Mutation frequency (%)"]
        for row in rows
        if not np.isnan(row["Mutation frequency (%)"])
    ]

    entropy_values = [
        row["Shannon entropy"]
        for row in rows
        if not np.isnan(row["Shannon entropy"])
    ]

    print("\nMetric summary:")
    print(f"  Positions analyzed: {len(rows)}")
    print(f"  Max mutation frequency: {max(mut_freq_values):.3f}%")
    print(f"  Max Shannon entropy: {max(entropy_values):.3f}")


if __name__ == "__main__":
    main()