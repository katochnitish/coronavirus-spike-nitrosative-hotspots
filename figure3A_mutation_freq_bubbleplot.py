# ==============================================================================
# Script Name: figure3A_mutation_freq_bubbleplot.py
# Author: Code made by Nitish Katoch
#
# Description:
#   Performs domain-specific mutation frequency analysis tracking key major 
#   variants of concern (Alpha, Beta, Delta, Omicron BA.2) relative to the 
#   ancestral Wuhan reference sequence.
#
# Plots Made:
#   - Multidimensional bubble plot mapping major structural domains (NTD, RBD, 
#     S1/S2, FP, HR1, HR2, TM/CT) where bubble diameters visually correspond 
#     to regional mutation burden frequencies.
#
# Inputs:
#   - 02_CuratedRawData/combined_curated_aligned.fasta
#
# Calculations:
#   - Compiles lineage-specific consensus sequence strings from raw alignment files.
#   - Pinpoints and tracks exact amino acid substitution identities against Wuhan.
#   - Tallies overall mutational density percentages normalized to the true 
#     sequence length of each defined glycoprotein domain.
#
# Outputs:
#   - 03_Results/Figure3A_consensus_mutation_names_by_variant.csv
#   - 03_Results/Figure3A_domain_mutation_frequency_by_variant.csv
#   - 03_Results/Figure3A_mutation_frequency_bubbleplot.png
# ==============================================================================

from pathlib import Path
from collections import Counter, defaultdict
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =========================================================
# Figure 3A combined analysis
# Mutation-name CSV + mutation-frequency CSV + bubble plot
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

ALIGNMENT_FILE = BASE_DIR / "02_CuratedRawData" / "combined_curated_aligned.fasta"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MUTATION_CSV = OUTPUT_DIR / "Figure3A_consensus_mutation_names_by_variant.csv"
OUTPUT_FREQUENCY_CSV = OUTPUT_DIR / "Figure3A_domain_mutation_frequency_by_variant.csv"
OUTPUT_BUBBLE_PNG = OUTPUT_DIR / "Figure3A_mutation_frequency_bubbleplot.png"


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

GROUP_COLORS = {
    "Wuhan": "#0B2545",
    "Alpha": "#B2182B",
    "Beta": "#A6611A",
    "Delta": "#D48806",
    "Omicron_BA2": "#5F8D3A",
}


# =========================================================
# Spike domain definitions
# True coordinates are used for mutation counting.
# Plot coordinates are visually expanded for clean labels.
# =========================================================

SPIKE_LENGTH = 1273

DOMAINS = {
    "NTD": {
        "start": 14,
        "end": 305,
        "plot_start": 14,
        "plot_end": 305,
        "color": "#163B73",
        "fontsize": 16.0,
    },
    "RBD/RBM": {
        "start": 319,
        "end": 541,
        "plot_start": 330,
        "plot_end": 540,
        "color": "#6F1D1B",
        "fontsize": 15.0,
    },
    "S1/S2": {
        "start": 675,
        "end": 692,
        "plot_start": 640,
        "plot_end": 725,
        "color": "#7B4F00",
        "fontsize": 14.0,
    },
    "FP": {
        "start": 816,
        "end": 833,
        "plot_start": 790,
        "plot_end": 865,
        "color": "#7F2F12",
        "fontsize": 14.0,
    },
    "HR1": {
        "start": 912,
        "end": 984,
        "plot_start": 905,
        "plot_end": 985,
        "color": "#0F5132",
        "fontsize": 15.0,
    },
    "HR2": {
        "start": 1163,
        "end": 1213,
        "plot_start": 1135,
        "plot_end": 1190,
        "color": "#352A86",
        "fontsize": 14.0,
    },
    "TM/CT": {
        "start": 1214,
        "end": 1273,
        "plot_start": 1210,
        "plot_end": 1273,
        "color": "#303030",
        "fontsize": 13.5,
    },
}


FUNCTIONAL_REGIONS = {
    "NTD antigenic supersite": (14, 305),
    "RBD/RBM immune escape": (437, 505),
    "S1/S2 cleavage site": (675, 692),
    "Fusion peptide": (816, 833),
}


HOTSPOT_REGIONS = {
    "NTD\nantigenic": (14, 305),
    "RBD/RBM\nimmune escape": (437, 505),
    "S1/S2\ncleavage site": (675, 692),
    "FP\nfusion peptide": (816, 833),
}


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
# Classification utilities
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
# Consensus sequence creation and coordinate cleaning
# =========================================================

def build_consensus_from_group(records):
    if not records:
        return None

    sequences = [seq for _, seq in records]
    max_len = max(len(seq) for seq in sequences)
    consensus = []

    for i in range(max_len):
        residues = []

        for seq in sequences:
            if i >= len(seq):
                continue

            aa = seq[i].upper()

            if aa in ["?", ".", "X", "B", "Z", "J", "U", "O", "*"]:
                continue

            residues.append(aa)

        if residues:
            consensus.append(Counter(residues).most_common(1)[0][0])
        else:
            consensus.append("-")

    return "".join(consensus)


def remove_reference_gap_columns(reference_seq, consensus_by_group):
    keep_indices = [i for i, aa in enumerate(reference_seq) if aa != "-"]

    cleaned = {}

    for group, seq in consensus_by_group.items():
        cleaned[group] = "".join(
            seq[i] if i < len(seq) else "-"
            for i in keep_indices
        )

    return cleaned


# =========================================================
# Mutation extraction and annotation
# =========================================================

def assign_domain(position):
    for domain_name, domain_info in DOMAINS.items():
        if domain_info["start"] <= position <= domain_info["end"]:
            return domain_name

    if 1 <= position <= 685:
        return "S1 other"

    if 686 <= position <= SPIKE_LENGTH:
        return "S2 other"

    return "Outside spike"


def assign_functional_region(position):
    matched = []

    for region_name, (start, end) in FUNCTIONAL_REGIONS.items():
        if start <= position <= end:
            matched.append(region_name)

    if matched:
        return "; ".join(matched)

    return "Other"


def is_valid_reference_aa(aa):
    return aa not in ["-", "?", ".", "X", "B", "Z", "J", "U", "O", "*"]


def is_ambiguous_query_aa(aa):
    return aa in ["?", ".", "X", "B", "Z", "J", "U", "O", "*"]


def extract_consensus_mutations(cleaned_sequences):
    reference = cleaned_sequences["Wuhan"]
    rows = []
    mutation_lookup = defaultdict(list)

    for group in GROUP_ORDER:
        if group == "Wuhan":
            continue

        if group not in cleaned_sequences:
            continue

        query = cleaned_sequences[group]
        max_len = min(len(reference), len(query), SPIKE_LENGTH)

        for idx in range(max_len):
            pos = idx + 1
            ref_aa = reference[idx]
            query_aa = query[idx]

            if not is_valid_reference_aa(ref_aa):
                continue

            if query_aa == "-":
                mutation_type = "Deletion"
                mutation_label = f"{ref_aa}{pos}del"
                is_mutation = True

            elif is_ambiguous_query_aa(query_aa):
                mutation_type = "Ambiguous"
                mutation_label = f"{ref_aa}{pos}?"
                is_mutation = False

            elif query_aa != ref_aa:
                mutation_type = "Substitution"
                mutation_label = f"{ref_aa}{pos}{query_aa}"
                is_mutation = True

            else:
                mutation_type = "Reference"
                mutation_label = f"{ref_aa}{pos}{query_aa}"
                is_mutation = False

            if not is_mutation:
                continue

            row = {
                "Group": group,
                "Display name": DISPLAY_NAMES[group],
                "Position": pos,
                "Reference AA": ref_aa,
                "Variant consensus AA": query_aa,
                "Mutation": mutation_label,
                "Mutation type": mutation_type,
                "Domain": assign_domain(pos),
                "Functional region": assign_functional_region(pos),
            }

            rows.append(row)
            mutation_lookup[group].append(row)

    return rows, mutation_lookup


def save_mutation_table(rows):
    fieldnames = [
        "Group",
        "Display name",
        "Position",
        "Reference AA",
        "Variant consensus AA",
        "Mutation",
        "Mutation type",
        "Domain",
        "Functional region",
    ]

    with open(OUTPUT_MUTATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =========================================================
# Domain-level mutation frequency
# =========================================================

def mutation_frequency_by_region(reference_seq, query_seq, start, end):
    valid = 0
    mismatch = 0
    substitution = 0
    deletion = 0

    mutation_names = []

    for pos in range(start, end + 1):
        idx = pos - 1

        if idx >= len(reference_seq) or idx >= len(query_seq):
            continue

        ref_aa = reference_seq[idx]
        query_aa = query_seq[idx]

        if not is_valid_reference_aa(ref_aa):
            continue

        if is_ambiguous_query_aa(query_aa):
            continue

        valid += 1

        if query_aa == "-":
            mismatch += 1
            deletion += 1
            mutation_names.append(f"{ref_aa}{pos}del")

        elif query_aa != ref_aa:
            mismatch += 1
            substitution += 1
            mutation_names.append(f"{ref_aa}{pos}{query_aa}")

    if valid == 0:
        frequency = np.nan
    else:
        frequency = 100.0 * mismatch / valid

    return frequency, mismatch, substitution, deletion, valid, mutation_names


def build_domain_frequency_table(cleaned_sequences):
    reference_seq = cleaned_sequences["Wuhan"]

    rows = []

    for group in GROUP_ORDER:
        if group not in cleaned_sequences:
            continue

        query_seq = cleaned_sequences[group]

        for domain_name, domain_info in DOMAINS.items():
            frequency, mismatch, substitution, deletion, valid, mutation_names = mutation_frequency_by_region(
                reference_seq=reference_seq,
                query_seq=query_seq,
                start=domain_info["start"],
                end=domain_info["end"],
            )

            rows.append(
                {
                    "Group": group,
                    "Display name": DISPLAY_NAMES[group],
                    "Domain": domain_name,
                    "Start": domain_info["start"],
                    "End": domain_info["end"],
                    "Valid residues": valid,
                    "Mutated residues": mismatch,
                    "Substitutions": substitution,
                    "Deletions": deletion,
                    "Mutation frequency (%)": frequency,
                    "Mutation names": "; ".join(mutation_names),
                }
            )

    return rows


def save_domain_frequency_table(rows):
    fieldnames = [
        "Group",
        "Display name",
        "Domain",
        "Start",
        "End",
        "Valid residues",
        "Mutated residues",
        "Substitutions",
        "Deletions",
        "Mutation frequency (%)",
        "Mutation names",
    ]

    with open(OUTPUT_FREQUENCY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = row.copy()

            if isinstance(out["Mutation frequency (%)"], float) and np.isnan(out["Mutation frequency (%)"]):
                out["Mutation frequency (%)"] = "NA"
            else:
                out["Mutation frequency (%)"] = f"{out['Mutation frequency (%)']:.3f}"

            writer.writerow(out)


# =========================================================
# Bubble plot
# =========================================================

def mutation_frequency_lookup(rows):
    lookup = {}

    for row in rows:
        lookup[(row["Group"], row["Domain"])] = row["Mutation frequency (%)"]

    return lookup


def draw_domain_box(ax, x1, x2, y, height, label, color, fontsize):
    ax.add_patch(
        Rectangle(
            (x1, y - height / 2),
            x2 - x1,
            height,
            facecolor=color,
            edgecolor="#111111",
            linewidth=0.75,
            alpha=0.97,
            zorder=2,
        )
    )

    ax.text(
        (x1 + x2) / 2,
        y,
        label,
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=3,
    )


def draw_mutation_frequency_bubbleplot(domain_frequency_rows):
    lookup = mutation_frequency_lookup(domain_frequency_rows)

    # Similar horizontal style to the Shannon entropy figure, with reduced height.
    fig, ax = plt.subplots(figsize=(18.0, 7.2), facecolor="white")
    ax.set_facecolor("white")

    xmin = 1
    xmax = SPIKE_LENGTH

    ax.set_xlim(xmin - 75, xmax + 85)

    # Reduced vertical height compared with previous bubble plot.
    ax.set_ylim(-2.40, len(GROUP_ORDER) + 1.15)

    ax.axis("off")

    domain_y = len(GROUP_ORDER) + 0.35
    domain_height = 0.50

    ax.plot(
        [xmin, xmax],
        [domain_y, domain_y],
        color="#606060",
        linewidth=1.55,
        alpha=0.55,
        zorder=1,
    )

    for domain_name, info in DOMAINS.items():
        draw_domain_box(
            ax,
            info["plot_start"],
            info["plot_end"],
            domain_y,
            domain_height,
            domain_name,
            info["color"],
            info["fontsize"],
        )

    row_y = {}

    for i, group in enumerate(GROUP_ORDER):
        y = len(GROUP_ORDER) - 1 - i
        row_y[group] = y

        ax.text(
            xmin - 10,
            y,
            DISPLAY_NAMES[group],
            fontsize=16.0,
            fontweight="bold",
            color="#2F2F2F",
            ha="right",
            va="center",
        )

    max_frequency = 0

    for row in domain_frequency_rows:
        freq = row["Mutation frequency (%)"]

        if isinstance(freq, float) and not np.isnan(freq):
            max_frequency = max(max_frequency, freq)

    if max_frequency == 0:
        max_frequency = 1

    # Reduced bubble sizes to reduce vertical dominance.
    min_size = 65
    max_size = 850

    for group in GROUP_ORDER:
        if group not in row_y:
            continue

        y = row_y[group]

        for domain_name, info in DOMAINS.items():
            freq = lookup.get((group, domain_name), np.nan)

            if isinstance(freq, float) and np.isnan(freq):
                continue

            if group == "Wuhan":
                size = 55
                alpha = 0.28
            else:
                size = min_size + (freq / max_frequency) * (max_size - min_size)
                alpha = 0.92 if freq > 0 else 0.07

            x = (info["plot_start"] + info["plot_end"]) / 2

            ax.scatter(
                x,
                y,
                s=size,
                color=GROUP_COLORS[group],
                edgecolor="#111111",
                linewidth=0.45,
                alpha=alpha,
                zorder=5,
            )

    hotspot_y = -0.55

    hotspot_colors = {
        "NTD\nantigenic": "#8B1E1E",
        "RBD/RBM\nimmune escape": "#8B1E1E",
        "S1/S2\ncleavage site": "#7B4F00",
        "FP\nfusion peptide": "#7F2F12",
    }

    hotspot_label_offsets = {
        "NTD\nantigenic": 0,
        "RBD/RBM\nimmune escape": 0,
        "S1/S2\ncleavage site": -18,
        "FP\nfusion peptide": 28,
    }

    for label, (start, end) in HOTSPOT_REGIONS.items():
        color = hotspot_colors.get(label, "#555555")
        center_x = (start + end) / 2 + hotspot_label_offsets.get(label, 0)

        ax.plot(
            [start, end],
            [hotspot_y, hotspot_y],
            color=color,
            linewidth=3.4,
            alpha=0.95,
            solid_capstyle="butt",
            zorder=2,
        )

        ax.text(
            center_x,
            hotspot_y - 0.18,
            label,
            fontsize=13.0,
            fontweight="bold",
            color="#4A4A4A",
            ha="center",
            va="top",
            linespacing=1.10,
        )

    # Bubble-size legend with larger text and reduced overlap.
    legend_x = xmax - 300
    legend_title_y = -1.12
    legend_y = -1.58
    label_y = -2.08

    ax.text(
        legend_x + 116,
        legend_title_y,
        "Mutation frequency",
        fontsize=14.0,
        fontweight="bold",
        color="#333333",
        ha="center",
        va="center",
        zorder=10,
    )

    guide_values = [5, 15, 30]
    guide_x_offsets = [0, 116, 250]

    for value, offset in zip(guide_values, guide_x_offsets):
        size = min_size + (value / max_frequency) * (max_size - min_size)
        size = min(size, max_size)

        ax.scatter(
            legend_x + offset,
            legend_y,
            s=size,
            color="#B5B5B5",
            edgecolor="#111111",
            linewidth=0.45,
            alpha=0.82,
            zorder=6,
        )

        ax.text(
            legend_x + offset,
            label_y,
            f"{value}%",
            fontsize=12.5,
            color="#4A4A4A",
            ha="center",
            va="top",
            zorder=10,
        )

    plt.savefig(
        OUTPUT_BUBBLE_PNG,
        dpi=900,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print("\nSaved bubble plot:")
    print(f"  {OUTPUT_BUBBLE_PNG}")


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

    print("\nGroup summary before consensus:")
    for group in GROUP_ORDER:
        print(f"  {DISPLAY_NAMES[group]}: n={len(grouped.get(group, []))}")

    if unclassified:
        print(f"\nUnclassified records: {len(unclassified)}")
        print("First 25 unclassified headers:")
        for header in unclassified[:25]:
            print(f"  {header}")

    consensus_by_group = {"Wuhan": wuhan_seq}

    for group in GROUP_ORDER:
        if group == "Wuhan":
            continue

        group_records = grouped.get(group, [])

        if not group_records:
            print(f"\nWarning: {DISPLAY_NAMES[group]} was not found and will be skipped.")
            continue

        consensus_by_group[group] = build_consensus_from_group(group_records)

    cleaned_sequences = remove_reference_gap_columns(
        reference_seq=consensus_by_group["Wuhan"],
        consensus_by_group=consensus_by_group,
    )

    available_groups = [group for group in GROUP_ORDER if group in cleaned_sequences]

    print("\nAvailable groups:")
    for group in available_groups:
        print(f"  {DISPLAY_NAMES[group]}")

    mutation_rows, mutation_lookup = extract_consensus_mutations(cleaned_sequences)

    save_mutation_table(mutation_rows)

    print("\nSaved mutation-name CSV:")
    print(f"  {OUTPUT_MUTATION_CSV}")

    print("\nConsensus mutation counts relative to Wuhan:")
    for group in available_groups:
        if group == "Wuhan":
            print(f"  {DISPLAY_NAMES[group]}: reference")
        else:
            print(f"  {DISPLAY_NAMES[group]}: {len(mutation_lookup.get(group, []))}")

    domain_frequency_rows = build_domain_frequency_table(cleaned_sequences)

    save_domain_frequency_table(domain_frequency_rows)

    print("\nSaved domain-frequency CSV:")
    print(f"  {OUTPUT_FREQUENCY_CSV}")

    draw_mutation_frequency_bubbleplot(domain_frequency_rows)


if __name__ == "__main__":
    main()