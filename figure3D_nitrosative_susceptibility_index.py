# ==============================================================================
# Script Name: figure3D_nitrosative_susceptibility_index.py
# Author: Code made by Nitish Katoch
#
# Description:
#   Implements a composite rule-based priority model to map out the Nitrosative 
#   Susceptibility Index (NSI) across individual amino acid residues of the Spike.
#
# Plots Made:
#   - Linear metric tracking chart mapping local smoothed and raw NSI scores 
#     across structural protein domain frames.
#
# Inputs:
#   - 02_CuratedRawData/combined_curated_aligned.fasta
#
# Calculations:
#   - Computes residue scoring using a multifactorial rule framework:
#     NSI = 0.35*reactivity + 0.25*exposure + 0.15*domain + 0.10*entropy + 0.15*ACE2_interface
#   - Normalizes data matrices after stripping non-wuhan gap columns.
#   - Smooths local scoring variations using a rolling arithmetic mean window.
#
# Outputs:
#   - 03_Results/Figure3D_nitrosative_susceptibility_index.png
#   - 03_Results/Figure3D_nitrosative_susceptibility_index.pdf
#   - 03_Results/Figure3D_nitrosative_susceptibility_index.csv
# ==============================================================================
from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =========================================================
# Figure 3D
# Nitrosative susceptibility index across SARS-CoV-2 spike
# =========================================================
#
# Input:
#   02_CuratedRawData/combined_curated_aligned.fasta
#
# Output:
#   03_Results/Figure3D_nitrosative_susceptibility_index.png
#   03_Results/Figure3D_nitrosative_susceptibility_index.pdf
#   03_Results/Figure3D_nitrosative_susceptibility_index.csv
#
# Included groups:
#   Wuhan, Alpha, Beta, Delta, Omicron BA.2
#
# The score is a transparent rule-based prioritization score:
#   NSI = 0.35*reactivity + 0.25*exposure + 0.15*domain
#         + 0.10*entropy + 0.15*ACE2_interface
#
# Replace the approximate exposure term with DSSP/FreeSASA/NetSurfP
# solvent accessibility values later, if available.
# =========================================================


BASE_DIR = Path(__file__).resolve().parent

ALIGNMENT_FILE = BASE_DIR / "02_CuratedRawData" / "combined_curated_aligned.fasta"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "Figure3D_nitrosative_susceptibility_index.png"
OUTPUT_PDF = OUTPUT_DIR / "Figure3D_nitrosative_susceptibility_index.pdf"
OUTPUT_CSV = OUTPUT_DIR / "Figure3D_nitrosative_susceptibility_index.csv"


GROUP_ORDER = ["Wuhan", "Alpha", "Beta", "Delta", "Omicron_BA2"]

DISPLAY_NAMES = {
    "Wuhan": "Wuhan",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Delta": "Delta",
    "Omicron_BA2": "Omicron BA.2",
}


SPIKE_LENGTH = 1273

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

FUNCTIONAL_REGIONS = [
    ("NTD antigenic\nregion", 14, 305, "#8A6A00"),
    ("RBD/RBM", 437, 505, "#1B7837"),
    ("S1/S2\ncleavage", 675, 692, "#222222"),
    ("Fusion\npeptide", 816, 833, "#558B2F"),
    ("HR1/HR2", 912, 1213, "#B36B00"),
    ("TM/CT", 1214, 1273, "#444444"),
]


# Residue-level chemistry score.
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

SMOOTHING_WINDOW = 5
INVALID_AA = set(["-", "?", ".", "X", "B", "Z", "J", "U", "O", "*"])


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
    if window <= 1:
        return np.array(values, dtype=float)

    values = np.array(values, dtype=float)
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


def calculate_positionwise_entropy(cleaned_sequences):
    reference = cleaned_sequences["Wuhan"][0]

    variant_groups = [
        group for group in GROUP_ORDER
        if group != "Wuhan" and group in cleaned_sequences and cleaned_sequences[group]
    ]

    entropy_values = []

    for idx in range(min(len(reference), SPIKE_LENGTH)):
        residues = []

        for group in variant_groups:
            for seq in cleaned_sequences[group]:
                if idx >= len(seq):
                    continue

                aa = seq[idx]

                if aa not in INVALID_AA:
                    residues.append(aa)

        entropy_values.append(shannon_entropy(residues))

    entropy_array = np.array(entropy_values, dtype=float)

    if np.all(np.isnan(entropy_array)):
        normalized_entropy = np.zeros_like(entropy_array)
    else:
        entropy_array_clean = np.nan_to_num(entropy_array, nan=0.0)
        max_entropy = np.max(entropy_array_clean)

        if max_entropy == 0:
            normalized_entropy = np.zeros_like(entropy_array_clean)
        else:
            normalized_entropy = entropy_array_clean / max_entropy

    normalized_entropy_smooth = rolling_mean(normalized_entropy, SMOOTHING_WINDOW)

    return entropy_array, normalized_entropy, normalized_entropy_smooth


def calculate_nsi(cleaned_sequences):
    reference = cleaned_sequences["Wuhan"][0]
    entropy_raw, entropy_norm, entropy_norm_smooth = calculate_positionwise_entropy(cleaned_sequences)

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
        entropy_score = entropy_norm_smooth[idx]
        ace2_score = ace2_interface_score(position)

        raw_nsi = (
            W_REACTIVITY * reactivity
            + W_EXPOSURE * exposure
            + W_DOMAIN * domain_score
            + W_ENTROPY * entropy_score
            + W_ACE2 * ace2_score
        )

        rows.append(
            {
                "Position": position,
                "AA": aa,
                "Domain": domain,
                "Residue reactivity score": reactivity,
                "Approximate exposure score": exposure,
                "Domain weight score": domain_score,
                "Raw entropy": entropy_raw[idx],
                "Normalized entropy": entropy_norm[idx],
                "Smoothed normalized entropy": entropy_score,
                "ACE2 interface score": ace2_score,
                "Raw NSI": raw_nsi,
            }
        )

    raw_scores = np.array([row["Raw NSI"] for row in rows], dtype=float)
    min_score = np.nanmin(raw_scores)
    max_score = np.nanmax(raw_scores)

    if max_score == min_score:
        normalized_scores = np.zeros_like(raw_scores)
    else:
        normalized_scores = (raw_scores - min_score) / (max_score - min_score)

    smoothed_scores = rolling_mean(normalized_scores, SMOOTHING_WINDOW)

    for row, nsi, nsi_smooth in zip(rows, normalized_scores, smoothed_scores):
        row["Nitrosative susceptibility index"] = nsi
        row["Smoothed NSI"] = nsi_smooth

        if row["AA"] in ["C", "Y", "M", "W", "H"]:
            row["Reactive residue"] = "Yes"
        else:
            row["Reactive residue"] = "No"

        if row["Position"] in ACE2_DIRECT_CONTACT_POSITIONS:
            row["ACE2 relation"] = "Direct contact"
        elif row["Position"] in ACE2_NEAR_INTERFACE_POSITIONS:
            row["ACE2 relation"] = "Near interface"
        else:
            row["ACE2 relation"] = "No"

    return rows


def save_nsi_table(rows):
    fieldnames = [
        "Position",
        "AA",
        "Domain",
        "Reactive residue",
        "ACE2 relation",
        "Residue reactivity score",
        "Approximate exposure score",
        "Domain weight score",
        "Raw entropy",
        "Normalized entropy",
        "Smoothed normalized entropy",
        "ACE2 interface score",
        "Raw NSI",
        "Nitrosative susceptibility index",
        "Smoothed NSI",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = row.copy()

            for key in [
                "Residue reactivity score",
                "Approximate exposure score",
                "Domain weight score",
                "Raw entropy",
                "Normalized entropy",
                "Smoothed normalized entropy",
                "ACE2 interface score",
                "Raw NSI",
                "Nitrosative susceptibility index",
                "Smoothed NSI",
            ]:
                value = out[key]
                out[key] = "NA" if np.isnan(value) else f"{value:.6f}"

            writer.writerow(out)


def add_domain_backgrounds(ax):
    for name, start, end, color in DOMAINS:
        alpha = 0.08

        if name in ["RBD/RBM", "S1/S2", "FP"]:
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


def plot_figure(rows):
    positions = np.array([row["Position"] for row in rows], dtype=int)
    nsi = np.array([row["Nitrosative susceptibility index"] for row in rows], dtype=float)
    nsi_smooth = np.array([row["Smoothed NSI"] for row in rows], dtype=float)
    entropy = np.array([row["Smoothed normalized entropy"] for row in rows], dtype=float)
    aa_list = [row["AA"] for row in rows]

    fig, ax = plt.subplots(figsize=(15.8, 5.8), facecolor="white")
    ax.set_facecolor("white")

    add_domain_backgrounds(ax)

    # Heat strip at the bottom using NSI.
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

    ax.plot(
        positions,
        nsi,
        color="#C51B7D",
        lw=0.95,
        alpha=0.45,
        label="Raw NSI",
        zorder=3,
    )

    ax.plot(
        positions,
        nsi_smooth,
        color="#7A0177",
        lw=1.85,
        label=f"Smoothed NSI, window={SMOOTHING_WINDOW}",
        zorder=4,
    )

    ax.fill_between(
        positions,
        nsi_smooth,
        0,
        color="#C51B7D",
        alpha=0.15,
        zorder=2,
    )

    ax.plot(
        positions,
        entropy,
        color="#2C5AA0",
        lw=1.0,
        alpha=0.40,
        label="Normalized entropy",
        zorder=3,
    )

    reactive_positions = []
    reactive_scores = []

    for pos, aa, score in zip(positions, aa_list, nsi_smooth):
        if aa in ["C", "Y", "M", "W", "H"] and score >= 0.55:
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

        ax.axvline(
            pos,
            color="#808080",
            linestyle=":",
            lw=0.65,
            alpha=0.65,
            zorder=1,
        )

        ax.text(
            pos,
            min(1.02, y + 0.10),
            label,
            ha="center",
            va="bottom",
            fontsize=8.0,
            rotation=90,
            fontweight="bold",
            color="#444444",
            zorder=6,
        )

    for pos in [449, 453, 489, 505]:
        idx = np.where(positions == pos)[0]

        if len(idx) == 0:
            continue

        ax.scatter(
            [pos],
            [nsi_smooth[idx[0]]],
            s=42,
            color="#1B7837",
            edgecolor="white",
            linewidth=0.7,
            zorder=7,
        )

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

    ax.text(
        -0.055,
        1.12,
        "D",
        transform=ax.transAxes,
        fontsize=21,
        fontweight="bold",
        ha="left",
        va="top",
    )

    ax.set_title(
        "Nitrosative susceptibility index across the SARS-CoV-2 spike glycoprotein",
        fontsize=15,
        fontweight="bold",
        pad=28,
    )

    fig.text(
        0.5,
        0.91,
        "Composite score based on residue chemistry, approximate exposure, domain context, local entropy, and ACE2-interface relevance",
        ha="center",
        fontsize=10.3,
        color="#555555",
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=8.7,
        handlelength=2.0,
    )

    plt.tight_layout(rect=[0.035, 0.08, 0.995, 0.90])

    fig.savefig(OUTPUT_PNG, dpi=900, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print("\nSaved Figure 3D:")
    print(f"  {OUTPUT_PNG}")
    print(f"  {OUTPUT_PDF}")


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

    rows = calculate_nsi(cleaned_sequences)

    save_nsi_table(rows)
    plot_figure(rows)

    print("\nSaved NSI table:")
    print(f"  {OUTPUT_CSV}")

    top_rows = sorted(rows, key=lambda r: r["Smoothed NSI"], reverse=True)[:30]

    print("\nTop 30 positions by smoothed NSI:")
    for row in top_rows:
        print(
            f"  {row['AA']}{row['Position']} | "
            f"{row['Domain']} | "
            f"NSI={row['Smoothed NSI']:.3f} | "
            f"reactivity={row['Residue reactivity score']:.2f} | "
            f"exposure={row['Approximate exposure score']:.2f} | "
            f"ACE2={row['ACE2 relation']}"
        )


if __name__ == "__main__":
    main()
