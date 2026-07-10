# ==============================================================================
# Script Name: figure2B_Sequences.py
# Author: Code made by Nitish Katoch
#
# Description:
#   Generates a publication-quality, two-panel sequence alignment diagram 
#   characterizing the Spike protein secondary structures, structural domains,
#   and ACE2-receptor binding residues across human, bat, and variant coronaviruses.
#
# Plots Made:
#   - High-resolution, multi-row consensus sequence layout maps highlighting 
#     helices, beta-strands, and functional contact residues between 
#     alignment coordinates 300 and 600.
#
# Inputs:
#   - 02_CuratedRawData/combined_curated_aligned.fasta
#
# Calculations:
#   - Compiles position-by-position majority consensus records for major virus lineages.
#   - Removes alignment gap columns relative to the reference strain.
#   - Extracts specific structural fragments (residues 300–600) and color-codes 
#     secondary elements alongside verified ACE2-binding contact footprints.
#
# Outputs:
#   - 03_Results/Consensus_sequence_diagram_300_600_900dpi.png
#   - 04_RepresentativeSequences/consensus_sequences_by_major_group.fasta
# ==============================================================================

from pathlib import Path
from collections import Counter, defaultdict

from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

ALIGNMENT_FILE = BASE_DIR / "02_CuratedRawData" / "combined_curated_aligned.fasta"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONSENSUS_DIR = BASE_DIR / "04_RepresentativeSequences"
CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)

CONSENSUS_FASTA = CONSENSUS_DIR / "consensus_sequences_by_major_group.fasta"

OUTPUT_FIGURE = OUTPUT_DIR / "Consensus_sequence_diagram_300_600_900dpi.png"


# =========================================================
# Major group order
# =========================================================

MAJOR_GROUP_ORDER = [
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
    "SARS_CoV_2": "SARS-CoV-2 consensus",
    "SARS_CoV": "SARS-CoV consensus",
    "MERS_CoV": "MERS-CoV consensus",
    "HCoV_HKU1": "HCoV-HKU1 consensus",
    "HCoV_OC43": "HCoV-OC43 consensus",
    "HCoV_229E": "HCoV-229E consensus",
    "HCoV_NL63": "HCoV-NL63 consensus",
    "Bat_CoV": "Bat-CoV consensus",
}


GROUP_COLORS = {
    "Wuhan": "#111111",
    "Alpha": "#D73027",
    "Beta": "#FC8D59",
    "Delta": "#7B3294",
    "Omicron_BA1": "#C51B7D",
    "Omicron_BA2": "#B2182B",
    "SARS_CoV_2": "#1F78B4",
    "SARS_CoV": "#33A02C",
    "MERS_CoV": "#6A3D9A",
    "HCoV_HKU1": "#00A6A6",
    "HCoV_OC43": "#1F78B4",
    "HCoV_229E": "#A65628",
    "HCoV_NL63": "#33A02C",
    "Bat_CoV": "#666666",
}


# =========================================================
# Region and annotations
# =========================================================

ALIGNMENT_START = 300
ALIGNMENT_END = 600

SARS_COV_2_ACE2_RESIDUES = [
    417, 446, 449, 453, 455, 456, 475, 486, 487,
    489, 493, 496, 498, 500, 501, 502, 505
]

SARS_COV_ACE2_ANALOG_RESIDUES = [
    426, 436, 442, 472, 473, 475, 479, 480, 487, 491
]

SECONDARY_ELEMENTS = [
    ("β1", 310, 318, "strand"),
    ("α1", 340, 350, "helix"),
    ("η1", 365, 372, "helix"),
    ("βc1", 375, 389, "strand"),
    ("α2", 405, 420, "helix"),
    ("βc2", 437, 449, "strand"),
    ("βc3", 455, 470, "strand"),
    ("η2", 475, 486, "helix"),
    ("α3", 488, 505, "helix"),
    ("βc4", 507, 523, "strand"),
    ("α1′", 525, 535, "helix"),
    ("β5", 540, 550, "strand"),
]

MOTIF_BOXES = [
    ("RBD", 319, 541, "#1A1A1A"),
    ("mAb1", 365, 420, "#5C4033"),
    ("mAb2/3", 455, 505, "#3B2F2F"),
    ("RBM core", 437, 505, "#2B2B2B"),
    ("ACE2 hot region", 486, 505, "#4A2C2A"),
]

CONSERVATION_CMAP = LinearSegmentedColormap.from_list(
    "conservation",
    ["#B2182B", "#FEE08B", "#1A9850"]
)

SEQUENCE_IDENTITY_COLOR = "#D7191C"


# =========================================================
# Classification utilities
# =========================================================

def normalize_text(text):
    text = str(text).lower()

    for ch in ["|", "-", ".", " ", "/", ":", ";", ",", "(", ")", "[", "]"]:
        text = text.replace(ch, "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def get_major_group_from_record_id(record_id):
    record_id = str(record_id)

    if "|" in record_id:
        first_part = record_id.split("|")[0].strip()
    else:
        first_part = record_id.strip()

    if first_part in MAJOR_GROUP_ORDER:
        return first_part

    text = normalize_text(first_part)

    # Wuhan reference
    if (
        "wuhan" in text
        or "refseq" in text
        or "yp_009724390" in text
        or "nc_045512" in text
    ):
        return "Wuhan"

    # SARS-CoV-2 variants
    if "alpha" in text or "b_1_1_7" in text:
        return "Alpha"

    if "beta" in text or "b_1_351" in text:
        return "Beta"

    if "delta" in text or "b_1_617_2" in text:
        return "Delta"

    # Omicron BA.1 detection
    # This catches labels such as:
    # Omicron_BA1, Omicron_BA_1, Omicron BA.1, BA.1, BA.1.1, B.1.1.529
    if (
        "omicron_ba1" in text
        or "omicron_ba_1" in text
        or "omicron_ba_1_1" in text
        or text == "ba1"
        or text.startswith("ba1_")
        or "_ba1" in text
        or "_ba1_" in text
        or text == "ba_1"
        or text.startswith("ba_1_")
        or "_ba_1" in text
        or "_ba_1_" in text
        or "b_1_1_529" in text
    ):
        return "Omicron_BA1"

    # Omicron BA.2 detection
    # This catches labels such as:
    # Omicron_BA2, Omicron_BA_2, Omicron BA.2, BA.2, BA.2.12.1
    if (
        "omicron_ba2" in text
        or "omicron_ba_2" in text
        or text == "ba2"
        or text.startswith("ba2_")
        or "_ba2" in text
        or "_ba2_" in text
        or text == "ba_2"
        or text.startswith("ba_2_")
        or "_ba_2" in text
        or "_ba_2_" in text
    ):
        return "Omicron_BA2"

    # Broader virus consensus groups
    if "sars_cov_2" in text or "sarscov2" in text:
        return "SARS_CoV_2"

    if "sars_cov" in text or "sarscov" in text:
        return "SARS_CoV"

    if "mers" in text:
        return "MERS_CoV"

    if "hku1" in text:
        return "HCoV_HKU1"

    if "oc43" in text:
        return "HCoV_OC43"

    if "229e" in text or "h229e" in text:
        return "HCoV_229E"

    if "nl63" in text:
        return "HCoV_NL63"

    if "bat" in text:
        return "Bat_CoV"

    return None


def find_wuhan_reference(alignment):
    for record in alignment:
        group = get_major_group_from_record_id(record.id)

        if group == "Wuhan":
            return record

    for record in alignment:
        group = get_major_group_from_record_id(record.id)

        if group == "SARS_CoV_2":
            return record

    raise ValueError("No Wuhan or SARS-CoV-2 reference sequence was found.")


# =========================================================
# Consensus sequence creation
# =========================================================

def build_consensus_sequence(records):
    seq_length = len(records[0].seq)
    consensus = []

    for i in range(seq_length):
        residues = []

        for record in records:
            aa = str(record.seq[i]).upper()

            if aa != "-":
                residues.append(aa)

        if residues:
            consensus.append(Counter(residues).most_common(1)[0][0])
        else:
            consensus.append("-")

    return "".join(consensus)


def create_consensus_records(alignment):
    grouped_records = defaultdict(list)

    reference_record = find_wuhan_reference(alignment)

    unmatched_records = []

    for record in alignment:
        group = get_major_group_from_record_id(record.id)

        if group is not None:
            grouped_records[group].append(record)
        else:
            unmatched_records.append(record.id)

    consensus_records = []

    print("\nConsensus group summary:")

    for group in MAJOR_GROUP_ORDER:
        records = grouped_records.get(group, [])

        if not records:
            print(f"  {DISPLAY_NAMES.get(group, group)}: n=0, skipped")
            continue

        if group == "Wuhan":
            consensus_seq = str(reference_record.seq).upper()
            description = f"Wuhan reference sequence from {reference_record.id}"
            n_records = 1
        else:
            consensus_seq = build_consensus_sequence(records)
            description = f"{DISPLAY_NAMES.get(group, group)} built from n={len(records)} aligned sequences"
            n_records = len(records)

        consensus_records.append(
            SeqRecord(
                Seq(consensus_seq),
                id=group,
                description=description
            )
        )

        print(f"  {DISPLAY_NAMES.get(group, group)}: n={n_records}")

    SeqIO.write(consensus_records, str(CONSENSUS_FASTA), "fasta")

    print("\nConsensus FASTA saved:")
    print(f"  {CONSENSUS_FASTA}")

    if unmatched_records:
        print("\nUnmatched record IDs:")
        for rid in unmatched_records[:30]:
            print(f"  {rid}")

        if len(unmatched_records) > 30:
            print(f"  ... and {len(unmatched_records) - 30} more unmatched records")

    return consensus_records


def remove_reference_gap_columns(consensus_records, reference_id="Wuhan"):
    reference_record = None

    for record in consensus_records:
        if record.id == reference_id:
            reference_record = record
            break

    if reference_record is None:
        raise ValueError("Wuhan reference was not found in consensus records.")

    reference_seq = str(reference_record.seq).upper()
    keep_columns = [i for i, aa in enumerate(reference_seq) if aa != "-"]

    cleaned_sequences = {}

    for record in consensus_records:
        seq = str(record.seq).upper()
        cleaned_sequences[record.id] = "".join(seq[i] for i in keep_columns if i < len(seq))

    return cleaned_sequences


def get_region_sequences(consensus_records, cleaned_sequences):
    region_sequences = {}

    for record in consensus_records:
        seq = cleaned_sequences.get(record.id)

        if seq is None:
            continue

        if len(seq) < ALIGNMENT_END:
            print(f"Warning: {record.id} shorter than {ALIGNMENT_END}. Skipped.")
            continue

        region_sequences[record.id] = seq[ALIGNMENT_START - 1:ALIGNMENT_END]

    return region_sequences


# =========================================================
# Analysis utilities
# =========================================================

def compute_conservation(region_sequences, sequence_ids, reference_id):
    reference_seq = region_sequences[reference_id]
    scores = []

    for i, ref_aa in enumerate(reference_seq):
        valid = 0
        same = 0

        for sid in sequence_ids:
            aa = region_sequences[sid][i]

            if aa == "-":
                continue

            valid += 1

            if aa == ref_aa:
                same += 1

        scores.append(same / valid if valid else 0)

    return scores


def compute_mutation_burden(region_sequences, sequence_ids, reference_id):
    reference_seq = region_sequences[reference_id]
    burden = {}

    for sid in sequence_ids:
        seq = region_sequences[sid]

        valid = 0
        mismatch = 0

        for ref_aa, aa in zip(reference_seq, seq):
            if ref_aa == "-" or aa == "-":
                continue

            valid += 1

            if aa != ref_aa:
                mismatch += 1

        burden[sid] = 100 * mismatch / valid if valid else 0

    return burden


# =========================================================
# Drawing utilities
# =========================================================

def residue_facecolor(reference_aa, query_aa, is_reference):
    if query_aa in ["-", "."]:
        return "#FFFFFF"

    if is_reference:
        return "#FFFFFF"

    if query_aa == reference_aa:
        return SEQUENCE_IDENTITY_COLOR

    return "#FFFFFF"


def residue_textcolor(reference_aa, query_aa, is_reference):
    if query_aa in ["-", "."]:
        return "#777777"

    if not is_reference and query_aa == reference_aa:
        return "white"

    if query_aa in ["D", "E"]:
        return "#D7191C"

    if query_aa in ["K", "R", "H"]:
        return "#2C7BB6"

    if query_aa in ["S", "T", "N", "Q"]:
        return "#C05A00"

    if query_aa in ["F", "W", "Y"]:
        return "#7B3294"

    if query_aa in ["C", "G", "P"]:
        return "#444444"

    return "#111111"


def draw_marker_square(ax, x, y, color="#B71C1C"):
    ax.add_patch(
        Rectangle(
            (x + 0.16, y),
            0.68,
            0.42,
            facecolor=color,
            edgecolor=color,
            linewidth=0.60,
            zorder=6
        )
    )


def draw_marker_triangle(ax, x, y, color="#2C3EBD"):
    ax.add_patch(
        Polygon(
            [
                (x + 0.50, y + 0.44),
                (x + 0.14, y),
                (x + 0.86, y),
            ],
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.60,
            zorder=6
        )
    )


def draw_secondary_structure(ax, block_start, block_end, y, x_scale=1.0):
    placed_labels = []
    label_offsets = [0.82, 1.28, 1.74, 2.20]

    for label, start, end, kind in SECONDARY_ELEMENTS:
        overlap_start = max(block_start, start)
        overlap_end = min(block_end, end)

        if overlap_start > overlap_end:
            continue

        x1 = (overlap_start - block_start) * x_scale
        x2 = (overlap_end - block_start + 1) * x_scale
        center_x = (x1 + x2) / 2

        if kind == "strand":
            ax.annotate(
                "",
                xy=(x2, y),
                xytext=(x1, y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#111111",
                    lw=2.1,
                    shrinkA=0,
                    shrinkB=0
                )
            )
        else:
            ax.plot([x1, x2], [y, y], color="#111111", linewidth=1.8)

            xx = x1 + 0.9 * x_scale

            while xx < x2 - 0.5 * x_scale:
                ax.text(
                    xx,
                    y + 0.08,
                    "∿",
                    fontsize=9.5,
                    ha="center",
                    va="center",
                    color="#555555"
                )
                xx += 2.5 * x_scale

        level = 0

        while level < len(label_offsets):
            clash = any(
                abs(center_x - old_x) < 8.0 * x_scale and level == old_level
                for old_x, old_level in placed_labels
            )

            if not clash:
                break

            level += 1

        if level >= len(label_offsets):
            level = len(label_offsets) - 1

        placed_labels.append((center_x, level))

        ax.text(
            center_x,
            y + label_offsets[level],
            label,
            fontsize=10.2,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="#111111"
        )


def draw_motif_boxes(ax, block_start, block_end, y_top, y_bottom, x_scale=1.0):
    for label, start, end, color in MOTIF_BOXES:
        overlap_start = max(block_start, start)
        overlap_end = min(block_end, end)

        if overlap_start > overlap_end:
            continue

        x1 = (overlap_start - block_start) * x_scale
        width = (overlap_end - overlap_start + 1) * x_scale

        ax.add_patch(
            Rectangle(
                (x1, y_bottom),
                width,
                y_top - y_bottom,
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                linestyle=(0, (1.2, 1.2)),
                zorder=4
            )
        )

        if block_start <= start <= block_end:
            ax.text(
                x1,
                y_bottom - 0.22,
                label,
                fontsize=8.9,
                color=color,
                fontweight="bold",
                ha="left",
                va="top"
            )


# =========================================================
# Main alignment block
# =========================================================

def draw_alignment_block(
    ax,
    region_sequences,
    sequence_ids,
    reference_id,
    block_start,
    block_end,
    x_scale=0.63,
    row_step=1.10,
    label_space_left=9.2,
    right_space=7.2,
    title_text=None,
):
    reference_seq = region_sequences[reference_id]
    conservation = compute_conservation(region_sequences, sequence_ids, reference_id)
    mutation_burden = compute_mutation_burden(region_sequences, sequence_ids, reference_id)

    n_sequences = len(sequence_ids)
    block_len = block_end - block_start + 1
    block_width = block_len * x_scale

    ax.axis("off")
    ax.set_xlim(-label_space_left, block_width + right_space)

    block_top_y = n_sequences * row_step + 5.0
    first_row_y = block_top_y - 1.30
    alignment_bottom_y = first_row_y - (n_sequences - 1) * row_step

    y_marker_square = alignment_bottom_y - 1.45
    y_marker_triangle = alignment_bottom_y - 2.38
    y_conservation = alignment_bottom_y - 3.55

    ax.set_ylim(y_conservation - 0.75, block_top_y + 3.05)

    if title_text is not None:
        ax.text(
            -label_space_left,
            block_top_y + 2.10,
            title_text,
            fontsize=13.2,
            fontweight="bold",
            ha="left",
            va="center",
            color="#222222"
        )

    draw_secondary_structure(
        ax,
        block_start,
        block_end,
        block_top_y + 1.55,
        x_scale=x_scale
    )

    tick_positions = [block_start]
    p = ((block_start + 24) // 25) * 25

    while p < block_end:
        tick_positions.append(p)
        p += 25

    if block_end not in tick_positions:
        tick_positions.append(block_end)

    for pos in tick_positions:
        x = (pos - block_start + 0.5) * x_scale

        ax.text(
            x,
            block_top_y - 0.12,
            str(pos),
            fontsize=7.6,
            rotation=90,
            ha="center",
            va="bottom",
            color="#333333"
        )

    for pos in SARS_COV_2_ACE2_RESIDUES:
        if block_start <= pos <= block_end:
            x = (pos - block_start + 0.5) * x_scale

            ax.text(
                x,
                block_top_y - 0.62,
                "*",
                fontsize=14,
                fontweight="bold",
                color="#B71C1C",
                ha="center",
                va="bottom"
            )

    for row_index, sid in enumerate(sequence_ids):
        y = first_row_y - row_index * row_step
        seq = region_sequences[sid]
        row_label = DISPLAY_NAMES.get(sid, sid)

        ax.text(
            -0.90,
            y + 0.5,
            row_label,
            fontsize=10.2,
            fontweight="bold",
            ha="right",
            va="center",
            color=GROUP_COLORS.get(sid, "#222222")
        )

        for i, pos in enumerate(range(block_start, block_end + 1)):
            seq_idx = pos - ALIGNMENT_START

            aa = seq[seq_idx] if seq_idx < len(seq) else "-"
            aa_show = "." if aa == "-" else aa

            ref_aa = reference_seq[seq_idx] if seq_idx < len(reference_seq) else "-"
            is_reference = sid == reference_id

            facecolor = residue_facecolor(ref_aa, aa, is_reference)
            textcolor = residue_textcolor(ref_aa, aa, is_reference)

            edgecolor = "#A7A7E8" if aa_show != "." else "#FFFFFF"
            linewidth = 0.54 if aa_show != "." else 0.0

            if pos in SARS_COV_2_ACE2_RESIDUES and sid == reference_id:
                edgecolor = "#B71C1C"
                linewidth = 1.25

            x0 = i * x_scale

            ax.add_patch(
                Rectangle(
                    (x0, y),
                    x_scale,
                    1.0,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    zorder=2
                )
            )

            ax.text(
                x0 + 0.5 * x_scale,
                y + 0.5,
                aa_show,
                fontsize=7.7,
                family="monospace",
                fontweight="bold",
                ha="center",
                va="center",
                color=textcolor,
                zorder=3
            )

        bar_x = block_width + 0.90
        burden_width = min(mutation_burden[sid] / 100 * 6.2, 6.2)

        ax.add_patch(
            Rectangle(
                (bar_x, y + 0.22),
                burden_width,
                0.56,
                facecolor=GROUP_COLORS.get(sid, "#222222"),
                edgecolor="#111111",
                linewidth=0.40,
                alpha=0.88
            )
        )

        if row_index == 0:
            ax.text(
                bar_x,
                y + 1.25,
                "Mismatch\nburden",
                fontsize=8.7,
                fontweight="bold",
                ha="left",
                va="bottom",
                color="#333333"
            )

    for pos in SARS_COV_2_ACE2_RESIDUES:
        if block_start <= pos <= block_end:
            draw_marker_square(
                ax,
                (pos - block_start) * x_scale + 0.22 * x_scale,
                y_marker_square,
                color="#B71C1C"
            )

    for pos in SARS_COV_ACE2_ANALOG_RESIDUES:
        if block_start <= pos <= block_end:
            draw_marker_triangle(
                ax,
                (pos - block_start) * x_scale + 0.16 * x_scale,
                y_marker_triangle,
                color="#2C3EBD"
            )

    for i, pos in enumerate(range(block_start, block_end + 1)):
        seq_idx = pos - ALIGNMENT_START
        score = conservation[seq_idx]
        color = CONSERVATION_CMAP(score)

        ax.add_patch(
            Rectangle(
                (i * x_scale, y_conservation),
                x_scale,
                0.70,
                facecolor=color,
                edgecolor="#FFFFFF",
                linewidth=0.22,
                zorder=2
            )
        )

    ax.text(
        -0.90,
        y_conservation + 0.35,
        "Conservation",
        fontsize=9.8,
        fontweight="bold",
        ha="right",
        va="center",
        color="#333333"
    )

    draw_motif_boxes(
        ax,
        block_start=block_start,
        block_end=block_end,
        y_top=block_top_y + 1.05,
        y_bottom=alignment_bottom_y + 0.02,
        x_scale=x_scale
    )


# =========================================================
# Final figure
# =========================================================

def draw_two_panel_sequence_diagram(region_sequences, sequence_ids, reference_id):
    fig = plt.figure(figsize=(18.6, 12.8), facecolor="white")

    fig.text(
        0.014,
        0.985,
        "C",
        fontsize=28,
        fontweight="bold",
        ha="left",
        va="top"
    )

    positions = [
        [0.055, 0.585, 0.895, 0.330],
        [0.055, 0.205, 0.895, 0.330],
    ]

    panel_blocks = [
        (300, 450),
        (450, 600),
    ]

    panel_titles = [
        "Spike RBD region 300-450",
        "Spike RBD region 450-600",
    ]

    for pos, block, title in zip(positions, panel_blocks, panel_titles):
        ax = fig.add_axes(pos)

        draw_alignment_block(
            ax,
            region_sequences,
            sequence_ids,
            reference_id,
            block_start=block[0],
            block_end=block[1],
            x_scale=0.63,
            row_step=1.10,
            label_space_left=9.2,
            right_space=7.2,
            title_text=title,
        )

    ax_leg = fig.add_axes([0.10, 0.055, 0.80, 0.080])
    ax_leg.axis("off")
    ax_leg.set_xlim(0, 100)
    ax_leg.set_ylim(0, 10)

    draw_marker_square(ax_leg, 1.8, 6.1, color="#B71C1C")

    ax_leg.text(
        3.3,
        6.35,
        "SARS-CoV-2 ACE2-contact residues",
        fontsize=10.6,
        fontweight="bold",
        ha="left",
        va="center"
    )

    draw_marker_triangle(ax_leg, 1.8, 3.2, color="#2C3EBD")

    ax_leg.text(
        3.3,
        3.45,
        "SARS-CoV ACE2-analog residues",
        fontsize=10.6,
        fontweight="bold",
        ha="left",
        va="center"
    )

    ax_leg.text(
        42.0,
        8.4,
        "Conservation heat map",
        fontsize=10.6,
        fontweight="bold",
        ha="left",
        va="center",
        color="#333333"
    )

    heat_x = 42.0
    heat_y = 4.9
    heat_w = 24.0
    heat_h = 1.2
    n_heat = 30

    for j in range(n_heat):
        x0 = heat_x + j * (heat_w / n_heat)

        ax_leg.add_patch(
            Rectangle(
                (x0, heat_y),
                heat_w / n_heat,
                heat_h,
                facecolor=CONSERVATION_CMAP(j / (n_heat - 1)),
                edgecolor="#FFFFFF",
                linewidth=0.25,
            )
        )

    ax_leg.text(
        heat_x,
        4.1,
        "Low",
        fontsize=9.6,
        ha="left",
        va="top",
        color="#333333"
    )

    ax_leg.text(
        heat_x + heat_w,
        4.1,
        "High",
        fontsize=9.6,
        ha="right",
        va="top",
        color="#333333"
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=900,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white"
    )

    plt.close(fig)

    print("\nSaved consensus sequence diagram:")
    print(f"  {OUTPUT_FIGURE}")


# =========================================================
# Main
# =========================================================

def main():
    if not ALIGNMENT_FILE.exists():
        raise FileNotFoundError(f"Alignment file not found:\n{ALIGNMENT_FILE}")

    alignment = AlignIO.read(str(ALIGNMENT_FILE), "fasta")

    print("\nLoaded updated aligned FASTA:")
    print(f"  {ALIGNMENT_FILE}")
    print(f"  Number of aligned records: {len(alignment)}")
    print(f"  Alignment length: {alignment.get_alignment_length()}")

    consensus_records = create_consensus_records(alignment)

    cleaned_sequences = remove_reference_gap_columns(
        consensus_records,
        reference_id="Wuhan"
    )

    region_sequences = get_region_sequences(
        consensus_records,
        cleaned_sequences
    )

    sequence_ids = [
        group for group in MAJOR_GROUP_ORDER
        if group in region_sequences
    ]

    if "Wuhan" not in sequence_ids:
        raise ValueError("Wuhan reference was not found in plotted sequences.")

    print("\nFinal plotting order:")
    for i, sid in enumerate(sequence_ids, start=1):
        print(f"  {i:02d}. {DISPLAY_NAMES.get(sid, sid)}")

    if "Omicron_BA1" not in sequence_ids:
        print("\nWarning: Omicron BA.1 is still not present in the final plot.")
        print("Possible reasons:")
        print("  1. No BA.1 sequence exists in the aligned FASTA.")
        print("  2. BA.1 sequence headers use an unexpected name.")
        print("  3. BA.1 sequence became shorter than residue 600 after cleaning.")

    if "Omicron_BA2" not in sequence_ids:
        print("\nWarning: Omicron BA.2 is still not present in the final plot.")
        print("Possible reasons:")
        print("  1. No BA.2 sequence exists in the aligned FASTA.")
        print("  2. BA.2 sequence headers use an unexpected name.")
        print("  3. BA.2 sequence became shorter than residue 600 after cleaning.")

    draw_two_panel_sequence_diagram(
        region_sequences=region_sequences,
        sequence_ids=sequence_ids,
        reference_id="Wuhan"
    )

    print("\nGenerated files:")
    print(f"  Figure: {OUTPUT_FIGURE}")
    print(f"  Consensus FASTA: {CONSENSUS_FASTA}")


if __name__ == "__main__":
    main()