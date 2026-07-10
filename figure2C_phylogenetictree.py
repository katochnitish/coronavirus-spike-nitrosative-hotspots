
# Script Name: figure2C_phylogenetictree.py
# Author: Code made by Nitish Katoch
#
# Description:
#   Constructs and visualizes evolutionary relationships across the broader 
#   Coronaviridae family along with focused sub-analyses of circulating 
#   SARS-CoV-2 lineages.
#
# Plots Made:
#   - Global Coronaviridae consensus phylogenetic tree with comprehensive legends.
#   - Focused SARS-CoV-2 variant zoom tree highlighting specific sub-lineages.
#   - Combined dual-panel publication layout grouping both trees into a single canvas.
#
# Inputs:
#   - 02_CuratedRawData/combined_curated_aligned.fasta
#
# Calculations:
#   - Groups matching fasta inputs to reconstruct unified consensus sequences.
#   - Calculates evolutionary distance matrices via Biopython's DistanceCalculator.
#   - Builds neighbor-joining phylogenetic trees with custom branch scaling and 
#     categorical color styling.
#
# Outputs:
#   - 03_Results/Figure_B_AllVirus_Consensus_Tree_WithLegend.png
#   - 03_Results/Figure_B_SARSCoV2_Variant_Consensus_ZoomTree_WithLegend.png
#   - 03_Results/Figure_B_AllTrees_Consensus_Together_WithLegend.png
#   - 04_RepresentativeSequences/consensus_sequences_for_phylogenetic_tree.fasta
# ==============================================================================
from pathlib import Path
from collections import Counter, defaultdict
import copy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

ALIGNMENT_FILE = BASE_DIR / "02_CuratedRawData" / "combined_curated_aligned.fasta"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_DIR = BASE_DIR / "04_RepresentativeSequences"
REPRESENTATIVE_DIR.mkdir(parents=True, exist_ok=True)

CONSENSUS_FASTA = REPRESENTATIVE_DIR / "consensus_sequences_for_phylogenetic_tree.fasta"

OUTPUT_ALL_TREE = OUTPUT_DIR / "Figure_B_AllVirus_Consensus_Tree_WithLegend.png"
OUTPUT_VARIANT_TREE = OUTPUT_DIR / "Figure_B_SARSCoV2_Variant_Consensus_ZoomTree_WithLegend.png"
OUTPUT_TOGETHER = OUTPUT_DIR / "Figure_B_AllTrees_Consensus_Together_WithLegend.png"


# =============================================================================
# Group settings
# =============================================================================

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

VARIANT_ORDER = [
    "Wuhan",
    "Alpha",
    "Beta",
    "Delta",
    "Omicron_BA1",
    "Omicron_BA2",
    "SARS_CoV_2",
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

SHORT_DISPLAY_NAMES = {
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


# =============================================================================
# Utility functions
# =============================================================================

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

    if first_part in GROUP_ORDER:
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
    # Handles labels such as:
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
    # Handles labels such as:
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

    raise ValueError("Wuhan reference or SARS-CoV-2 reference was not found.")


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


def create_consensus_records(alignment, order):
    grouped_records = defaultdict(list)
    unmatched_records = []
    reference_record = find_wuhan_reference(alignment)

    for record in alignment:
        group = get_major_group_from_record_id(record.id)

        if group is not None:
            grouped_records[group].append(record)
        else:
            unmatched_records.append(record.id)

    consensus_records = []

    print("\nConsensus group summary:")

    for group in order:
        records = grouped_records.get(group, [])

        if not records:
            print(f"  {SHORT_DISPLAY_NAMES.get(group, group)}: n=0, skipped")
            continue

        if group == "Wuhan":
            consensus_seq = str(reference_record.seq).upper()
            description = f"Wuhan reference sequence from {reference_record.id}"
            n_records = 1
        else:
            consensus_seq = build_consensus_sequence(records)
            description = f"{SHORT_DISPLAY_NAMES.get(group, group)} consensus from n={len(records)}"
            n_records = len(records)

        consensus_records.append(
            SeqRecord(
                Seq(consensus_seq),
                id=group,
                description=description,
            )
        )

        print(f"  {SHORT_DISPLAY_NAMES.get(group, group)}: n={n_records}")

    if unmatched_records:
        print("\nUnmatched record IDs:")
        for rid in unmatched_records[:40]:
            print(f"  {rid}")

        if len(unmatched_records) > 40:
            print(f"  ... and {len(unmatched_records) - 40} more unmatched records")

    return consensus_records


def remove_reference_gap_columns(records, reference_id="Wuhan"):
    reference_record = None

    for record in records:
        if record.id == reference_id:
            reference_record = record
            break

    if reference_record is None:
        raise ValueError("Wuhan reference was not found in consensus records.")

    reference_seq = str(reference_record.seq).upper()
    keep_columns = [i for i, aa in enumerate(reference_seq) if aa != "-"]

    cleaned_records = []

    for record in records:
        seq = str(record.seq).upper()
        cleaned_seq = "".join(seq[i] for i in keep_columns if i < len(seq))

        cleaned_records.append(
            SeqRecord(
                Seq(cleaned_seq),
                id=record.id,
                description=record.description,
            )
        )

    return cleaned_records


def build_tree_from_records(records):
    if len(records) < 2:
        raise ValueError("At least two records are required to build a tree.")

    alignment = MultipleSeqAlignment(records)

    calculator = DistanceCalculator("identity")
    constructor = DistanceTreeConstructor(calculator, "nj")

    tree = constructor.build_tree(alignment)

    return tree


def group_id_from_name(name):
    name = str(name)

    if name in GROUP_ORDER:
        return name

    return get_major_group_from_record_id(name) or "Unknown"


def color_from_name(name):
    group_id = group_id_from_name(name)
    return GROUP_COLORS.get(group_id, "#222222")


# =============================================================================
# Tree layout
# =============================================================================

def reorder_tree_by_group(tree, order):
    order_map = {group: index for index, group in enumerate(order)}

    def sort_key(clade):
        if clade.is_terminal():
            return order_map.get(group_id_from_name(clade.name), 999)

        terminal_orders = [
            order_map.get(group_id_from_name(terminal.name), 999)
            for terminal in clade.get_terminals()
        ]

        return min(terminal_orders) if terminal_orders else 999

    for clade in tree.find_clades(order="postorder"):
        if clade.clades:
            clade.clades.sort(key=sort_key)


def root_tree_with_wuhan(tree):
    wuhan_terminal = next(
        (
            terminal
            for terminal in tree.get_terminals()
            if group_id_from_name(terminal.name) == "Wuhan"
        ),
        None,
    )

    if wuhan_terminal is not None:
        tree.root_with_outgroup(wuhan_terminal)

    return tree


def compute_x_positions(tree):
    x_pos = {}

    def walk(clade, x):
        branch_length = clade.branch_length if clade.branch_length is not None else 0.0
        x_pos[clade] = x + branch_length

        for child in clade.clades:
            walk(child, x + branch_length)

    walk(tree.root, 0.0)

    return x_pos


def compute_y_positions(tree, y_gap=1.0):
    y_pos = {}

    for index, terminal in enumerate(tree.get_terminals()):
        y_pos[terminal] = index * y_gap

    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            continue

        child_ys = [y_pos[child] for child in clade.clades if child in y_pos]

        if child_ys:
            y_pos[clade] = sum(child_ys) / len(child_ys)

    return y_pos


def rescale_x_positions(x_pos, target_max):
    max_x = max(x_pos.values()) if x_pos else 1.0

    if max_x <= 0:
        return x_pos

    scale_factor = target_max / max_x

    return {
        clade: value * scale_factor
        for clade, value in x_pos.items()
    }


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_tree_clade(ax, clade, x_pos, y_pos, terminal_lw=3.2, internal_lw=2.8):
    x_here = x_pos[clade]

    if clade.clades:
        child_ys = [y_pos[child] for child in clade.clades]

        ax.plot(
            [x_here, x_here],
            [min(child_ys), max(child_ys)],
            color="#151515",
            linewidth=internal_lw,
            solid_capstyle="round",
            zorder=1,
        )

        for child in clade.clades:
            x_child = x_pos[child]
            y_child = y_pos[child]

            line_color = color_from_name(child.name) if child.is_terminal() else "#151515"
            line_width = terminal_lw if child.is_terminal() else internal_lw

            ax.plot(
                [x_here, x_child],
                [y_child, y_child],
                color=line_color,
                linewidth=line_width,
                solid_capstyle="round",
                zorder=2,
            )

            draw_tree_clade(ax, child, x_pos, y_pos, terminal_lw, internal_lw)


def draw_terminal_labels(
    ax,
    tree,
    x_pos,
    y_pos,
    label_offset,
    fontsize_normal,
    marker_size=50,
):
    for terminal in tree.get_terminals():
        group_id = group_id_from_name(terminal.name)
        label = SHORT_DISPLAY_NAMES.get(group_id, terminal.name)

        x = x_pos[terminal]
        y = y_pos[terminal]
        color = GROUP_COLORS.get(group_id, "#222222")

        ax.scatter(
            x,
            y,
            s=marker_size,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

        ax.text(
            x + label_offset,
            y,
            label,
            fontsize=fontsize_normal,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            zorder=5,
        )


def style_tree_axis(
    ax,
    xlim,
    xticks,
    xtick_labels,
    xlabel,
    fontsize_axis=13,
    fontsize_label=15,
):
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel(
        xlabel,
        fontsize=fontsize_label,
        fontweight="bold",
        labelpad=8,
    )

    ax.set_yticks([])

    ax.tick_params(
        axis="x",
        labelsize=fontsize_axis,
        width=2.0,
        length=6,
        direction="out",
        pad=6,
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=1.0,
        alpha=0.30,
    )

    ax.spines["bottom"].set_linewidth(2.2)

    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)


def build_legend_items(group_ids):
    legend_items = []

    for group in GROUP_ORDER:
        if group not in group_ids:
            continue

        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=SHORT_DISPLAY_NAMES.get(group, group),
                markerfacecolor=GROUP_COLORS.get(group, "#222222"),
                markeredgecolor="white",
                markersize=8,
            )
        )

    return legend_items


def add_axis_legend(ax, group_ids, ncol=3, fontsize=8.5):
    legend_items = build_legend_items(group_ids)

    ax.legend(
        handles=legend_items,
        loc="upper right",
        bbox_to_anchor=(1.00, 1.00),
        ncol=ncol,
        frameon=True,
        fontsize=fontsize,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.95,
        columnspacing=0.65,
        handletextpad=0.35,
        borderpad=0.50,
        labelspacing=0.30,
        markerscale=1.0,
    )


def add_shared_legend(fig, group_ids):
    legend_items = build_legend_items(group_ids)

    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=7,
        frameon=True,
        fontsize=10,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.95,
        columnspacing=0.80,
        handletextpad=0.35,
        borderpad=0.55,
        labelspacing=0.35,
        markerscale=1.0,
    )


# =============================================================================
# Tree drawing functions
# =============================================================================

def draw_all_virus_tree(
    ax,
    tree,
    group_ids,
    show_legend=True,
):
    tree_to_draw = copy.deepcopy(tree)

    root_tree_with_wuhan(tree_to_draw)
    reorder_tree_by_group(tree_to_draw, GROUP_ORDER)

    x_pos_raw = compute_x_positions(tree_to_draw)
    x_pos = rescale_x_positions(x_pos_raw, target_max=5.5)

    y_pos = compute_y_positions(tree_to_draw, y_gap=1.0)
    max_y = max(y_pos.values()) if y_pos else 1.0

    draw_tree_clade(
        ax,
        tree_to_draw.root,
        x_pos,
        y_pos,
        terminal_lw=3.1,
        internal_lw=2.7,
    )

    draw_terminal_labels(
        ax,
        tree_to_draw,
        x_pos,
        y_pos,
        label_offset=0.07,
        fontsize_normal=12,
        marker_size=38,
    )

    ax.set_ylim(-0.8, max_y + 1.2)
    ax.invert_yaxis()

    style_tree_axis(
        ax,
        xlim=(-0.35, 6.3),
        xticks=[0, 1, 2, 3, 4, 5, 5.5],
        xtick_labels=["0", "1", "2", "3", "4", "5", "5.5"],
        xlabel="Evolutionary divergence",
        fontsize_axis=12,
        fontsize_label=15,
    )

    if show_legend:
        add_axis_legend(
            ax,
            group_ids,
            ncol=3,
            fontsize=8.2,
        )


def draw_variant_zoom_tree(
    ax,
    tree,
    group_ids,
    show_legend=True,
):
    tree_to_draw = copy.deepcopy(tree)

    root_tree_with_wuhan(tree_to_draw)
    reorder_tree_by_group(tree_to_draw, VARIANT_ORDER)

    x_pos_raw = compute_x_positions(tree_to_draw)
    x_pos = rescale_x_positions(x_pos_raw, target_max=0.00355)

    y_pos = compute_y_positions(tree_to_draw, y_gap=1.0)
    max_y = max(y_pos.values()) if y_pos else 1.0

    draw_tree_clade(
        ax,
        tree_to_draw.root,
        x_pos,
        y_pos,
        terminal_lw=3.5,
        internal_lw=3.0,
    )

    draw_terminal_labels(
        ax,
        tree_to_draw,
        x_pos,
        y_pos,
        label_offset=0.00008,
        fontsize_normal=18,
        marker_size=70,
    )

    ax.set_ylim(-0.7, max_y + 0.7)
    ax.invert_yaxis()

    style_tree_axis(
        ax,
        xlim=(-0.00013, 0.00385),
        xticks=[
            0.0000,
            0.0005,
            0.0010,
            0.0015,
            0.0020,
            0.0025,
            0.0030,
            0.0035,
        ],
        xtick_labels=[
            "0.0000",
            "0.0005",
            "0.0010",
            "0.0015",
            "0.0020",
            "0.0025",
            "0.0030",
            "0.0035",
        ],
        xlabel="Evolutionary divergence from Wuhan reference",
        fontsize_axis=15,
        fontsize_label=18,
    )

    if show_legend:
        add_axis_legend(
            ax,
            group_ids,
            ncol=2,
            fontsize=9.5,
        )


# =============================================================================
# Main
# =============================================================================

def main():
    if not ALIGNMENT_FILE.exists():
        raise FileNotFoundError(
            f"Alignment file was not found:\n{ALIGNMENT_FILE}"
        )

    alignment = AlignIO.read(str(ALIGNMENT_FILE), "fasta")

    print("\nLoaded alignment:")
    print(f"  {ALIGNMENT_FILE}")
    print(f"  Records: {len(alignment)}")
    print(f"  Alignment length: {alignment.get_alignment_length()}")

    all_consensus_records = create_consensus_records(
        alignment,
        GROUP_ORDER,
    )

    variant_consensus_records = create_consensus_records(
        alignment,
        VARIANT_ORDER,
    )

    all_cleaned_records = remove_reference_gap_columns(
        all_consensus_records,
        reference_id="Wuhan",
    )

    variant_cleaned_records = remove_reference_gap_columns(
        variant_consensus_records,
        reference_id="Wuhan",
    )

    SeqIO.write(all_cleaned_records, str(CONSENSUS_FASTA), "fasta")

    print("\nConsensus FASTA saved:")
    print(f"  {CONSENSUS_FASTA}")

    all_group_ids = {record.id for record in all_cleaned_records}
    variant_group_ids = {record.id for record in variant_cleaned_records}

    print("\nFinal all-virus tree groups:")
    for i, group in enumerate(GROUP_ORDER, start=1):
        if group in all_group_ids:
            print(f"  {i:02d}. {SHORT_DISPLAY_NAMES.get(group, group)}")

    print("\nFinal SARS-CoV-2 variant tree groups:")
    for i, group in enumerate(VARIANT_ORDER, start=1):
        if group in variant_group_ids:
            print(f"  {i:02d}. {SHORT_DISPLAY_NAMES.get(group, group)}")

    if "Omicron_BA1" not in variant_group_ids:
        print("\nWarning: Omicron BA.1 is still not present in the variant tree.")
        print("Possible reasons:")
        print("  1. No BA.1 sequence exists in the aligned FASTA.")
        print("  2. BA.1 sequence headers use an unexpected name.")
        print("  3. BA.1 sequence was removed during earlier curation.")

    if "Omicron_BA2" not in variant_group_ids:
        print("\nWarning: Omicron BA.2 is still not present in the variant tree.")
        print("Possible reasons:")
        print("  1. No BA.2 sequence exists in the aligned FASTA.")
        print("  2. BA.2 sequence headers use an unexpected name.")
        print("  3. BA.2 sequence was removed during earlier curation.")

    all_tree = build_tree_from_records(all_cleaned_records)
    variant_tree = build_tree_from_records(variant_cleaned_records)

    # -------------------------------------------------------------------------
    # Figure 1: all virus tree
    # -------------------------------------------------------------------------

    fig1, ax1 = plt.subplots(
        figsize=(16.0, 4.3),
        facecolor="white",
    )

    draw_all_virus_tree(
        ax1,
        all_tree,
        all_group_ids,
        show_legend=True,
    )

    fig1.savefig(
        OUTPUT_ALL_TREE,
        dpi=900,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: SARS-CoV-2 variant zoom tree
    # -------------------------------------------------------------------------

    fig2, ax2 = plt.subplots(
        figsize=(16.0, 4.3),
        facecolor="white",
    )

    draw_variant_zoom_tree(
        ax2,
        variant_tree,
        variant_group_ids,
        show_legend=True,
    )

    fig2.savefig(
        OUTPUT_VARIANT_TREE,
        dpi=900,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Figure 3: combined trees
    # -------------------------------------------------------------------------

    fig3, axes = plt.subplots(
        2,
        1,
        figsize=(16.0, 8.8),
        facecolor="white",
        gridspec_kw={
            "height_ratios": [1.15, 1.0],
            "hspace": 0.42,
        },
    )

    draw_all_virus_tree(
        axes[0],
        all_tree,
        all_group_ids,
        show_legend=False,
    )

    draw_variant_zoom_tree(
        axes[1],
        variant_tree,
        variant_group_ids,
        show_legend=False,
    )

    add_shared_legend(
        fig3,
        group_ids=all_group_ids,
    )

    fig3.subplots_adjust(bottom=0.16)

    fig3.savefig(
        OUTPUT_TOGETHER,
        dpi=900,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig3)

    print("\nSaved figures:")
    print(f"  {OUTPUT_ALL_TREE}")
    print(f"  {OUTPUT_VARIANT_TREE}")
    print(f"  {OUTPUT_TOGETHER}")


if __name__ == "__main__":
    main()