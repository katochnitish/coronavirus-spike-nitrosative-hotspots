
# Script Name: figure3C_domainwise_mutation_burden.py
# Author: Code made by Nitish Katoch
#
# Description:
#   Quantifies and compares structural distribution patterns of mutational density 
#   and evolutionary burden clustered across specific functional protein domains.
#
# Plots Made:
#   - High-impact violin plots displaying distribution density curves of positional 
#     mutation burden within structural protein segments.
#
# Inputs:
#   - 02_CuratedRawData/combined_curated_aligned.fasta
#
# Calculations:
#   - Evaluates point mutation rates across distinct variant sequences.
#   - Maps position-wise metric distributions directly into specific sub-regions 
#     (NTD, RBD, FP, HR1, HR2, etc.).
#   - Normalizes domain mutation density ratios per 100 amino acids for uniform testing.
#
# Outputs:
#   - 03_Results/Figure3C_domainwise_mutation_burden_violin.png
#   - 03_Results/Figure3C_domainwise_mutation_burden_violin.csv
# ==============================================================================
from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Figure 3C
# Domain-wise mutation burden in SARS-CoV-2 spike variants
# Violin plot version
# =========================================================


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

ALIGNMENT_FILE = BASE_DIR / "02_CuratedRawData" / "combined_curated_aligned.fasta"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "Figure3C_domainwise_mutation_burden_violin.png"
OUTPUT_CSV = OUTPUT_DIR / "Figure3C_domainwise_mutation_burden_violin.csv"


# =========================================================
# Selected SARS-CoV-2 groups
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
# Wuhan-Hu-1 spike domain definitions
# =========================================================

SPIKE_LENGTH = 1273

DOMAINS = [
    ("NTD", 14, 305, "#B2182B"),
    ("RBD/RBM", 319, 541, "#C43C39"),
    ("S1/S2", 681, 686, "#A6611A"),
    ("FP", 816, 833, "#8C2D04"),
    ("HR1", 912, 984, "#D6604D"),
    ("HR2", 1163, 1213, "#7F2704"),
    ("TM/CT", 1214, 1273, "#6B6B6B"),
]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
INVALID_AA = set(["?", ".", "X", "B", "Z", "J", "U", "O", "*"])


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
# Alignment cleaning
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
# Metrics
# =========================================================

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


def domain_positions(start, end):
    return list(range(start, end + 1))


def calculate_domain_metrics(cleaned_sequences):
    reference = cleaned_sequences["Wuhan"][0]

    variant_groups = [
        group for group in GROUP_ORDER
        if group != "Wuhan" and group in cleaned_sequences and len(cleaned_sequences[group]) > 0
    ]

    rows = []

    for domain_name, start, end, domain_color in DOMAINS:
        positions = domain_positions(start, end)
        domain_length = len(positions)

        for group in variant_groups:
            seqs = cleaned_sequences[group]

            substitution_count_total = 0
            deletion_count_total = 0
            valid_comparisons_total = 0
            mutated_positions = set()

            for seq in seqs:
                for pos in positions:
                    idx = pos - 1

                    if idx >= len(reference) or idx >= len(seq):
                        continue

                    ref_aa = reference[idx]
                    query_aa = seq[idx]

                    if ref_aa in INVALID_AA or ref_aa == "-":
                        continue

                    if query_aa in INVALID_AA:
                        continue

                    valid_comparisons_total += 1

                    if query_aa == "-":
                        deletion_count_total += 1
                        mutated_positions.add(pos)
                    elif query_aa != ref_aa:
                        substitution_count_total += 1
                        mutated_positions.add(pos)

            mutation_count_total = substitution_count_total + deletion_count_total

            n_sequences = len(seqs)

            mean_mutations_per_sequence = (
                mutation_count_total / n_sequences
                if n_sequences else np.nan
            )

            mutation_density_per_100aa = (
                mean_mutations_per_sequence / domain_length * 100.0
                if domain_length > 0 else np.nan
            )

            mutated_position_fraction = (
                len(mutated_positions) / domain_length * 100.0
                if domain_length > 0 else np.nan
            )

            rows.append(
                {
                    "Group": group,
                    "Display name": DISPLAY_NAMES[group],
                    "Domain": domain_name,
                    "Domain start": start,
                    "Domain end": end,
                    "Domain length": domain_length,
                    "Sequences in group": n_sequences,
                    "Substitution count": substitution_count_total,
                    "Deletion count": deletion_count_total,
                    "Total mutation events": mutation_count_total,
                    "Mean mutations per sequence": mean_mutations_per_sequence,
                    "Mutation density per 100 aa": mutation_density_per_100aa,
                    "Mutated positions": len(mutated_positions),
                    "Mutated position fraction (%)": mutated_position_fraction,
                }
            )

    entropy_by_domain = {}

    for domain_name, start, end, _ in DOMAINS:
        entropies = []

        for pos in domain_positions(start, end):
            idx = pos - 1

            if idx >= len(reference):
                continue

            ref_aa = reference[idx]

            if ref_aa in INVALID_AA or ref_aa == "-":
                continue

            residues = []

            for group in variant_groups:
                for seq in cleaned_sequences[group]:
                    if idx >= len(seq):
                        continue

                    aa = seq[idx]

                    if aa in INVALID_AA or aa == "-":
                        continue

                    residues.append(aa)

            ent = shannon_entropy(residues)

            if not np.isnan(ent):
                entropies.append(ent)

        mean_entropy = np.mean(entropies) if entropies else np.nan
        max_entropy = np.max(entropies) if entropies else np.nan

        entropy_by_domain[domain_name] = {
            "Mean entropy": mean_entropy,
            "Max entropy": max_entropy,
            "Entropy values": entropies,
        }

    for row in rows:
        entropy_info = entropy_by_domain.get(row["Domain"], {})
        row["Mean entropy"] = entropy_info.get("Mean entropy", np.nan)
        row["Max entropy"] = entropy_info.get("Max entropy", np.nan)

    return rows, entropy_by_domain


def calculate_domain_summary(rows, entropy_by_domain):
    summary_rows = []

    for domain_name, start, end, domain_color in DOMAINS:
        domain_rows = [row for row in rows if row["Domain"] == domain_name]

        if not domain_rows:
            continue

        summary_rows.append(
            {
                "Domain": domain_name,
                "Domain start": start,
                "Domain end": end,
                "Domain length": end - start + 1,
                "Mean mutation density per 100 aa": np.mean(
                    [row["Mutation density per 100 aa"] for row in domain_rows]
                ),
                "Mean mutations per sequence": np.mean(
                    [row["Mean mutations per sequence"] for row in domain_rows]
                ),
                "Mean mutated position fraction (%)": np.mean(
                    [row["Mutated position fraction (%)"] for row in domain_rows]
                ),
                "Mean entropy": entropy_by_domain[domain_name]["Mean entropy"],
                "Max entropy": entropy_by_domain[domain_name]["Max entropy"],
                "Color": domain_color,
            }
        )

    return summary_rows


def save_table(rows, summary_rows):
    fieldnames = [
        "Group",
        "Display name",
        "Domain",
        "Domain start",
        "Domain end",
        "Domain length",
        "Sequences in group",
        "Substitution count",
        "Deletion count",
        "Total mutation events",
        "Mean mutations per sequence",
        "Mutation density per 100 aa",
        "Mutated positions",
        "Mutated position fraction (%)",
        "Mean entropy",
        "Max entropy",
    ]

    summary_fieldnames = [
        "Domain",
        "Domain start",
        "Domain end",
        "Domain length",
        "Mean mutation density per 100 aa",
        "Mean mutations per sequence",
        "Mean mutated position fraction (%)",
        "Mean entropy",
        "Max entropy",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Table"] + fieldnames)
        writer.writeheader()

        for row in rows:
            out = {"Table": "Variant-domain"}
            out.update(row)

            for key in [
                "Mean mutations per sequence",
                "Mutation density per 100 aa",
                "Mutated position fraction (%)",
                "Mean entropy",
                "Max entropy",
            ]:
                value = out[key]
                out[key] = "NA" if np.isnan(value) else f"{value:.6f}"

            writer.writerow(out)

        writer.writerow({})

        writer = csv.DictWriter(f, fieldnames=["Table"] + summary_fieldnames)
        writer.writeheader()

        for row in summary_rows:
            out = {"Table": "Domain-summary"}

            for key in summary_fieldnames:
                value = row[key]
                if isinstance(value, float):
                    out[key] = "NA" if np.isnan(value) else f"{value:.6f}"
                else:
                    out[key] = value

            writer.writerow(out)


# =========================================================
# Violin plotting
# =========================================================

def clean_values(values):
    values = [float(v) for v in values if np.isfinite(v)]

    if len(values) == 0:
        return [0.0]

    if len(values) == 1:
        return [values[0], values[0] + 1e-6]

    return values


def make_violin_axis(ax, data, domains, colors, ylabel):
    positions = np.arange(1, len(domains) + 1)

    violin_parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.58,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )

    for body, color in zip(violin_parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_linewidth(0.9)
        body.set_alpha(0.78)

    for key in ["cmeans", "cmedians"]:
        if key in violin_parts:
            violin_parts[key].set_color("#111111")
            violin_parts[key].set_linewidth(1.5)

    rng = np.random.default_rng(42)

    for i, values in enumerate(data):
        jitter = rng.normal(0, 0.030, size=len(values))
        ax.scatter(
            np.full(len(values), positions[i]) + jitter,
            values,
            s=28,
            color="#111111",
            alpha=0.62,
            zorder=4,
            linewidth=0,
        )

    ax.set_ylabel(
        ylabel,
        fontsize=12.0,
        fontweight="bold",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        domains,
        fontsize=10.5,
        fontweight="bold",
        rotation=0,
        ha="center",
    )

    ax.tick_params(axis="y", labelsize=10.5)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.grid(axis="y", color="#E0E0E0", lw=0.7, alpha=0.8)

    # Full box around each panel
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)


def plot_figure(rows, summary_rows, entropy_by_domain):
    domains = [row["Domain"] for row in summary_rows]
    colors = [row["Color"] for row in summary_rows]

    density_data = []
    mean_mutation_data = []
    entropy_data = []

    for domain in domains:
        domain_rows = [row for row in rows if row["Domain"] == domain]

        density_values = [
            row["Mutation density per 100 aa"]
            for row in domain_rows
        ]

        mean_mutation_values = [
            row["Mean mutations per sequence"]
            for row in domain_rows
        ]

        entropy_values = entropy_by_domain[domain]["Entropy values"]

        density_data.append(clean_values(density_values))
        mean_mutation_data.append(clean_values(mean_mutation_values))
        entropy_data.append(clean_values(entropy_values))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(21.5, 4.1),
        gridspec_kw={"wspace": 0.30},
        facecolor="white",
    )

    ax1, ax2, ax3 = axes

    make_violin_axis(
        ax1,
        density_data,
        domains,
        colors,
        "Mutation density\nper 100 aa",
    )

    make_violin_axis(
        ax2,
        mean_mutation_data,
        domains,
        colors,
        "Mean mutations\nper sequence",
    )

    make_violin_axis(
        ax3,
        entropy_data,
        domains,
        colors,
        "Shannon entropy",
    )

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        if ymax > 0:
            ax.set_ylim(0, ymax * 1.05)

    plt.tight_layout(rect=[0.025, 0.08, 0.995, 0.98])

    fig.savefig(
        OUTPUT_PNG,
        dpi=900,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print("\nSaved Figure 3C violin plot:")
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

    rows, entropy_by_domain = calculate_domain_metrics(cleaned_sequences)
    summary_rows = calculate_domain_summary(rows, entropy_by_domain)

    save_table(rows, summary_rows)
    plot_figure(rows, summary_rows, entropy_by_domain)

    print("\nSaved domain-wise table:")
    print(f"  {OUTPUT_CSV}")

    print("\nDomain-wise summary:")
    for row in summary_rows:
        print(
            f"  {row['Domain']}: "
            f"density={row['Mean mutation density per 100 aa']:.3f}, "
            f"mean mutations/seq={row['Mean mutations per sequence']:.3f}, "
            f"mean entropy={row['Mean entropy']:.4f}"
        )


if __name__ == "__main__":
    main()