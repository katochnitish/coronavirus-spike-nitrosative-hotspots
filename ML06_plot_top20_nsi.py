from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "09_AI_NitroSpike" / "nsi_residue_level_ranking.csv"

OUTPUT_DIR = BASE_DIR / "10_results_AI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "top20_nsi_residue_ranking_with_components.png"

TOP_N = 20


# ==========================================================
# Helper
# ==========================================================

def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of these columns were found: {candidates}")


# ==========================================================
# Main plotting function
# ==========================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    residue_col = find_column(df, ["Residue_Label", "Residue", "Candidate_Residue"])
    mean_col = find_column(df, ["Mean_NSI", "Mean_NSI_Score", "NSI_Mean"])
    sd_col = find_column(df, ["SD_NSI", "NSI_SD", "Std_NSI", "NSI_Std"])
    n_col = find_column(df, ["Variant_Record_Count", "N_Variants", "Variant_Count", "Record_Count"])

    component_map = {
        "Solvent\naccessibility\nscore": find_column(
            df,
            ["Solvent_Accessibility_Score", "Accessibility_Component_mean", "Accessibility_Component"]
        ),
        "Local chemical\nmicroenvironment\nscore": find_column(
            df,
            ["Local_Microenvironment_Score", "Chemistry_Component_mean", "Chemistry_Component"]
        ),
        "Spike functional\nrelevance\nscore": find_column(
            df,
            ["Spike_Functional_Relevance_Score", "Functional_Context_Component_mean", "Functional_Context_Component"]
        ),
        "Evolutionary\nconservation\nscore": find_column(
            df,
            ["Evolutionary_Conservation_Score", "Evolution_Component_mean", "Evolution_Component"]
        ),
    }

    top = df.sort_values(mean_col, ascending=False).head(TOP_N).copy()
    top = top.iloc[::-1]

    sns.set_theme(style="white", context="paper")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 500,
        "axes.linewidth": 1.0,
    })

    fig = plt.figure(figsize=(14.2, 9.6))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 1.35],
        wspace=0.30
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ======================================================
    # Panel A: Bar plot with error bars
    # ======================================================

    y = np.arange(len(top))
    residues = top[residue_col].astype(str).values
    mean_vals = top[mean_col].astype(float).values
    sd_vals = top[sd_col].astype(float).fillna(0).values
    n_vals = top[n_col].astype(int).values

    colors = sns.color_palette("Blues", n_colors=len(top) + 5)[5:]

    bars = ax1.barh(
        y,
        mean_vals,
        xerr=sd_vals,
        color=colors,
        edgecolor="black",
        linewidth=0.55,
        height=0.68,
        error_kw={
            "elinewidth": 0.85,
            "capsize": 2.7,
            "capthick": 0.85,
            "ecolor": "black"
        }
    )

    ax1.set_yticks(y)
    ax1.set_yticklabels(residues)
    ax1.set_xlabel("Mean NSI score, ± SD")
    ax1.set_ylabel("Residue")
    ax1.set_title("Residue-level NSI ranking", fontweight="bold", pad=12)

    xmax = max(mean_vals + sd_vals) * 1.32
    ax1.set_xlim(0, xmax)

    ax1.grid(axis="x", color="0.88", linewidth=0.7)
    ax1.set_axisbelow(True)

    # Keep all borders to make the bar plot appear inside a box.
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # Add value labels with more distance from bar and error bar.
    label_offset = xmax * 0.035
    for i, (m, sd, n) in enumerate(zip(mean_vals, sd_vals, n_vals)):
        x_text = m + sd + label_offset
        ax1.text(
            x_text,
            i,
            f"{m:.3f}  (n={n})",
            va="center",
            ha="left",
            fontsize=8.8,
            color="black"
        )

    # Small legend-like note below Panel A.
    ax1.plot([], [], color="black", marker="_", linestyle="-", label="± 1 SD across variants")
    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, -0.18),
        frameon=True,
        edgecolor="0.4",
        fontsize=8.8
    )

    # ======================================================
    # Panel B: Heatmap of component scores
    # ======================================================

    heat = top[[component_map[k] for k in component_map]].astype(float)
    heat.columns = list(component_map.keys())
    heat.index = residues

    sns.heatmap(
        heat,
        ax=ax2,
        cmap="vlag",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8.0},
        linewidths=0.55,
        linecolor="white",
        square=False,
        cbar=True,
        cbar_kws={
            "label": "Component score",
            "fraction": 0.04,
            "pad": 0.03,
            "shrink": 0.70
        }
    )

    ax2.set_title("NSI component scores", fontweight="bold", pad=18)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    # Put component labels on top and avoid overlap.
    ax2.xaxis.tick_top()
    ax2.tick_params(
        axis="x",
        labelrotation=0,
        labelsize=8.5,
        pad=8,
        length=0
    )
    ax2.tick_params(axis="y", labelrotation=0, labelsize=9)

    for label in ax2.get_xticklabels():
        label.set_horizontalalignment("center")
        label.set_linespacing(1.15)

    # Box around heatmap.
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # # Panel letters.
    # ax1.text(
    #     -0.20,
    #     1.04,
    #     "A",
    #     transform=ax1.transAxes,
    #     fontsize=18,
    #     fontweight="bold"
    # )

    # ax2.text(
    #     -0.16,
    #     1.04,
    #     "B",
    #     transform=ax2.transAxes,
    #     fontsize=18,
    #     fontweight="bold"
    # )

    # Bottom explanation.
    fig.text(
        0.07,
        0.035,
        "Bars show residue-level mean Nitrosative Susceptibility Index (NSI); error bars indicate ±1 SD across contributing variant records. "
        "Heatmap values show mean component scores used to derive residue-level NSI ranking.",
        fontsize=8.8,
        color="0.25"
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.87,
        bottom=0.16
    )

    plt.savefig(
        OUTPUT_PNG,
        dpi=500,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()
    print(f"Saved figure:\n{OUTPUT_PNG}")


if __name__ == "__main__":
    main()