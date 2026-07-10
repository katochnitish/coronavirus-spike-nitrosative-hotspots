"""
Figure 5C: Comparison of RBD-ACE2 docking and interface NSI metrics.

Code developed by
-----------------
Nitish Katoch

Purpose
-------
This script compares RBD-ACE2 docking scores with residue-level NSI metrics at
predefined ACE2-contact positions for Wuhan, Alpha, and Omicron BA.2. It creates
a three-panel publication figure and exports the corresponding group-level
summary statistics.

Input
-----
03_Results/Figure4_delta_NSI_vs_Wuhan.csv

The input table must contain:
    Group
    Variant
    Position
    Smoothed_NSI
    Delta_NSI_vs_Wuhan

Additional user-defined inputs
------------------------------
HADDOCK_SCORES:
    Predefined docking scores for Wuhan, Alpha, and Omicron BA.2.

INTERFACE_RESIDUES:
    Spike RBD positions treated as direct ACE2-interface residues.

Main calculations
-----------------
1. Selects the predefined ACE2-interface residues for the three groups.
2. Calculates the mean and standard error of the mean for interface NSI.
3. Calculates the mean and standard error of the mean for interface delta NSI
   relative to Wuhan.
4. Counts residues with positive delta NSI and calculates their percentage.
5. Combines these metrics with the predefined HADDOCK docking scores.

Plots generated
---------------
Panel A, bar plot:
    HADDOCK score for Wuhan, Alpha, and Omicron BA.2.

Panel B, bar and scatter plot:
    Mean interface NSI with standard-error bars. Individual interface-residue
    NSI values are overlaid as jittered points.

Panel C, bar and scatter plot:
    Mean interface delta NSI relative to Wuhan with standard-error bars.
    Individual residue-level delta NSI values are overlaid as jittered points.

Outputs
-------
03_Results/Figure5C_refined_reddish_boxed_style.png
    High-resolution raster figure.

03_Results/Figure5C_refined_reddish_boxed_style.pdf
    Vector-format publication figure.

03_Results/Figure5C_refined_reddish_boxed_style_summary.csv
    Group-level docking score, interface NSI, interface delta NSI, standard
    error, and positive-delta residue summary.

Dependencies
------------
Python 3, pandas, NumPy, and Matplotlib.

Notes
-----
HADDOCK scores are supplied directly in the script and are not calculated from
molecular structures. NSI values are read from the preceding Figure 4 analysis.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Figure 5C, refined reddish style
# HADDOCK score + interface NSI + interface Delta NSI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "03_Results" / "Figure4_delta_NSI_vs_Wuhan.csv"

OUTPUT_DIR = BASE_DIR / "03_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "Figure5C_refined_reddish_boxed_style.png"
OUTPUT_PDF = OUTPUT_DIR / "Figure5C_refined_reddish_boxed_style.pdf"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "Figure5C_refined_reddish_boxed_style_summary.csv"


# =========================================================
# User data
# =========================================================

HADDOCK_SCORES = {
    "Wuhan": -135,
    "Alpha": -135,
    "Omicron_BA2": -95,
}

INTERFACE_RESIDUES = [
    417, 449, 453, 455, 456, 475,
    486, 487, 489, 493, 498, 501, 505
]

GROUP_ORDER = ["Wuhan", "Alpha", "Omicron_BA2"]

DISPLAY_NAMES = {
    "Wuhan": "Wuhan",
    "Alpha": "Alpha",
    "Omicron_BA2": "Omicron\nBA.2",
}

GROUP_COLORS = {
    "Wuhan": "#4A2F2A",       # dark brown-red
    "Alpha": "#B96B6B",       # muted red
    "Omicron_BA2": "#8D4F69", # mauve-red
}


# =========================================================
# Figure style
# =========================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.9,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =========================================================
# Load NSI data
# =========================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input file was not found:\n{INPUT_CSV}\n\n"
        "Please run your Figure 4 NSI script first."
    )

df = pd.read_csv(INPUT_CSV)

required_cols = [
    "Group",
    "Variant",
    "Position",
    "Smoothed_NSI",
    "Delta_NSI_vs_Wuhan",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
df["Smoothed_NSI"] = pd.to_numeric(df["Smoothed_NSI"], errors="coerce")
df["Delta_NSI_vs_Wuhan"] = pd.to_numeric(df["Delta_NSI_vs_Wuhan"], errors="coerce")

df = df.dropna(subset=["Position", "Smoothed_NSI", "Delta_NSI_vs_Wuhan"])
df["Position"] = df["Position"].astype(int)

interface_df = df[
    (df["Position"].isin(INTERFACE_RESIDUES)) &
    (df["Group"].isin(GROUP_ORDER))
].copy()

if interface_df.empty:
    raise ValueError("No interface residues were found.")


# =========================================================
# Summary table
# =========================================================

summary_rows = []

for group in GROUP_ORDER:
    gdf = interface_df[interface_df["Group"] == group].copy()

    if gdf.empty:
        continue

    summary_rows.append(
        {
            "Group": group,
            "Variant": DISPLAY_NAMES[group],
            "HADDOCK_score": HADDOCK_SCORES.get(group, np.nan),
            "Interface_residue_count": len(gdf),
            "Mean_interface_NSI": gdf["Smoothed_NSI"].mean(),
            "SEM_interface_NSI": gdf["Smoothed_NSI"].std(ddof=1) / np.sqrt(len(gdf)),
            "Mean_interface_Delta_NSI_vs_Wuhan": gdf["Delta_NSI_vs_Wuhan"].mean(),
            "SEM_interface_Delta_NSI_vs_Wuhan": gdf["Delta_NSI_vs_Wuhan"].std(ddof=1) / np.sqrt(len(gdf)),
            "Positive_Delta_NSI_residues": int((gdf["Delta_NSI_vs_Wuhan"] > 0).sum()),
            "Percent_positive_Delta_NSI_residues": 100 * (gdf["Delta_NSI_vs_Wuhan"] > 0).sum() / len(gdf),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)


# =========================================================
# Plot helper
# =========================================================

def boxed_axis(ax):
    """
    Keeps all four plot borders visible for a clean boxed publication style.
    """
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color("black")

    ax.tick_params(axis="both", width=0.9, length=3.5, color="black")
    ax.grid(axis="y", color="#D8D0CD", lw=0.45, alpha=0.70)
    ax.set_axisbelow(True)


# =========================================================
# Create figure
# =========================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.4, 2.65),
    facecolor="white",
    gridspec_kw={"width_ratios": [0.95, 1.05, 1.05], "wspace": 0.52},
)

ax0, ax1, ax2 = axes

x = np.arange(len(GROUP_ORDER))
labels = [DISPLAY_NAMES[g] for g in GROUP_ORDER]
colors = [GROUP_COLORS[g] for g in GROUP_ORDER]


# =========================================================
# Panel A: HADDOCK score
# =========================================================

haddock_values = [HADDOCK_SCORES[g] for g in GROUP_ORDER]

ax0.bar(
    x,
    haddock_values,
    color=colors,
    edgecolor="black",
    linewidth=0.65,
    width=0.62,
)

ax0.axhline(0, color="black", lw=0.8)

ax0.set_xticks(x)
ax0.set_xticklabels(labels, rotation=0, ha="center")
ax0.tick_params(axis="x", pad=1)

ax0.set_ylabel("HADDOCK score")
ax0.set_title("RBD-ACE2 docking")
ax0.set_ylim(-142, 0)

# No values on bars in Panel A.
boxed_axis(ax0)


# =========================================================
# Panel B: Mean interface NSI with residue points
# =========================================================

mean_nsi = [
    summary_df.loc[summary_df["Group"] == g, "Mean_interface_NSI"].values[0]
    for g in GROUP_ORDER
]

sem_nsi = [
    summary_df.loc[summary_df["Group"] == g, "SEM_interface_NSI"].values[0]
    for g in GROUP_ORDER
]

ax1.bar(
    x,
    mean_nsi,
    yerr=sem_nsi,
    color=colors,
    edgecolor="black",
    linewidth=0.65,
    width=0.62,
    capsize=2.2,
    error_kw={"elinewidth": 0.8, "capthick": 0.8},
)

rng = np.random.default_rng(7)

for i, group in enumerate(GROUP_ORDER):
    gdf = interface_df[interface_df["Group"] == group]
    jitter = rng.normal(0, 0.045, size=len(gdf))

    ax1.scatter(
        np.full(len(gdf), i) + jitter,
        gdf["Smoothed_NSI"],
        s=22,
        facecolor="white",
        edgecolor="#2B2B2B",
        linewidth=0.75,
        zorder=3,
    )

ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=0, ha="center")
ax1.tick_params(axis="x", pad=2)

ax1.set_ylabel("Interface NSI")
ax1.set_title("Chemical susceptibility")
ax1.set_ylim(0, max(interface_df["Smoothed_NSI"]) * 1.22)

boxed_axis(ax1)


# =========================================================
# Panel C: Mean interface Delta NSI with residue points
# =========================================================

mean_delta = [
    summary_df.loc[summary_df["Group"] == g, "Mean_interface_Delta_NSI_vs_Wuhan"].values[0]
    for g in GROUP_ORDER
]

sem_delta = [
    summary_df.loc[summary_df["Group"] == g, "SEM_interface_Delta_NSI_vs_Wuhan"].values[0]
    for g in GROUP_ORDER
]

ax2.bar(
    x,
    mean_delta,
    yerr=sem_delta,
    color=colors,
    edgecolor="black",
    linewidth=0.65,
    width=0.62,
    capsize=2.2,
    error_kw={"elinewidth": 0.8, "capthick": 0.8},
)

for i, group in enumerate(GROUP_ORDER):
    gdf = interface_df[interface_df["Group"] == group]
    jitter = rng.normal(0, 0.045, size=len(gdf))

    ax2.scatter(
        np.full(len(gdf), i) + jitter,
        gdf["Delta_NSI_vs_Wuhan"],
        s=22,
        facecolor="white",
        edgecolor="#2B2B2B",
        linewidth=0.75,
        zorder=3,
    )

ax2.axhline(0, color="black", lw=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=0, ha="center")
ax2.tick_params(axis="x", pad=2)

ax2.set_ylabel("Interface ΔNSI")
ax2.set_title("Change relative to Wuhan")

delta_abs = max(
    abs(interface_df["Delta_NSI_vs_Wuhan"].min()),
    abs(interface_df["Delta_NSI_vs_Wuhan"].max()),
)

if delta_abs == 0:
    delta_abs = 0.05

ax2.set_ylim(-delta_abs * 1.30, delta_abs * 1.30)

boxed_axis(ax2)


# =========================================================
# Final formatting
# =========================================================

plt.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=900, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nSaved refined boxed Figure 5C:")
print(f"  {OUTPUT_PNG}")
print(f"  {OUTPUT_PDF}")
print(f"  {OUTPUT_SUMMARY_CSV}")

print("\nSummary:")
print(
    summary_df[
        [
            "Variant",
            "HADDOCK_score",
            "Mean_interface_NSI",
            "Mean_interface_Delta_NSI_vs_Wuhan",
            "Percent_positive_Delta_NSI_residues",
        ]
    ].to_string(index=False)
)