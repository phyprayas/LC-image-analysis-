
# ============================================================
# PUBLICATION-QUALITY GLOBAL / LOCAL S2D COMPARISON
# 30:70 vs 50:50 NaOH-treated CPC:glycerol
#
# Input:
#   STEP10_MASTER_PER_FOV_30_70_vs_50_50.csv
#
# Output:
#   PNG + PDF + plotted-data CSV
#
# Plot style:
#   - raw FOV points
#   - mean ± SD
#   - no bars
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. PATHS — EDIT THESE
# ============================================================

INPUT_CSV = r"D:\New results\FINAL_30_70_vs_50_50_COMPARISON\STEP10_MASTER_PER_FOV_30_70_vs_50_50.csv"

OUTPUT_FOLDER = r"D:\New results\FINAL_30_70_vs_50_50_COMPARISON\Publication_Figures"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

required = [
    "Composition",
    "global_S2D",
    "mean_local_S2D"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise KeyError(
        f"Missing required columns: {missing}\n"
        f"Available columns:\n{list(df.columns)}"
    )

# Keep only the two compositions
df = df[df["Composition"].isin(["30:70", "50:50"])].copy()

# Convert to numeric safely
df["global_S2D"] = pd.to_numeric(
    df["global_S2D"],
    errors="coerce"
)

df["mean_local_S2D"] = pd.to_numeric(
    df["mean_local_S2D"],
    errors="coerce"
)

# Save exactly what is plotted
plot_data = df[
    ["Composition", "FOV_key", "global_S2D", "mean_local_S2D"]
].copy()

plot_data.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "S2D_raw_FOV_plot_data.csv"
    ),
    index=False
)

# ============================================================
# 3. SUMMARY TABLE
# ============================================================

summary_rows = []

for comp in ["30:70", "50:50"]:

    sub = df[df["Composition"] == comp]

    for metric in ["global_S2D", "mean_local_S2D"]:

        vals = sub[metric].dropna()

        summary_rows.append({
            "Composition": comp,
            "Metric": metric,
            "N_FOV": len(vals),
            "Mean": vals.mean(),
            "SD": vals.std(ddof=1),
            "SEM": vals.sem(ddof=1)
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "S2D_mean_SD_SEM_summary.csv"
    ),
    index=False
)

# ============================================================
# 4. PUBLICATION FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 5.6))

# x positions:
# Global: 0,1
# Local:  3,4
positions = {
    ("30:70", "global_S2D"): 0.0,
    ("50:50", "global_S2D"): 1.0,
    ("30:70", "mean_local_S2D"): 3.0,
    ("50:50", "mean_local_S2D"): 4.0,
}

rng = np.random.default_rng(42)

for comp in ["30:70", "50:50"]:

    for metric in ["global_S2D", "mean_local_S2D"]:

        vals = (
            df.loc[
                df["Composition"] == comp,
                metric
            ]
            .dropna()
            .to_numpy()
        )

        x0 = positions[(comp, metric)]

        # deterministic jitter
        if len(vals) > 1:
            jitter = np.linspace(
                -0.09,
                0.09,
                len(vals)
            )
        else:
            jitter = np.array([0.0])

        # raw points
        ax.scatter(
            np.full(len(vals), x0) + jitter,
            vals,
            s=55,
            zorder=3,
            edgecolors="black",
            linewidths=0.7
        )

        # mean ± SD
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0

        ax.errorbar(
            x0,
            mean,
            yerr=sd,
            fmt="_",
            markersize=20,
            markeredgewidth=2.2,
            capsize=5,
            elinewidth=1.6,
            zorder=4
        )

# x ticks
ax.set_xticks(
    [0, 1, 3, 4]
)

ax.set_xticklabels(
    [
        "30:70",
        "50:50",
        "30:70",
        "50:50"
    ]
)

# group labels below x-axis
ax.text(
    0.5,
    -0.15,
    "Global $S_{2D}$",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=11
)

ax.text(
    3.5,
    -0.15,
    "Local $S_{2D}$",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=11
)

# separator
ax.axvline(
    2.0,
    linestyle="--",
    linewidth=1
)

ax.set_ylabel(
    "$S_{2D}$",
    fontsize=12
)

ax.set_ylim(
    0,
    0.8
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    "Global and local orientational order",
    fontsize=13
)

plt.tight_layout()

png_path = os.path.join(
    OUTPUT_FOLDER,
    "FIG_Global_Local_S2D_raw_points_mean_SD.png"
)

pdf_path = os.path.join(
    OUTPUT_FOLDER,
    "FIG_Global_Local_S2D_raw_points_mean_SD.pdf"
)

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 5. PRINT SUMMARY
# ============================================================

print("\n" + "="*72)
print("GLOBAL / LOCAL S2D FIGURE COMPLETE")
print("="*72)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved files:")
print(png_path)
print(pdf_path)
print(
    os.path.join(
        OUTPUT_FOLDER,
        "S2D_raw_FOV_plot_data.csv"
    )
)
print(
    os.path.join(
        OUTPUT_FOLDER,
        "S2D_mean_SD_SEM_summary.csv"
    )
)
