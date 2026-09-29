
# ============================================================
# PUBLICATION FIGURE:
# ORIENTATION ENTROPY + MEAN DOMAIN AREA
# 30:70 vs 50:50 NaOH-treated CPC:glycerol
#
# Raw FOV points + mean ± SD
# Two-panel figure:
#   (a) Normalized orientation entropy
#   (b) Mean domain area
#
# INPUT:
#   Step-9 ALL-FOV summary CSV for 30:70
#   Step-9 ALL-FOV summary CSV for 50:50
#
# OUTPUT:
#   PNG, PDF, raw plotting CSV, summary CSV
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. EDIT THESE PATHS
# ============================================================

CSV_30_70 = r"D:\New results\3070\step 9 domain area, entropy, angular orientation\STEP9_ALL_4FOV_summary.csv"

CSV_50_50 = r"D:\New results\5070\step 9 domain area, entropy, angular dispersion\STEP9_ALL_4FOV_summary.csv"

OUTPUT_FOLDER = r"D:\New results\FINAL_30_70_vs_50_50_COMPARISON\Publication_Figures"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

df30 = pd.read_csv(CSV_30_70)
df50 = pd.read_csv(CSV_50_50)

required_cols = [
    "FOV",
    "normalized_orientation_entropy",
    "mean_domain_area_um2"
]

for label, df in [("30:70", df30), ("50:50", df50)]:
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise KeyError(
            f"{label}: missing columns {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

# Keep only the required columns
df30 = df30[required_cols].copy()
df50 = df50[required_cols].copy()

df30["Composition"] = "30:70"
df50["Composition"] = "50:50"

combined = pd.concat(
    [df30, df50],
    ignore_index=True
)

# Convert safely to numeric
for col in [
    "normalized_orientation_entropy",
    "mean_domain_area_um2"
]:
    combined[col] = pd.to_numeric(
        combined[col],
        errors="coerce"
    )

# Save exact raw data used for plotting
raw_csv = os.path.join(
    OUTPUT_FOLDER,
    "FIG2f_entropy_domain_area_raw_FOV_data.csv"
)

combined.to_csv(
    raw_csv,
    index=False
)

# ============================================================
# 3. SUMMARY STATISTICS
# ============================================================

summary_rows = []

for comp in ["30:70", "50:50"]:

    sub = combined[
        combined["Composition"] == comp
    ]

    for metric in [
        "normalized_orientation_entropy",
        "mean_domain_area_um2"
    ]:

        vals = sub[metric].dropna()

        summary_rows.append({
            "Composition": comp,
            "Metric": metric,
            "N_FOV": len(vals),
            "Mean": vals.mean(),
            "SD": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "SEM": vals.sem(ddof=1) if len(vals) > 1 else np.nan
        })

summary = pd.DataFrame(
    summary_rows
)

summary_csv = os.path.join(
    OUTPUT_FOLDER,
    "FIG2f_entropy_domain_area_mean_SD_SEM.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)

# ============================================================
# 4. PLOTTING FUNCTION
# ============================================================

def raw_points_mean_sd(
    ax,
    data,
    metric,
    ylabel,
    panel_label
):

    compositions = [
        "30:70",
        "50:50"
    ]

    x_positions = [
        0,
        1
    ]

    for x, comp in zip(
        x_positions,
        compositions
    ):

        vals = (
            data.loc[
                data["Composition"] == comp,
                metric
            ]
            .dropna()
            .to_numpy()
        )

        # deterministic horizontal spread for raw points
        if len(vals) > 1:
            jitter = np.linspace(
                -0.08,
                0.08,
                len(vals)
            )
        else:
            jitter = np.array([0.0])

        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            s=58,
            edgecolors="black",
            linewidths=0.7,
            zorder=3
        )

        if len(vals):

            mean = np.mean(
                vals
            )

            sd = (
                np.std(
                    vals,
                    ddof=1
                )
                if len(vals) > 1
                else 0
            )

            # mean ± SD
            ax.errorbar(
                x,
                mean,
                yerr=sd,
                fmt="_",
                markersize=22,
                markeredgewidth=2.2,
                capsize=5,
                elinewidth=1.6,
                zorder=4
            )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        compositions
    )

    ax.set_xlabel(
        "CPC:glycerol composition",
        fontsize=11
    )

    ax.set_ylabel(
        ylabel,
        fontsize=11
    )

    # box on all four sides
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1.0,
        labelsize=10
    )

    ax.grid(
        False
    )

    ax.text(
        -0.14,
        1.03,
        panel_label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold"
    )

# ============================================================
# 5. TWO-PANEL PUBLICATION FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.0, 4.5)
)

# Panel (a): entropy
raw_points_mean_sd(
    axes[0],
    combined,
    "normalized_orientation_entropy",
    "Normalized orientation entropy",
    "(a)"
)

# Panel (b): domain area
raw_points_mean_sd(
    axes[1],
    combined,
    "mean_domain_area_um2",
    r"Mean domain area ($\mu$m$^2$)",
    "(b)"
)

# Sensible y-limits based on your actual data
axes[0].set_ylim(
    0.90,
    1.00
)

# Add a little vertical headroom for domain area
domain_vals = combined[
    "mean_domain_area_um2"
].dropna()

if len(domain_vals):
    ymin = max(
        0,
        domain_vals.min() * 0.85
    )

    ymax = domain_vals.max() * 1.15

    axes[1].set_ylim(
        ymin,
        ymax
    )

plt.tight_layout()

# ============================================================
# 6. SAVE FIGURE
# ============================================================

png_path = os.path.join(
    OUTPUT_FOLDER,
    "FIG2f_entropy_and_mean_domain_area_raw_points_mean_SD.png"
)

pdf_path = os.path.join(
    OUTPUT_FOLDER,
    "FIG2f_entropy_and_mean_domain_area_raw_points_mean_SD.pdf"
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
# 7. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 82)
print("FIGURE 2f COMPLETE")
print("=" * 82)

print("\nSummary statistics:")
print(
    summary.to_string(
        index=False
    )
)

print("\nSaved files:")
print(png_path)
print(pdf_path)
print(raw_csv)
print(summary_csv)
