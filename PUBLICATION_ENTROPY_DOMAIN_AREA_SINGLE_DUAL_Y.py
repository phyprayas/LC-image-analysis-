
# ============================================================
# SINGLE PUBLICATION PLOT WITH DUAL Y-AXES
# 30:70 vs 50:50 NaOH-treated CPC:glycerol
#
# LEFT Y-AXIS  : Normalized orientation entropy
# RIGHT Y-AXIS : Mean domain area (µm²)
#
# Shows:
#   - 4 raw FOV points for each composition
#   - mean ± SD
#   - no bars
#   - no (a)/(b) labels
#   - one single figure
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

required = [
    "FOV",
    "normalized_orientation_entropy",
    "mean_domain_area_um2"
]

for label, df in [("30:70", df30), ("50:50", df50)]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing columns {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

df30 = df30[required].copy()
df50 = df50[required].copy()

df30["Composition"] = "30:70"
df50["Composition"] = "50:50"

data = pd.concat([df30, df50], ignore_index=True)

for col in [
    "normalized_orientation_entropy",
    "mean_domain_area_um2"
]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# ============================================================
# 3. SAVE RAW DATA + SUMMARY
# ============================================================

data.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Entropy_DomainArea_raw_FOV_data.csv"
    ),
    index=False
)

summary_rows = []

for comp in ["30:70", "50:50"]:
    sub = data[data["Composition"] == comp]

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

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Entropy_DomainArea_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 4. SINGLE DUAL-Y FIGURE
# ============================================================

fig, ax1 = plt.subplots(figsize=(7.2, 5.5))
ax2 = ax1.twinx()

x_positions = {
    "30:70": 0,
    "50:50": 1
}

# Small horizontal offsets so entropy and domain-area
# points do not overlap at exactly the same x-position
OFFSET_ENTROPY = -0.10
OFFSET_AREA = +0.10

# ------------------------------------------------------------
# Entropy -> left y-axis
# ------------------------------------------------------------

for comp in ["30:70", "50:50"]:

    vals = (
        data.loc[
            data["Composition"] == comp,
            "normalized_orientation_entropy"
        ]
        .dropna()
        .to_numpy()
    )

    x0 = x_positions[comp] + OFFSET_ENTROPY

    if len(vals) > 1:
        jitter = np.linspace(-0.035, 0.035, len(vals))
    else:
        jitter = np.array([0.0])

    ax1.scatter(
        np.full(len(vals), x0) + jitter,
        vals,
        s=60,
        marker="o",
        edgecolors="black",
        linewidths=0.7,
        zorder=4,
        label="Orientation entropy" if comp == "30:70" else None
    )

    mean = np.mean(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0

    ax1.errorbar(
        x0,
        mean,
        yerr=sd,
        fmt="_",
        markersize=22,
        markeredgewidth=2.2,
        capsize=5,
        elinewidth=1.7,
        zorder=5
    )

# ------------------------------------------------------------
# Mean domain area -> right y-axis
# ------------------------------------------------------------

for comp in ["30:70", "50:50"]:

    vals = (
        data.loc[
            data["Composition"] == comp,
            "mean_domain_area_um2"
        ]
        .dropna()
        .to_numpy()
    )

    x0 = x_positions[comp] + OFFSET_AREA

    if len(vals) > 1:
        jitter = np.linspace(-0.035, 0.035, len(vals))
    else:
        jitter = np.array([0.0])

    ax2.scatter(
        np.full(len(vals), x0) + jitter,
        vals,
        s=64,
        marker="s",
        edgecolors="black",
        linewidths=0.7,
        zorder=4,
        label="Mean domain area" if comp == "30:70" else None
    )

    mean = np.mean(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0

    ax2.errorbar(
        x0,
        mean,
        yerr=sd,
        fmt="_",
        markersize=22,
        markeredgewidth=2.2,
        capsize=5,
        elinewidth=1.7,
        zorder=5
    )

# ============================================================
# 5. AXES
# ============================================================

ax1.set_xticks([0, 1])
ax1.set_xticklabels(["30:70", "50:50"])

ax1.set_xlabel(
    "CPC:glycerol composition",
    fontsize=12
)

ax1.set_ylabel(
    "Normalized orientation entropy",
    fontsize=12
)

ax2.set_ylabel(
    r"Mean domain area ($\mu$m$^2$)",
    fontsize=12
)

# Fixed/clean ranges for your current dataset
ax1.set_ylim(0.90, 1.00)

area_vals = data["mean_domain_area_um2"].dropna()
if len(area_vals):
    ax2.set_ylim(
        max(0, area_vals.min() - 1.0),
        area_vals.max() + 1.5
    )

ax1.set_xlim(-0.45, 1.45)

# Outer box / ticks on all sides
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.1)

ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_linewidth(1.1)

ax1.tick_params(
    axis="both",
    direction="in",
    top=True,
    bottom=True,
    left=True,
    length=5,
    width=1.0,
    labelsize=10
)

ax2.tick_params(
    axis="y",
    direction="in",
    right=True,
    length=5,
    width=1.0,
    labelsize=10
)

ax1.grid(False)
ax2.grid(False)

# ============================================================
# 6. COMBINED LEGEND
# ============================================================

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    frameon=False,
    loc="best",
    fontsize=10
)

plt.tight_layout()

# ============================================================
# 7. SAVE
# ============================================================

png_path = os.path.join(
    OUTPUT_FOLDER,
    "FIG_Entropy_DomainArea_SINGLE_DUAL_Y_raw_points_mean_SD.png"
)

pdf_path = os.path.join(
    OUTPUT_FOLDER,
    "FIG_Entropy_DomainArea_SINGLE_DUAL_Y_raw_points_mean_SD.pdf"
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
# 8. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SINGLE DUAL-Y ENTROPY / DOMAIN-AREA FIGURE COMPLETE")
print("=" * 80)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(png_path)
print(pdf_path)
