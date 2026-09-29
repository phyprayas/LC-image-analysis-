
# ============================================================
# DEFECT DENSITY BAR PLOT — QDLC1 vs QDLC2
#
# QDLC1 = 30:70
# QDLC2 = 50:50
#
# Plot:
#   - Bar = mean defect density
#   - Error bar = SD across FOVs
#   - Raw FOV points overlaid
#   - Different colors for both samples
#
# Expected input column:
#   Composition
#   density_total_per_um2
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT
# ============================================================

# Use your STEP10 master comparison CSV
CSV_PATH = r"D:\New results\STEP10_FINAL_30_70_vs_50_50\STEP10_MASTER_PER_FOV_30_70_vs_50_50.csv"

OUTPUT_DIR = r"D:\New results\Figure_2_Defect_Analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Different colors requested
COLOR_QDLC1 = "#2878B5"   # blue
COLOR_QDLC2 = "#E45756"   # red-orange

DPI = 600

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Columns found:")
print(list(df.columns))

# Flexible search for the density column
density_candidates = [
    "density_total_per_um2",
    "defect_density_um_minus2",
    "defect_density_per_um2",
    "defect_density_um2"
]

density_col = None

for c in density_candidates:
    if c in df.columns:
        density_col = c
        break

if density_col is None:
    raise KeyError(
        "Defect-density column not found.\n"
        f"Available columns:\n{list(df.columns)}"
    )

if "Composition" not in df.columns:
    raise KeyError(
        "Column 'Composition' not found in the CSV."
    )

# ============================================================
# 3. RENAME SAMPLES
# ============================================================

sample_map = {
    "30:70": "QDLC1",
    "50:50": "QDLC2"
}

df["Sample"] = df["Composition"].astype(str).map(sample_map)

df = df[
    df["Sample"].isin(["QDLC1", "QDLC2"])
].copy()

df[density_col] = pd.to_numeric(
    df[density_col],
    errors="coerce"
)

df = df.dropna(
    subset=[density_col]
)

# ============================================================
# 4. CALCULATE MEAN ± SD
# ============================================================

order = [
    "QDLC1",
    "QDLC2"
]

means = []
sds = []
ns = []
raw_values = {}

for sample in order:

    vals = df.loc[
        df["Sample"] == sample,
        density_col
    ].to_numpy(float)

    raw_values[sample] = vals

    means.append(
        np.mean(vals)
    )

    sds.append(
        np.std(vals, ddof=1)
        if len(vals) > 1
        else 0
    )

    ns.append(
        len(vals)
    )

print("\nDefect density results:")
for sample, mean, sd, n in zip(
    order,
    means,
    sds,
    ns
):
    print(
        f"{sample}: {mean:.6f} ± {sd:.6f} µm^-2  (n={n} FOVs)"
    )

# ============================================================
# 5. BAR PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.0, 5.5)
)

x = np.arange(
    len(order)
)

colors = [
    COLOR_QDLC1,
    COLOR_QDLC2
]

bars = ax.bar(
    x,
    means,
    yerr=sds,
    width=0.62,
    color=colors,
    edgecolor="black",
    linewidth=1.2,
    capsize=7,
    error_kw={
        "elinewidth": 1.7,
        "capthick": 1.7
    },
    zorder=2
)

# ============================================================
# 6. RAW FOV POINTS
# ============================================================

for i, sample in enumerate(order):

    vals = raw_values[
        sample
    ]

    if len(vals) == 1:
        jitter = np.array([0.0])
    else:
        jitter = np.linspace(
            -0.07,
            0.07,
            len(vals)
        )

    ax.scatter(
        np.full(
            len(vals),
            i
        ) + jitter,
        vals,
        s=48,
        facecolor="white",
        edgecolor=colors[i],
        linewidth=1.5,
        zorder=4
    )

# ============================================================
# 7. LABELS
# ============================================================

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        "QDLC1\n(30:70)",
        "QDLC2\n(50:50)"
    ],
    fontsize=12
)

ax.set_ylabel(
    r"Defect density ($\mu$m$^{-2}$)",
    fontsize=13
)

ax.set_xlabel(
    "Sample",
    fontsize=13
)

# For manuscript panel, usually no large title is needed.
# Uncomment if you want one:
# ax.set_title(
#     "Defect density",
#     fontsize=14
# )

# ============================================================
# 8. VALUE LABELS ABOVE BARS
# ============================================================

for i, (mean, sd) in enumerate(
    zip(
        means,
        sds
    )
):

    ax.text(
        i,
        mean + sd + max(means) * 0.035,
        f"{mean:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# ============================================================
# 9. PUBLICATION STYLE
# ============================================================

ax.tick_params(
    axis="both",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.1,
    labelsize=11
)

for spine in ax.spines.values():
    spine.set_linewidth(
        1.2
    )

ax.grid(
    False
)

ax.set_ylim(
    bottom=0
)

plt.tight_layout()

# ============================================================
# 10. SAVE
# ============================================================

png_path = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_DEFECT_DENSITY_BAR.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_DEFECT_DENSITY_BAR.pdf"
)

svg_path = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_DEFECT_DENSITY_BAR.svg"
)

fig.savefig(
    png_path,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

fig.savefig(
    svg_path,
    bbox_inches="tight"
)

plt.show()

plt.close(
    fig
)

# ============================================================
# 11. SAVE SUMMARY CSV
# ============================================================

summary = pd.DataFrame({
    "Sample": order,
    "Composition": [
        "30:70",
        "50:50"
    ],
    "N_FOV": ns,
    "Mean_defect_density_um_minus2": means,
    "SD_defect_density_um_minus2": sds
})

summary_path = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_DEFECT_DENSITY_SUMMARY.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\nSaved:")
print(png_path)
print(pdf_path)
print(svg_path)
print(summary_path)
