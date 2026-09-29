
# ============================================================
# TEXTURE STUDY — STEP 3
# T1 vs T2 vs T3 QUANTITATIVE COMPARISON
# JUPYTER / VS CODE FRIENDLY
#
# INPUT:
#   STEP2_TEXTURE_INTENSITY_SUMMARY_PER_ROI.csv
#
# OUTPUTS:
#   - clean comparison table
#   - raw-points + mean±SD plots
#   - normalized heatmap
#   - optional descriptive statistics
#
# IMPORTANT:
# Multiple ROIs from ONE image are spatial subsamples,
# NOT independent material replicates.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT
# ============================================================

STEP2_DIR = r"D:\New results\texture_specific_study\STEP2_INTENSITY_ANALYSIS"

INPUT_CSV = os.path.join(
    STEP2_DIR,
    "STEP2_TEXTURE_INTENSITY_SUMMARY_PER_ROI.csv"
)

OUTPUT_DIR = r"D:\New results\texture_specific_study\STEP3_TEXTURE_COMPARISON"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DPI = 300

TEXTURE_ORDER = [
    "T1",
    "T2",
    "T3"
]

TEXTURE_LABELS = {
    "T1": "T1\nOrange/brown circular",
    "T2": "T2\nGreen curved",
    "T3": "T3\nYellow-green four-brush"
}

# Main metrics to compare
METRICS = [
    "Mean_intensity",
    "r_peak_um",
    "FWHM_um",
    "A2_over_A0",
    "A4_over_A0"
]

METRIC_LABELS = {
    "Mean_intensity": "Mean POM intensity (a.u.)",
    "r_peak_um": r"Peak radius $r_{peak}$ ($\mu$m)",
    "FWHM_um": r"Radial peak FWHM ($\mu$m)",
    "A2_over_A0": r"Two-fold anisotropy $A_2/A_0$",
    "A4_over_A0": r"Four-fold anisotropy $A_4/A_0$"
}


# ============================================================
# 2. LOAD DATA
# ============================================================

if not os.path.isfile(
    INPUT_CSV
):
    raise FileNotFoundError(
        f"Step-2 summary not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(
    INPUT_CSV
)

print("\nLoaded Step-2 data:")
print(df)


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "ROI_ID",
    "Texture",
    "Texture_name"
] + METRICS

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise KeyError(
        f"Missing columns: {missing}\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# 4. CLEAN NUMERIC VALUES
# ============================================================

for metric in METRICS:
    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce"
    )


# ============================================================
# 5. PER-TEXTURE SUMMARY TABLE
# ============================================================

summary_rows = []

for texture in TEXTURE_ORDER:

    sub = df[
        df["Texture"] == texture
    ]

    for metric in METRICS:

        vals = sub[
            metric
        ].dropna().to_numpy(
            dtype=float
        )

        summary_rows.append({
            "Texture": texture,
            "Metric": metric,
            "N_ROI": len(vals),
            "Mean": np.mean(vals) if len(vals) else np.nan,
            "SD": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "SEM": (
                np.std(vals, ddof=1) / np.sqrt(len(vals))
                if len(vals) > 1
                else np.nan
            ),
            "Median": np.median(vals) if len(vals) else np.nan,
            "Min": np.min(vals) if len(vals) else np.nan,
            "Max": np.max(vals) if len(vals) else np.nan
        })

summary = pd.DataFrame(
    summary_rows
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "STEP3_TEXTURE_COMPARISON_SUMMARY.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)

print("\nTexture comparison summary:")
print(summary)


# ============================================================
# 6. WIDE TABLE FOR EASY VIEWING
# ============================================================

wide_rows = []

for texture in TEXTURE_ORDER:

    row = {
        "Texture": texture
    }

    for metric in METRICS:

        s = summary[
            (summary["Texture"] == texture)
            &
            (summary["Metric"] == metric)
        ]

        if len(s) == 1:

            mean = s.iloc[0]["Mean"]
            sd = s.iloc[0]["SD"]
            n = int(s.iloc[0]["N_ROI"])

            row[
                f"{metric}_Mean"
            ] = mean

            row[
                f"{metric}_SD"
            ] = sd

            row[
                f"{metric}_N"
            ] = n

    wide_rows.append(
        row
    )

wide = pd.DataFrame(
    wide_rows
)

wide_csv = os.path.join(
    OUTPUT_DIR,
    "STEP3_TEXTURE_COMPARISON_WIDE.csv"
)

wide.to_csv(
    wide_csv,
    index=False
)


# ============================================================
# 7. RAW POINTS + MEAN ± SD PLOTS
# ============================================================

for metric in METRICS:

    fig, ax = plt.subplots(
        figsize=(6.4, 5.2)
    )

    for xpos, texture in enumerate(
        TEXTURE_ORDER
    ):

        vals = df.loc[
            df["Texture"] == texture,
            metric
        ].dropna().to_numpy(
            dtype=float
        )

        if len(vals) == 0:
            continue

        # deterministic small horizontal spread
        if len(vals) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(
                -0.07,
                0.07,
                len(vals)
            )

        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=65,
            edgecolors="black",
            linewidths=0.7,
            zorder=3
        )

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

        ax.errorbar(
            xpos,
            mean,
            yerr=sd,
            fmt="_",
            markersize=24,
            markeredgewidth=2.0,
            capsize=5,
            zorder=4
        )

    ax.set_xticks(
        range(
            len(TEXTURE_ORDER)
        )
    )

    ax.set_xticklabels(
        [
            TEXTURE_LABELS[t]
            for t in TEXTURE_ORDER
        ]
    )

    ax.set_ylabel(
        METRIC_LABELS[
            metric
        ]
    )

    ax.set_xlabel(
        "Texture class"
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    ax.set_title(
        f"T1 vs T2 vs T3: {METRIC_LABELS[metric]}"
    )

    plt.tight_layout()

    out_png = os.path.join(
        OUTPUT_DIR,
        f"STEP3_{metric}_T1_T2_T3.png"
    )

    out_pdf = os.path.join(
        OUTPUT_DIR,
        f"STEP3_{metric}_T1_T2_T3.pdf"
    )

    fig.savefig(
        out_png,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 8. NORMALIZED HEATMAP
# ============================================================

# mean value of each metric for each texture
mean_matrix = []

for texture in TEXTURE_ORDER:

    row_vals = []

    for metric in METRICS:

        vals = df.loc[
            df["Texture"] == texture,
            metric
        ].dropna().to_numpy(
            dtype=float
        )

        row_vals.append(
            np.mean(vals)
            if len(vals)
            else np.nan
        )

    mean_matrix.append(
        row_vals
    )

mean_matrix = np.asarray(
    mean_matrix,
    dtype=float
)

# Column-wise z-score normalization
norm_matrix = np.full_like(
    mean_matrix,
    np.nan
)

for j in range(
    mean_matrix.shape[1]
):

    col = mean_matrix[
        :,
        j
    ]

    valid = np.isfinite(
        col
    )

    if np.sum(valid) >= 2:

        mu = np.mean(
            col[valid]
        )

        sd = np.std(
            col[valid],
            ddof=0
        )

        if sd > 0:

            norm_matrix[
                valid,
                j
            ] = (
                col[valid]
                -
                mu
            ) / sd

        else:

            norm_matrix[
                valid,
                j
            ] = 0


fig, ax = plt.subplots(
    figsize=(8.5, 4.5)
)

im = ax.imshow(
    norm_matrix,
    aspect="auto"
)

ax.set_yticks(
    range(
        len(TEXTURE_ORDER)
    )
)

ax.set_yticklabels(
    TEXTURE_ORDER
)

ax.set_xticks(
    range(
        len(METRICS)
    )
)

ax.set_xticklabels(
    [
        "Mean I",
        "r_peak",
        "FWHM",
        "A2/A0",
        "A4/A0"
    ],
    rotation=35,
    ha="right"
)

ax.set_title(
    "Normalized texture-property comparison"
)

cbar = plt.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    "Within-metric z-score"
)

# write z-values
for i in range(
    norm_matrix.shape[0]
):

    for j in range(
        norm_matrix.shape[1]
    ):

        value = norm_matrix[
            i,
            j
        ]

        if np.isfinite(
            value
        ):

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

plt.tight_layout()

heatmap_png = os.path.join(
    OUTPUT_DIR,
    "STEP3_NORMALIZED_TEXTURE_HEATMAP.png"
)

fig.savefig(
    heatmap_png,
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 9. HARMONIC INTERPRETATION TABLE
# ============================================================

harmonic_rows = []

for texture in TEXTURE_ORDER:

    sub = df[
        df["Texture"] == texture
    ]

    a2 = sub[
        "A2_over_A0"
    ].dropna().to_numpy(
        dtype=float
    )

    a4 = sub[
        "A4_over_A0"
    ].dropna().to_numpy(
        dtype=float
    )

    mean_a2 = (
        np.mean(a2)
        if len(a2)
        else np.nan
    )

    mean_a4 = (
        np.mean(a4)
        if len(a4)
        else np.nan
    )

    if np.isfinite(mean_a2) and np.isfinite(mean_a4):

        if mean_a4 > mean_a2:
            descriptive_pattern = "four-fold component numerically stronger than two-fold"
        elif mean_a2 > mean_a4:
            descriptive_pattern = "two-fold component numerically stronger than four-fold"
        else:
            descriptive_pattern = "similar two-fold and four-fold components"

    else:
        descriptive_pattern = "insufficient data"

    harmonic_rows.append({
        "Texture": texture,
        "Mean_A2_over_A0": mean_a2,
        "Mean_A4_over_A0": mean_a4,
        "Descriptive_pattern": descriptive_pattern
    })

harmonic_table = pd.DataFrame(
    harmonic_rows
)

harmonic_csv = os.path.join(
    OUTPUT_DIR,
    "STEP3_HARMONIC_INTERPRETATION.csv"
)

harmonic_table.to_csv(
    harmonic_csv,
    index=False
)

print("\nHarmonic summary:")
print(harmonic_table)


# ============================================================
# 10. IMPORTANT SAMPLE-SIZE MESSAGE
# ============================================================

counts = (
    df[
        "Texture"
    ]
    .value_counts()
    .reindex(
        TEXTURE_ORDER,
        fill_value=0
    )
)

print("\n======================================")
print("ROI COUNTS")
print("======================================")
print(counts)

if np.any(
    counts.values < 3
):

    print(
        "\nNOTE:"
        "\nAt least one texture class has fewer than 3 ROIs."
        "\nUse the results as DESCRIPTIVE texture comparison only."
        "\nDo not perform inferential statistics yet."
    )

else:

    print(
        "\nYou have >=3 ROIs per texture class."
        "\nHowever, because these ROIs come from the same image/FOV,"
        "\nthey are spatial subsamples rather than independent material replicates."
    )


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n======================================")
print("STEP 3 COMPLETE")
print("======================================")

print("\nSaved summary:")
print(summary_csv)

print("\nSaved wide table:")
print(wide_csv)

print("\nSaved normalized heatmap:")
print(heatmap_png)

print("\nSaved harmonic interpretation:")
print(harmonic_csv)
