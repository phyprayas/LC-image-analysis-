
# ============================================================
# TEXTURE STUDY — STEP 6
# FINAL INTEGRATED T1 vs T2 vs T3 MASTER TABLE + FIGURES
# JUPYTER / VS CODE FRIENDLY
#
# INPUTS:
#   STEP 2 intensity/harmonic summary
#   STEP 4 orientation-intensity coupling summary
#   STEP 5 core-boundary-exterior summary
#
# OUTPUTS:
#   1) integrated per-ROI master table
#   2) texture-class mean/SD summary
#   3) publication-ready comparison plots
#   4) compact manuscript metrics table
#
# IMPORTANT:
# Multiple ROIs from ONE FOV are spatial subsamples,
# not independent material replicates.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT
# ============================================================

STEP2_CSV = r"D:\New results\texture_specific_study\STEP2_INTENSITY_ANALYSIS\STEP2_TEXTURE_INTENSITY_SUMMARY_PER_ROI.csv"

STEP4_CSV = r"D:\New results\texture_specific_study\STEP4_ORIENTATION_INTENSITY_COUPLING\STEP4_ORIENTATION_INTENSITY_COUPLING_PER_ROI.csv"

STEP5_CSV = r"D:\New results\texture_specific_study\STEP5_CORE_BOUNDARY_EXTERIOR\STEP5_CORE_BOUNDARY_EXTERIOR_PER_ROI.csv"

OUTPUT_DIR = r"D:\New results\texture_specific_study\STEP6_FINAL_INTEGRATED"

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


# ============================================================
# 2. LOAD FILES
# ============================================================

for path in [
    STEP2_CSV,
    STEP4_CSV,
    STEP5_CSV
]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )

step2 = pd.read_csv(
    STEP2_CSV
)

step4 = pd.read_csv(
    STEP4_CSV
)

step5 = pd.read_csv(
    STEP5_CSV
)

print("\nLoaded:")
print("Step 2:", len(step2), "rows")
print("Step 4:", len(step4), "rows")
print("Step 5:", len(step5), "rows")


# ============================================================
# 3. STANDARDIZE ROI / TEXTURE COLUMNS
# ============================================================

def normalize_columns(df):

    rename_map = {}

    if "roi_id" in df.columns and "ROI_ID" not in df.columns:
        rename_map["roi_id"] = "ROI_ID"

    if "texture_code" in df.columns and "Texture" not in df.columns:
        rename_map["texture_code"] = "Texture"

    if "texture_name" in df.columns and "Texture_name" not in df.columns:
        rename_map["texture_name"] = "Texture_name"

    return df.rename(
        columns=rename_map
    )


step2 = normalize_columns(
    step2
)

step4 = normalize_columns(
    step4
)

step5 = normalize_columns(
    step5
)

for name, df in [
    ("Step2", step2),
    ("Step4", step4),
    ("Step5", step5)
]:

    if "ROI_ID" not in df.columns:
        raise KeyError(
            f"{name} does not contain ROI_ID"
        )


# ============================================================
# 4. SELECT COLUMNS TO MERGE
# ============================================================

step2_keep = [
    c for c in [
        "ROI_ID",
        "Texture",
        "Texture_name",
        "Mean_intensity",
        "Median_intensity",
        "SD_intensity",
        "r_peak_um",
        "I_peak",
        "FWHM_um",
        "A2_over_A0",
        "A4_over_A0",
        "phi2_deg",
        "phi4_deg"
    ]
    if c in step2.columns
]

step4_keep = [
    c for c in [
        "ROI_ID",
        "N_vectors",
        "Mean_Psi_deg",
        "Mean_gradPsi",
        "Spearman_I_vs_sin2_2Psi",
        "Spearman_I_vs_gradPsi",
        "Harmonic_I_Psi_R2",
        "Orientation_Amp2_over_meanI",
        "Orientation_Amp4_over_meanI"
    ]
    if c in step4.columns
]

step5_keep = [
    c for c in [
        "ROI_ID",
        "Core_N",
        "Boundary_N",
        "Exterior_N",
        "Core_intensity",
        "Boundary_intensity",
        "Exterior_intensity",
        "Boundary_minus_core_intensity",
        "Boundary_minus_exterior_intensity",
        "Core_gradPsi",
        "Boundary_gradPsi",
        "Exterior_gradPsi",
        "Boundary_minus_core_gradPsi",
        "Boundary_minus_exterior_gradPsi",
        "Core_Slocal",
        "Boundary_Slocal",
        "Exterior_Slocal",
        "Boundary_minus_core_Slocal",
        "Boundary_minus_exterior_Slocal"
    ]
    if c in step5.columns
]


# ============================================================
# 5. MERGE ALL STEPS
# ============================================================

master = (
    step2[
        step2_keep
    ]
    .merge(
        step4[
            step4_keep
        ],
        on="ROI_ID",
        how="outer",
        suffixes=("", "_step4")
    )
    .merge(
        step5[
            step5_keep
        ],
        on="ROI_ID",
        how="outer",
        suffixes=("", "_step5")
    )
)

# restore texture columns if only present in step2
if "Texture" not in master.columns:
    raise KeyError(
        "Texture column missing after merge."
    )

master_csv = os.path.join(
    OUTPUT_DIR,
    "STEP6_MASTER_PER_ROI.csv"
)

master.to_csv(
    master_csv,
    index=False
)

print(
    "\nIntegrated master table:"
)

print(
    master
)


# ============================================================
# 6. DEFINE MAIN MANUSCRIPT METRICS
# ============================================================

MAIN_METRICS = [
    "Mean_intensity",
    "r_peak_um",
    "FWHM_um",
    "A2_over_A0",
    "A4_over_A0",
    "Mean_gradPsi",
    "Spearman_I_vs_sin2_2Psi",
    "Spearman_I_vs_gradPsi",
    "Harmonic_I_Psi_R2",
    "Boundary_minus_core_gradPsi",
    "Boundary_minus_exterior_gradPsi",
    "Boundary_minus_core_Slocal",
    "Boundary_minus_exterior_Slocal"
]

MAIN_METRICS = [
    m for m in MAIN_METRICS
    if m in master.columns
]


# ============================================================
# 7. TEXTURE-CLASS SUMMARY
# ============================================================

summary_rows = []

for texture in TEXTURE_ORDER:

    sub = master[
        master["Texture"] == texture
    ]

    for metric in MAIN_METRICS:

        vals = pd.to_numeric(
            sub[metric],
            errors="coerce"
        ).dropna().to_numpy(float)

        summary_rows.append({

            "Texture":
                texture,

            "Metric":
                metric,

            "N_ROI":
                len(vals),

            "Mean":
                np.mean(vals)
                if len(vals)
                else np.nan,

            "SD":
                np.std(
                    vals,
                    ddof=1
                )
                if len(vals) > 1
                else np.nan,

            "SEM":
                (
                    np.std(
                        vals,
                        ddof=1
                    )
                    /
                    np.sqrt(
                        len(vals)
                    )
                )
                if len(vals) > 1
                else np.nan,

            "Median":
                np.median(vals)
                if len(vals)
                else np.nan
        })


summary = pd.DataFrame(
    summary_rows
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "STEP6_TEXTURE_CLASS_MEAN_SD_SUMMARY.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# 8. COMPACT MANUSCRIPT TABLE
# ============================================================

compact_metrics = [
    "Mean_intensity",
    "A2_over_A0",
    "A4_over_A0",
    "Mean_gradPsi",
    "Spearman_I_vs_sin2_2Psi",
    "Spearman_I_vs_gradPsi",
    "Harmonic_I_Psi_R2",
    "Boundary_minus_core_gradPsi",
    "Boundary_minus_exterior_Slocal"
]

compact_metrics = [
    m for m in compact_metrics
    if m in master.columns
]

compact_rows = []

for texture in TEXTURE_ORDER:

    row = {
        "Texture":
            texture
    }

    sub = master[
        master["Texture"] == texture
    ]

    for metric in compact_metrics:

        vals = pd.to_numeric(
            sub[metric],
            errors="coerce"
        ).dropna().to_numpy(float)

        mean = (
            np.mean(vals)
            if len(vals)
            else np.nan
        )

        sd = (
            np.std(
                vals,
                ddof=1
            )
            if len(vals) > 1
            else np.nan
        )

        row[
            f"{metric}_Mean"
        ] = mean

        row[
            f"{metric}_SD"
        ] = sd

        row[
            f"{metric}_N"
        ] = len(vals)

    compact_rows.append(
        row
    )


compact = pd.DataFrame(
    compact_rows
)

compact_csv = os.path.join(
    OUTPUT_DIR,
    "STEP6_COMPACT_MANUSCRIPT_METRICS.csv"
)

compact.to_csv(
    compact_csv,
    index=False
)


# ============================================================
# 9. PUBLICATION PLOT FUNCTION
# ============================================================

def plot_metric(metric, ylabel):

    if metric not in master.columns:
        return

    fig, ax = plt.subplots(
        figsize=(6.3, 5.0)
    )

    for xpos, texture in enumerate(
        TEXTURE_ORDER
    ):

        vals = pd.to_numeric(
            master.loc[
                master["Texture"] == texture,
                metric
            ],
            errors="coerce"
        ).dropna().to_numpy(float)

        if len(vals) == 0:
            continue

        jitter = (
            np.array([0.0])
            if len(vals) == 1
            else np.linspace(
                -0.07,
                0.07,
                len(vals)
            )
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
            markeredgewidth=2,
            capsize=5,
            zorder=4
        )

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        [
            TEXTURE_LABELS[t]
            for t in TEXTURE_ORDER
        ]
    )

    ax.set_xlabel(
        "Texture class"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"STEP6_{metric}_T1_T2_T3.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"STEP6_{metric}_T1_T2_T3.pdf"
        ),
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 10. MAIN FINAL FIGURES
# ============================================================

plot_metric(
    "Mean_intensity",
    "Mean POM intensity (a.u.)"
)

plot_metric(
    "A2_over_A0",
    r"Two-fold anisotropy $A_2/A_0$"
)

plot_metric(
    "A4_over_A0",
    r"Four-fold anisotropy $A_4/A_0$"
)

plot_metric(
    "Mean_gradPsi",
    r"Mean $|\nabla\Psi|$ (rad $\mu$m$^{-1}$)"
)

plot_metric(
    "Spearman_I_vs_sin2_2Psi",
    r"Spearman $\rho$: $I$ vs $\sin^2(2\Psi)$"
)

plot_metric(
    "Spearman_I_vs_gradPsi",
    r"Spearman $\rho$: $I$ vs $|\nabla\Psi|$"
)

plot_metric(
    "Boundary_minus_core_gradPsi",
    r"$\Delta|\nabla\Psi|_{\mathrm{boundary-core}}$"
)

plot_metric(
    "Boundary_minus_exterior_Slocal",
    r"$\Delta S_{\mathrm{local,\ boundary-exterior}}$"
)


# ============================================================
# 11. NORMALIZED MASTER HEATMAP
# ============================================================

heat_metrics = [
    m for m in [
        "Mean_intensity",
        "A2_over_A0",
        "A4_over_A0",
        "Mean_gradPsi",
        "Spearman_I_vs_sin2_2Psi",
        "Spearman_I_vs_gradPsi",
        "Boundary_minus_core_gradPsi",
        "Boundary_minus_exterior_Slocal"
    ]
    if m in master.columns
]

mean_matrix = []

for texture in TEXTURE_ORDER:

    sub = master[
        master["Texture"] == texture
    ]

    row = []

    for metric in heat_metrics:

        vals = pd.to_numeric(
            sub[metric],
            errors="coerce"
        ).dropna().to_numpy(float)

        row.append(
            np.mean(vals)
            if len(vals)
            else np.nan
        )

    mean_matrix.append(
        row
    )

mean_matrix = np.asarray(
    mean_matrix,
    dtype=float
)

z = np.full_like(
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
            col[valid]
        )

        if sd > 0:

            z[
                valid,
                j
            ] = (
                col[valid]
                -
                mu
            ) / sd

        else:

            z[
                valid,
                j
            ] = 0


fig, ax = plt.subplots(
    figsize=(
        max(
            8,
            1.25 * len(heat_metrics)
        ),
        4.2
    )
)

im = ax.imshow(
    z,
    aspect="auto"
)

ax.set_yticks(
    [0, 1, 2]
)

ax.set_yticklabels(
    TEXTURE_ORDER
)

ax.set_xticks(
    range(
        len(heat_metrics)
    )
)

ax.set_xticklabels(
    heat_metrics,
    rotation=45,
    ha="right"
)

ax.set_title(
    "Integrated normalized texture-property map"
)

cbar = plt.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    "Within-metric z-score"
)

for i in range(
    z.shape[0]
):

    for j in range(
        z.shape[1]
    ):

        if np.isfinite(
            z[i, j]
        ):

            ax.text(
                j,
                i,
                f"{z[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "STEP6_INTEGRATED_NORMALIZED_HEATMAP.png"
)

fig.savefig(
    heatmap_path,
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 12. DATA COMPLETENESS CHECK
# ============================================================

completeness = []

for texture in TEXTURE_ORDER:

    sub = master[
        master["Texture"] == texture
    ]

    completeness.append({
        "Texture":
            texture,

        "N_total_ROI":
            len(sub),

        "N_with_Step2":
            sub[
                "Mean_intensity"
            ].notna().sum()
            if "Mean_intensity" in sub.columns
            else 0,

        "N_with_Step4":
            sub[
                "Mean_gradPsi"
            ].notna().sum()
            if "Mean_gradPsi" in sub.columns
            else 0,

        "N_with_Step5":
            sub[
                "Boundary_minus_core_gradPsi"
            ].notna().sum()
            if "Boundary_minus_core_gradPsi" in sub.columns
            else 0
    })


completeness_df = pd.DataFrame(
    completeness
)

completeness_csv = os.path.join(
    OUTPUT_DIR,
    "STEP6_DATA_COMPLETENESS_CHECK.csv"
)

completeness_df.to_csv(
    completeness_csv,
    index=False
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n======================================")
print("STEP 6 COMPLETE")
print("======================================")

print("\nIntegrated per-ROI master:")
print(master_csv)

print("\nTexture-class mean/SD summary:")
print(summary_csv)

print("\nCompact manuscript table:")
print(compact_csv)

print("\nIntegrated heatmap:")
print(heatmap_path)

print("\nData completeness:")
print(completeness_df)

print(
    "\nNOTE: If only one ROI exists for each texture class, "
    "use these results descriptively. For stronger texture-class "
    "comparison, repeat ROI selection for several T1/T2/T3 regions "
    "from the same image and rerun Steps 2–6."
)
