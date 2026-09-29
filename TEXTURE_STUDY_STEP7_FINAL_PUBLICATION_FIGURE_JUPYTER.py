
# ============================================================
# TEXTURE STUDY — STEP 7
# FINAL PUBLICATION FIGURE + KEY METRICS TABLE
# JUPYTER / VS CODE FRIENDLY
#
# INPUTS:
#   STEP1 ROI master
#   STEP6 integrated master table
#
# FIGURE LAYOUT:
#   (a) T1 representative texture
#   (b) T2 representative texture
#   (c) T3 representative texture
#   (d) A4/A0 comparison
#   (e) Mean |gradPsi| comparison
#   (f) Boundary-core |gradPsi| contrast
#
# NOTES:
# - Representative ROI = first available ROI of each class
# - If multiple ROIs exist, quantitative panels show all raw points
#   + mean ± SD.
# - Multiple ROIs from one image are spatial subsamples,
#   not independent material replicates.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 1. USER INPUT
# ============================================================

STEP1_MASTER_CSV = r"D:\New results\texture_specific_study\STEP1_ROI_SELECTION\TEXTURE_ROI_MASTER.csv"

STEP6_MASTER_CSV = r"D:\New results\texture_specific_study\STEP6_FINAL_INTEGRATED\STEP6_MASTER_PER_ROI.csv"

OUTPUT_DIR = r"D:\New results\texture_specific_study\STEP7_FINAL_FIGURE"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DPI = 600

TEXTURE_ORDER = [
    "T1",
    "T2",
    "T3"
]

TEXTURE_TITLES = {
    "T1": "T1: orange/brown circular",
    "T2": "T2: green curved",
    "T3": "T3: yellow-green four-brush"
}

# ============================================================
# 2. LOAD TABLES
# ============================================================

if not os.path.isfile(STEP1_MASTER_CSV):
    raise FileNotFoundError(
        f"Step-1 master not found:\n{STEP1_MASTER_CSV}"
    )

if not os.path.isfile(STEP6_MASTER_CSV):
    raise FileNotFoundError(
        f"Step-6 master not found:\n{STEP6_MASTER_CSV}"
    )

step1 = pd.read_csv(
    STEP1_MASTER_CSV
)

step6 = pd.read_csv(
    STEP6_MASTER_CSV
)

# ============================================================
# 3. NORMALIZE COLUMN NAMES
# ============================================================

def normalize(df):

    rename = {}

    if "roi_id" in df.columns and "ROI_ID" not in df.columns:
        rename["roi_id"] = "ROI_ID"

    if "texture_code" in df.columns and "Texture" not in df.columns:
        rename["texture_code"] = "Texture"

    if "texture_name" in df.columns and "Texture_name" not in df.columns:
        rename["texture_name"] = "Texture_name"

    return df.rename(
        columns=rename
    )


step1 = normalize(
    step1
)

step6 = normalize(
    step6
)

# Easy Step-1 code used crop_path / masked_crop_path
crop_col = None

for candidate in [
    "masked_crop_path",
    "crop_path",
    "masked_rgb_png",
    "crop_rgb_png"
]:
    if candidate in step1.columns:
        crop_col = candidate
        break

if crop_col is None:
    raise KeyError(
        "Could not find ROI crop path column in Step-1 master.\n"
        f"Available columns: {list(step1.columns)}"
    )

# ============================================================
# 4. SELECT REPRESENTATIVE ROI OF EACH TEXTURE
# ============================================================

representatives = {}

for texture in TEXTURE_ORDER:

    sub = step1[
        step1["Texture"] == texture
    ]

    if len(sub) == 0:
        raise ValueError(
            f"No ROI found for {texture}"
        )

    # first selected ROI = representative example
    row = sub.iloc[0]

    crop_path = str(
        row[crop_col]
    )

    if not os.path.isfile(crop_path):
        raise FileNotFoundError(
            f"Representative crop not found:\n{crop_path}"
        )

    representatives[
        texture
    ] = {
        "ROI_ID": row["ROI_ID"],
        "crop_path": crop_path
    }

# save representative list
rep_df = pd.DataFrame([
    {
        "Texture": t,
        "ROI_ID": representatives[t]["ROI_ID"],
        "crop_path": representatives[t]["crop_path"]
    }
    for t in TEXTURE_ORDER
])

rep_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "STEP7_REPRESENTATIVE_ROIS_USED.csv"
    ),
    index=False
)

# ============================================================
# 5. HELPER FOR QUANTITATIVE PANELS
# ============================================================

def plot_metric_on_axis(
    ax,
    metric,
    ylabel,
    zero_line=False
):

    if metric not in step6.columns:
        ax.text(
            0.5,
            0.5,
            f"{metric}\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        ax.axis("off")
        return

    for xpos, texture in enumerate(
        TEXTURE_ORDER
    ):

        vals = pd.to_numeric(
            step6.loc[
                step6["Texture"] == texture,
                metric
            ],
            errors="coerce"
        ).dropna().to_numpy(float)

        if len(vals) == 0:
            continue

        if len(vals) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(
                -0.06,
                0.06,
                len(vals)
            )

        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=42,
            edgecolors="black",
            linewidths=0.6,
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
            markersize=20,
            markeredgewidth=1.8,
            capsize=4,
            zorder=4
        )

    if zero_line:
        ax.axhline(
            0,
            linestyle="--",
            linewidth=0.8
        )

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        ["T1", "T2", "T3"]
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

# ============================================================
# 6. CREATE FINAL 6-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12.5, 8.2)
)

# ------------------------------------------------------------
# Top row: representative textures
# ------------------------------------------------------------

for j, texture in enumerate(
    TEXTURE_ORDER
):

    img = np.asarray(
        Image.open(
            representatives[texture]["crop_path"]
        ).convert("RGB")
    )

    axes[0, j].imshow(
        img
    )

    axes[0, j].set_title(
        TEXTURE_TITLES[
            texture
        ],
        fontsize=11
    )

    axes[0, j].axis(
        "off"
    )

# ------------------------------------------------------------
# Bottom row: key quantitative comparisons
# ------------------------------------------------------------

plot_metric_on_axis(
    axes[1, 0],
    "A4_over_A0",
    r"Four-fold anisotropy $A_4/A_0$"
)

plot_metric_on_axis(
    axes[1, 1],
    "Mean_gradPsi",
    r"Mean $|\nabla\Psi|$ (rad $\mu$m$^{-1}$)"
)

plot_metric_on_axis(
    axes[1, 2],
    "Boundary_minus_core_gradPsi",
    r"$\Delta|\nabla\Psi|_{\mathrm{boundary-core}}$",
    zero_line=True
)

# ------------------------------------------------------------
# Panel labels
# ------------------------------------------------------------

panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)",
    "(f)"
]

for ax, label in zip(
    axes.ravel(),
    panel_labels
):

    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top"
    )

plt.tight_layout()

png_path = os.path.join(
    OUTPUT_DIR,
    "STEP7_FINAL_TEXTURE_FIGURE.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    "STEP7_FINAL_TEXTURE_FIGURE.pdf"
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

plt.show()
plt.close(fig)

# ============================================================
# 7. KEY METRICS TABLE
# ============================================================

key_metrics = [
    "Mean_intensity",
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

key_metrics = [
    m for m in key_metrics
    if m in step6.columns
]

rows = []

for texture in TEXTURE_ORDER:

    sub = step6[
        step6["Texture"] == texture
    ]

    for metric in key_metrics:

        vals = pd.to_numeric(
            sub[metric],
            errors="coerce"
        ).dropna().to_numpy(float)

        rows.append({

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
                else np.nan
        })

key_table = pd.DataFrame(
    rows
)

key_table_path = os.path.join(
    OUTPUT_DIR,
    "STEP7_KEY_MANUSCRIPT_METRICS.csv"
)

key_table.to_csv(
    key_table_path,
    index=False
)

# ============================================================
# 8. FINAL MESSAGE
# ============================================================

print("\n======================================")
print("STEP 7 COMPLETE")
print("======================================")

print("\nFinal publication figure:")
print(png_path)

print("\nPDF:")
print(pdf_path)

print("\nRepresentative ROI list:")
print(
    os.path.join(
        OUTPUT_DIR,
        "STEP7_REPRESENTATIVE_ROIS_USED.csv"
    )
)

print("\nKey manuscript metrics:")
print(key_table_path)

print(
    "\nNEXT STEP:"
    "\nUse STEP7_KEY_MANUSCRIPT_METRICS.csv + the final figure "
    "to write the Results/Discussion and figure caption."
)
