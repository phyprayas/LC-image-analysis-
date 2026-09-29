
# ============================================================
# PUBLICATION-QUALITY C2(r) COMPARISON CURVES
# MANUAL FILE PATH VERSION
# 30:70 vs 50:50 NaOH-treated CPC:glycerol
#
# HOW TO USE:
# 1. Paste the full path of each 30:70 C2(r) CSV below.
# 2. Paste the full path of each 50:50 C2(r) CSV below.
# 3. Set OUTPUT_FOLDER.
# 4. Run the script.
#
# IMPORTANT:
# Use the individual Step-3 C2(r) curve CSV files,
# NOT the Step-3 summary CSV.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. MANUALLY ADD YOUR FILE PATHS HERE
# ============================================================

FILES_30_70 = [
    r"D:\New results\3070\step 3 orientation order correlation\FOV1\YOUR_C2_FILE.csv",
    r"D:\New results\3070\step 3 orientation order correlation\FOV2\YOUR_C2_FILE.csv",
    r"D:\New results\3070\step 3 orientation order correlation\FOV3\YOUR_C2_FILE.csv",
    r"D:\New results\3070\step 3 orientation order correlation\FOV4\YOUR_C2_FILE.csv",
]

FILES_50_50 = [
    r"D:\New results\5070\step 3 orientation order correlation\FOV1\YOUR_C2_FILE.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV2\YOUR_C2_FILE.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV3\YOUR_C2_FILE.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV4\YOUR_C2_FILE.csv",
]

OUTPUT_FOLDER = r"D:\New results\FINAL_30_70_vs_50_50_COMPARISON\Publication_Figures_C2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. COLUMN ALIASES
# ============================================================

R_ALIASES = [
    "r_um", "r", "distance_um", "radius_um", "separation_um",
    "lag_um", "r_microns"
]

C2_ALIASES = [
    "C2", "c2", "C_2", "correlation", "C2_r",
    "C2_mean", "C2_value"
]

FIT_ALIASES = [
    "fit", "C2_fit", "fit_C2", "fit_curve",
    "exp_fit", "fitted_C2"
]

# ============================================================
# 3. HELPERS
# ============================================================

def check_files(files, label):
    missing = [p for p in files if not os.path.isfile(p)]

    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(
            f"\n{label}: these files were not found:\n{msg}\n\n"
            "Replace YOUR_C2_FILE.csv with the real CSV filenames."
        )


def detect_column(df, aliases, label):
    for col in aliases:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find {label} column.\n"
        f"Tried: {aliases}\n"
        f"Available columns: {list(df.columns)}"
    )


def load_curve(path):
    df = pd.read_csv(path)

    r_col = detect_column(df, R_ALIASES, "r")
    c2_col = detect_column(df, C2_ALIASES, "C2")

    fit_col = None
    for col in FIT_ALIASES:
        if col in df.columns:
            fit_col = col
            break

    out = pd.DataFrame({
        "r_um": pd.to_numeric(df[r_col], errors="coerce"),
        "C2": pd.to_numeric(df[c2_col], errors="coerce")
    })

    if fit_col is not None:
        out["fit"] = pd.to_numeric(df[fit_col], errors="coerce")
    else:
        out["fit"] = np.nan

    out = (
        out.dropna(subset=["r_um", "C2"])
           .sort_values("r_um")
           .drop_duplicates("r_um")
    )

    return out


def make_common_grid(curves):
    mins = []
    maxs = []
    diffs = []

    for d in curves:
        r = d["r_um"].to_numpy()

        mins.append(np.nanmin(r))
        maxs.append(np.nanmax(r))

        dr = np.diff(r)
        dr = dr[np.isfinite(dr) & (dr > 0)]

        if len(dr):
            diffs.append(np.median(dr))

    r_min = max(mins)
    r_max = min(maxs)

    if r_max <= r_min:
        raise ValueError("No common r-range across selected curves.")

    step = np.median(diffs) if diffs else 0.1

    grid = np.arange(
        r_min,
        r_max + step / 2,
        step
    )

    return grid


def align_curves(curves, grid):
    c2_all = []
    fit_all = []

    for d in curves:

        r = d["r_um"].to_numpy()
        c2 = d["C2"].to_numpy()

        c2_all.append(
            np.interp(grid, r, c2)
        )

        fit = d["fit"].to_numpy()

        valid = np.isfinite(fit)

        if valid.sum() >= 2:
            fit_all.append(
                np.interp(
                    grid,
                    r[valid],
                    fit[valid]
                )
            )
        else:
            fit_all.append(
                np.full_like(
                    grid,
                    np.nan,
                    dtype=float
                )
            )

    return (
        np.asarray(c2_all),
        np.asarray(fit_all)
    )


def process_group(files, label):

    curves = [
        load_curve(p)
        for p in files
    ]

    grid = make_common_grid(curves)

    c2, fits = align_curves(
        curves,
        grid
    )

    mean_c2 = np.nanmean(
        c2,
        axis=0
    )

    if c2.shape[0] > 1:
        sd_c2 = np.nanstd(
            c2,
            axis=0,
            ddof=1
        )
    else:
        sd_c2 = np.zeros_like(
            mean_c2
        )

    sem_c2 = (
        sd_c2 /
        np.sqrt(c2.shape[0])
    )

    if np.isfinite(fits).any():
        mean_fit = np.nanmean(
            fits,
            axis=0
        )
    else:
        mean_fit = np.full_like(
            mean_c2,
            np.nan
        )

    summary = pd.DataFrame({
        "r_um": grid,
        f"{label}_mean_C2": mean_c2,
        f"{label}_SD_C2": sd_c2,
        f"{label}_SEM_C2": sem_c2,
        f"{label}_mean_fit": mean_fit
    })

    aligned = pd.DataFrame({
        "r_um": grid
    })

    for i in range(c2.shape[0]):
        aligned[
            f"{label}_FOV{i+1}_C2"
        ] = c2[i]

    return summary, aligned

# ============================================================
# 4. CHECK FILE PATHS
# ============================================================

check_files(
    FILES_30_70,
    "30:70"
)

check_files(
    FILES_50_50,
    "50:50"
)

# ============================================================
# 5. PROCESS DATA
# ============================================================

summary_3070, aligned_3070 = process_group(
    FILES_30_70,
    "30_70"
)

summary_5050, aligned_5050 = process_group(
    FILES_50_50,
    "50_50"
)

# common grid between compositions
r_min = max(
    summary_3070["r_um"].min(),
    summary_5050["r_um"].min()
)

r_max = min(
    summary_3070["r_um"].max(),
    summary_5050["r_um"].max()
)

step = min(
    np.median(np.diff(summary_3070["r_um"])),
    np.median(np.diff(summary_5050["r_um"]))
)

grid = np.arange(
    r_min,
    r_max + step/2,
    step
)


def interp_col(df, col):
    return np.interp(
        grid,
        df["r_um"],
        df[col]
    )


comparison = pd.DataFrame({
    "r_um": grid,

    "C2_30_70_mean":
        interp_col(
            summary_3070,
            "30_70_mean_C2"
        ),

    "C2_30_70_SD":
        interp_col(
            summary_3070,
            "30_70_SD_C2"
        ),

    "C2_30_70_SEM":
        interp_col(
            summary_3070,
            "30_70_SEM_C2"
        ),

    "Fit_30_70_mean":
        interp_col(
            summary_3070,
            "30_70_mean_fit"
        ),

    "C2_50_50_mean":
        interp_col(
            summary_5050,
            "50_50_mean_C2"
        ),

    "C2_50_50_SD":
        interp_col(
            summary_5050,
            "50_50_SD_C2"
        ),

    "C2_50_50_SEM":
        interp_col(
            summary_5050,
            "50_50_SEM_C2"
        ),

    "Fit_50_50_mean":
        interp_col(
            summary_5050,
            "50_50_mean_fit"
        ),
})

# ============================================================
# 6. SAVE NUMERICAL DATA
# ============================================================

comparison.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "C2r_comparison_mean_SD_SEM.csv"
    ),
    index=False
)

aligned_3070.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "C2r_30_70_aligned_FOV_curves.csv"
    ),
    index=False
)

aligned_5050.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "C2r_50_50_aligned_FOV_curves.csv"
    ),
    index=False
)

pd.DataFrame({
    "Composition":
        ["30:70"] * len(FILES_30_70)
        +
        ["50:50"] * len(FILES_50_50),

    "Input_file":
        FILES_30_70
        +
        FILES_50_50
}).to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "C2r_selected_input_files.csv"
    ),
    index=False
)

# ============================================================
# 7. PUBLICATION PLOT — MEAN ± SD
# ============================================================

r = comparison["r_um"].to_numpy()

y30 = comparison[
    "C2_30_70_mean"
].to_numpy()

sd30 = comparison[
    "C2_30_70_SD"
].to_numpy()

y50 = comparison[
    "C2_50_50_mean"
].to_numpy()

sd50 = comparison[
    "C2_50_50_SD"
].to_numpy()

fit30 = comparison[
    "Fit_30_70_mean"
].to_numpy()

fit50 = comparison[
    "Fit_50_50_mean"
].to_numpy()

fig, ax = plt.subplots(
    figsize=(7.0, 5.4)
)

ax.plot(
    r,
    y30,
    linewidth=2.2,
    label="30:70"
)

ax.fill_between(
    r,
    y30 - sd30,
    y30 + sd30,
    alpha=0.22
)

ax.plot(
    r,
    y50,
    linewidth=2.2,
    label="50:50"
)

ax.fill_between(
    r,
    y50 - sd50,
    y50 + sd50,
    alpha=0.22
)

if np.isfinite(fit30).any():
    ax.plot(
        r,
        fit30,
        linestyle="--",
        linewidth=1.4,
        label="30:70 fit"
    )

if np.isfinite(fit50).any():
    ax.plot(
        r,
        fit50,
        linestyle="--",
        linewidth=1.4,
        label="50:50 fit"
    )

ax.set_xlabel(
    "Distance, r (µm)",
    fontsize=12
)

ax.set_ylabel(
    r"$C_2(r)$",
    fontsize=12
)

ax.set_title(
    "Orientational correlation",
    fontsize=13
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

ax.legend(
    frameon=False
)

plt.tight_layout()

png_path = os.path.join(
    OUTPUT_FOLDER,
    "FIG_C2r_30_70_vs_50_50_mean_SD.png"
)

pdf_path = os.path.join(
    OUTPUT_FOLDER,
    "FIG_C2r_30_70_vs_50_50_mean_SD.pdf"
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

print("\n" + "="*78)
print("C2(r) COMPARISON COMPLETE")
print("="*78)

print("\nSaved:")
print(png_path)
print(pdf_path)

print("\n30:70 files used:")
for f in FILES_30_70:
    print(" ", f)

print("\n50:50 files used:")
for f in FILES_50_50:
    print(" ", f)
