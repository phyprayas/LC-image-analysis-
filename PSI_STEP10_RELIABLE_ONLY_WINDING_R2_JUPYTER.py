
# ============================================================
# PSI STUDY — STEP 10
# RELIABLE-ONLY WINDING-FIT QUALITY + FINAL DEFECT SUMMARY
# JUPYTER / VS CODE FRIENDLY
#
# PURPOSE
# -------
# Correct the winding-fit quality summary by calculating R² only
# from defects that passed the Step-4 reliable-fit criteria.
#
# This avoids using the mean R² of all attempted winding fits as
# if it represented only the validated defects.
#
# INPUT:
#   Step-4 reliable-defects CSV for all 8 FOVs
#
# OUTPUT:
#   1) STEP10_RELIABLE_WINDING_PER_FOV.csv
#   2) STEP10_RELIABLE_WINDING_COMPOSITION_SUMMARY.csv
#   3) reliable-only R² comparison plot
#   4) reliable-only |s_fit| comparison plot
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT
# ============================================================

STEP4_ROOT = r"D:\New results\psi map results\New folder\step 4"
OUTPUT_DIR = r"D:\New results\psi map results\New folder\step 10"

DATASETS = [
    ("30_70_FOV1", os.path.join(STEP4_ROOT, "30_70_FOV1", "30_70_FOV1_STEP4_reliable_defects_only.csv")),
    ("30_70_FOV2", os.path.join(STEP4_ROOT, "30_70_FOV2", "30_70_FOV2_STEP4_reliable_defects_only.csv")),
    ("30_70_FOV3", os.path.join(STEP4_ROOT, "30_70_FOV3", "30_70_FOV3_STEP4_reliable_defects_only.csv")),
    ("30_70_FOV4", os.path.join(STEP4_ROOT, "30_70_FOV4", "30_70_FOV4_STEP4_reliable_defects_only.csv")),

    ("50_50_FOV1", os.path.join(STEP4_ROOT, "50_50_FOV1", "50_50_FOV1_STEP4_reliable_defects_only.csv")),
    ("50_50_FOV2", os.path.join(STEP4_ROOT, "50_50_FOV2", "50_50_FOV2_STEP4_reliable_defects_only.csv")),
    ("50_50_FOV3", os.path.join(STEP4_ROOT, "50_50_FOV3", "50_50_FOV3_STEP4_reliable_defects_only.csv")),
    ("50_50_FOV4", os.path.join(STEP4_ROOT, "50_50_FOV4", "50_50_FOV4_STEP4_reliable_defects_only.csv")),
]

DPI = 600
SHOW_PLOTS_IN_NOTEBOOK = True

# ============================================================
# 2. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_csv_robust(path):
    for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def comp_from_label(label):
    return "30:70" if str(label).startswith("30_70") else "50:50"

def mean_sd_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan, 0

    mean = np.mean(values)

    if len(values) > 1:
        sd = np.std(values, ddof=1)
        sem = sd / np.sqrt(len(values))
    else:
        sd = np.nan
        sem = np.nan

    return mean, sd, sem, len(values)

# ============================================================
# 3. PROCESS ALL 8 FOVs
# ============================================================

ensure_dir(OUTPUT_DIR)

per_fov_rows = []

for label, path in DATASETS:

    print("\nProcessing:", label)

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["R2", "s_fit"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing {missing} in:\n{path}\n"
            f"Available columns: {list(df.columns)}"
        )

    # this file should already contain only reliable fits
    r2 = pd.to_numeric(
        df["R2"],
        errors="coerce"
    ).dropna().to_numpy(float)

    sfit = pd.to_numeric(
        df["s_fit"],
        errors="coerce"
    ).dropna().to_numpy(float)

    abs_s = np.abs(sfit)

    row = {
        "Label": label,
        "Composition": comp_from_label(label),
        "N_reliable_defects": len(df),

        "Reliable_R2_mean": np.mean(r2) if len(r2) else np.nan,
        "Reliable_R2_SD": np.std(r2, ddof=1) if len(r2) > 1 else np.nan,
        "Reliable_R2_median": np.median(r2) if len(r2) else np.nan,
        "Reliable_R2_min": np.min(r2) if len(r2) else np.nan,
        "Reliable_R2_max": np.max(r2) if len(r2) else np.nan,

        "Reliable_abs_s_mean": np.mean(abs_s) if len(abs_s) else np.nan,
        "Reliable_abs_s_SD": np.std(abs_s, ddof=1) if len(abs_s) > 1 else np.nan,
        "Reliable_abs_s_median": np.median(abs_s) if len(abs_s) else np.nan,

        "Fraction_close_to_half_charge": (
            np.mean(np.abs(abs_s - 0.5) <= 0.20)
            if len(abs_s)
            else np.nan
        )
    }

    per_fov_rows.append(row)

per_fov = pd.DataFrame(per_fov_rows)

per_fov_path = os.path.join(
    OUTPUT_DIR,
    "STEP10_RELIABLE_WINDING_PER_FOV.csv"
)

per_fov.to_csv(
    per_fov_path,
    index=False
)

print("\nPER-FOV RELIABLE-ONLY SUMMARY")
print(per_fov)

# ============================================================
# 4. COMPOSITION SUMMARY
# ============================================================

metrics = [
    "Reliable_R2_mean",
    "Reliable_abs_s_mean",
    "Fraction_close_to_half_charge",
    "N_reliable_defects"
]

rows = []

for composition in ["30:70", "50:50"]:

    sub = per_fov[
        per_fov["Composition"] == composition
    ]

    for metric in metrics:

        vals = pd.to_numeric(
            sub[metric],
            errors="coerce"
        ).dropna().to_numpy(float)

        mean, sd, sem, n = mean_sd_sem(vals)

        rows.append({
            "Composition": composition,
            "Metric": metric,
            "N_FOV": n,
            "Mean": mean,
            "SD": sd,
            "SEM": sem
        })

composition_summary = pd.DataFrame(rows)

composition_path = os.path.join(
    OUTPUT_DIR,
    "STEP10_RELIABLE_WINDING_COMPOSITION_SUMMARY.csv"
)

composition_summary.to_csv(
    composition_path,
    index=False
)

# ============================================================
# 5. PLOT — RELIABLE-ONLY R²
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.2, 5.0)
)

for xpos, composition in enumerate(["30:70", "50:50"]):

    vals = per_fov.loc[
        per_fov["Composition"] == composition,
        "Reliable_R2_mean"
    ].dropna().to_numpy(float)

    jitter = np.linspace(
        -0.06,
        0.06,
        len(vals)
    ) if len(vals) > 1 else np.array([0.0])

    ax.scatter(
        np.full(len(vals), xpos) + jitter,
        vals,
        s=65,
        edgecolors="black",
        linewidths=0.7
    )

    if len(vals):
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0

        ax.errorbar(
            xpos,
            mean,
            yerr=sd,
            fmt="_",
            markersize=24,
            markeredgewidth=2,
            capsize=5
        )

ax.axhline(
    0.70,
    linestyle="--",
    linewidth=1,
    label=r"Reliability threshold $R^2=0.70$"
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["30:70", "50:50"])
ax.set_ylabel(r"Mean reliable-only winding-fit $R^2$")
ax.set_xlabel("CPC:glycerol composition")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False)
ax.tick_params(direction="in", top=True, right=True)

plt.tight_layout()

r2_png = os.path.join(
    OUTPUT_DIR,
    "STEP10_RELIABLE_ONLY_R2_30_70_vs_50_50.png"
)

r2_pdf = os.path.join(
    OUTPUT_DIR,
    "STEP10_RELIABLE_ONLY_R2_30_70_vs_50_50.pdf"
)

fig.savefig(
    r2_png,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    r2_pdf,
    bbox_inches="tight"
)

if SHOW_PLOTS_IN_NOTEBOOK:
    plt.show()

plt.close(fig)

# ============================================================
# 6. PLOT — RELIABLE |s_fit|
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.2, 5.0)
)

for xpos, composition in enumerate(["30:70", "50:50"]):

    vals = per_fov.loc[
        per_fov["Composition"] == composition,
        "Reliable_abs_s_mean"
    ].dropna().to_numpy(float)

    jitter = np.linspace(
        -0.06,
        0.06,
        len(vals)
    ) if len(vals) > 1 else np.array([0.0])

    ax.scatter(
        np.full(len(vals), xpos) + jitter,
        vals,
        s=65,
        edgecolors="black",
        linewidths=0.7
    )

    if len(vals):
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0

        ax.errorbar(
            xpos,
            mean,
            yerr=sd,
            fmt="_",
            markersize=24,
            markeredgewidth=2,
            capsize=5
        )

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label=r"$|s|=1/2$"
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["30:70", "50:50"])
ax.set_ylabel(r"Mean reliable $|s_{\mathrm{fit}}|$")
ax.set_xlabel("CPC:glycerol composition")
ax.legend(frameon=False)
ax.tick_params(direction="in", top=True, right=True)

plt.tight_layout()

s_png = os.path.join(
    OUTPUT_DIR,
    "STEP10_RELIABLE_ABS_SFIT_30_70_vs_50_50.png"
)

s_pdf = os.path.join(
    OUTPUT_DIR,
    "STEP10_RELIABLE_ABS_SFIT_30_70_vs_50_50.pdf"
)

fig.savefig(
    s_png,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    s_pdf,
    bbox_inches="tight"
)

if SHOW_PLOTS_IN_NOTEBOOK:
    plt.show()

plt.close(fig)

# ============================================================
# 7. FINISH
# ============================================================

print("\n" + "="*80)
print("PSI STEP 10 COMPLETE")
print("="*80)

print("Per-FOV reliable-only summary:")
print(per_fov_path)

print("Composition summary:")
print(composition_path)

print("Reliable-only R2 plot:")
print(r2_png)

print("Reliable |s_fit| plot:")
print(s_png)
