
# ============================================================
# ORDER RECOVERY S(r) + lambda_S
# QDLC1 (30:70) vs QDLC2 (50:50)
# Manuscript-ready Figure 2(e)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

# ============================================================
# 1. USER INPUT
# ============================================================

QDLC1_PROFILE_FILES = [
    r"D:\New results\3070\step 7 order recovery\FOV1_RECOVERY_PROFILE.csv",
    r"D:\New results\3070\step 7 order recovery\FOV2_RECOVERY_PROFILE.csv",
    r"D:\New results\3070\step 7 order recovery\FOV3_RECOVERY_PROFILE.csv",
    r"D:\New results\3070\step 7 order recovery\FOV4_RECOVERY_PROFILE.csv",
]

QDLC2_PROFILE_FILES = [
    r"D:\New results\5050\step 7 order recovery\FOV1_RECOVERY_PROFILE.csv",
    r"D:\New results\5050\step 7 order recovery\FOV2_RECOVERY_PROFILE.csv",
    r"D:\New results\5050\step 7 order recovery\FOV3_RECOVERY_PROFILE.csv",
    r"D:\New results\5050\step 7 order recovery\FOV4_RECOVERY_PROFILE.csv",
]

OUTPUT_DIR = r"D:\New results\Figure_2_Defect_Analysis_Pretty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_RADIUS_UM = 8.0
DR_UM = 0.25
MIN_FOV_PER_RADIUS = 2

ERROR_BAND = "SD"   # "SD" or "SEM"

QDLC1_COLOR = "#1565C0"
QDLC2_COLOR = "#E31A1C"
QDLC1_FILL = "#90CAF9"
QDLC2_FILL = "#EF9A9A"

DPI = 600

# ============================================================
# 2. POSSIBLE COLUMN NAMES
# ============================================================

RADIUS_ALIASES = [
    "r_um",
    "radius_um",
    "distance_um",
    "Distance_um",
    "radial_distance_um",
    "distance_from_defect_um",
    "r",
]

ORDER_ALIASES = [
    "mean_Slocal",
    "Mean_Slocal",
    "Slocal_mean",
    "S_local_mean",
    "mean_local_S2D",
    "mean_local_order",
    "S_mean",
    "S_r",
    "Slocal",
    "S_local",
    "local_S2D",
]

# ============================================================
# 3. HELPERS
# ============================================================

def find_column(df, aliases, label):
    for c in aliases:
        if c in df.columns:
            return c

    lower_map = {c.lower(): c for c in df.columns}

    for c in aliases:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    raise KeyError(
        f"Could not find {label} column.\n"
        f"Tried: {aliases}\n"
        f"Available columns: {list(df.columns)}"
    )


def load_profile(path, fov_name, sample_name):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found:\n{path}\n\n"
            "Replace the example filename with your actual Step-7 recovery-profile CSV."
        )

    df = pd.read_csv(path)

    rcol = find_column(df, RADIUS_ALIASES, "radius")
    scol = find_column(df, ORDER_ALIASES, "order / Slocal")

    out = pd.DataFrame({
        "r_um": pd.to_numeric(df[rcol], errors="coerce"),
        "S": pd.to_numeric(df[scol], errors="coerce"),
    }).dropna()

    out = out[
        (out["r_um"] >= 0)
        & (out["r_um"] <= MAX_RADIUS_UM)
        & (out["S"] >= 0)
        & (out["S"] <= 1.2)
    ].copy()

    out = out.sort_values("r_um")

    if len(out) < 4:
        raise ValueError(f"Too few valid points in:\n{path}")

    out["FOV"] = fov_name
    out["Sample"] = sample_name

    print(
        f"{sample_name} {fov_name}: "
        f"{len(out)} points | r={rcol} | S={scol}"
    )

    return out


def load_sample_profiles(file_list, sample_name):
    frames = []
    for i, path in enumerate(file_list, start=1):
        frames.append(load_profile(path, f"FOV{i}", sample_name))
    return frames


def aggregate_profiles(profile_frames):

    common_r = np.arange(
        0,
        MAX_RADIUS_UM + DR_UM/2,
        DR_UM
    )

    curves = []

    for df in profile_frames:
        r = df["r_um"].to_numpy(float)
        s = df["S"].to_numpy(float)

        interp = np.full(common_r.shape, np.nan)

        valid_range = (
            (common_r >= np.min(r))
            & (common_r <= np.max(r))
        )

        interp[valid_range] = np.interp(
            common_r[valid_range],
            r,
            s
        )

        curves.append(interp)

    arr = np.vstack(curves)

    rows = []

    for j, rr in enumerate(common_r):
        vals = arr[:, j]
        vals = vals[np.isfinite(vals)]

        if len(vals) < MIN_FOV_PER_RADIUS:
            continue

        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
        sem = sd / np.sqrt(len(vals)) if len(vals) > 1 else np.nan

        rows.append({
            "r_um": rr,
            "Mean_S": mean,
            "SD": sd,
            "SEM": sem,
            "N_FOV": len(vals)
        })

    return pd.DataFrame(rows)


def recovery_model(r, S_inf, A, lambda_S):
    return S_inf - A * np.exp(-r / lambda_S)


def fit_recovery_curve(summary_df):

    r = summary_df["r_um"].to_numpy(float)
    S = summary_df["Mean_S"].to_numpy(float)

    valid = np.isfinite(r) & np.isfinite(S)
    r = r[valid]
    S = S[valid]

    if len(r) < 5:
        raise ValueError("Not enough points to fit recovery model.")

    S_inf0 = float(np.nanmean(S[-max(3, len(S)//4):]))
    S_inf0 = min(max(S_inf0, 0.1), 1.0)

    A0 = max(0.05, S_inf0 - S[0])
    lambda0 = 2.5

    popt, pcov = curve_fit(
        recovery_model,
        r,
        S,
        p0=[S_inf0, A0, lambda0],
        bounds=([0.0, 0.0, 0.05], [1.2, 1.5, 50.0]),
        maxfev=20000
    )

    perr = np.sqrt(np.diag(pcov))
    pred = recovery_model(r, *popt)

    ss_res = np.sum((S - pred)**2)
    ss_tot = np.sum((S - np.mean(S))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return {
        "S_inf": popt[0],
        "S_inf_error": perr[0],
        "A": popt[1],
        "A_error": perr[1],
        "lambda_S_um": popt[2],
        "lambda_S_error_um": perr[2],
        "R2": r2
    }

# ============================================================
# 4. LOAD + AGGREGATE
# ============================================================

qdlc1_frames = load_sample_profiles(QDLC1_PROFILE_FILES, "QDLC1")
qdlc2_frames = load_sample_profiles(QDLC2_PROFILE_FILES, "QDLC2")

qdlc1_summary = aggregate_profiles(qdlc1_frames)
qdlc2_summary = aggregate_profiles(qdlc2_frames)

fit1 = fit_recovery_curve(qdlc1_summary)
fit2 = fit_recovery_curve(qdlc2_summary)

# ============================================================
# 5. SAVE NUMERICAL DATA
# ============================================================

qdlc1_summary.to_csv(
    os.path.join(OUTPUT_DIR, "QDLC1_ORDER_RECOVERY_MEAN_SD.csv"),
    index=False
)

qdlc2_summary.to_csv(
    os.path.join(OUTPUT_DIR, "QDLC2_ORDER_RECOVERY_MEAN_SD.csv"),
    index=False
)

pd.DataFrame([
    {"Sample": "QDLC1", **fit1},
    {"Sample": "QDLC2", **fit2}
]).to_csv(
    os.path.join(OUTPUT_DIR, "QDLC1_QDLC2_ORDER_RECOVERY_FIT_SUMMARY.csv"),
    index=False
)

# ============================================================
# 6. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6.5, 5.2))

# QDLC1
r1 = qdlc1_summary["r_um"].to_numpy(float)
s1 = qdlc1_summary["Mean_S"].to_numpy(float)
e1 = qdlc1_summary[ERROR_BAND].to_numpy(float)

ax.fill_between(
    r1, s1-e1, s1+e1,
    color=QDLC1_FILL,
    alpha=0.40,
    linewidth=0
)

ax.scatter(
    r1, s1,
    s=20,
    color=QDLC1_COLOR,
    zorder=4
)

r1_fit = np.linspace(np.min(r1), np.max(r1), 400)

ax.plot(
    r1_fit,
    recovery_model(
        r1_fit,
        fit1["S_inf"],
        fit1["A"],
        fit1["lambda_S_um"]
    ),
    color=QDLC1_COLOR,
    linewidth=2.4
)

# QDLC2
r2 = qdlc2_summary["r_um"].to_numpy(float)
s2 = qdlc2_summary["Mean_S"].to_numpy(float)
e2 = qdlc2_summary[ERROR_BAND].to_numpy(float)

ax.fill_between(
    r2, s2-e2, s2+e2,
    color=QDLC2_FILL,
    alpha=0.38,
    linewidth=0
)

ax.scatter(
    r2, s2,
    s=20,
    color=QDLC2_COLOR,
    zorder=4
)

r2_fit = np.linspace(np.min(r2), np.max(r2), 400)

ax.plot(
    r2_fit,
    recovery_model(
        r2_fit,
        fit2["S_inf"],
        fit2["A"],
        fit2["lambda_S_um"]
    ),
    color=QDLC2_COLOR,
    linewidth=2.4
)

# lambda markers
lambda1 = fit1["lambda_S_um"]
lambda2 = fit2["lambda_S_um"]

ax.axvline(
    lambda1,
    color=QDLC1_COLOR,
    linestyle="--",
    linewidth=1.2,
    alpha=0.8
)

ax.axvline(
    lambda2,
    color=QDLC2_COLOR,
    linestyle="--",
    linewidth=1.2,
    alpha=0.8
)

# compact lambda labels
ax.text(
    lambda1 + 0.08,
    0.93,
    rf"$\lambda_S$={lambda1:.2f} $\mu$m",
    color=QDLC1_COLOR,
    fontsize=10,
    rotation=90,
    va="top"
)

ax.text(
    lambda2 + 0.08,
    0.72,
    rf"$\lambda_S$={lambda2:.2f} $\mu$m",
    color=QDLC2_COLOR,
    fontsize=10,
    rotation=90,
    va="top"
)

# labels
ax.set_xlabel(
    r"Distance from defect, $r$ ($\mu$m)",
    fontsize=13
)

ax.set_ylabel(
    r"$S(r)$",
    fontsize=14
)

ax.set_xlim(0, MAX_RADIUS_UM)
ax.set_ylim(0, 1.0)

legend_handles = [
    Line2D([0], [0], color=QDLC1_COLOR, lw=2.4, marker="o", markersize=4, label="QDLC1"),
    Line2D([0], [0], color=QDLC2_COLOR, lw=2.4, marker="o", markersize=4, label="QDLC2"),
]

ax.legend(
    handles=legend_handles,
    frameon=False,
    fontsize=11,
    loc="lower right"
)

# Full border
for side in ["top", "right", "bottom", "left"]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.5)

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.2,
    labelsize=11
)

ax.grid(False)

ax.text(
    -0.13, 1.06, "(e)",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold"
)

ax.text(
    0.00, 1.035,
    r"Order recovery $S(r)$",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold"
)

plt.tight_layout()

# ============================================================
# 7. SAVE FIGURE
# ============================================================

PNG_PATH = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_ORDER_RECOVERY_LAMBDA_S.png"
)

PDF_PATH = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_ORDER_RECOVERY_LAMBDA_S.pdf"
)

SVG_PATH = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_ORDER_RECOVERY_LAMBDA_S.svg"
)

fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight")
fig.savefig(PDF_PATH, bbox_inches="tight")
fig.savefig(SVG_PATH, bbox_inches="tight")

plt.show()
plt.close(fig)

print("\nDONE")
print(f"QDLC1 lambda_S = {fit1['lambda_S_um']:.3f} ± {fit1['lambda_S_error_um']:.3f} µm")
print(f"QDLC2 lambda_S = {fit2['lambda_S_um']:.3f} ± {fit2['lambda_S_error_um']:.3f} µm")
print(PNG_PATH)
print(PDF_PATH)
print(SVG_PATH)
