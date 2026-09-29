
# ============================================================
# QDLC1 vs QDLC2
# LOCAL ORDER vs DISTANCE TO NEAREST DEFECT
# EXPLICIT 8-FILE VERSION (4 FOVs + 4 FOVs)
#
# IMPORTANT:
# Do NOT give STEP5_ALL_4FOV_summary.csv here.
# This plot needs POINTWISE Step-5 CSVs containing:
#   - nearest defect distance (um)
#   - local S2D / S_local
#
# QDLC1 = 30:70
# QDLC2 = 50:50
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# 1. USER INPUT — PUT THE 8 POINTWISE CSV FILES HERE
# ============================================================

QDLC1_FILES = [
    r"D:\New results\3070\step 5 order defect coupling\FOV1_POINTWISE.csv",
    r"D:\New results\3070\step 5 order defect coupling\FOV2_POINTWISE.csv",
    r"D:\New results\3070\step 5 order defect coupling\FOV3_POINTWISE.csv",
    r"D:\New results\3070\step 5 order defect coupling\FOV4_POINTWISE.csv",
]

QDLC2_FILES = [
    r"D:\New results\5050\step 5 order defect coupling\FOV1_POINTWISE.csv",
    r"D:\New results\5050\step 5 order defect coupling\FOV2_POINTWISE.csv",
    r"D:\New results\5050\step 5 order defect coupling\FOV3_POINTWISE.csv",
    r"D:\New results\5050\step 5 order defect coupling\FOV4_POINTWISE.csv",
]

OUTPUT_DIR = r"D:\New results\Figure_2_Defect_Analysis_Pretty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. SETTINGS
# ============================================================

BIN_WIDTH_UM = 0.5
MAX_DISTANCE_UM = 20.0
MIN_POINTS_PER_BIN_PER_FOV = 5

ERROR_BAND = "SD"   # "SD" or "SEM"

APPLY_SMOOTHING = True
SMOOTH_WINDOW = 7
SMOOTH_POLYORDER = 2

QDLC1_COLOR = "#1565C0"
QDLC2_COLOR = "#E31A1C"
QDLC1_FILL = "#90CAF9"
QDLC2_FILL = "#EF9A9A"

DPI = 600

# ============================================================
# 3. COLUMN ALIASES
# ============================================================

DISTANCE_ALIASES = [
    "nearest_defect_distance_um",
    "nearest_distance_um",
    "distance_to_nearest_defect_um",
    "nearest_defect_dist_um",
    "defect_distance_um",
    "r_nearest_defect_um",
    "distance_um",
]

SLOCAL_ALIASES = [
    "Slocal",
    "S_local",
    "local_S2D",
    "local_s2d",
    "S2D_local",
    "Local_S2D",
    "local_order",
    "local_order_parameter",
    "S_local_2D",
]

def find_column(df, aliases):
    # exact
    for c in aliases:
        if c in df.columns:
            return c

    # case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    for c in aliases:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None

# ============================================================
# 4. LOAD ONE FOV FILE
# ============================================================

def load_one_file(path, fov_name, sample_name):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nFile not found:\n{path}\n\n"
            "Replace the example filename with your actual POINTWISE Step-5 CSV."
        )

    df = pd.read_csv(path)

    dcol = find_column(df, DISTANCE_ALIASES)
    scol = find_column(df, SLOCAL_ALIASES)

    if dcol is None or scol is None:
        raise KeyError(
            f"\nRequired pointwise columns not found in:\n{path}\n\n"
            f"Available columns:\n{list(df.columns)}\n\n"
            "This is probably a SUMMARY CSV, not the pointwise Step-5 CSV."
        )

    out = pd.DataFrame({
        "Distance_um": pd.to_numeric(df[dcol], errors="coerce"),
        "Slocal": pd.to_numeric(df[scol], errors="coerce"),
    })

    out["FOV"] = fov_name
    out["Sample"] = sample_name
    out["Source_file"] = os.path.basename(path)

    out = out.dropna(subset=["Distance_um", "Slocal"])

    out = out[
        (out["Distance_um"] >= 0)
        & (out["Distance_um"] <= MAX_DISTANCE_UM)
        & (out["Slocal"] >= 0)
        & (out["Slocal"] <= 1.2)
    ].copy()

    print(
        f"{sample_name} {fov_name}: "
        f"{len(out)} points | distance={dcol} | Slocal={scol}"
    )

    return out

# ============================================================
# 5. LOAD ALL 8 FOV FILES
# ============================================================

def load_sample(file_list, sample_name):

    frames = []

    for i, path in enumerate(file_list, start=1):
        frames.append(
            load_one_file(
                path,
                f"FOV{i}",
                sample_name
            )
        )

    return pd.concat(frames, ignore_index=True)

qdlc1_points = load_sample(QDLC1_FILES, "QDLC1")
qdlc2_points = load_sample(QDLC2_FILES, "QDLC2")

# ============================================================
# 6. BIN EACH FOV SEPARATELY
# ============================================================

def bin_each_fov(point_df):

    edges = np.arange(
        0,
        MAX_DISTANCE_UM + BIN_WIDTH_UM,
        BIN_WIDTH_UM
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []

    for fov, sub in point_df.groupby("FOV"):

        dist = sub["Distance_um"].to_numpy(float)
        sloc = sub["Slocal"].to_numpy(float)

        idx = np.digitize(dist, edges) - 1

        for i, center in enumerate(centers):

            vals = sloc[idx == i]

            if len(vals) < MIN_POINTS_PER_BIN_PER_FOV:
                continue

            rows.append({
                "FOV": fov,
                "Distance_um": center,
                "Mean_Slocal": np.mean(vals),
                "N_points": len(vals)
            })

    return pd.DataFrame(rows)

qdlc1_binned = bin_each_fov(qdlc1_points)
qdlc2_binned = bin_each_fov(qdlc2_points)

# ============================================================
# 7. COMPOSITION MEAN ± SD/SEM ACROSS FOVs
# ============================================================

def composition_curve(fov_binned):

    rows = []

    for dist, sub in fov_binned.groupby("Distance_um"):

        vals = sub["Mean_Slocal"].dropna().to_numpy(float)

        if len(vals) == 0:
            continue

        mean = np.mean(vals)

        sd = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
        sem = sd / np.sqrt(len(vals)) if len(vals) > 1 else np.nan

        rows.append({
            "Distance_um": dist,
            "Mean_Slocal": mean,
            "SD": sd,
            "SEM": sem,
            "N_FOV": len(vals)
        })

    return pd.DataFrame(rows).sort_values("Distance_um")

curve1 = composition_curve(qdlc1_binned)
curve2 = composition_curve(qdlc2_binned)

# ============================================================
# 8. OPTIONAL SMOOTHING
# ============================================================

def smooth(y):
    y = np.asarray(y, float)

    if not APPLY_SMOOTHING or len(y) < SMOOTH_WINDOW:
        return y

    win = SMOOTH_WINDOW

    if win % 2 == 0:
        win += 1

    if win > len(y):
        win = len(y)
        if win % 2 == 0:
            win -= 1

    if win <= SMOOTH_POLYORDER:
        return y

    return savgol_filter(
        y,
        window_length=win,
        polyorder=SMOOTH_POLYORDER,
        mode="interp"
    )

# ============================================================
# 9. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6.4, 5.0))

# ---- QDLC1
x1 = curve1["Distance_um"].to_numpy(float)
y1 = curve1["Mean_Slocal"].to_numpy(float)
e1 = curve1[ERROR_BAND].to_numpy(float)

ax.fill_between(
    x1,
    y1 - e1,
    y1 + e1,
    color=QDLC1_FILL,
    alpha=0.45,
    linewidth=0
)

ax.plot(
    x1,
    smooth(y1),
    color=QDLC1_COLOR,
    linewidth=2.4,
    label="QDLC1"
)

# ---- QDLC2
x2 = curve2["Distance_um"].to_numpy(float)
y2 = curve2["Mean_Slocal"].to_numpy(float)
e2 = curve2[ERROR_BAND].to_numpy(float)

ax.fill_between(
    x2,
    y2 - e2,
    y2 + e2,
    color=QDLC2_FILL,
    alpha=0.42,
    linewidth=0
)

ax.plot(
    x2,
    smooth(y2),
    color=QDLC2_COLOR,
    linewidth=2.4,
    label="QDLC2"
)

# ============================================================
# 10. STYLE
# ============================================================

ax.set_xlim(0, MAX_DISTANCE_UM)
ax.set_ylim(0, 1.0)

ax.set_xlabel(
    r"Distance to nearest defect ($\mu$m)",
    fontsize=13
)

ax.set_ylabel(
    r"$S_{\mathrm{local}}$",
    fontsize=14
)

ax.legend(
    frameon=False,
    fontsize=11,
    loc="upper right"
)

# Full boundary box
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

# Panel title
ax.text(
    -0.13,
    1.06,
    "(d)",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold"
)

ax.text(
    0.00,
    1.035,
    "Local order vs. distance to defect",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold"
)

ax.grid(False)

plt.tight_layout()

# ============================================================
# 11. SAVE
# ============================================================

png = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_LOCAL_ORDER_vs_DEFECT_DISTANCE.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_LOCAL_ORDER_vs_DEFECT_DISTANCE.pdf"
)

svg = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_LOCAL_ORDER_vs_DEFECT_DISTANCE.svg"
)

fig.savefig(png, dpi=DPI, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")

plt.show()

# save numerical outputs
qdlc1_binned.to_csv(
    os.path.join(OUTPUT_DIR, "QDLC1_FOV_BINNED.csv"),
    index=False
)

qdlc2_binned.to_csv(
    os.path.join(OUTPUT_DIR, "QDLC2_FOV_BINNED.csv"),
    index=False
)

curve1.to_csv(
    os.path.join(OUTPUT_DIR, "QDLC1_FINAL_CURVE.csv"),
    index=False
)

curve2.to_csv(
    os.path.join(OUTPUT_DIR, "QDLC2_FINAL_CURVE.csv"),
    index=False
)

print("\nDONE")
print("PNG:", png)
print("PDF:", pdf)
print("SVG:", svg)
