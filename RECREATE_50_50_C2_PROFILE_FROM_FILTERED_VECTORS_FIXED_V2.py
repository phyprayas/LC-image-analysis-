
# ============================================================
# RECREATE C2_profile.csv FROM 50:50 STEP-3 filtered_vectors.csv
# MATCHED TO THE STEP-3 ORIENTATIONAL CORRELATION METHOD
#
# PURPOSE:
# 50:50 Step-3 saved the C2 PNG but not the numerical C2 profile.
# This script recalculates the C2(r) profile from each FOV's
# filtered_vectors.csv and saves:
#       C2_profile.csv
#
# IMPORTANT:
# Use the SAME correlation binning as the analysis you want to compare.
# For strict comparison to the completed 30:70 analysis:
#       CORR_BIN_PX = 20
#       MAX_CORR_DIST_PX = 500
#
# Calibration:
#       10 µm = 157 px
# ============================================================

import os
import numpy as np
import pandas as pd

# -----------------------------
# EDIT THESE 4 PATHS
# -----------------------------
FILTERED_VECTOR_FILES = [
    r"D:\New results\5070\step 3 orientation order correlation\FOV1\filtered_vectors.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV2\filtered_vectors.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV3\filtered_vectors.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV4\filtered_vectors.csv",
]

# -----------------------------
# MATCHED SETTINGS
# -----------------------------
UM_PER_PX = 10.0 / 157.0
CORR_BIN_PX = 20
MAX_CORR_DIST_PX = 500

# Column aliases
X_ALIASES = ["X", "x", "x_px", "X_px"]
Y_ALIASES = ["Y", "y", "y_px", "Y_px"]
THETA_ALIASES = [
    "Orientation", "orientation", "Orientation_deg",
    "theta_deg", "theta", "theta_rad", "angle_deg", "Angle"
]

def find_col(df, aliases, label):
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find {label} column.\n"
        f"Tried: {aliases}\n"
        f"Available: {list(df.columns)}"
    )

def nematic_c2_from_vectors(df):
    xcol = find_col(df, X_ALIASES, "X")
    ycol = find_col(df, Y_ALIASES, "Y")
    tcol = find_col(df, THETA_ALIASES, "orientation")

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
    if tcol == "theta_rad":
        th = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(th)
        x, y, th = x[good], y[good], th[good]
    else:
        th_deg = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(th_deg)
        x, y, th_deg = x[good], y[good], th_deg[good]
        th = np.deg2rad(th_deg)

    if len(x) < 2:
        raise ValueError("Too few valid orientation vectors.")

    # Compute pairwise distances and nematic angular correlation.
    # This is O(N^2); OrientationJ grid datasets of a few thousand
    # points are manageable but can take some time.
    bins = np.arange(
        0,
        MAX_CORR_DIST_PX + CORR_BIN_PX,
        CORR_BIN_PX,
        dtype=float
    )

    sums = np.zeros(len(bins) - 1, dtype=float)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)

    n = len(x)

    for i in range(n - 1):
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        rr = np.sqrt(dx*dx + dy*dy)

        valid = rr <= MAX_CORR_DIST_PX
        if not np.any(valid):
            continue

        rr = rr[valid]
        dtheta = th[i+1:][valid] - th[i]

        c2 = np.cos(2.0 * dtheta)

        inds = np.digitize(rr, bins) - 1
        inside = (inds >= 0) & (inds < len(sums))

        np.add.at(sums, inds[inside], c2[inside])
        np.add.at(counts, inds[inside], 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c2 = sums / counts

    r_center_px = 0.5 * (bins[:-1] + bins[1:])
    r_um = r_center_px * UM_PER_PX

    out = pd.DataFrame({
        "r_px": r_center_px,
        "r_um": r_um,
        "C2": mean_c2,
        "pair_count": counts
    })

    out = out[
        np.isfinite(out["C2"]) &
        (out["pair_count"] > 0)
    ].reset_index(drop=True)

    return out

for path in FILTERED_VECTOR_FILES:

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found:\n{path}\n\n"
            "Edit FILTERED_VECTOR_FILES with your real paths."
        )

    print("\nProcessing:")
    print(path)

    df = pd.read_csv(path)

    c2 = nematic_c2_from_vectors(df)

    out_dir = os.path.dirname(path)
    out_csv = os.path.join(
        out_dir,
        "C2_profile.csv"
    )

    c2.to_csv(
        out_csv,
        index=False
    )

    print("Saved:")
    print(out_csv)

print("\nDONE — C2_profile.csv created for all 50:50 FOVs.")
