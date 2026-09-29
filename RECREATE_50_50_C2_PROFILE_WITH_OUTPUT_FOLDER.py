
# ============================================================
# RECREATE 50:50 C2_profile.csv FROM filtered_vectors.csv
# WITH MANUAL OUTPUT FOLDER
#
# INPUT:
#   4 filtered_vectors.csv files
#
# OUTPUT:
#   4 C2_profile.csv files saved in OUTPUT_FOLDER
#   filenames:
#       C2_profile_FOV1.csv
#       C2_profile_FOV2.csv
#       C2_profile_FOV3.csv
#       C2_profile_FOV4.csv
#
# MATCHED SETTINGS:
#   10 µm = 157 px
#   correlation bin = 20 px
#   maximum distance = 500 px
# ============================================================

import os
import numpy as np
import pandas as pd

# ============================================================
# 1. EDIT INPUT FILE PATHS
# ============================================================

FILTERED_VECTOR_FILES = [
    r"D:\New results\5070\step 3 orientation order correlation\FOV1\filtered_vectors.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV2\filtered_vectors.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV3\filtered_vectors.csv",
    r"D:\New results\5070\step 3 orientation order correlation\FOV4\filtered_vectors.csv",
]

# ============================================================
# 2. EDIT OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = r"D:\New results\5070\step 3 orientation order correlation\C2_profiles_for_comparison"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. MATCHED SETTINGS
# ============================================================

UM_PER_PX = 10.0 / 157.0
CORR_BIN_PX = 20
MAX_CORR_DIST_PX = 500

X_ALIASES = ["X", "x", "x_px", "X_px"]
Y_ALIASES = ["Y", "y", "y_px", "Y_px"]

THETA_ALIASES = [
    "Orientation",
    "orientation",
    "Orientation_deg",
    "theta_deg",
    "theta",
    "theta_rad",
    "angle_deg",
    "Angle"
]

# ============================================================
# 4. HELPERS
# ============================================================

def find_col(df, aliases, label):
    for c in aliases:
        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find {label} column.\n"
        f"Tried: {aliases}\n"
        f"Available columns: {list(df.columns)}"
    )


def nematic_c2_from_vectors(df):

    xcol = find_col(df, X_ALIASES, "X")
    ycol = find_col(df, Y_ALIASES, "Y")
    tcol = find_col(df, THETA_ALIASES, "orientation")

    x = pd.to_numeric(
        df[xcol],
        errors="coerce"
    ).to_numpy(float)

    y = pd.to_numeric(
        df[ycol],
        errors="coerce"
    ).to_numpy(float)

    if tcol == "theta_rad":

        th = pd.to_numeric(
            df[tcol],
            errors="coerce"
        ).to_numpy(float)

        good = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(th)
        )

        x = x[good]
        y = y[good]
        th = th[good]

    else:

        th_deg = pd.to_numeric(
            df[tcol],
            errors="coerce"
        ).to_numpy(float)

        good = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(th_deg)
        )

        x = x[good]
        y = y[good]
        th_deg = th_deg[good]

        th = np.deg2rad(
            th_deg
        )

    if len(x) < 2:
        raise ValueError(
            "Too few valid orientation vectors."
        )

    bins = np.arange(
        0,
        MAX_CORR_DIST_PX + CORR_BIN_PX,
        CORR_BIN_PX,
        dtype=float
    )

    sums = np.zeros(
        len(bins) - 1,
        dtype=float
    )

    counts = np.zeros(
        len(bins) - 1,
        dtype=np.int64
    )

    n = len(x)

    for i in range(n - 1):

        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]

        rr = np.sqrt(
            dx * dx + dy * dy
        )

        valid = (
            rr <= MAX_CORR_DIST_PX
        )

        if not np.any(valid):
            continue

        rr = rr[valid]

        dtheta = (
            th[i+1:][valid]
            - th[i]
        )

        c2 = np.cos(
            2.0 * dtheta
        )

        inds = (
            np.digitize(
                rr,
                bins
            ) - 1
        )

        inside = (
            (inds >= 0)
            & (inds < len(sums))
        )

        np.add.at(
            sums,
            inds[inside],
            c2[inside]
        )

        np.add.at(
            counts,
            inds[inside],
            1
        )

    with np.errstate(
        invalid="ignore",
        divide="ignore"
    ):

        mean_c2 = (
            sums / counts
        )

    r_center_px = (
        0.5
        * (
            bins[:-1]
            + bins[1:]
        )
    )

    r_um = (
        r_center_px
        * UM_PER_PX
    )

    out = pd.DataFrame({
        "r_px": r_center_px,
        "r_um": r_um,
        "C2": mean_c2,
        "pair_count": counts
    })

    out = (
        out[
            np.isfinite(
                out["C2"]
            )
            &
            (
                out["pair_count"]
                > 0
            )
        ]
        .reset_index(
            drop=True
        )
    )

    return out

# ============================================================
# 5. PROCESS ALL 4 FOVs
# ============================================================

summary_rows = []

for i, path in enumerate(
    FILTERED_VECTOR_FILES,
    start=1
):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"\nFOV{i} file not found:\n"
            f"{path}\n\n"
            "Edit FILTERED_VECTOR_FILES with the correct paths."
        )

    print(
        f"\nProcessing FOV{i}:"
    )

    print(
        path
    )

    df = pd.read_csv(
        path
    )

    c2 = nematic_c2_from_vectors(
        df
    )

    output_csv = os.path.join(
        OUTPUT_FOLDER,
        f"C2_profile_FOV{i}.csv"
    )

    c2.to_csv(
        output_csv,
        index=False
    )

    summary_rows.append({
        "FOV": f"FOV{i}",
        "input_file": path,
        "output_file": output_csv,
        "n_C2_bins": len(c2),
        "r_min_um": c2["r_um"].min(),
        "r_max_um": c2["r_um"].max()
    })

    print(
        "Saved:"
    )

    print(
        output_csv
    )

# ============================================================
# 6. SAVE RUN SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_csv = os.path.join(
    OUTPUT_FOLDER,
    "C2_profile_generation_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)

print(
    "\n" + "=" * 80
)

print(
    "DONE — ALL 50:50 C2 PROFILES CREATED"
)

print(
    "=" * 80
)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_FOLDER
)

print(
    "\nSummary:"
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nSummary CSV:"
)

print(
    summary_csv
)
