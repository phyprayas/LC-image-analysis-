
# ============================================================
# STEP 3 — 30:70 + NaOH, 4 FOVs
# PATH-BASED BATCH ANALYSIS
#
# Outputs per FOV:
#   1) Filtered vectors after mask + coherency threshold
#   2) Global S2D
#   3) Local S2D
#   4) Nematic orientational correlation C2(r)
#   5) Correlation length xi
#   6) Raw ±1/2 winding-derived defect candidates
#
# Final outputs:
#   - STEP3_ALL_4FOV_summary.csv
#   - STEP3_30_70_GROUP_mean_SD_SEM.csv
#
# IMPORTANT:
#   Edit ONLY the file/folder paths in SECTION 1.
#   Keep all analysis settings identical for all 4 FOVs.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

CSV_1 = r"C:\Users\Prayas Singh\Desktop\3070\OJ-Table-Vector-Field-3070-1.csv"
CSV_2 = r"C:\Users\Prayas Singh\Desktop\3070\OJ-Table-Vector-Field-3070-2.csv"
CSV_3 = r"C:\Users\Prayas Singh\Desktop\3070\OJ-Table-Vector-Field-3070-3.csv"
CSV_4 = r"C:\Users\Prayas Singh\Desktop\3070\OJ-Table-Vector-Field-3070-4.csv"

MASK_1 = r"C:\Users\Prayas Singh\Desktop\3070\FOV1\02_valid_region_mask.png"
MASK_2 = r"C:\Users\Prayas Singh\Desktop\3070\FOV2\02_valid_region_mask.png"
MASK_3 = r"C:\Users\Prayas Singh\Desktop\3070\FOV3\02_valid_region_mask.png"
MASK_4 = r"C:\Users\Prayas Singh\Desktop\3070\FOV4\02_valid_region_mask.png"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS"

# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

# Same threshold for all images
COHERENCY_THRESHOLD = 0.30

# OrientationJ grid spacing used during export
GRID_SPACING_PX = 20.0

# Local S2D neighborhood:
# 2 grid spacings on each side ~ 40 px radius
LOCAL_RADIUS_PX = 40.0

# Correlation
CORR_BIN_WIDTH_PX = 20.0
CORR_MAX_R_PX = 500.0

# Defect detection tolerance
DEFECT_TOL = 0.20

# Physical calibration for 30:70 + NaOH
# 10 µm scale bar = 157 px
PIXEL_SIZE_UM = 10.0 / 157.0

# ============================================================
# 3. DATASET DEFINITION
# ============================================================

DATASETS = [
    {
        "name": "30_70_NaOH_FOV1",
        "csv": CSV_1,
        "mask": MASK_1,
    },
    {
        "name": "30_70_NaOH_FOV2",
        "csv": CSV_2,
        "mask": MASK_2,
    },
    {
        "name": "30_70_NaOH_FOV3",
        "csv": CSV_3,
        "mask": MASK_3,
    },
    {
        "name": "30_70_NaOH_FOV4",
        "csv": CSV_4,
        "mask": MASK_4,
    },
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def load_mask(mask_path):
    """
    White/nonzero = valid
    Black = excluded
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 127


def sample_mask(mask, x, y):
    """
    Sample binary mask at OrientationJ X/Y coordinates.
    """
    h, w = mask.shape

    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)

    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)

    return mask[yi, xi]


def global_s2d(theta_deg):
    """
    2D nematic orientational order:
    S2D = | < exp(i 2 theta) > |
    """
    theta = np.deg2rad(theta_deg)

    z = np.exp(1j * 2.0 * theta)

    return np.abs(np.mean(z))


def local_s2d(coords, theta_deg, radius_px):
    """
    Local nematic order in a circular neighborhood.
    """
    tree = cKDTree(coords)

    theta = np.deg2rad(theta_deg)
    z = np.exp(1j * 2.0 * theta)

    local_values = np.full(len(coords), np.nan)

    neighbor_lists = tree.query_ball_point(
        coords,
        r=radius_px
    )

    for i, ids in enumerate(neighbor_lists):

        # center point + at least 2 neighbors
        if len(ids) >= 3:
            local_values[i] = np.abs(
                np.mean(z[ids])
            )

    return local_values


def compute_c2(
    coords,
    theta_deg,
    bin_width_px,
    max_r_px
):
    """
    Nematic correlation:

    C2(r) = < cos[2(theta_i - theta_j)] >

    Uses unique point pairs within max_r_px.
    """

    theta = np.deg2rad(theta_deg)

    tree = cKDTree(coords)

    pair_set = tree.query_pairs(
        r=max_r_px
    )

    pairs = np.array(
        list(pair_set),
        dtype=int
    )

    bins = np.arange(
        0,
        max_r_px + bin_width_px,
        bin_width_px
    )

    centers = 0.5 * (
        bins[:-1] + bins[1:]
    )

    C2 = np.full(
        len(centers),
        np.nan
    )

    counts = np.zeros(
        len(centers),
        dtype=int
    )

    if len(pairs) == 0:
        return centers, C2, counts

    i = pairs[:, 0]
    j = pairs[:, 1]

    distances = np.linalg.norm(
        coords[i] - coords[j],
        axis=1
    )

    corr = np.cos(
        2.0 * (
            theta[i] - theta[j]
        )
    )

    bin_id = np.digitize(
        distances,
        bins
    ) - 1

    for b in range(len(centers)):

        sel = (
            bin_id == b
        )

        counts[b] = np.sum(sel)

        if counts[b] > 0:
            C2[b] = np.mean(
                corr[sel]
            )

    return centers, C2, counts


def c2_model(
    r,
    A,
    xi,
    C_inf
):
    """
    C2(r) = A exp(-r/xi) + C_inf
    """
    return (
        A * np.exp(-r / xi)
        + C_inf
    )


def fit_c2(
    r,
    C2,
    counts
):
    """
    Fit orientational correlation profile.
    Only bins with >=10 pairs are used.
    """

    good = (
        np.isfinite(C2)
        & (counts >= 10)
    )

    x = r[good]
    y = C2[good]

    if len(x) < 5:
        return None

    A0 = max(
        y[0] - y[-1],
        0.05
    )

    xi0 = 50.0
    C0 = y[-1]

    try:

        popt, pcov = curve_fit(
            c2_model,
            x,
            y,
            p0=[
                A0,
                xi0,
                C0
            ],
            bounds=(
                [0.0, 1.0, -1.0],
                [2.0, 5000.0, 1.0]
            ),
            maxfev=30000
        )

        fit_y = c2_model(
            x,
            *popt
        )

        residuals = (
            y - fit_y
        )

        ss_res = np.sum(
            residuals ** 2
        )

        ss_tot = np.sum(
            (y - np.mean(y)) ** 2
        )

        if ss_tot > 0:
            R2 = (
                1.0
                - ss_res / ss_tot
            )
        else:
            R2 = np.nan

        errors = np.sqrt(
            np.diag(pcov)
        )

        return {
            "A": popt[0],
            "xi_px": popt[1],
            "C_inf": popt[2],
            "A_err": errors[0],
            "xi_px_err": errors[1],
            "C_inf_err": errors[2],
            "R2": R2,
            "fit_x": x,
            "fit_y": fit_y
        }

    except Exception as e:

        print(
            "C2 fitting failed:",
            e
        )

        return None


def nematic_diff(
    theta2,
    theta1
):
    """
    Wrapped nematic angular difference.
    """
    d = theta2 - theta1

    return (
        0.5
        * np.arctan2(
            np.sin(2*d),
            np.cos(2*d)
        )
    )


def detect_half_defects(
    df
):
    """
    Winding-based ±1/2 defect candidate detection
    on adjacent OrientationJ grid cells.
    """

    xs = np.sort(
        df["X"].unique()
    )

    ys = np.sort(
        df["Y"].unique()
    )

    theta_pivot = df.pivot(
        index="Y",
        columns="X",
        values="Orientation"
    ).reindex(
        index=ys,
        columns=xs
    )

    keep_pivot = df.pivot(
        index="Y",
        columns="X",
        values="keep"
    ).reindex(
        index=ys,
        columns=xs
    )

    theta = np.deg2rad(
        theta_pivot.values
    )

    keep = (
        keep_pivot
        .fillna(False)
        .values
        .astype(bool)
    )

    defects = []

    for iy in range(
        len(ys) - 1
    ):

        for ix in range(
            len(xs) - 1
        ):

            # four corners
            valid = (
                keep[iy, ix]
                and keep[iy, ix+1]
                and keep[iy+1, ix+1]
                and keep[iy+1, ix]
            )

            if not valid:
                continue

            t0 = theta[
                iy,
                ix
            ]

            t1 = theta[
                iy,
                ix+1
            ]

            t2 = theta[
                iy+1,
                ix+1
            ]

            t3 = theta[
                iy+1,
                ix
            ]

            total = (
                nematic_diff(t1, t0)
                + nematic_diff(t2, t1)
                + nematic_diff(t3, t2)
                + nematic_diff(t0, t3)
            )

            charge = (
                total
                / (2.0 * np.pi)
            )

            defect_type = None

            if abs(
                charge - 0.5
            ) <= DEFECT_TOL:

                defect_type = "+1/2"

            elif abs(
                charge + 0.5
            ) <= DEFECT_TOL:

                defect_type = "-1/2"

            if defect_type is not None:

                defects.append({
                    "x_px":
                        0.5
                        * (
                            xs[ix]
                            + xs[ix+1]
                        ),

                    "y_px":
                        0.5
                        * (
                            ys[iy]
                            + ys[iy+1]
                        ),

                    "charge":
                        charge,

                    "defect_type":
                        defect_type
                })

    return pd.DataFrame(
        defects
    )


# ============================================================
# 5. PROCESS EACH FOV
# ============================================================

all_summary = []

for dataset in DATASETS:

    name = dataset["name"]
    csv_path = dataset["csv"]
    mask_path = dataset["mask"]

    print(
        "\n"
        + "="*80
    )

    print(
        "Processing:",
        name
    )

    print(
        "="*80
    )

    # --------------------------------------------------------
    # Path checks
    # --------------------------------------------------------

    if not os.path.exists(
        csv_path
    ):
        raise FileNotFoundError(
            f"CSV not found:\n{csv_path}"
        )

    if not os.path.exists(
        mask_path
    ):
        raise FileNotFoundError(
            f"Mask not found:\n{mask_path}"
        )

    # output subfolder
    out_dir = os.path.join(
        OUTPUT_FOLDER,
        name
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load OrientationJ CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path
    )

    required_cols = [
        "X",
        "Y",
        "Orientation",
        "Coherency"
    ]

    missing = [
        c
        for c in required_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}"
        )

    for col in required_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_cols
    ).copy()

    # --------------------------------------------------------
    # Apply valid-region mask
    # --------------------------------------------------------

    mask = load_mask(
        mask_path
    )

    df["mask_valid"] = sample_mask(
        mask,
        df["X"].values,
        df["Y"].values
    )

    # --------------------------------------------------------
    # Apply coherency threshold
    # --------------------------------------------------------

    df["keep"] = (
        df["mask_valid"]
        & (
            df["Coherency"]
            >= COHERENCY_THRESHOLD
        )
    )

    use = df[
        df["keep"]
    ].copy()

    print(
        "Total vectors:",
        len(df)
    )

    print(
        "Retained vectors:",
        len(use)
    )

    if len(use) < 20:
        raise RuntimeError(
            f"{name}: too few valid vectors."
        )

    # --------------------------------------------------------
    # Coordinates and orientation
    # --------------------------------------------------------

    coords = use[
        ["X", "Y"]
    ].to_numpy()

    theta_deg = use[
        "Orientation"
    ].to_numpy()

    # --------------------------------------------------------
    # Global S2D
    # --------------------------------------------------------

    global_S = global_s2d(
        theta_deg
    )

    # --------------------------------------------------------
    # Local S2D
    # --------------------------------------------------------

    local_S = local_s2d(
        coords,
        theta_deg,
        radius_px=LOCAL_RADIUS_PX
    )

    use[
        "local_S2D"
    ] = local_S

    use.to_csv(
        os.path.join(
            out_dir,
            "filtered_vectors_with_local_S2D.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # C2(r)
    # --------------------------------------------------------

    r_px, C2, pair_counts = compute_c2(
        coords,
        theta_deg,
        bin_width_px=
            CORR_BIN_WIDTH_PX,
        max_r_px=
            CORR_MAX_R_PX
    )

    c2_fit = fit_c2(
        r_px,
        C2,
        pair_counts
    )

    C2_table = pd.DataFrame({
        "r_px":
            r_px,

        "r_um":
            r_px
            * PIXEL_SIZE_UM,

        "C2":
            C2,

        "pair_count":
            pair_counts
    })

    C2_table.to_csv(
        os.path.join(
            out_dir,
            "C2_profile.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Raw defect candidates
    # --------------------------------------------------------

    defects = detect_half_defects(
        df
    )

    if len(defects) > 0:

        defects[
            "x_um"
        ] = (
            defects["x_px"]
            * PIXEL_SIZE_UM
        )

        defects[
            "y_um"
        ] = (
            defects["y_px"]
            * PIXEL_SIZE_UM
        )

    defects.to_csv(
        os.path.join(
            out_dir,
            "raw_half_defect_candidates.csv"
        ),
        index=False
    )

    n_plus = (
        int(
            np.sum(
                defects["defect_type"]
                == "+1/2"
            )
        )
        if len(defects)
        else 0
    )

    n_minus = (
        int(
            np.sum(
                defects["defect_type"]
                == "-1/2"
            )
        )
        if len(defects)
        else 0
    )

    n_total_def = (
        n_plus
        + n_minus
    )

    # --------------------------------------------------------
    # Plot 1 — orientation field
    # --------------------------------------------------------

    theta_rad = np.deg2rad(
        theta_deg
    )

    u = np.cos(
        theta_rad
    )

    v = np.sin(
        theta_rad
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.quiver(
        use["X"],
        use["Y"],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=0.12,
        width=0.002
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "X (px)"
    )

    plt.ylabel(
        "Y (px)"
    )

    plt.title(
        f"{name}\n"
        "Image-derived orientation field"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "01_orientation_field.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2 — local S2D
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    sc = plt.scatter(
        use["X"],
        use["Y"],
        c=local_S,
        s=18,
        cmap="viridis"
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "X (px)"
    )

    plt.ylabel(
        "Y (px)"
    )

    plt.title(
        f"{name}\n"
        "Local orientational order"
    )

    cbar = plt.colorbar(
        sc
    )

    cbar.set_label(
        "Local $S_{2D}$"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "02_local_S2D.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3 — C2(r)
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    good = np.isfinite(
        C2
    )

    plt.plot(
        r_px[good]
        * PIXEL_SIZE_UM,
        C2[good],
        "o",
        markersize=5,
        label="$C_2(r)$"
    )

    if c2_fit is not None:

        plt.plot(
            c2_fit["fit_x"]
            * PIXEL_SIZE_UM,
            c2_fit["fit_y"],
            "--",
            linewidth=2,
            label=(
                f"Fit: "
                f"$\\xi$="
                f"{c2_fit['xi_px']*PIXEL_SIZE_UM:.2f} µm, "
                f"$R^2$="
                f"{c2_fit['R2']:.3f}"
            )
        )

    plt.xlabel(
        "Distance $r$ (µm)"
    )

    plt.ylabel(
        "$C_2(r)$"
    )

    plt.title(
        f"{name}\n"
        "Nematic orientational correlation"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "03_C2_correlation.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 4 — raw defects
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    if len(defects) > 0:

        plus = defects[
            defects["defect_type"]
            == "+1/2"
        ]

        minus = defects[
            defects["defect_type"]
            == "-1/2"
        ]

        if len(plus):

            plt.scatter(
                plus["x_px"],
                plus["y_px"],
                marker="^",
                s=60,
                label="+1/2"
            )

        if len(minus):

            plt.scatter(
                minus["x_px"],
                minus["y_px"],
                marker="v",
                s=60,
                label="-1/2"
            )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "X (px)"
    )

    plt.ylabel(
        "Y (px)"
    )

    plt.title(
        f"{name}\n"
        "Raw winding-derived defect candidates"
    )

    if len(defects) > 0:
        plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "04_raw_defect_candidates.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # FOV summary
    # --------------------------------------------------------

    summary = {
        "FOV":
            name,

        "total_vectors":
            len(df),

        "retained_vectors":
            len(use),

        "retained_fraction":
            len(use) / len(df),

        "global_S2D":
            global_S,

        "mean_local_S2D":
            np.nanmean(
                local_S
            ),

        "median_local_S2D":
            np.nanmedian(
                local_S
            ),

        "SD_local_S2D":
            np.nanstd(
                local_S,
                ddof=1
            ),

        "raw_plus_half":
            n_plus,

        "raw_minus_half":
            n_minus,

        "raw_total_half_defects":
            n_total_def,

        "xi_px":
            (
                c2_fit["xi_px"]
                if c2_fit
                else np.nan
            ),

        "xi_um":
            (
                c2_fit["xi_px"]
                * PIXEL_SIZE_UM
                if c2_fit
                else np.nan
            ),

        "xi_error_px":
            (
                c2_fit["xi_px_err"]
                if c2_fit
                else np.nan
            ),

        "xi_error_um":
            (
                c2_fit["xi_px_err"]
                * PIXEL_SIZE_UM
                if c2_fit
                else np.nan
            ),

        "C2_fit_R2":
            (
                c2_fit["R2"]
                if c2_fit
                else np.nan
            )
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            out_dir,
            "STEP3_FOV_summary.csv"
        ),
        index=False
    )

    all_summary.append(
        summary
    )

    print(
        "Global S2D:",
        global_S
    )

    print(
        "Mean local S2D:",
        np.nanmean(local_S)
    )

    print(
        "Raw +1/2:",
        n_plus
    )

    print(
        "Raw -1/2:",
        n_minus
    )

    if c2_fit:

        print(
            "xi:",
            c2_fit["xi_px"]
            * PIXEL_SIZE_UM,
            "µm"
        )

        print(
            "C2 fit R2:",
            c2_fit["R2"]
        )

# ============================================================
# 6. FINAL 4-FOV TABLE
# ============================================================

summary_df = pd.DataFrame(
    all_summary
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP3_ALL_4FOV_summary.csv"
    ),
    index=False
)

# ============================================================
# 7. GROUP MEAN ± SD ± SEM
# ============================================================

group_metrics = [
    "global_S2D",
    "mean_local_S2D",
    "median_local_S2D",
    "xi_um",
    "C2_fit_R2",
    "raw_total_half_defects"
]

group_rows = []

for metric in group_metrics:

    values = pd.to_numeric(
        summary_df[metric],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        continue

    group_rows.append({
        "Parameter":
            metric,

        "N_FOV":
            len(values),

        "Mean":
            values.mean(),

        "SD":
            (
                values.std(ddof=1)
                if len(values) > 1
                else np.nan
            ),

        "SEM":
            (
                values.sem(ddof=1)
                if len(values) > 1
                else np.nan
            )
    })

group_df = pd.DataFrame(
    group_rows
)

group_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP3_30_70_GROUP_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 8. FINAL TERMINAL SUMMARY
# ============================================================

print(
    "\n"
    + "="*90
)

print(
    "STEP 3 COMPLETE — 30:70 + NaOH, 4 FOVs"
)

print(
    "="*90
)

print(
    "\nPer-FOV results:\n"
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nGroup summary:\n"
)

print(
    group_df.to_string(
        index=False
    )
)

print(
    "\nOutput folder:\n",
    OUTPUT_FOLDER
)

print(
    "\nIMPORTANT:"
)

print(
    "These defect counts are RAW winding-derived candidates."
)

print(
    "Final merged/unique defects, density and nearest-neighbor"
)

print(
    "statistics will be calculated in STEP 4."
)
