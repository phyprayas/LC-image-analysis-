
# ============================================================
# PSI STUDY — STEP 3
# DEFECT-CENTERED AZIMUTHAL Ψ(φ) ANALYSIS
# JUPYTER / VS CODE FRIENDLY
#
# PURPOSE
# -------
# Use the ORIGINAL retained orientation vectors (not dense interpolated pixels)
# together with the existing merged-defect coordinates.
#
# For every merged defect candidate:
#   1) build a circular/annular ROI around the defect
#   2) calculate azimuthal angle φ around the defect center
#   3) collect local Ψ values in that annulus
#   4) save wrapped Ψ(φ)
#   5) save unwrapped doubled-angle 2Ψ(φ)
#   6) calculate objective ROI quality:
#         - number of vectors
#         - angular coverage
#         - nearest-defect distance
#         - effective analysis radius
#
# IMPORTANT
# ---------
# STEP 3 does NOT yet assign/validate topological charge from the fit.
# Formal winding fitting and fitted s = m/2 will be done in PSI STEP 4.
#
# This code uses the ORIGINAL OrientationJ/Step-3 vectors for quantitative
# azimuthal analysis. Dense Ψ maps remain visualization products.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT — EDIT THESE PATHS
# ============================================================
#
# Each tuple:
# (
#   LABEL,
#   ORIGINAL/FILTERED ORIENTATION VECTOR CSV,
#   MERGED DEFECT CSV
# )
#
# Orientation input examples:
#   filtered_vectors_with_local_S2D.csv
#   filtered_vectors.csv
#
# Defect input:
#   your Step-4 merged-defect CSV for the same FOV
# ============================================================

DATASETS = [
    (
        "30_70_FOV1",
        r"D:\YOUR_PATH\30_70_FOV1\filtered_vectors_with_local_S2D.csv",
        r"D:\YOUR_PATH\30_70_FOV1\merged_defects.csv"
    ),
    (
        "30_70_FOV2",
        r"D:\YOUR_PATH\30_70_FOV2\filtered_vectors_with_local_S2D.csv",
        r"D:\YOUR_PATH\30_70_FOV2\merged_defects.csv"
    ),
    (
        "30_70_FOV3",
        r"D:\YOUR_PATH\30_70_FOV3\filtered_vectors_with_local_S2D.csv",
        r"D:\YOUR_PATH\30_70_FOV3\merged_defects.csv"
    ),
    (
        "30_70_FOV4",
        r"D:\YOUR_PATH\30_70_FOV4\filtered_vectors_with_local_S2D.csv",
        r"D:\YOUR_PATH\30_70_FOV4\merged_defects.csv"
    ),

    (
        "50_50_FOV1",
        r"D:\YOUR_PATH\50_50_FOV1\filtered_vectors.csv",
        r"D:\YOUR_PATH\50_50_FOV1\merged_defects.csv"
    ),
    (
        "50_50_FOV2",
        r"D:\YOUR_PATH\50_50_FOV2\filtered_vectors.csv",
        r"D:\YOUR_PATH\50_50_FOV2\merged_defects.csv"
    ),
    (
        "50_50_FOV3",
        r"D:\YOUR_PATH\50_50_FOV3\filtered_vectors.csv",
        r"D:\YOUR_PATH\50_50_FOV3\merged_defects.csv"
    ),
    (
        "50_50_FOV4",
        r"D:\YOUR_PATH\50_50_FOV4\filtered_vectors.csv",
        r"D:\YOUR_PATH\50_50_FOV4\merged_defects.csv"
    ),
]

OUTPUT_DIR = r"D:\YOUR_PATH\PSI_STUDY_STEP3"

# ============================================================
# 2. FIXED STUDY SETTINGS
# ============================================================

# Established calibration:
UM_PER_PX = 10.0 / 157.0

# Existing OrientationJ coherency threshold
COHERENCY_THRESHOLD = 0.30

# Annular ROI settings
R_INNER_UM = 0.8
R_OUTER_MAX_UM = 4.0

# If True, outer radius is reduced in crowded defect regions
# using a fraction of the nearest-defect distance.
USE_ADAPTIVE_OUTER_RADIUS = True
NN_RADIUS_FRACTION = 0.45
R_OUTER_MIN_UM = 2.2

# Objective eligibility criteria
MIN_VECTORS_IN_ANNULUS = 10

# Angular coverage:
# 12 bins across 0–360°. A good candidate should occupy
# at least this fraction of azimuth bins.
N_AZIMUTH_BINS = 12
MIN_ANGULAR_COVERAGE = 0.67

# Plot only the best candidates per FOV to avoid hundreds of figures.
# Numerical data are still saved for every analyzed candidate.
MAX_PLOTS_PER_FOV = 12

DPI = 600
SHOW_PLOTS_IN_NOTEBOOK = False

# ============================================================
# 3. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_robust(path):
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise last_error


def find_col(df, names):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:
        key = str(name).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def to_boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    if numeric.notna().sum() > 0:
        return numeric.fillna(0).to_numpy() > 0

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return text.isin(
        ["true", "yes", "1", "valid"]
    ).to_numpy()


# ============================================================
# 4. LOAD ORIGINAL ORIENTATION VECTORS
# ============================================================

def load_orientation_vectors(path):
    df = read_csv_robust(path)
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    print("Orientation columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    x_um_col = find_col(
        df,
        ["x_um"]
    )

    y_um_col = find_col(
        df,
        ["y_um"]
    )

    if x_um_col is not None and y_um_col is not None:

        x_um = pd.to_numeric(
            df[x_um_col],
            errors="coerce"
        ).to_numpy(float)

        y_um = pd.to_numeric(
            df[y_um_col],
            errors="coerce"
        ).to_numpy(float)

        coordinate_source = "x_um/y_um"

    else:

        x_col = find_col(
            df,
            ["X", "x", "X_px", "x_px"]
        )

        y_col = find_col(
            df,
            ["Y", "y", "Y_px", "y_px"]
        )

        if x_col is None or y_col is None:
            raise KeyError(
                "Orientation CSV must contain either x_um/y_um or X/Y."
            )

        x_um = (
            pd.to_numeric(
                df[x_col],
                errors="coerce"
            ).to_numpy(float)
            * UM_PER_PX
        )

        y_um = (
            pd.to_numeric(
                df[y_col],
                errors="coerce"
            ).to_numpy(float)
            * UM_PER_PX
        )

        coordinate_source = "X/Y converted using 10/157 µm/px"

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    rad_col = find_col(
        df,
        [
            "theta_rad",
            "orientation_rad",
            "Psi_rad"
        ]
    )

    deg_col = find_col(
        df,
        [
            "Orientation_deg",
            "Orientation",
            "orientation_deg",
            "orientation",
            "theta_deg",
            "Angle",
            "angle"
        ]
    )

    if rad_col is not None:

        theta = pd.to_numeric(
            df[rad_col],
            errors="coerce"
        ).to_numpy(float)

        orientation_source = rad_col

    elif deg_col is not None:

        theta_deg = pd.to_numeric(
            df[deg_col],
            errors="coerce"
        ).to_numpy(float)

        theta = np.deg2rad(
            theta_deg
        )

        orientation_source = deg_col

    else:
        raise KeyError(
            "Could not find orientation-angle column."
        )

    psi = np.mod(
        theta,
        np.pi
    )

    # --------------------------------------------------------
    # Validity
    # --------------------------------------------------------

    valid = (
        np.isfinite(x_um)
        & np.isfinite(y_um)
        & np.isfinite(psi)
    )

    valid_col = find_col(
        df,
        [
            "valid_texture",
            "valid",
            "is_valid",
            "texture_valid"
        ]
    )

    if valid_col is not None:
        valid &= to_boolean(
            df[valid_col]
        )

    coh_col = find_col(
        df,
        [
            "Coherency",
            "coherency",
            "Coherence",
            "coherence"
        ]
    )

    if coh_col is not None:

        coherence = pd.to_numeric(
            df[coh_col],
            errors="coerce"
        ).to_numpy(float)

        valid &= (
            np.isfinite(coherence)
            & (coherence >= COHERENCY_THRESHOLD)
        )

    vectors = pd.DataFrame({
        "x_um": x_um[valid],
        "y_um": y_um[valid],
        "Psi_rad": psi[valid],
        "Psi_deg": np.rad2deg(
            psi[valid]
        )
    })

    if len(vectors) < 20:
        raise ValueError(
            f"Only {len(vectors)} valid orientation vectors remain."
        )

    return (
        vectors,
        coordinate_source,
        orientation_source
    )


# ============================================================
# 5. LOAD MERGED DEFECT COORDINATES
# ============================================================

def load_defects(path):
    df = read_csv_robust(path)
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    print("Defect columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Coordinates: prefer µm
    # --------------------------------------------------------

    x_um_col = find_col(
        df,
        [
            "x_um",
            "X_um",
            "defect_x_um",
            "center_x_um",
            "cx_um"
        ]
    )

    y_um_col = find_col(
        df,
        [
            "y_um",
            "Y_um",
            "defect_y_um",
            "center_y_um",
            "cy_um"
        ]
    )

    if x_um_col is not None and y_um_col is not None:

        x_um = pd.to_numeric(
            df[x_um_col],
            errors="coerce"
        ).to_numpy(float)

        y_um = pd.to_numeric(
            df[y_um_col],
            errors="coerce"
        ).to_numpy(float)

    else:

        x_px_col = find_col(
            df,
            [
                "X",
                "x",
                "x_px",
                "X_px",
                "defect_x",
                "center_x",
                "cx"
            ]
        )

        y_px_col = find_col(
            df,
            [
                "Y",
                "y",
                "y_px",
                "Y_px",
                "defect_y",
                "center_y",
                "cy"
            ]
        )

        if x_px_col is None or y_px_col is None:
            raise KeyError(
                "Defect CSV must contain defect coordinates in µm or pixels."
            )

        x_um = (
            pd.to_numeric(
                df[x_px_col],
                errors="coerce"
            ).to_numpy(float)
            * UM_PER_PX
        )

        y_um = (
            pd.to_numeric(
                df[y_px_col],
                errors="coerce"
            ).to_numpy(float)
            * UM_PER_PX
        )

    # --------------------------------------------------------
    # Existing defect sign / charge if available
    # --------------------------------------------------------

    sign_col = find_col(
        df,
        [
            "charge",
            "sign",
            "defect_charge",
            "topological_charge",
            "winding",
            "s"
        ]
    )

    if sign_col is not None:
        sign_values = df[
            sign_col
        ].astype(str).to_numpy()
    else:
        sign_values = np.array(
            ["unknown"] * len(df),
            dtype=object
        )

    valid = (
        np.isfinite(x_um)
        & np.isfinite(y_um)
    )

    defects = pd.DataFrame({
        "defect_id": np.arange(
            1,
            np.sum(valid) + 1
        ),
        "x_um": x_um[valid],
        "y_um": y_um[valid],
        "existing_sign": sign_values[valid]
    })

    if len(defects) == 0:
        raise ValueError(
            "No valid merged defect coordinates found."
        )

    return defects


# ============================================================
# 6. NEAREST-DEFECT DISTANCE
# ============================================================

def nearest_defect_distances(defects):
    xy = defects[
        ["x_um", "y_um"]
    ].to_numpy(float)

    n = len(xy)

    if n == 1:
        return np.array(
            [np.nan]
        )

    dmat = np.sqrt(
        (
            (
                xy[:, None, :]
                - xy[None, :, :]
            ) ** 2
        ).sum(axis=2)
    )

    np.fill_diagonal(
        dmat,
        np.inf
    )

    return np.min(
        dmat,
        axis=1
    )


# ============================================================
# 7. DEFECT-CENTERED ANNULAR DATA
# ============================================================

def effective_outer_radius(nn_distance):
    if not USE_ADAPTIVE_OUTER_RADIUS:
        return R_OUTER_MAX_UM

    if not np.isfinite(nn_distance):
        return R_OUTER_MAX_UM

    adaptive = (
        NN_RADIUS_FRACTION
        * nn_distance
    )

    return float(
        np.clip(
            adaptive,
            R_OUTER_MIN_UM,
            R_OUTER_MAX_UM
        )
    )


def angular_coverage_fraction(phi_rad):
    if len(phi_rad) == 0:
        return 0.0

    phi_0_2pi = np.mod(
        phi_rad,
        2.0*np.pi
    )

    counts, _ = np.histogram(
        phi_0_2pi,
        bins=N_AZIMUTH_BINS,
        range=(0, 2*np.pi)
    )

    occupied = np.sum(
        counts > 0
    )

    return occupied / N_AZIMUTH_BINS


def defect_annulus_data(
    vectors,
    xc,
    yc,
    r_inner,
    r_outer
):
    dx = (
        vectors["x_um"].to_numpy(float)
        - xc
    )

    dy = (
        vectors["y_um"].to_numpy(float)
        - yc
    )

    r = np.sqrt(
        dx**2 + dy**2
    )

    phi = np.arctan2(
        dy,
        dx
    )

    use = (
        np.isfinite(r)
        & np.isfinite(phi)
        & (r >= r_inner)
        & (r <= r_outer)
    )

    data = vectors.loc[
        use
    ].copy()

    if len(data) == 0:
        return data

    data["r_um"] = r[use]

    data["phi_rad"] = phi[use]

    data["phi_deg"] = np.rad2deg(
        phi[use]
    )

    # sort from -π to +π
    data = data.sort_values(
        "phi_rad"
    ).reset_index(
        drop=True
    )

    # doubled-angle values
    two_psi = (
        2.0
        * data["Psi_rad"].to_numpy(float)
    )

    # unwrap doubled angle for later Step-4 winding fit
    two_psi_unwrapped = np.unwrap(
        two_psi
    )

    data[
        "twoPsi_rad_wrapped"
    ] = np.mod(
        two_psi,
        2*np.pi
    )

    data[
        "twoPsi_rad_unwrapped"
    ] = two_psi_unwrapped

    data[
        "twoPsi_deg_unwrapped"
    ] = np.rad2deg(
        two_psi_unwrapped
    )

    return data


# ============================================================
# 8. PLOTS
# ============================================================

def plot_roi_orientation(
    label,
    defect_row,
    roi_data,
    r_inner,
    r_outer,
    output_path
):
    xc = defect_row["x_um"]
    yc = defect_row["y_um"]

    fig, ax = plt.subplots(
        figsize=(6.3, 6.0)
    )

    sc = ax.scatter(
        roi_data["x_um"],
        roi_data["y_um"],
        c=roi_data["Psi_deg"],
        s=45,
        vmin=0,
        vmax=180
    )

    # orientation sticks
    stick_len = 0.55

    for _, row in roi_data.iterrows():

        theta = row["Psi_rad"]

        dx = (
            0.5
            * stick_len
            * np.cos(theta)
        )

        dy = (
            0.5
            * stick_len
            * np.sin(theta)
        )

        ax.plot(
            [
                row["x_um"] - dx,
                row["x_um"] + dx
            ],
            [
                row["y_um"] - dy,
                row["y_um"] + dy
            ],
            linewidth=0.8
        )

    ax.scatter(
        [xc],
        [yc],
        marker="x",
        s=90,
        linewidths=2
    )

    # annular guides
    circle1 = plt.Circle(
        (xc, yc),
        r_inner,
        fill=False,
        linestyle="--",
        linewidth=1
    )

    circle2 = plt.Circle(
        (xc, yc),
        r_outer,
        fill=False,
        linestyle="--",
        linewidth=1
    )

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    margin = (
        r_outer
        + 0.5
    )

    ax.set_xlim(
        xc - margin,
        xc + margin
    )

    ax.set_ylim(
        yc + margin,
        yc - margin
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        r"X ($\mu$m)"
    )

    ax.set_ylabel(
        r"Y ($\mu$m)"
    )

    ax.set_title(
        f"{label} — defect {int(defect_row['defect_id'])}"
    )

    cb = fig.colorbar(
        sc,
        ax=ax
    )

    cb.set_label(
        r"$\Psi$ (deg)"
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()

    plt.close(fig)


def plot_psi_phi(
    label,
    defect_row,
    roi_data,
    output_path
):
    fig, ax = plt.subplots(
        figsize=(7.0, 4.8)
    )

    ax.scatter(
        roi_data["phi_deg"],
        roi_data["Psi_deg"],
        s=45
    )

    ax.set_xlabel(
        r"Azimuth $\phi$ (deg)"
    )

    ax.set_ylabel(
        r"$\Psi$ (deg)"
    )

    ax.set_xlim(
        -180,
        180
    )

    ax.set_ylim(
        0,
        180
    )

    ax.set_title(
        f"{label} — defect {int(defect_row['defect_id'])}: "
        r"$\Psi(\phi)$"
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()

    plt.close(fig)


def plot_unwrapped_two_psi(
    label,
    defect_row,
    roi_data,
    output_path
):
    fig, ax = plt.subplots(
        figsize=(7.0, 4.8)
    )

    ax.scatter(
        roi_data["phi_deg"],
        roi_data["twoPsi_deg_unwrapped"],
        s=45
    )

    ax.set_xlabel(
        r"Azimuth $\phi$ (deg)"
    )

    ax.set_ylabel(
        r"Unwrapped $2\Psi$ (deg)"
    )

    ax.set_xlim(
        -180,
        180
    )

    ax.set_title(
        f"{label} — defect {int(defect_row['defect_id'])}: "
        r"unwrapped $2\Psi(\phi)$"
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()

    plt.close(fig)


# ============================================================
# 9. ANALYZE ONE FOV
# ============================================================

def analyze_fov(
    label,
    orientation_path,
    defect_path
):
    print("\n" + "="*80)
    print("PROCESSING:", label)
    print("Orientation CSV:", orientation_path)
    print("Defect CSV:", defect_path)

    if not os.path.isfile(
        orientation_path
    ):
        raise FileNotFoundError(
            orientation_path
        )

    if not os.path.isfile(
        defect_path
    ):
        raise FileNotFoundError(
            defect_path
        )

    vectors, coord_source, orientation_source = (
        load_orientation_vectors(
            orientation_path
        )
    )

    defects = load_defects(
        defect_path
    )

    defects["nearest_defect_um"] = (
        nearest_defect_distances(
            defects
        )
    )

    fov_folder = os.path.join(
        OUTPUT_DIR,
        label
    )

    ensure_dir(
        fov_folder
    )

    candidate_rows = []

    all_roi_tables = []

    for idx, defect in defects.iterrows():

        defect_id = int(
            defect["defect_id"]
        )

        nn = defect[
            "nearest_defect_um"
        ]

        r_outer = effective_outer_radius(
            nn
        )

        roi = defect_annulus_data(
            vectors,
            defect["x_um"],
            defect["y_um"],
            R_INNER_UM,
            r_outer
        )

        n_vectors = len(
            roi
        )

        coverage = (
            angular_coverage_fraction(
                roi["phi_rad"].to_numpy(float)
            )
            if n_vectors > 0
            else 0.0
        )

        eligible = (
            n_vectors >= MIN_VECTORS_IN_ANNULUS
            and coverage >= MIN_ANGULAR_COVERAGE
        )

        candidate_rows.append({
            "Label": label,
            "defect_id": defect_id,
            "x_um": defect["x_um"],
            "y_um": defect["y_um"],
            "existing_sign": defect["existing_sign"],
            "nearest_defect_um": nn,
            "r_inner_um": R_INNER_UM,
            "r_outer_um": r_outer,
            "N_vectors_annulus": n_vectors,
            "angular_coverage_fraction": coverage,
            "eligible_for_step4": eligible
        })

        if n_vectors > 0:

            roi = roi.copy()

            roi.insert(
                0,
                "Label",
                label
            )

            roi.insert(
                1,
                "defect_id",
                defect_id
            )

            roi.insert(
                2,
                "existing_sign",
                defect["existing_sign"]
            )

            all_roi_tables.append(
                roi
            )

            roi.to_csv(
                os.path.join(
                    fov_folder,
                    f"{label}_defect_{defect_id:03d}_azimuthal_data.csv"
                ),
                index=False
            )

    candidates = pd.DataFrame(
        candidate_rows
    )

    candidates.to_csv(
        os.path.join(
            fov_folder,
            f"{label}_STEP3_candidate_summary.csv"
        ),
        index=False
    )

    if all_roi_tables:

        all_data = pd.concat(
            all_roi_tables,
            ignore_index=True
        )

        all_data.to_csv(
            os.path.join(
                fov_folder,
                f"{label}_STEP3_all_defect_azimuthal_data.csv"
            ),
            index=False
        )

    # --------------------------------------------------------
    # Rank objectively for representative plots
    # --------------------------------------------------------

    eligible_table = candidates[
        candidates[
            "eligible_for_step4"
        ]
    ].copy()

    eligible_table = eligible_table.sort_values(
        by=[
            "angular_coverage_fraction",
            "N_vectors_annulus"
        ],
        ascending=[
            False,
            False
        ]
    )

    plot_ids = (
        eligible_table[
            "defect_id"
        ]
        .head(
            MAX_PLOTS_PER_FOV
        )
        .astype(int)
        .tolist()
    )

    for defect_id in plot_ids:

        defect = defects.loc[
            defects[
                "defect_id"
            ] == defect_id
        ].iloc[0]

        row = candidates.loc[
            candidates[
                "defect_id"
            ] == defect_id
        ].iloc[0]

        roi_path = os.path.join(
            fov_folder,
            f"{label}_defect_{defect_id:03d}_azimuthal_data.csv"
        )

        roi = pd.read_csv(
            roi_path
        )

        plot_roi_orientation(
            label,
            defect,
            roi,
            row["r_inner_um"],
            row["r_outer_um"],
            os.path.join(
                fov_folder,
                f"{label}_defect_{defect_id:03d}_ROI_map.png"
            )
        )

        plot_psi_phi(
            label,
            defect,
            roi,
            os.path.join(
                fov_folder,
                f"{label}_defect_{defect_id:03d}_Psi_vs_phi.png"
            )
        )

        plot_unwrapped_two_psi(
            label,
            defect,
            roi,
            os.path.join(
                fov_folder,
                f"{label}_defect_{defect_id:03d}_twoPsi_unwrapped_vs_phi.png"
            )
        )

    print(
        "Total merged defects:",
        len(candidates)
    )

    print(
        "Eligible for Step 4:",
        int(
            candidates[
                "eligible_for_step4"
            ].sum()
        )
    )

    return {
        "Label": label,
        "Orientation_file": orientation_path,
        "Defect_file": defect_path,
        "Coordinate_source": coord_source,
        "Orientation_source": orientation_source,
        "N_merged_defects": len(candidates),
        "N_step4_eligible": int(
            candidates[
                "eligible_for_step4"
            ].sum()
        ),
        "Fraction_step4_eligible": (
            candidates[
                "eligible_for_step4"
            ].mean()
            if len(candidates) > 0
            else np.nan
        ),
        "Mean_angular_coverage": candidates[
            "angular_coverage_fraction"
        ].mean(),
        "Median_vectors_per_annulus": candidates[
            "N_vectors_annulus"
        ].median()
    }


# ============================================================
# 10. RUN ALL 8 FOVs
# ============================================================

ensure_dir(
    OUTPUT_DIR
)

master_rows = []

for (
    label,
    orientation_path,
    defect_path
) in DATASETS:

    result = analyze_fov(
        label,
        orientation_path,
        defect_path
    )

    master_rows.append(
        result
    )

master = pd.DataFrame(
    master_rows
)

master["Composition"] = master[
    "Label"
].apply(
    lambda x:
    "30:70"
    if str(x).startswith(
        "30_70"
    )
    else "50:50"
)

master_path = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP3_ALL_8_FOV_SUMMARY.csv"
)

master.to_csv(
    master_path,
    index=False
)

# composition summary
composition_summary = (
    master.groupby(
        "Composition"
    )
    .agg(
        N_FOV=(
            "Label",
            "count"
        ),
        Mean_merged_defects=(
            "N_merged_defects",
            "mean"
        ),
        Mean_step4_eligible=(
            "N_step4_eligible",
            "mean"
        ),
        Mean_fraction_step4_eligible=(
            "Fraction_step4_eligible",
            "mean"
        ),
        Mean_angular_coverage=(
            "Mean_angular_coverage",
            "mean"
        ),
        Median_vectors_per_annulus=(
            "Median_vectors_per_annulus",
            "median"
        )
    )
    .reset_index()
)

composition_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "PSI_STEP3_COMPOSITION_SUMMARY.csv"
    ),
    index=False
)

print("\n" + "="*80)
print("PSI STEP 3 COMPLETE")
print("="*80)
print("Master summary:")
print(master_path)
print("\nNext:")
print(
    "Use candidates with eligible_for_step4 = True "
    "for formal winding fitting in PSI Step 4."
)
