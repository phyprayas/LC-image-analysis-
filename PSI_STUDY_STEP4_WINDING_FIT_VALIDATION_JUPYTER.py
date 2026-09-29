
# ============================================================
# PSI STUDY — STEP 4
# FORMAL WINDING FIT / DEFECT VALIDATION
# JUPYTER / VS CODE FRIENDLY
#
# INPUT:
#   PSI Step 3 per-defect azimuthal CSV files:
#       *_defect_XXX_azimuthal_data.csv
#
# OR:
#   PSI Step 3 combined file:
#       *_STEP3_all_defect_azimuthal_data.csv
#
# METHOD:
#   Fit:
#       2Psi_unwrapped = m * phi + C
#
#   Then:
#       s_fit = m / 2
#
# IMPORTANT:
#   Because phi is in radians and 2Psi is in radians,
#   the slope m is dimensionless.
#
#   Expected ideal values:
#       s = +1/2  -> m = +1
#       s = -1/2  -> m = -1
#       s = +1    -> m = +2
#       s = -1    -> m = -2
#
# OUTPUT:
#   - per-defect fitted slope m
#   - fitted topological charge s_fit
#   - R^2
#   - RMSE
#   - angular coverage
#   - N points
#   - agreement with existing sign if available
#   - fit plots
#   - all-defect master summary
#   - composition-level summary
#
# NOTE:
#   This is a validation of image-derived texture-orientation winding.
#   It should be described as winding-derived defect validation, not
#   as a direct molecular director measurement.
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT
# ============================================================
#
# Point this to the main Step-3 output folder.
# Expected structure:
#
# PSI_STUDY_STEP3/
#   30_70_FOV1/
#   30_70_FOV2/
#   ...
#   50_50_FOV4/
#
# Each subfolder contains:
#   *_STEP3_candidate_summary.csv
#   *_defect_XXX_azimuthal_data.csv
# ============================================================

STEP3_ROOT = r"D:\YOUR_PATH\PSI_STUDY_STEP3"

OUTPUT_DIR = r"D:\YOUR_PATH\PSI_STUDY_STEP4"

# ============================================================
# 2. FIT QUALITY SETTINGS
# ============================================================

# Minimum number of annular vectors
MIN_POINTS = 10

# Minimum angular coverage inherited from Step 3
MIN_ANGULAR_COVERAGE = 0.67

# Minimum R^2 to call the linear winding fit reliable
MIN_R2 = 0.70

# Optional RMSE ceiling in radians for 2Psi fit
# Set None if you do not want to use it as a hard criterion.
MAX_RMSE_RAD = None

# Charge matching tolerance
# Example:
# fitted s = 0.46 is considered consistent with +1/2
CHARGE_TOLERANCE = 0.20

# Which ideal charges to classify against
IDEAL_CHARGES = np.array([
    -1.0,
    -0.5,
     0.0,
     0.5,
     1.0
])

# Save fit plots only for reliable candidates
PLOT_ONLY_RELIABLE = True

DPI = 600
SHOW_PLOTS_IN_NOTEBOOK = False


# ============================================================
# 3. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_robust(path):
    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin1",
        "cp1252"
    ]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc
            )
        except Exception as e:
            last_error = e

    raise last_error


def normalize_existing_sign(value):
    """
    Convert common sign/charge text to numeric values when possible.
    Returns np.nan if unknown/unreadable.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    replacements = {
        "+1/2": 0.5,
        "1/2": 0.5,
        "+0.5": 0.5,
        "0.5": 0.5,

        "-1/2": -0.5,
        "-0.5": -0.5,

        "+1": 1.0,
        "1": 1.0,
        "1.0": 1.0,

        "-1": -1.0,
        "-1.0": -1.0,

        "positive": 0.5,
        "negative": -0.5,
    }

    if text in replacements:
        return replacements[text]

    # Try direct float
    try:
        return float(text)
    except Exception:
        return np.nan


def nearest_ideal_charge(s_fit):
    idx = np.argmin(
        np.abs(
            IDEAL_CHARGES
            - s_fit
        )
    )

    ideal = float(
        IDEAL_CHARGES[idx]
    )

    distance = float(
        np.abs(
            s_fit
            - ideal
        )
    )

    return ideal, distance


def linear_fit(x, y):
    """
    Simple OLS linear fit y = m x + c.
    Returns m, c, yhat, R2, RMSE.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[good]
    y = y[good]

    if len(x) < 3:
        raise ValueError(
            "Need at least 3 valid points for linear fit."
        )

    coeff = np.polyfit(
        x,
        y,
        1
    )

    m = float(
        coeff[0]
    )

    c = float(
        coeff[1]
    )

    yhat = (
        m*x
        + c
    )

    residual = (
        y
        - yhat
    )

    ss_res = np.sum(
        residual**2
    )

    ss_tot = np.sum(
        (
            y
            - np.mean(y)
        )**2
    )

    if ss_tot > 0:
        r2 = (
            1.0
            - ss_res/ss_tot
        )
    else:
        r2 = np.nan

    rmse = float(
        np.sqrt(
            np.mean(
                residual**2
            )
        )
    )

    return (
        m,
        c,
        yhat,
        r2,
        rmse
    )


# ============================================================
# 4. FIT ONE DEFECT
# ============================================================

def fit_one_defect(
    csv_path,
    candidate_info=None
):
    df = read_csv_robust(
        csv_path
    )

    required = [
        "phi_rad",
        "twoPsi_rad_unwrapped"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns {missing} in:\n{csv_path}\n"
            f"Available: {list(df.columns)}"
        )

    x = pd.to_numeric(
        df["phi_rad"],
        errors="coerce"
    ).to_numpy(float)

    y = pd.to_numeric(
        df["twoPsi_rad_unwrapped"],
        errors="coerce"
    ).to_numpy(float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[good]
    y = y[good]

    n = len(x)

    if n < MIN_POINTS:
        return {
            "N_points": n,
            "fit_possible": False,
            "reliable_fit": False,
            "failure_reason": (
                f"N_points < {MIN_POINTS}"
            )
        }

    # Step-3 file is already sorted by phi,
    # but sort again for safety.
    order = np.argsort(
        x
    )

    x = x[order]
    y = y[order]

    m, c, yhat, r2, rmse = linear_fit(
        x,
        y
    )

    s_fit = (
        m / 2.0
    )

    nearest_charge, charge_distance = nearest_ideal_charge(
        s_fit
    )

    # Angular coverage
    if candidate_info is not None:
        coverage = float(
            candidate_info.get(
                "angular_coverage_fraction",
                np.nan
            )
        )
    else:
        # fallback calculation from phi
        phi_0_2pi = np.mod(
            x,
            2*np.pi
        )

        counts, _ = np.histogram(
            phi_0_2pi,
            bins=12,
            range=(
                0,
                2*np.pi
            )
        )

        coverage = (
            np.sum(
                counts > 0
            )
            / 12.0
        )

    # Existing sign if present
    existing_sign_raw = None
    existing_sign_num = np.nan

    if "existing_sign" in df.columns:
        existing_sign_raw = df[
            "existing_sign"
        ].iloc[0]

        existing_sign_num = normalize_existing_sign(
            existing_sign_raw
        )

    elif candidate_info is not None:
        existing_sign_raw = candidate_info.get(
            "existing_sign",
            None
        )

        existing_sign_num = normalize_existing_sign(
            existing_sign_raw
        )

    sign_agreement = np.nan

    if np.isfinite(
        existing_sign_num
    ):
        sign_agreement = (
            np.sign(
                existing_sign_num
            )
            == np.sign(
                s_fit
            )
        )

    reliable = (
        (coverage >= MIN_ANGULAR_COVERAGE)
        and np.isfinite(r2)
        and (r2 >= MIN_R2)
        and (charge_distance <= CHARGE_TOLERANCE)
    )

    if (
        MAX_RMSE_RAD is not None
        and np.isfinite(rmse)
    ):
        reliable = (
            reliable
            and (
                rmse
                <= MAX_RMSE_RAD
            )
        )

    result = {
        "N_points": n,
        "angular_coverage_fraction": coverage,
        "slope_m": m,
        "intercept_C_rad": c,
        "s_fit": s_fit,
        "nearest_ideal_charge": nearest_charge,
        "charge_distance": charge_distance,
        "R2": r2,
        "RMSE_rad": rmse,
        "existing_sign_raw": existing_sign_raw,
        "existing_sign_numeric": existing_sign_num,
        "sign_agreement": sign_agreement,
        "fit_possible": True,
        "reliable_fit": reliable,
        "failure_reason": ""
    }

    return (
        result,
        x,
        y,
        yhat
    )


# ============================================================
# 5. PLOT ONE FIT
# ============================================================

def plot_fit(
    label,
    defect_id,
    x,
    y,
    yhat,
    result,
    output_path
):
    fig, ax = plt.subplots(
        figsize=(7.0, 4.9)
    )

    ax.scatter(
        np.rad2deg(
            x
        ),
        np.rad2deg(
            y
        ),
        s=45,
        label="Measured"
    )

    ax.plot(
        np.rad2deg(
            x
        ),
        np.rad2deg(
            yhat
        ),
        linewidth=2,
        label="Linear fit"
    )

    ax.set_xlabel(
        r"Azimuth $\phi$ (deg)"
    )

    ax.set_ylabel(
        r"Unwrapped $2\Psi$ (deg)"
    )

    ax.set_title(
        f"{label} — defect {defect_id}\n"
        f"s_fit={result['s_fit']:.3f}, "
        f"R²={result['R2']:.3f}, "
        f"ideal={result['nearest_ideal_charge']:+.1f}"
    )

    ax.legend(
        frameon=False
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

    plt.close(
        fig
    )


# ============================================================
# 6. ANALYZE ONE FOV
# ============================================================

def analyze_one_fov(
    fov_folder
):
    label = os.path.basename(
        os.path.normpath(
            fov_folder
        )
    )

    print("\n" + "="*80)
    print("FOV:", label)

    candidate_files = glob.glob(
        os.path.join(
            fov_folder,
            "*_STEP3_candidate_summary.csv"
        )
    )

    if len(candidate_files) == 0:
        raise FileNotFoundError(
            f"No Step-3 candidate summary found in:\n{fov_folder}"
        )

    candidate_summary = read_csv_robust(
        candidate_files[0]
    )

    if "defect_id" not in candidate_summary.columns:
        raise KeyError(
            "Step-3 candidate summary has no defect_id column."
        )

    defect_csvs = glob.glob(
        os.path.join(
            fov_folder,
            "*_defect_*_azimuthal_data.csv"
        )
    )

    if len(defect_csvs) == 0:
        raise FileNotFoundError(
            f"No per-defect azimuthal CSVs found in:\n{fov_folder}"
        )

    output_fov = os.path.join(
        OUTPUT_DIR,
        label
    )

    ensure_dir(
        output_fov
    )

    rows = []

    for csv_path in sorted(
        defect_csvs
    ):

        base = os.path.basename(
            csv_path
        )

        # Parse defect ID from filename
        # example: label_defect_003_azimuthal_data.csv
        try:
            defect_part = (
                base.split(
                    "_defect_"
                )[1]
                .split(
                    "_"
                )[0]
            )

            defect_id = int(
                defect_part
            )

        except Exception:
            print(
                "Could not parse defect ID from:",
                base
            )
            continue

        candidate_row = candidate_summary.loc[
            candidate_summary[
                "defect_id"
            ] == defect_id
        ]

        candidate_info = None

        if len(
            candidate_row
        ) > 0:
            candidate_info = (
                candidate_row.iloc[0]
                .to_dict()
            )

        fit_output = fit_one_defect(
            csv_path,
            candidate_info
        )

        # fit may be impossible if too few points
        if isinstance(
            fit_output,
            dict
        ):
            result = fit_output

            x = None
            y = None
            yhat = None

        else:
            (
                result,
                x,
                y,
                yhat
            ) = fit_output

        result_row = {
            "Label": label,
            "defect_id": defect_id,
            "azimuthal_csv": csv_path
        }

        if candidate_info is not None:
            for key in [
                "x_um",
                "y_um",
                "existing_sign",
                "nearest_defect_um",
                "r_inner_um",
                "r_outer_um",
                "N_vectors_annulus",
                "angular_coverage_fraction",
                "eligible_for_step4"
            ]:
                if key in candidate_info:
                    result_row[key] = candidate_info[
                        key
                    ]

        result_row.update(
            result
        )

        rows.append(
            result_row
        )

        if (
            result.get(
                "fit_possible",
                False
            )
            and (
                (
                    not PLOT_ONLY_RELIABLE
                )
                or result.get(
                    "reliable_fit",
                    False
                )
            )
        ):

            plot_path = os.path.join(
                output_fov,
                f"{label}_defect_{defect_id:03d}_winding_fit.png"
            )

            plot_fit(
                label,
                defect_id,
                x,
                y,
                yhat,
                result,
                plot_path
            )

    result_df = pd.DataFrame(
        rows
    )

    result_path = os.path.join(
        output_fov,
        f"{label}_STEP4_winding_fit_results.csv"
    )

    result_df.to_csv(
        result_path,
        index=False
    )

    reliable_df = result_df[
        result_df[
            "reliable_fit"
        ] == True
    ].copy()

    reliable_path = os.path.join(
        output_fov,
        f"{label}_STEP4_reliable_defects_only.csv"
    )

    reliable_df.to_csv(
        reliable_path,
        index=False
    )

    print(
        "Total fitted candidates:",
        len(result_df)
    )

    print(
        "Reliable winding fits:",
        len(reliable_df)
    )

    if len(
        result_df
    ) > 0:

        fraction_reliable = (
            len(reliable_df)
            / len(result_df)
        )

    else:
        fraction_reliable = np.nan

    # Sign agreement only where existing sign is readable
    valid_sign = result_df[
        "sign_agreement"
    ].dropna()

    if len(
        valid_sign
    ) > 0:

        sign_agreement_fraction = (
            valid_sign.astype(bool)
            .mean()
        )

    else:
        sign_agreement_fraction = np.nan

    return {
        "Label": label,
        "N_candidates_fit_attempted": len(result_df),
        "N_reliable_winding_fits": len(reliable_df),
        "Fraction_reliable_winding_fits": fraction_reliable,
        "Mean_R2_all_possible": result_df.loc[
            result_df[
                "fit_possible"
            ] == True,
            "R2"
        ].mean(),
        "Mean_abs_s_fit_reliable": (
            np.abs(
                reliable_df[
                    "s_fit"
                ]
            ).mean()
            if len(
                reliable_df
            ) > 0
            else np.nan
        ),
        "Sign_agreement_fraction": sign_agreement_fraction,
        "Result_CSV": result_path,
        "Reliable_CSV": reliable_path
    }


# ============================================================
# 7. RUN ALL FOV FOLDERS
# ============================================================

ensure_dir(
    OUTPUT_DIR
)

fov_folders = [
    p
    for p in glob.glob(
        os.path.join(
            STEP3_ROOT,
            "*"
        )
    )
    if os.path.isdir(
        p
    )
]

if len(
    fov_folders
) == 0:
    raise FileNotFoundError(
        f"No FOV folders found inside:\n{STEP3_ROOT}"
    )

master_rows = []

for folder in sorted(
    fov_folders
):
    master_rows.append(
        analyze_one_fov(
            folder
        )
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
    "PSI_STEP4_ALL_FOV_WINDING_SUMMARY.csv"
)

master.to_csv(
    master_path,
    index=False
)

# ============================================================
# 8. COMPOSITION SUMMARY
# ============================================================

comp_rows = []

for composition in [
    "30:70",
    "50:50"
]:

    sub = master[
        master[
            "Composition"
        ] == composition
    ]

    comp_rows.append({
        "Composition": composition,
        "N_FOV": len(sub),
        "Mean_fraction_reliable": sub[
            "Fraction_reliable_winding_fits"
        ].mean(),
        "SD_fraction_reliable": sub[
            "Fraction_reliable_winding_fits"
        ].std(ddof=1),
        "Mean_R2": sub[
            "Mean_R2_all_possible"
        ].mean(),
        "Mean_abs_s_fit_reliable": sub[
            "Mean_abs_s_fit_reliable"
        ].mean(),
        "Mean_sign_agreement_fraction": sub[
            "Sign_agreement_fraction"
        ].mean()
    })

composition_summary = pd.DataFrame(
    comp_rows
)

composition_summary_path = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP4_COMPOSITION_SUMMARY.csv"
)

composition_summary.to_csv(
    composition_summary_path,
    index=False
)

# ============================================================
# 9. PLOT RELIABLE FRACTION BY FOV
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.4, 5.0)
)

for xpos, composition in enumerate(
    [
        "30:70",
        "50:50"
    ]
):

    vals = master.loc[
        master[
            "Composition"
        ] == composition,
        "Fraction_reliable_winding_fits"
    ].dropna().to_numpy(float)

    if len(
        vals
    ) == 0:
        continue

    jitter = np.linspace(
        -0.06,
        0.06,
        len(vals)
    )

    ax.scatter(
        np.full(
            len(vals),
            xpos
        )
        + jitter,
        vals,
        s=65,
        edgecolors="black",
        linewidths=0.7
    )

    mean = np.mean(
        vals
    )

    sd = (
        np.std(
            vals,
            ddof=1
        )
        if len(
            vals
        ) > 1
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
        elinewidth=1.5
    )

ax.set_xticks(
    [0, 1]
)

ax.set_xticklabels(
    [
        "30:70",
        "50:50"
    ]
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "Fraction of reliable winding fits"
)

ax.set_xlabel(
    "CPC:glycerol composition"
)

ax.tick_params(
    direction="in",
    top=True,
    right=True
)

plt.tight_layout()

png = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP4_reliable_fraction_30_70_vs_50_50.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP4_reliable_fraction_30_70_vs_50_50.pdf"
)

fig.savefig(
    png,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    pdf,
    bbox_inches="tight"
)

if SHOW_PLOTS_IN_NOTEBOOK:
    plt.show()

plt.close(
    fig
)

print("\n" + "="*80)
print("PSI STEP 4 COMPLETE")
print("="*80)
print("Master summary:")
print(master_path)
print("Composition summary:")
print(composition_summary_path)
