
# ============================================================
# STEP 7 — 50:50 + NaOH | 4 FOVs
# ORDER RECOVERY LENGTH lambda_S
# EXACTLY MATCHED TO THE 30:70 WORKFLOW
#
# INPUT PER FOV
#   Step 5 exact-matched output:
#       order_defect_pointwise_table.csv
#
# MODEL
#   S(r) = S_inf - A * exp(-r / lambda_S)
#
# PRIMARY OUTPUTS
#   - fitted lambda_S (µm)
#   - S_inf
#   - A
#   - fit R^2
#   - parameter uncertainty
#   - per-FOV reliability flag
#   - 4-FOV mean ± SD ± SEM
#
# MATCHED SETTINGS TO 30:70
#   radial bin width = 1.0 µm
#   minimum points per bin = 20
#   fit up to 90th percentile of nearest-defect distance
#   reliability:
#       R^2 >= 0.70
#       relative lambda error <= 0.75
#       no fit-bound hit
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

FOV1_NAME = "50_50_NaOH_FOV1"
FOV1_TABLE = r"C:\Users\Prayas Singh\Desktop\5050\STEP5_50_50_ORDER_DEFECT_COUPLING_MATCHED\50_50_NaOH_FOV1\order_defect_pointwise_table.csv"

FOV2_NAME = "50_50_NaOH_FOV2"
FOV2_TABLE = r"C:\Users\Prayas Singh\Desktop\5050\STEP5_50_50_ORDER_DEFECT_COUPLING_MATCHED\50_50_NaOH_FOV2\order_defect_pointwise_table.csv"

FOV3_NAME = "50_50_NaOH_FOV3"
FOV3_TABLE = r"C:\Users\Prayas Singh\Desktop\5050\STEP5_50_50_ORDER_DEFECT_COUPLING_MATCHED\50_50_NaOH_FOV3\order_defect_pointwise_table.csv"

FOV4_NAME = "50_50_NaOH_FOV4"
FOV4_TABLE = r"C:\Users\Prayas Singh\Desktop\5050\STEP5_50_50_ORDER_DEFECT_COUPLING_MATCHED\50_50_NaOH_FOV4\order_defect_pointwise_table.csv"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\5050\STEP7_50_50_ORDER_RECOVERY_LENGTH"

DATASETS = [
    {"name": FOV1_NAME, "table": FOV1_TABLE},
    {"name": FOV2_NAME, "table": FOV2_TABLE},
    {"name": FOV3_NAME, "table": FOV3_TABLE},
    {"name": FOV4_NAME, "table": FOV4_TABLE},
]

# ============================================================
# 2. SETTINGS — SAME AS 30:70
# ============================================================

BIN_WIDTH_UM = 1.0
MIN_POINTS_PER_BIN = 20
FIT_PERCENTILE = 90.0

R2_MIN_RELIABLE = 0.70
MAX_REL_LAMBDA_ERR = 0.75

# Fit bounds
A_MIN = 0.0
A_MAX = 1.5

LAMBDA_MIN_UM = 0.05
LAMBDA_MAX_UM = 100.0

SINF_MIN = 0.0
SINF_MAX = 1.2

BOUND_TOL_FRAC = 0.01

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. MODEL
# ============================================================

def recovery_model(r, S_inf, A, lambda_s):
    return S_inf - A * np.exp(-r / lambda_s)

# ============================================================
# 4. HELPERS
# ============================================================

def load_pointwise_table(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Step 5 pointwise table not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "nearest_defect_distance_um",
        "local_S2D"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    df["nearest_defect_distance_um"] = pd.to_numeric(
        df["nearest_defect_distance_um"],
        errors="coerce"
    )

    df["local_S2D"] = pd.to_numeric(
        df["local_S2D"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "nearest_defect_distance_um",
            "local_S2D"
        ]
    ).copy()

    df = df[
        (df["nearest_defect_distance_um"] >= 0)
        &
        (df["local_S2D"] >= 0)
    ].copy()

    return df


def radial_bin_profile(distance_um, Slocal):

    distance_um = np.asarray(
        distance_um,
        dtype=float
    )

    Slocal = np.asarray(
        Slocal,
        dtype=float
    )

    valid = (
        np.isfinite(distance_um)
        &
        np.isfinite(Slocal)
    )

    distance_um = distance_um[valid]
    Slocal = Slocal[valid]

    if len(distance_um) == 0:
        return pd.DataFrame()

    max_r = np.max(distance_um)

    edges = np.arange(
        0,
        max_r + BIN_WIDTH_UM,
        BIN_WIDTH_UM
    )

    if len(edges) < 2:
        edges = np.array(
            [0, BIN_WIDTH_UM]
        )

    centers = 0.5 * (
        edges[:-1]
        + edges[1:]
    )

    which = np.digitize(
        distance_um,
        edges
    ) - 1

    rows = []

    for i, center in enumerate(centers):

        vals = Slocal[
            which == i
        ]

        n = len(vals)

        if n == 0:
            mean = np.nan
            sd = np.nan
            sem = np.nan
        else:
            mean = np.mean(vals)

            sd = (
                np.std(vals, ddof=1)
                if n > 1
                else np.nan
            )

            sem = (
                sd / np.sqrt(n)
                if n > 1
                else np.nan
            )

        rows.append({
            "r_center_um": center,
            "mean_local_S2D": mean,
            "sd_local_S2D": sd,
            "sem_local_S2D": sem,
            "n_points": n
        })

    return pd.DataFrame(
        rows
    )


def fit_recovery(profile_df, raw_distances):

    prof = profile_df.copy()

    prof = prof[
        prof["n_points"] >= MIN_POINTS_PER_BIN
    ].copy()

    prof = prof.dropna(
        subset=[
            "r_center_um",
            "mean_local_S2D"
        ]
    )

    if len(prof) < 4:
        return None

    fit_max_r = np.percentile(
        raw_distances,
        FIT_PERCENTILE
    )

    prof_fit = prof[
        prof["r_center_um"] <= fit_max_r
    ].copy()

    if len(prof_fit) < 4:
        return None

    x = prof_fit[
        "r_center_um"
    ].to_numpy()

    y = prof_fit[
        "mean_local_S2D"
    ].to_numpy()

    # Initial guesses
    S_inf_0 = np.mean(
        y[-min(3, len(y)):]
    )

    S0 = y[0]

    A0 = max(
        S_inf_0 - S0,
        0.05
    )

    lambda0 = max(
        np.median(x),
        1.0
    )

    lower = [
        SINF_MIN,
        A_MIN,
        LAMBDA_MIN_UM
    ]

    upper = [
        SINF_MAX,
        A_MAX,
        LAMBDA_MAX_UM
    ]

    try:

        popt, pcov = curve_fit(
            recovery_model,
            x,
            y,
            p0=[
                S_inf_0,
                A0,
                lambda0
            ],
            bounds=(
                lower,
                upper
            ),
            maxfev=50000
        )

        S_inf, A, lambda_s = popt

        perr = np.sqrt(
            np.diag(pcov)
        )

        S_inf_err, A_err, lambda_err = perr

        y_fit = recovery_model(
            x,
            *popt
        )

        ss_res = np.sum(
            (y - y_fit)**2
        )

        ss_tot = np.sum(
            (y - np.mean(y))**2
        )

        R2 = (
            1 - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        )

        rel_lambda_err = (
            lambda_err / lambda_s
            if lambda_s > 0
            else np.inf
        )

        # Bound-hit check
        lambda_hit_low = (
            abs(lambda_s - LAMBDA_MIN_UM)
            <=
            BOUND_TOL_FRAC
            * max(LAMBDA_MIN_UM, 1.0)
        )

        lambda_hit_high = (
            abs(lambda_s - LAMBDA_MAX_UM)
            <=
            BOUND_TOL_FRAC
            * LAMBDA_MAX_UM
        )

        S_inf_hit_low = (
            abs(S_inf - SINF_MIN)
            <=
            BOUND_TOL_FRAC
        )

        S_inf_hit_high = (
            abs(S_inf - SINF_MAX)
            <=
            BOUND_TOL_FRAC
            * SINF_MAX
        )

        A_hit_low = (
            abs(A - A_MIN)
            <=
            BOUND_TOL_FRAC
        )

        A_hit_high = (
            abs(A - A_MAX)
            <=
            BOUND_TOL_FRAC
            * A_MAX
        )

        hit_bound = any([
            lambda_hit_low,
            lambda_hit_high,
            S_inf_hit_low,
            S_inf_hit_high,
            A_hit_low,
            A_hit_high
        ])

        reliable = (
            np.isfinite(R2)
            and
            R2 >= R2_MIN_RELIABLE
            and
            np.isfinite(rel_lambda_err)
            and
            rel_lambda_err <= MAX_REL_LAMBDA_ERR
            and
            not hit_bound
        )

        return {
            "profile_used": prof_fit,
            "x_fit": x,
            "y_fit_data": y,
            "y_fit_model": y_fit,
            "S_inf": S_inf,
            "S_inf_err": S_inf_err,
            "A": A,
            "A_err": A_err,
            "lambda_S_um": lambda_s,
            "lambda_S_err_um": lambda_err,
            "relative_lambda_error": rel_lambda_err,
            "R2": R2,
            "fit_max_r_um": fit_max_r,
            "hit_fit_bound": hit_bound,
            "reliable": reliable
        }

    except Exception as e:

        print(
            "Fit failed:",
            e
        )

        return None


# ============================================================
# 5. PROCESS EACH FOV
# ============================================================

summary_rows = []

for item in DATASETS:

    name = item["name"]

    print("\n" + "="*95)
    print("Processing:", name)
    print("="*95)

    out_dir = os.path.join(
        OUTPUT_FOLDER,
        name
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    df = load_pointwise_table(
        item["table"]
    )

    profile = radial_bin_profile(
        df[
            "nearest_defect_distance_um"
        ].values,
        df[
            "local_S2D"
        ].values
    )

    profile.to_csv(
        os.path.join(
            out_dir,
            "radial_order_recovery_profile.csv"
        ),
        index=False
    )

    fit_result = fit_recovery(
        profile,
        df[
            "nearest_defect_distance_um"
        ].values
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    good = (
        profile["n_points"]
        >= MIN_POINTS_PER_BIN
    )

    good &= (
        np.isfinite(
            profile["mean_local_S2D"]
        )
    )

    plt.errorbar(
        profile.loc[
            good,
            "r_center_um"
        ],
        profile.loc[
            good,
            "mean_local_S2D"
        ],
        yerr=profile.loc[
            good,
            "sem_local_S2D"
        ],
        fmt="o",
        capsize=3,
        label="Binned data"
    )

    if fit_result is not None:

        x_smooth = np.linspace(
            0,
            max(
                fit_result[
                    "fit_max_r_um"
                ],
                0.1
            ),
            400
        )

        y_smooth = recovery_model(
            x_smooth,
            fit_result["S_inf"],
            fit_result["A"],
            fit_result["lambda_S_um"]
        )

        plt.plot(
            x_smooth,
            y_smooth,
            linewidth=2,
            label=(
                r"Fit: $S(r)=S_\infty-Ae^{-r/\lambda_S}$"
            )
        )

        plt.axhline(
            fit_result["S_inf"],
            linestyle="--",
            linewidth=1,
            label=(
                f"$S_\\infty$ = "
                f"{fit_result['S_inf']:.3f}"
            )
        )

        title = (
            f"{name}\n"
            f"$\\lambda_S$ = "
            f"{fit_result['lambda_S_um']:.2f} ± "
            f"{fit_result['lambda_S_err_um']:.2f} µm, "
            f"$R^2$ = {fit_result['R2']:.3f}"
        )

    else:

        title = (
            f"{name}\n"
            "Order-recovery fit unavailable"
        )

    plt.xlabel(
        "Distance to nearest defect (µm)"
    )

    plt.ylabel(
        "Mean local $S_{2D}$"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "01_characteristic_order_recovery_length.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Additional plot — EXACT 30:70-style binned local order
    # versus defect distance
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.errorbar(
        profile.loc[
            good,
            "r_center_um"
        ],
        profile.loc[
            good,
            "mean_local_S2D"
        ],
        yerr=profile.loc[
            good,
            "sem_local_S2D"
        ],
        fmt="o-",
        capsize=3
    )

    plt.xlabel(
        "Distance to nearest defect (µm)"
    )

    plt.ylabel(
        "Binned mean local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        "Binned local order vs defect distance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "02_binned_local_order_vs_defect_distance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # ADDITIONAL 30:70-STYLE FIGURE
    # Characteristic normalized order-recovery scale
    #
    # Normalized recovery fraction:
    #     f(r) = 1 - exp(-r/lambda_S)
    #
    # At r = lambda_S:
    #     f = 1 - exp(-1) = 0.632120...
    # --------------------------------------------------------

    if fit_result is not None:

        lambda_s = fit_result["lambda_S_um"]

        # Show recovery out to ~4 lambda, but at least 8 µm
        rmax_char = max(
            4.0 * lambda_s,
            8.0
        )

        r_char = np.linspace(
            0,
            rmax_char,
            500
        )

        recovery_fraction = (
            1.0
            - np.exp(
                -r_char / lambda_s
            )
        )

        frac_at_lambda = (
            1.0
            - np.exp(-1.0)
        )

        plt.figure(
            figsize=(8, 5.5)
        )

        plt.plot(
            r_char,
            recovery_fraction,
            linewidth=2
        )

        plt.axvline(
            lambda_s,
            linestyle="--",
            linewidth=1.8,
            label=r"$r=\lambda_S$"
        )

        plt.axhline(
            frac_at_lambda,
            linestyle=":",
            linewidth=1.8
        )

        plt.xlabel(
            "Distance from defect (µm)"
        )

        plt.ylabel(
            "Normalized recovery fraction"
        )

        plt.ylim(
            -0.05,
            1.03
        )

        plt.xlim(
            0,
            rmax_char
        )

        plt.title(
            f"{name}\n"
            "Characteristic order-recovery scale"
        )

        plt.legend(
            loc="upper left"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                out_dir,
                "03_characteristic_normalized_order_recovery_scale.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    if fit_result is None:

        summary = {
            "FOV": name,
            "n_orientation_points": len(df),
            "lambda_S_um": np.nan,
            "lambda_S_err_um": np.nan,
            "relative_lambda_error": np.nan,
            "S_inf": np.nan,
            "S_inf_err": np.nan,
            "A": np.nan,
            "A_err": np.nan,
            "R2": np.nan,
            "fit_max_r_um": np.nan,
            "hit_fit_bound": True,
            "reliable_fit": False
        }

    else:

        summary = {
            "FOV": name,
            "n_orientation_points": len(df),
            "lambda_S_um":
                fit_result["lambda_S_um"],
            "lambda_S_err_um":
                fit_result["lambda_S_err_um"],
            "relative_lambda_error":
                fit_result["relative_lambda_error"],
            "S_inf":
                fit_result["S_inf"],
            "S_inf_err":
                fit_result["S_inf_err"],
            "A":
                fit_result["A"],
            "A_err":
                fit_result["A_err"],
            "R2":
                fit_result["R2"],
            "fit_max_r_um":
                fit_result["fit_max_r_um"],
            "hit_fit_bound":
                fit_result["hit_fit_bound"],
            "reliable_fit":
                fit_result["reliable"]
        }

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            out_dir,
            "STEP7_FOV_summary.csv"
        ),
        index=False
    )

    summary_rows.append(
        summary
    )

# ============================================================
# 6. ALL-FOV SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP7_ALL_4FOV_summary.csv"
    ),
    index=False
)

# ============================================================
# 7. GROUP STATS — RELIABLE FITS ONLY
# ============================================================

reliable_df = summary_df[
    summary_df[
        "reliable_fit"
    ] == True
].copy()

group_metrics = [
    "lambda_S_um",
    "R2",
    "S_inf",
    "A"
]

group_rows = []

for metric in group_metrics:

    vals = pd.to_numeric(
        reliable_df[metric],
        errors="coerce"
    ).dropna()

    group_rows.append({
        "Parameter": metric,
        "N_reliable_FOV": len(vals),
        "Mean":
            vals.mean()
            if len(vals) > 0
            else np.nan,
        "SD":
            vals.std(ddof=1)
            if len(vals) > 1
            else np.nan,
        "SEM":
            vals.sem(ddof=1)
            if len(vals) > 1
            else np.nan
    })

group_df = pd.DataFrame(
    group_rows
)

group_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP7_50_50_GROUP_RELIABLE_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 8. 4-FOV LAMBDA PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

x = np.arange(
    len(summary_df)
)

y = summary_df[
    "lambda_S_um"
].to_numpy()

yerr = summary_df[
    "lambda_S_err_um"
].to_numpy()

plt.bar(
    x,
    y,
    yerr=yerr,
    capsize=4
)

plt.xticks(
    x,
    summary_df["FOV"],
    rotation=25,
    ha="right"
)

plt.ylabel(
    r"Order recovery length $\lambda_S$ (µm)"
)

plt.title(
    "50:50 + NaOH — defect-associated order recovery"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "04_4FOV_characteristic_order_recovery_length.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 9. TERMINAL REPORT
# ============================================================

print("\n" + "="*95)
print("STEP 7 COMPLETE — 50:50 + NaOH, matched to 30:70")
print("="*95)

print("\nPer-FOV fit summary:\n")

print(
    summary_df.to_string(
        index=False
    )
)

print("\nReliable-fit group summary:\n")

print(
    group_df.to_string(
        index=False
    )
)

print("\nReliability criteria:")
print(f"  R2 >= {R2_MIN_RELIABLE}")
print(f"  relative lambda error <= {MAX_REL_LAMBDA_ERR}")
print("  no fit-bound hit")

print("\nMain Step 7 plots per FOV:")
print("  01_characteristic_order_recovery_length.png")
print("  02_binned_local_order_vs_defect_distance.png")
print("  03_characteristic_normalized_order_recovery_scale.png")

print("\nPrimary manuscript parameter:")
print(r"  lambda_S from S(r) = S_inf - A exp(-r/lambda_S)")

print("\nOutput folder:")
print(
    OUTPUT_FOLDER
)
