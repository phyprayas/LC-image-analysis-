
# ============================================================
# STEP 7 — 30:70 + NaOH, 4 FOVs
# PATH-BASED ORDER-RECOVERY LENGTH ANALYSIS
#
# MODEL:
#   S(r) = S_inf - A * exp(-r / lambda_S)
#
# PURPOSE:
#   For each FOV:
#     - use Step 5 pointwise order–defect table
#     - bin local S2D by nearest-defect distance
#     - fit exponential recovery
#     - report lambda_S, uncertainty, R2
#     - classify fit reliability
#
# IMPORTANT:
#   Edit ONLY SECTION 1 PATHS.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

POINTWISE_1 = r"C:\Users\Prayas Singh\Desktop\3070\STEP5_ORDER_DEFECT_COUPLING\30_70_NaOH_FOV1\order_defect_pointwise_table.csv"
POINTWISE_2 = r"C:\Users\Prayas Singh\Desktop\3070\STEP5_ORDER_DEFECT_COUPLING\30_70_NaOH_FOV2\order_defect_pointwise_table.csv"
POINTWISE_3 = r"C:\Users\Prayas Singh\Desktop\3070\STEP5_ORDER_DEFECT_COUPLING\30_70_NaOH_FOV3\order_defect_pointwise_table.csv"
POINTWISE_4 = r"C:\Users\Prayas Singh\Desktop\3070\STEP5_ORDER_DEFECT_COUPLING\30_70_NaOH_FOV4\order_defect_pointwise_table.csv"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\3070\STEP7_ORDER_RECOVERY_LENGTH"

# ============================================================
# 2. SETTINGS
# ============================================================

BIN_WIDTH_UM = 1.0
MIN_POINTS_PER_BIN = 20

# Fit only out to this percentile of observed distances
# to reduce domination by very sparse far-field tails
MAX_DISTANCE_PERCENTILE = 90

# Fit reliability thresholds
MIN_R2_FOR_RELIABLE = 0.70
MAX_RELATIVE_LAMBDA_ERROR = 0.75

DATASETS = [
    {"name": "30_70_NaOH_FOV1", "pointwise": POINTWISE_1},
    {"name": "30_70_NaOH_FOV2", "pointwise": POINTWISE_2},
    {"name": "30_70_NaOH_FOV3", "pointwise": POINTWISE_3},
    {"name": "30_70_NaOH_FOV4", "pointwise": POINTWISE_4},
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. MODEL + HELPERS
# ============================================================

def recovery_model(r, S_inf, A, lam):
    return S_inf - A * np.exp(-r / lam)


def make_binned_profile(distance_um, local_s2d):
    x = np.asarray(distance_um, dtype=float)
    y = np.asarray(local_s2d, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 10:
        return pd.DataFrame()

    max_fit_distance = np.percentile(
        x,
        MAX_DISTANCE_PERCENTILE
    )

    x = x[x <= max_fit_distance]
    y = y[:len(x)] if len(y) == len(x) else y

    # safer synchronized filtering
    original = pd.DataFrame({"x": distance_um, "y": local_s2d}).dropna()
    original = original[original["x"] <= max_fit_distance]

    x = original["x"].to_numpy()
    y = original["y"].to_numpy()

    edges = np.arange(
        0,
        np.max(x) + BIN_WIDTH_UM,
        BIN_WIDTH_UM
    )

    rows = []

    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]

        if i < len(edges) - 2:
            sel = (x >= lo) & (x < hi)
        else:
            sel = (x >= lo) & (x <= hi)

        vals = y[sel]

        if len(vals) < MIN_POINTS_PER_BIN:
            continue

        rows.append({
            "r_center_um": 0.5 * (lo + hi),
            "mean_local_S2D": np.mean(vals),
            "SD_local_S2D": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "SEM_local_S2D": (
                np.std(vals, ddof=1) / np.sqrt(len(vals))
                if len(vals) > 1 else np.nan
            ),
            "n_points": len(vals)
        })

    return pd.DataFrame(rows)


def fit_recovery(profile):
    if len(profile) < 5:
        return None

    x = profile["r_center_um"].to_numpy()
    y = profile["mean_local_S2D"].to_numpy()

    # Reasonable starting guesses
    S_inf0 = np.nanmax(y)
    A0 = max(S_inf0 - y[0], 0.05)
    lam0 = max(np.median(x), 1.0)

    try:
        popt, pcov = curve_fit(
            recovery_model,
            x,
            y,
            p0=[S_inf0, A0, lam0],
            bounds=(
                [0.0, 0.0, 0.05],
                [1.2, 1.2, 100.0]
            ),
            maxfev=50000
        )

        S_inf, A, lam = popt
        perr = np.sqrt(np.diag(pcov))

        yfit = recovery_model(x, *popt)

        ss_res = np.sum((y - yfit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rmse = np.sqrt(
            np.mean(
                (y - yfit) ** 2
            )
        )

        rel_err = (
            perr[2] / lam
            if lam > 0
            else np.inf
        )

        # reliability decision
        reliable = (
            np.isfinite(r2)
            and r2 >= MIN_R2_FOR_RELIABLE
            and np.isfinite(rel_err)
            and rel_err <= MAX_RELATIVE_LAMBDA_ERROR
            and S_inf < 1.19
            and lam < 99.0
        )

        reliability_label = (
            "RELIABLE"
            if reliable
            else "CAUTIOUS/UNRESOLVED"
        )

        return {
            "S_inf": S_inf,
            "A": A,
            "lambda_um": lam,
            "S_inf_err": perr[0],
            "A_err": perr[1],
            "lambda_err_um": perr[2],
            "lambda_relative_error": rel_err,
            "R2": r2,
            "RMSE": rmse,
            "reliability": reliability_label,
            "fit_x": x,
            "fit_y": yfit
        }

    except Exception as e:
        print("Fit failed:", e)
        return None


# ============================================================
# 4. PROCESS EACH FOV
# ============================================================

all_summary = []

for dataset in DATASETS:

    name = dataset["name"]
    pointwise_path = dataset["pointwise"]

    print("\n" + "="*90)
    print("Processing:", name)
    print("="*90)

    if not os.path.exists(pointwise_path):
        raise FileNotFoundError(
            f"Pointwise CSV not found:\n{pointwise_path}"
        )

    out_dir = os.path.join(
        OUTPUT_FOLDER,
        name
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    df = pd.read_csv(pointwise_path)

    needed = [
        "local_S2D",
        "nearest_defect_distance_um"
    ]

    missing = [
        c for c in needed
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}"
        )

    for c in needed:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna(
        subset=needed
    ).copy()

    # --------------------------------------------------------
    # Binned profile
    # --------------------------------------------------------

    profile = make_binned_profile(
        df["nearest_defect_distance_um"],
        df["local_S2D"]
    )

    profile.to_csv(
        os.path.join(
            out_dir,
            "binned_order_recovery_profile.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    fit = fit_recovery(
        profile
    )

    # --------------------------------------------------------
    # Plot 1: binned data only
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    if len(profile) > 0:

        plt.errorbar(
            profile["r_center_um"],
            profile["mean_local_S2D"],
            yerr=profile["SEM_local_S2D"],
            marker="o",
            capsize=3,
            linestyle="-"
        )

    plt.xlabel(
        "Distance from nearest defect (µm)"
    )

    plt.ylabel(
        "Mean local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        "Binned local order vs defect distance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "01_binned_local_order_vs_defect_distance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: recovery fit
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    if len(profile) > 0:

        plt.errorbar(
            profile["r_center_um"],
            profile["mean_local_S2D"],
            yerr=profile["SEM_local_S2D"],
            marker="o",
            capsize=3,
            label="Binned data"
        )

    if fit is not None:

        xfine = np.linspace(
            np.min(fit["fit_x"]),
            np.max(fit["fit_x"]),
            300
        )

        yfine = recovery_model(
            xfine,
            fit["S_inf"],
            fit["A"],
            fit["lambda_um"]
        )

        plt.plot(
            xfine,
            yfine,
            "--",
            linewidth=2,
            label=(
                f"$\\lambda_S$ = "
                f"{fit['lambda_um']:.2f} ± "
                f"{fit['lambda_err_um']:.2f} µm\n"
                f"$R^2$ = {fit['R2']:.3f} | "
                f"{fit['reliability']}"
            )
        )

    plt.xlabel(
        "Distance from nearest defect (µm)"
    )

    plt.ylabel(
        "Mean local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        "Order recovery away from defect candidates"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "02_order_recovery_fit.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3: normalized characteristic recovery
    # --------------------------------------------------------

    if fit is not None:

        xnorm = np.linspace(
            0,
            max(
                4 * fit["lambda_um"],
                fit["lambda_um"] + 1
            ),
            300
        )

        recovery_fraction = (
            1
            - np.exp(
                -xnorm
                / fit["lambda_um"]
            )
        )

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            xnorm,
            recovery_fraction
        )

        plt.axvline(
            fit["lambda_um"],
            linestyle="--",
            label="$r=\\lambda_S$"
        )

        plt.axhline(
            1 - np.exp(-1),
            linestyle=":"
        )

        plt.xlabel(
            "Distance from defect (µm)"
        )

        plt.ylabel(
            "Normalized recovery fraction"
        )

        plt.title(
            f"{name}\n"
            "Characteristic order-recovery scale"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                out_dir,
                "03_characteristic_order_recovery_length.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------------
    # Per-FOV summary
    # --------------------------------------------------------

    if fit is not None:

        summary = {
            "FOV":
                name,

            "n_pointwise_values":
                len(df),

            "n_recovery_bins":
                len(profile),

            "S_inf":
                fit["S_inf"],

            "S_inf_error":
                fit["S_inf_err"],

            "A":
                fit["A"],

            "A_error":
                fit["A_err"],

            "lambda_S_um":
                fit["lambda_um"],

            "lambda_S_error_um":
                fit["lambda_err_um"],

            "lambda_relative_error":
                fit["lambda_relative_error"],

            "R2":
                fit["R2"],

            "RMSE":
                fit["RMSE"],

            "fit_reliability":
                fit["reliability"]
        }

    else:

        summary = {
            "FOV": name,
            "n_pointwise_values": len(df),
            "n_recovery_bins": len(profile),
            "S_inf": np.nan,
            "S_inf_error": np.nan,
            "A": np.nan,
            "A_error": np.nan,
            "lambda_S_um": np.nan,
            "lambda_S_error_um": np.nan,
            "lambda_relative_error": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "fit_reliability": "FIT_FAILED"
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

    all_summary.append(
        summary
    )

    print(
        pd.DataFrame(
            [summary]
        ).to_string(
            index=False
        )
    )

# ============================================================
# 5. COMBINED 4-FOV SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    all_summary
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP7_ALL_4FOV_summary.csv"
    ),
    index=False
)

# ============================================================
# 6. GROUP SUMMARY — RELIABLE FITS ONLY
# ============================================================

reliable_df = summary_df[
    summary_df[
        "fit_reliability"
    ] == "RELIABLE"
].copy()

group_rows = []

for metric in [
    "lambda_S_um",
    "R2",
    "S_inf",
    "A"
]:

    vals = pd.to_numeric(
        reliable_df[
            metric
        ],
        errors="coerce"
    ).dropna()

    if len(vals) == 0:
        continue

    group_rows.append({
        "Parameter":
            metric,

        "N_reliable_FOV":
            len(vals),

        "Mean":
            vals.mean(),

        "SD":
            (
                vals.std(
                    ddof=1
                )
                if len(vals) > 1
                else np.nan
            ),

        "SEM":
            (
                vals.sem(
                    ddof=1
                )
                if len(vals) > 1
                else np.nan
            )
    })

group_df = pd.DataFrame(
    group_rows
)

group_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP7_GROUP_reliable_fits_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 7. FINAL GROUP PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

labels = summary_df[
    "FOV"
].tolist()

x = np.arange(
    len(labels)
)

y = summary_df[
    "lambda_S_um"
].to_numpy()

yerr = summary_df[
    "lambda_S_error_um"
].to_numpy()

plt.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    capsize=4
)

plt.xticks(
    x,
    labels,
    rotation=25,
    ha="right"
)

plt.ylabel(
    "$\\lambda_S$ (µm)"
)

plt.title(
    "30:70 + NaOH: order-recovery length across 4 FOVs"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_4FOV_order_recovery_length.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 8. FINAL TERMINAL OUTPUT
# ============================================================

print(
    "\n"
    + "="*95
)

print(
    "STEP 7 COMPLETE — 30:70 + NaOH, 4 FOVs"
)

print(
    "="*95
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
    "\nReliable-fit group summary:\n"
)

if len(group_df) > 0:
    print(
        group_df.to_string(
            index=False
        )
    )
else:
    print(
        "No FOV met the reliability criteria."
    )

print(
    "\nIMPORTANT:"
)

print(
    "Use lambda_S as a core manuscript parameter ONLY if multiple FOVs "
    "are classified RELIABLE and show reasonably consistent values."
)

print(
    "If fits are unstable or highly variable, retain the robust Step 5 "
    "order–defect coupling metrics as the primary result and treat lambda_S "
    "as supplementary/phenomenological."
)
