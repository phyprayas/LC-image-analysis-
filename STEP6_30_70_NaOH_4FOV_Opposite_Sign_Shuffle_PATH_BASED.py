
# ============================================================
# STEP 6 — 30:70 + NaOH, 4 FOVs
# PATH-BASED OPPOSITE-SIGN DEFECT ASSOCIATION
# WITH SIGN-SHUFFLE NULL TEST
#
# PURPOSE
#   For each FOV:
#     1. Use merged/unique ±1/2 defect candidates from Step 4
#     2. Find each defect's nearest neighbor
#     3. Measure fraction whose nearest neighbor has opposite sign
#     4. Shuffle defect signs many times while keeping positions fixed
#     5. Compare observed opposite-sign NN fraction to shuffle null
#
# PRIMARY OUTPUT
#   observed unlike-sign NN fraction
#   null mean ± SD
#   empirical p-value
#
# IMPORTANT
#   Edit ONLY SECTION 1 paths.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

DEFECTS_1 = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS\30_70_NaOH_FOV1\merged_unique_defect_candidates.csv"
DEFECTS_2 = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS\30_70_NaOH_FOV2\merged_unique_defect_candidates.csv"
DEFECTS_3 = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS\30_70_NaOH_FOV3\merged_unique_defect_candidates.csv"
DEFECTS_4 = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS\30_70_NaOH_FOV4\merged_unique_defect_candidates.csv"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\3070\STEP6_OPPOSITE_SIGN_ASSOCIATION"

# ============================================================
# 2. SETTINGS
# ============================================================

N_SHUFFLES = 5000
RANDOM_SEED = 42

PIXEL_SIZE_UM = 10.0 / 157.0

rng = np.random.default_rng(RANDOM_SEED)

DATASETS = [
    {"name": "30_70_NaOH_FOV1", "defects": DEFECTS_1},
    {"name": "30_70_NaOH_FOV2", "defects": DEFECTS_2},
    {"name": "30_70_NaOH_FOV3", "defects": DEFECTS_3},
    {"name": "30_70_NaOH_FOV4", "defects": DEFECTS_4},
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. HELPERS
# ============================================================

def validate_defect_table(df, name):
    required = ["x_px", "y_px", "defect_type"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name}: missing required columns: {missing}"
        )

    df = df.copy()

    df["x_px"] = pd.to_numeric(df["x_px"], errors="coerce")
    df["y_px"] = pd.to_numeric(df["y_px"], errors="coerce")

    df = df.dropna(
        subset=["x_px", "y_px", "defect_type"]
    ).copy()

    df = df[
        df["defect_type"].isin(["+1/2", "-1/2"])
    ].copy()

    return df


def nearest_neighbor_indices(coords):
    tree = cKDTree(coords)

    distances, indices = tree.query(
        coords,
        k=2
    )

    return distances[:, 1], indices[:, 1]


def unlike_fraction(signs, nn_indices):
    nn_signs = signs[nn_indices]
    unlike = nn_signs != signs

    return float(np.mean(unlike)), unlike


def empirical_upper_tail_p(null_values, observed):
    """
    Tests whether observed unlike-sign NN fraction is greater
    than expected under random sign assignment.

    +1 correction gives a valid permutation-style p-value.
    """
    return (
        np.sum(null_values >= observed) + 1
    ) / (
        len(null_values) + 1
    )


# ============================================================
# 4. PROCESS EACH FOV
# ============================================================

all_summary = []

for dataset in DATASETS:

    name = dataset["name"]
    defect_path = dataset["defects"]

    print("\n" + "="*90)
    print("Processing:", name)
    print("="*90)

    if not os.path.exists(defect_path):
        raise FileNotFoundError(
            f"Defect CSV not found:\n{defect_path}"
        )

    out_dir = os.path.join(
        OUTPUT_FOLDER,
        name
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    df = pd.read_csv(defect_path)
    df = validate_defect_table(df, name)

    if len(df) < 2:
        raise RuntimeError(
            f"{name}: at least 2 defects are required."
        )

    coords = df[
        ["x_px", "y_px"]
    ].to_numpy()

    signs = df[
        "defect_type"
    ].to_numpy()

    # --------------------------------------------------------
    # Nearest-neighbor geometry
    # --------------------------------------------------------

    nn_distance_px, nn_index = nearest_neighbor_indices(
        coords
    )

    nn_distance_um = (
        nn_distance_px
        * PIXEL_SIZE_UM
    )

    # --------------------------------------------------------
    # Observed unlike-sign fraction
    # --------------------------------------------------------

    observed_fraction, observed_unlike = unlike_fraction(
        signs,
        nn_index
    )

    # --------------------------------------------------------
    # Sign-shuffle null
    # Positions stay fixed.
    # Number of + and - defects stays fixed.
    # --------------------------------------------------------

    null_values = np.empty(
        N_SHUFFLES,
        dtype=float
    )

    for i in range(N_SHUFFLES):

        shuffled_signs = rng.permutation(
            signs
        )

        null_values[i], _ = unlike_fraction(
            shuffled_signs,
            nn_index
        )

    null_mean = float(
        np.mean(null_values)
    )

    null_sd = float(
        np.std(
            null_values,
            ddof=1
        )
    )

    p_empirical = empirical_upper_tail_p(
        null_values,
        observed_fraction
    )

    z_effect = (
        (observed_fraction - null_mean) / null_sd
        if null_sd > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Direction-resolved descriptive fractions
    # --------------------------------------------------------

    plus_mask = signs == "+1/2"
    minus_mask = signs == "-1/2"

    nn_signs = signs[
        nn_index
    ]

    plus_to_minus = (
        np.mean(
            nn_signs[plus_mask]
            == "-1/2"
        )
        if np.sum(plus_mask) > 0
        else np.nan
    )

    minus_to_plus = (
        np.mean(
            nn_signs[minus_mask]
            == "+1/2"
        )
        if np.sum(minus_mask) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Save pointwise nearest-neighbor table
    # --------------------------------------------------------

    pointwise = df.copy()

    pointwise[
        "nearest_neighbor_index"
    ] = nn_index

    pointwise[
        "nearest_neighbor_distance_px"
    ] = nn_distance_px

    pointwise[
        "nearest_neighbor_distance_um"
    ] = nn_distance_um

    pointwise[
        "nearest_neighbor_sign"
    ] = nn_signs

    pointwise[
        "nearest_neighbor_unlike_sign"
    ] = observed_unlike

    pointwise.to_csv(
        os.path.join(
            out_dir,
            "nearest_neighbor_sign_table.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save null distribution
    # --------------------------------------------------------

    null_df = pd.DataFrame({
        "shuffle_index":
            np.arange(
                1,
                N_SHUFFLES + 1
            ),

        "unlike_sign_NN_fraction":
            null_values
    })

    null_df.to_csv(
        os.path.join(
            out_dir,
            "sign_shuffle_null_distribution.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Plot: null histogram + observed line
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        null_values,
        bins=30,
        alpha=0.75
    )

    plt.axvline(
        observed_fraction,
        linestyle="--",
        linewidth=2,
        label=(
            f"Observed = "
            f"{observed_fraction:.3f}"
        )
    )

    plt.axvline(
        null_mean,
        linestyle=":",
        linewidth=2,
        label=(
            f"Null mean = "
            f"{null_mean:.3f}"
        )
    )

    plt.xlabel(
        "Unlike-sign nearest-neighbor fraction"
    )

    plt.ylabel(
        "Shuffle count"
    )

    plt.title(
        f"{name}\n"
        f"Sign-shuffle test, p = {p_empirical:.4f}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "01_sign_shuffle_null_test.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "FOV":
            name,

        "n_defects":
            len(df),

        "n_plus_half":
            int(
                np.sum(
                    signs == "+1/2"
                )
            ),

        "n_minus_half":
            int(
                np.sum(
                    signs == "-1/2"
                )
            ),

        "mean_NN_distance_um":
            float(
                np.mean(
                    nn_distance_um
                )
            ),

        "median_NN_distance_um":
            float(
                np.median(
                    nn_distance_um
                )
            ),

        "observed_unlike_sign_NN_fraction":
            observed_fraction,

        "shuffle_null_mean":
            null_mean,

        "shuffle_null_SD":
            null_sd,

        "empirical_p_upper_tail":
            p_empirical,

        "z_effect_vs_shuffle":
            z_effect,

        "plus_defect_nearest_neighbor_minus_fraction":
            plus_to_minus,

        "minus_defect_nearest_neighbor_plus_fraction":
            minus_to_plus,

        "n_shuffles":
            N_SHUFFLES
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            out_dir,
            "STEP6_FOV_summary.csv"
        ),
        index=False
    )

    all_summary.append(
        summary
    )

    print(
        f"Observed unlike-sign NN fraction = "
        f"{observed_fraction:.3f}"
    )

    print(
        f"Shuffle null = "
        f"{null_mean:.3f} ± {null_sd:.3f}"
    )

    print(
        f"Empirical p = "
        f"{p_empirical:.4f}"
    )

    print(
        f"z-like effect = "
        f"{z_effect:.2f}"
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
        "STEP6_ALL_4FOV_summary.csv"
    ),
    index=False
)

# ============================================================
# 6. GROUP MEAN ± SD ± SEM
# ============================================================

group_metrics = [
    "observed_unlike_sign_NN_fraction",
    "shuffle_null_mean",
    "z_effect_vs_shuffle",
    "plus_defect_nearest_neighbor_minus_fraction",
    "minus_defect_nearest_neighbor_plus_fraction",
    "mean_NN_distance_um",
    "median_NN_distance_um"
]

group_rows = []

for metric in group_metrics:

    vals = pd.to_numeric(
        summary_df[
            metric
        ],
        errors="coerce"
    ).dropna()

    if len(vals) == 0:
        continue

    group_rows.append({
        "Parameter":
            metric,

        "N_FOV":
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
        "STEP6_30_70_GROUP_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 7. GROUP PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

x = np.arange(
    len(summary_df)
)

width = 0.35

plt.bar(
    x - width/2,
    summary_df[
        "observed_unlike_sign_NN_fraction"
    ],
    width,
    label="Observed"
)

plt.bar(
    x + width/2,
    summary_df[
        "shuffle_null_mean"
    ],
    width,
    label="Sign-shuffled null"
)

plt.xticks(
    x,
    summary_df[
        "FOV"
    ],
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Unlike-sign NN fraction"
)

plt.title(
    "30:70 + NaOH: opposite-sign nearest-neighbor association"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_4FOV_observed_vs_shuffle.png"
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
    "STEP 6 COMPLETE — 30:70 + NaOH, 4 FOVs"
)

print(
    "="*95
)

print(
    "\nPer-FOV summary:\n"
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nGroup mean ± SD ± SEM:\n"
)

print(
    group_df.to_string(
        index=False
    )
)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_FOLDER
)

print(
    "\nINTERPRETATION:"
)

print(
    "If observed unlike-sign NN fraction is consistently above the shuffled "
    "null and empirical p is small, this supports preferential opposite-sign "
    "nearest-neighbor organization within the analyzed field."
)

print(
    "This is evidence of spatial topological organization, not by itself a "
    "direct thermodynamic interaction potential."
)
