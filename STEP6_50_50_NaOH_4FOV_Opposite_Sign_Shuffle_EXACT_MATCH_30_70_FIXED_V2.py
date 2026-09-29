
# ============================================================
# STEP 6 — 50:50 + NaOH | 4 FOVs
# EXACTLY MATCHED TO 30:70
# OPPOSITE-SIGN NEAREST-NEIGHBOR ASSOCIATION
# + SIGN-SHUFFLE NULL TEST
#
# INPUT PER FOV
#   Step 4:
#       merged_defects.csv
#
# PRIMARY OUTPUTS
#   - observed unlike-sign nearest-neighbor fraction
#   - sign-shuffled null distribution
#   - empirical one-sided p-value
#   - z-like standardized effect
#   - 4-FOV mean ± SD ± SEM
#
# MATCHED SETTINGS TO 30:70
#   N_SHUFFLES = 5000
#   RANDOM_SEED = 42
#
# IMPORTANT
#   Each FOV is analysed independently.
#   Coordinates from different FOVs are NOT pooled for NN analysis.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

FOV1_NAME = "50_50_NaOH_FOV1"
FOV1_DEFECTS = r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV1\merged_defects.csv"

FOV2_NAME = "50_50_NaOH_FOV2"
FOV2_DEFECTS = r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV2\merged_defects.csv"

FOV3_NAME = "50_50_NaOH_FOV3"
FOV3_DEFECTS = r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV3\merged_defects.csv"

FOV4_NAME = "50_50_NaOH_FOV4"
FOV4_DEFECTS = r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV4\merged_defects.csv"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\5050\STEP6_50_50_OPPOSITE_SIGN_SHUFFLE_MATCHED"

DATASETS = [
    {"name": FOV1_NAME, "defects": FOV1_DEFECTS},
    {"name": FOV2_NAME, "defects": FOV2_DEFECTS},
    {"name": FOV3_NAME, "defects": FOV3_DEFECTS},
    {"name": FOV4_NAME, "defects": FOV4_DEFECTS},
]

# ============================================================
# 2. SETTINGS — SAME AS 30:70
# ============================================================

N_SHUFFLES = 5000
RANDOM_SEED = 42

VALID_CLASSES = ["+1/2", "-1/2"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. HELPERS
# ============================================================

def load_half_defects(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Defect file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "x_um",
        "y_um",
        "class"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[
        df["class"].isin(
            VALID_CLASSES
        )
    ].copy()

    df["x_um"] = pd.to_numeric(
        df["x_um"],
        errors="coerce"
    )

    df["y_um"] = pd.to_numeric(
        df["y_um"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["x_um", "y_um"]
    ).reset_index(drop=True)

    df["sign"] = np.where(
        df["class"] == "+1/2",
        +1,
        -1
    )

    return df


def get_nn_graph(coords):
    """
    For each defect, identify its nearest spatial neighbor.
    """

    if len(coords) < 2:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float)
        )

    tree = cKDTree(
        coords
    )

    d, idx = tree.query(
        coords,
        k=2
    )

    # index 0 = self; index 1 = true nearest neighbor
    return (
        idx[:, 1],
        d[:, 1]
    )


def calculate_observed(df):

    if len(df) < 2:
        return {
            "overall_fraction": np.nan,
            "plus_to_minus_fraction": np.nan,
            "minus_to_plus_fraction": np.nan,
            "nn_index": np.array([], dtype=int),
            "nn_distance_um": np.array([], dtype=float)
        }

    coords = df[
        ["x_um", "y_um"]
    ].to_numpy()

    signs = df[
        "sign"
    ].to_numpy()

    nn_idx, nn_dist = get_nn_graph(
        coords
    )

    nn_sign = signs[
        nn_idx
    ]

    unlike = (
        signs != nn_sign
    )

    plus_mask = (
        signs == +1
    )

    minus_mask = (
        signs == -1
    )

    plus_to_minus = (
        np.mean(
            nn_sign[plus_mask] == -1
        )
        if np.any(plus_mask)
        else np.nan
    )

    minus_to_plus = (
        np.mean(
            nn_sign[minus_mask] == +1
        )
        if np.any(minus_mask)
        else np.nan
    )

    return {
        "overall_fraction":
            float(
                np.mean(unlike)
            ),

        "plus_to_minus_fraction":
            float(plus_to_minus)
            if np.isfinite(plus_to_minus)
            else np.nan,

        "minus_to_plus_fraction":
            float(minus_to_plus)
            if np.isfinite(minus_to_plus)
            else np.nan,

        "nn_index":
            nn_idx,

        "nn_distance_um":
            nn_dist
    }


def shuffle_null(df, n_shuffles, rng):
    """
    Fixed defect positions.
    Fixed nearest-neighbor graph.
    Only defect signs are shuffled.

    This preserves:
      - total defect number
      - spatial positions
      - number of +1/2 defects
      - number of -1/2 defects
    """

    if len(df) < 2:
        return np.array([])

    coords = df[
        ["x_um", "y_um"]
    ].to_numpy()

    observed_signs = df[
        "sign"
    ].to_numpy()

    nn_idx, _ = get_nn_graph(
        coords
    )

    null = np.zeros(
        n_shuffles,
        dtype=float
    )

    for i in range(
        n_shuffles
    ):

        shuffled_signs = rng.permutation(
            observed_signs
        )

        neighbor_signs = shuffled_signs[
            nn_idx
        ]

        null[i] = np.mean(
            shuffled_signs
            != neighbor_signs
        )

    return null


def empirical_upper_p(observed, null):

    if (
        len(null) == 0
        or not np.isfinite(observed)
    ):
        return np.nan

    return (
        np.sum(
            null >= observed
        )
        + 1
    ) / (
        len(null)
        + 1
    )


def standardized_effect(observed, null):

    if (
        len(null) < 2
        or not np.isfinite(observed)
    ):
        return np.nan

    mu = np.mean(
        null
    )

    sd = np.std(
        null,
        ddof=1
    )

    if sd == 0:
        return np.nan

    return (
        observed - mu
    ) / sd


# ============================================================
# 4. ANALYSE EACH FOV
# ============================================================

summary_rows = []

for fov_index, item in enumerate(DATASETS):

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

    defects = load_half_defects(
        item["defects"]
    )

    n_plus = int(
        np.sum(
            defects["class"] == "+1/2"
        )
    )

    n_minus = int(
        np.sum(
            defects["class"] == "-1/2"
        )
    )

    observed = calculate_observed(
        defects
    )

    # --------------------------------------------------------
    # A. Save observed NN table
    # --------------------------------------------------------

    if len(defects) >= 2:

        nn_table = defects.copy()

        nn_table[
            "nearest_neighbor_index"
        ] = observed[
            "nn_index"
        ]

        nn_table[
            "nearest_neighbor_distance_um"
        ] = observed[
            "nn_distance_um"
        ]

        nn_table[
            "nearest_neighbor_class"
        ] = defects.iloc[
            observed["nn_index"]
        ]["class"].to_numpy()

        nn_table[
            "unlike_sign_nearest_neighbor"
        ] = (
            nn_table[
                "class"
            ].to_numpy()
            !=
            nn_table[
                "nearest_neighbor_class"
            ].to_numpy()
        )

        nn_table.to_csv(
            os.path.join(
                out_dir,
                "nearest_neighbor_sign_table.csv"
            ),
            index=False
        )

    # --------------------------------------------------------
    # B. Shuffle null
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
        + fov_index
    )

    null = shuffle_null(
        defects,
        N_SHUFFLES,
        rng
    )

    null_mean = (
        np.mean(null)
        if len(null) > 0
        else np.nan
    )

    null_sd = (
        np.std(
            null,
            ddof=1
        )
        if len(null) > 1
        else np.nan
    )

    p_value = empirical_upper_p(
        observed[
            "overall_fraction"
        ],
        null
    )

    z_like = standardized_effect(
        observed[
            "overall_fraction"
        ],
        null
    )

    # --------------------------------------------------------
    # C. Save shuffle distribution
    # --------------------------------------------------------

    pd.DataFrame({
        "shuffle_iteration":
            np.arange(
                1,
                len(null) + 1
            ),

        "unlike_sign_NN_fraction":
            null
    }).to_csv(
        os.path.join(
            out_dir,
            "sign_shuffle_null_distribution.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # D. Null histogram
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    if len(null) > 0:

        plt.hist(
            null,
            bins=30,
            alpha=0.8,
            label="Sign-shuffled null"
        )

        plt.axvline(
            null_mean,
            linestyle="--",
            linewidth=1.5,
            label=f"Null mean = {null_mean:.3f}"
        )

        plt.axvline(
            observed[
                "overall_fraction"
            ],
            linestyle="-",
            linewidth=2,
            label=(
                f"Observed = "
                f"{observed['overall_fraction']:.3f}"
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
        "Opposite-sign nearest-neighbor shuffle test"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "01_sign_shuffle_null_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # E. Observed versus null
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.bar(
        ["Observed", "Shuffle null"],
        [
            observed[
                "overall_fraction"
            ],
            null_mean
        ],
        yerr=[
            0,
            null_sd
        ],
        capsize=5
    )

    plt.ylabel(
        "Unlike-sign nearest-neighbor fraction"
    )

    plt.ylim(
        0,
        1
    )

    if np.isfinite(
        p_value
    ):
        plt.title(
            f"{name}\n"
            f"Empirical p = {p_value:.4g}"
        )
    else:
        plt.title(
            name
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "02_observed_vs_shuffle_null.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # F. Save FOV summary
    # --------------------------------------------------------

    summary = {

        "FOV":
            name,

        "n_half_defects":
            len(defects),

        "n_plus_half":
            n_plus,

        "n_minus_half":
            n_minus,

        "observed_unlike_sign_NN_fraction":
            observed[
                "overall_fraction"
            ],

        "plus_to_minus_NN_fraction":
            observed[
                "plus_to_minus_fraction"
            ],

        "minus_to_plus_NN_fraction":
            observed[
                "minus_to_plus_fraction"
            ],

        "shuffle_null_mean":
            null_mean,

        "shuffle_null_SD":
            null_sd,

        "z_like_standardized_effect":
            z_like,

        "empirical_one_sided_p":
            p_value,

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

    summary_rows.append(
        summary
    )

# ============================================================
# 5. 4-FOV SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
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

    "plus_to_minus_NN_fraction",

    "minus_to_plus_NN_fraction",

    "shuffle_null_mean",

    "z_like_standardized_effect"
]

group_rows = []

for metric in group_metrics:

    vals = pd.to_numeric(
        summary_df[metric],
        errors="coerce"
    ).dropna()

    group_rows.append({

        "Parameter":
            metric,

        "N_FOV":
            len(vals),

        "Mean":
            vals.mean()
            if len(vals) > 0
            else np.nan,

        "SD":
            vals.std(
                ddof=1
            )
            if len(vals) > 1
            else np.nan,

        "SEM":
            vals.sem(
                ddof=1
            )
            if len(vals) > 1
            else np.nan
    })

group_df = pd.DataFrame(
    group_rows
)

group_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP6_50_50_GROUP_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 7. 4-FOV COMPARISON FIGURE
# ============================================================

x = np.arange(
    len(summary_df)
)

width = 0.36

plt.figure(
    figsize=(9, 6)
)

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
    yerr=summary_df[
        "shuffle_null_SD"
    ],
    capsize=4,
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
    "Unlike-sign nearest-neighbor fraction"
)

plt.ylim(
    0,
    1
)

plt.title(
    "50:50 + NaOH — opposite-sign nearest-neighbor organization"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_4FOV_observed_vs_shuffle_null.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 8. TERMINAL REPORT
# ============================================================

print("\n" + "="*95)
print("STEP 6 COMPLETE — 50:50 + NaOH, matched to 30:70")
print("="*95)

print("\nPer-FOV summary:\n")

print(
    summary_df.to_string(
        index=False
    )
)

print("\nGroup mean ± SD ± SEM:\n")

print(
    group_df.to_string(
        index=False
    )
)

print("\nMain parameter:")
print(
    "Observed unlike-sign nearest-neighbor fraction "
    "vs sign-shuffled null expectation"
)

print("\nInterpretation:")
print(
    "Observed > null with a small empirical one-sided p-value "
    "supports preferential opposite-sign nearest-neighbor organization."
)

print(
    "Do not describe this alone as direct thermodynamic attraction."
)

print("\nOutput folder:")
print(
    OUTPUT_FOLDER
)
