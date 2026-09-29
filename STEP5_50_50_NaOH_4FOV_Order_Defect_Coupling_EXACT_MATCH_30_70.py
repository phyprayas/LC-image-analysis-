
# ============================================================
# STEP 5 — 50:50 + NaOH | 4 FOVs
# EXACTLY MATCHED TO THE 30:70 ORDER–DEFECT COUPLING ANALYSIS
#
# INPUTS PER FOV
#   Step 3:
#       filtered_vectors.csv
#       (must contain X, Y, local_S2D)
#
#   Step 4:
#       merged_defects.csv
#       (must contain x_px, y_px and class)
#
# OUTPUTS PER FOV
#   - order_defect_pointwise_table.csv
#   - 01_Slocal_vs_nearest_defect_distance.png
#   - 02_Slocal_vs_local_defect_density.png
#   - 03_binned_Slocal_vs_defect_distance.png
#   - 04_binned_Slocal_vs_local_defect_density.png
#   - STEP5_FOV_summary.csv
#
# GROUP OUTPUTS
#   - STEP5_ALL_4FOV_summary.csv
#   - STEP5_50_50_GROUP_mean_SD_SEM.csv
#
# PRIMARY METRICS
#   rho(Slocal, nearest defect distance)
#   rho(Slocal, local defect density)
#
# MATCHED SETTINGS TO 30:70
#   PIXEL_SIZE = 10/157 µm/px
#   local defect-density radius = 120 px ≈ 7.64 µm
#   12 diagnostic bins
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.stats import spearmanr

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

# -------------------- FOV 1 --------------------
FOV1_NAME = "50_50_NaOH_FOV1"
FOV1_VECTORS = r"C:\Users\Prayas Singh\Desktop\5050\STEP3_50_50_RESULTS\50_50_NaOH_FOV1\filtered_vectors.csv"
FOV1_DEFECTS = r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV1\merged_defects.csv"

# -------------------- FOV 2 --------------------
FOV2_NAME = "50_50_NaOH_FOV2"
FOV2_VECTORS = r"C:\Users\Prayas Singh\Desktop\5050\STEP3_50_50_RESULTS\50_50_NaOH_FOV2\filtered_vectors.csv"
FOV2_DEFECTS = r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV2\merged_defects.csv"

# -------------------- FOV 3 --------------------
FOV3_NAME = "50_50_NaOH_FOV3"
FOV3_VECTORS = r"C:\Users\Prayas Singh\Desktop\5050\STEP3_50_50_RESULTS\50_50_NaOH_FOV3\filtered_vectors.csv"
FOV3_DEFECTS = r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV3\merged_defects.csv"

# -------------------- FOV 4 --------------------
FOV4_NAME = "50_50_NaOH_FOV4"
FOV4_VECTORS = r"C:\Users\Prayas Singh\Desktop\5050\STEP3_50_50_RESULTS\50_50_NaOH_FOV4\filtered_vectors.csv"
FOV4_DEFECTS = r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV4\merged_defects.csv"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\5050\STEP5_50_50_ORDER_DEFECT_COUPLING_MATCHED"

DATASETS = [
    {"name": FOV1_NAME, "vectors": FOV1_VECTORS, "defects": FOV1_DEFECTS},
    {"name": FOV2_NAME, "vectors": FOV2_VECTORS, "defects": FOV2_DEFECTS},
    {"name": FOV3_NAME, "vectors": FOV3_VECTORS, "defects": FOV3_DEFECTS},
    {"name": FOV4_NAME, "vectors": FOV4_VECTORS, "defects": FOV4_DEFECTS},
]

# ============================================================
# 2. SETTINGS — SAME AS 30:70
# ============================================================

PIXEL_SIZE_UM = 10.0 / 157.0

# Local defect-density radius used in the 30:70 Step 5
LOCAL_DEFECT_RADIUS_PX = 120.0
LOCAL_DEFECT_RADIUS_UM = LOCAL_DEFECT_RADIUS_PX * PIXEL_SIZE_UM

# Diagnostic bin count
N_BINS = 12

# Use half defects as the primary nematic defects
VALID_DEFECT_CLASSES = ["+1/2", "-1/2"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. HELPERS
# ============================================================

def load_vectors(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector file not found:\n{path}")

    df = pd.read_csv(path)

    required = {"X", "Y", "local_S2D"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df = df.copy()

    for c in ["X", "Y", "local_S2D"]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna(
        subset=["X", "Y", "local_S2D"]
    ).copy()

    df["x_um"] = df["X"] * PIXEL_SIZE_UM
    df["y_um"] = df["Y"] * PIXEL_SIZE_UM

    return df


def load_defects(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Defect file not found:\n{path}")

    df = pd.read_csv(path)

    required = {"x_px", "y_px", "class"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df = df.copy()

    df = df[
        df["class"].isin(
            VALID_DEFECT_CLASSES
        )
    ].copy()

    df["x_px"] = pd.to_numeric(
        df["x_px"],
        errors="coerce"
    )

    df["y_px"] = pd.to_numeric(
        df["y_px"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["x_px", "y_px"]
    ).copy()

    df["x_um"] = df["x_px"] * PIXEL_SIZE_UM
    df["y_um"] = df["y_px"] * PIXEL_SIZE_UM

    return df


def binned_mean(x, y, n_bins=12):
    """
    Equal-width diagnostic bins.
    Returns center, mean, SEM, count.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return pd.DataFrame(
            columns=[
                "x_center",
                "y_mean",
                "y_sem",
                "n"
            ]
        )

    xmin = np.min(x)
    xmax = np.max(x)

    if np.isclose(
        xmin,
        xmax
    ):
        return pd.DataFrame({
            "x_center": [xmin],
            "y_mean": [np.mean(y)],
            "y_sem": [
                np.std(y, ddof=1) / np.sqrt(len(y))
                if len(y) > 1
                else np.nan
            ],
            "n": [len(y)]
        })

    edges = np.linspace(
        xmin,
        xmax,
        n_bins + 1
    )

    centers = 0.5 * (
        edges[:-1]
        + edges[1:]
    )

    rows = []

    which = np.digitize(
        x,
        edges
    ) - 1

    which[
        which == n_bins
    ] = n_bins - 1

    for i in range(n_bins):

        vals = y[
            which == i
        ]

        if len(vals) == 0:
            rows.append({
                "x_center": centers[i],
                "y_mean": np.nan,
                "y_sem": np.nan,
                "n": 0
            })

        else:
            rows.append({
                "x_center": centers[i],
                "y_mean": np.mean(vals),
                "y_sem":
                    np.std(vals, ddof=1) / np.sqrt(len(vals))
                    if len(vals) > 1
                    else np.nan,
                "n": len(vals)
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 4. PROCESS EACH FOV
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

    vectors = load_vectors(
        item["vectors"]
    )

    defects = load_defects(
        item["defects"]
    )

    if len(defects) == 0:
        raise RuntimeError(
            f"{name}: no +1/2 or -1/2 merged defects available."
        )

    vector_xy_px = vectors[
        ["X", "Y"]
    ].to_numpy()

    defect_xy_px = defects[
        ["x_px", "y_px"]
    ].to_numpy()

    defect_tree = cKDTree(
        defect_xy_px
    )

    # --------------------------------------------------------
    # A. Distance to nearest defect
    # --------------------------------------------------------

    nearest_dist_px, nearest_idx = defect_tree.query(
        vector_xy_px,
        k=1
    )

    vectors[
        "nearest_defect_distance_px"
    ] = nearest_dist_px

    vectors[
        "nearest_defect_distance_um"
    ] = (
        nearest_dist_px
        * PIXEL_SIZE_UM
    )

    vectors[
        "nearest_defect_class"
    ] = defects.iloc[
        nearest_idx
    ]["class"].to_numpy()

    # --------------------------------------------------------
    # B. Local defect density around every orientation point
    # --------------------------------------------------------

    local_counts = np.array([
        len(
            defect_tree.query_ball_point(
                p,
                r=LOCAL_DEFECT_RADIUS_PX
            )
        )
        for p in vector_xy_px
    ], dtype=float)

    circle_area_um2 = (
        np.pi
        * LOCAL_DEFECT_RADIUS_UM**2
    )

    local_density = (
        local_counts
        / circle_area_um2
    )

    vectors[
        "local_defect_count"
    ] = local_counts

    vectors[
        "local_defect_density_per_um2"
    ] = local_density

    # --------------------------------------------------------
    # C. Spearman correlations
    # --------------------------------------------------------

    rho_distance, p_distance = spearmanr(
        vectors["local_S2D"],
        vectors[
            "nearest_defect_distance_um"
        ]
    )

    rho_density, p_density = spearmanr(
        vectors["local_S2D"],
        vectors[
            "local_defect_density_per_um2"
        ]
    )

    # --------------------------------------------------------
    # D. Save pointwise table
    # --------------------------------------------------------

    vectors.to_csv(
        os.path.join(
            out_dir,
            "order_defect_pointwise_table.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # E. Diagnostic binned profiles
    # --------------------------------------------------------

    binned_distance = binned_mean(
        vectors[
            "nearest_defect_distance_um"
        ],
        vectors[
            "local_S2D"
        ],
        N_BINS
    )

    binned_distance.to_csv(
        os.path.join(
            out_dir,
            "binned_Slocal_vs_defect_distance.csv"
        ),
        index=False
    )

    binned_density = binned_mean(
        vectors[
            "local_defect_density_per_um2"
        ],
        vectors[
            "local_S2D"
        ],
        N_BINS
    )

    binned_density.to_csv(
        os.path.join(
            out_dir,
            "binned_Slocal_vs_local_defect_density.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # F. Plot 1 — Slocal vs distance to nearest defect
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        vectors[
            "nearest_defect_distance_um"
        ],
        vectors[
            "local_S2D"
        ],
        s=10,
        alpha=0.35
    )

    plt.xlabel(
        "Distance to nearest defect (µm)"
    )

    plt.ylabel(
        "Local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        f"Spearman ρ = {rho_distance:.3f}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "01_Slocal_vs_nearest_defect_distance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # G. Plot 2 — Slocal vs local defect density
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        vectors[
            "local_defect_density_per_um2"
        ],
        vectors[
            "local_S2D"
        ],
        s=10,
        alpha=0.35
    )

    plt.xlabel(
        "Local defect density (µm$^{-2}$)"
    )

    plt.ylabel(
        "Local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        f"Spearman ρ = {rho_density:.3f}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "02_Slocal_vs_local_defect_density.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # H. Plot 3 — binned Slocal vs defect distance
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    good = (
        binned_distance["n"] > 0
    )

    plt.errorbar(
        binned_distance.loc[
            good,
            "x_center"
        ],
        binned_distance.loc[
            good,
            "y_mean"
        ],
        yerr=binned_distance.loc[
            good,
            "y_sem"
        ],
        marker="o",
        capsize=3
    )

    plt.xlabel(
        "Distance to nearest defect (µm)"
    )

    plt.ylabel(
        "Mean local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        "Binned order–defect distance relationship"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "03_binned_Slocal_vs_defect_distance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # I. Plot 4 — binned Slocal vs local defect density
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    good = (
        binned_density["n"] > 0
    )

    plt.errorbar(
        binned_density.loc[
            good,
            "x_center"
        ],
        binned_density.loc[
            good,
            "y_mean"
        ],
        yerr=binned_density.loc[
            good,
            "y_sem"
        ],
        marker="o",
        capsize=3
    )

    plt.xlabel(
        "Local defect density (µm$^{-2}$)"
    )

    plt.ylabel(
        "Mean local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        "Binned order–defect density relationship"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "04_binned_Slocal_vs_local_defect_density.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # J. Per-FOV summary
    # --------------------------------------------------------

    summary = {

        "FOV":
            name,

        "n_orientation_points":
            len(vectors),

        "n_half_defects":
            len(defects),

        "mean_local_S2D":
            vectors[
                "local_S2D"
            ].mean(),

        "median_local_S2D":
            vectors[
                "local_S2D"
            ].median(),

        "mean_nearest_defect_distance_um":
            vectors[
                "nearest_defect_distance_um"
            ].mean(),

        "median_nearest_defect_distance_um":
            vectors[
                "nearest_defect_distance_um"
            ].median(),

        "local_defect_density_radius_px":
            LOCAL_DEFECT_RADIUS_PX,

        "local_defect_density_radius_um":
            LOCAL_DEFECT_RADIUS_UM,

        "rho_Slocal_nearest_defect_distance":
            rho_distance,

        "p_Slocal_nearest_defect_distance":
            p_distance,

        "rho_Slocal_local_defect_density":
            rho_density,

        "p_Slocal_local_defect_density":
            p_density
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            out_dir,
            "STEP5_FOV_summary.csv"
        ),
        index=False
    )

    summary_rows.append(
        summary
    )

# ============================================================
# 5. COMBINED 4-FOV SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP5_ALL_4FOV_summary.csv"
    ),
    index=False
)

# ============================================================
# 6. GROUP MEAN ± SD ± SEM
# ============================================================

group_metrics = [

    "mean_local_S2D",

    "median_local_S2D",

    "mean_nearest_defect_distance_um",

    "median_nearest_defect_distance_um",

    "rho_Slocal_nearest_defect_distance",

    "rho_Slocal_local_defect_density"
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
        "STEP5_50_50_GROUP_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 7. GROUP SUMMARY PLOTS
# ============================================================

x = np.arange(
    len(summary_df)
)

# rho distance
plt.figure(
    figsize=(8, 5)
)

plt.bar(
    x,
    summary_df[
        "rho_Slocal_nearest_defect_distance"
    ]
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
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
    "Spearman ρ"
)

plt.title(
    "50:50 + NaOH\n"
    "$S_{local}$ vs nearest-defect distance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "05_4FOV_rho_Slocal_vs_defect_distance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# rho density
plt.figure(
    figsize=(8, 5)
)

plt.bar(
    x,
    summary_df[
        "rho_Slocal_local_defect_density"
    ]
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
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
    "Spearman ρ"
)

plt.title(
    "50:50 + NaOH\n"
    "$S_{local}$ vs local defect density"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "06_4FOV_rho_Slocal_vs_local_defect_density.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 8. TERMINAL REPORT
# ============================================================

print("\n" + "="*95)
print("STEP 5 COMPLETE — 50:50 + NaOH, matched to 30:70")
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

print("\nPrimary parameters to compare with 30:70:")
print("  rho(Slocal, nearest defect distance)")
print("  rho(Slocal, local defect density)")
print("  mean local S2D")
print("  median local S2D")

print("\nNOTE:")
print(
    "Grid-level p-values are diagnostic only because spatial points "
    "are not independent experimental replicates."
)

print(
    "For the manuscript, compare the effect sizes and consistency "
    "across the four FOVs."
)

print("\nOutput folder:")
print(
    OUTPUT_FOLDER
)
