
# ============================================================
# STEP 5 — 30:70 + NaOH, 4 FOVs
# PATH-BASED ORDER–DEFECT COUPLING ANALYSIS
#
# PURPOSE
#   For each FOV, quantify:
#
#   (1) Spearman rho between local orientational order and
#       distance to the nearest merged defect candidate:
#
#       rho( S_local , d_nearest_defect )
#
#   (2) Spearman rho between local orientational order and
#       local defect density around each orientation point:
#
#       rho( S_local , rho_defect_local )
#
#   Also saves:
#       - per-point coupling table
#       - scatter plots
#       - binned trend plot
#       - per-FOV summary
#       - 4-FOV group mean ± SD ± SEM
#
# IMPORTANT:
#   Edit ONLY SECTION 1 PATHS.
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

# Step 3 filtered vectors with local S2D
LOCAL_1 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV1\filtered_vectors_with_local_S2D.csv"
LOCAL_2 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV2\filtered_vectors_with_local_S2D.csv"
LOCAL_3 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV3\filtered_vectors_with_local_S2D.csv"
LOCAL_4 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV4\filtered_vectors_with_local_S2D.csv"

# Step 4 merged unique defect candidates
DEFECTS_1 = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS\30_70_NaOH_FOV1\merged_unique_defect_candidates.csv"
DEFECTS_2 = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS\30_70_NaOH_FOV2\merged_unique_defect_candidates.csv"
DEFECTS_3 = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS\30_70_NaOH_FOV3\merged_unique_defect_candidates.csv"
DEFECTS_4 = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS\30_70_NaOH_FOV4\merged_unique_defect_candidates.csv"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\3070\STEP5_ORDER_DEFECT_COUPLING"

# ============================================================
# 2. SETTINGS
# ============================================================

# physical calibration
PIXEL_SIZE_UM = 10.0 / 157.0

# local defect-density counting radius
# matched to earlier workflow: 120 px
LOCAL_DEFECT_RADIUS_PX = 120.0
LOCAL_DEFECT_RADIUS_UM = LOCAL_DEFECT_RADIUS_PX * PIXEL_SIZE_UM

# number of bins for diagnostic binned trend
N_BINS = 12

# ============================================================
# 3. DATASETS
# ============================================================

DATASETS = [
    {
        "name": "30_70_NaOH_FOV1",
        "local": LOCAL_1,
        "defects": DEFECTS_1,
    },
    {
        "name": "30_70_NaOH_FOV2",
        "local": LOCAL_2,
        "defects": DEFECTS_2,
    },
    {
        "name": "30_70_NaOH_FOV3",
        "local": LOCAL_3,
        "defects": DEFECTS_3,
    },
    {
        "name": "30_70_NaOH_FOV4",
        "local": LOCAL_4,
        "defects": DEFECTS_4,
    },
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 4. HELPERS
# ============================================================

def binned_mean(x, y, n_bins=12):
    """
    Equal-width bins in x.
    Returns center, mean y, SEM y, count.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 5:
        return pd.DataFrame(
            columns=["x_center", "mean_y", "sem_y", "count"]
        )

    edges = np.linspace(
        np.min(x),
        np.max(x),
        n_bins + 1
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    rows = []

    for i in range(n_bins):

        if i < n_bins - 1:
            sel = (
                (x >= edges[i])
                & (x < edges[i+1])
            )
        else:
            sel = (
                (x >= edges[i])
                & (x <= edges[i+1])
            )

        vals = y[sel]

        if len(vals) == 0:
            continue

        mean_y = np.mean(vals)

        if len(vals) > 1:
            sem_y = np.std(
                vals,
                ddof=1
            ) / np.sqrt(len(vals))
        else:
            sem_y = np.nan

        rows.append({
            "x_center":
                centers[i],

            "mean_y":
                mean_y,

            "sem_y":
                sem_y,

            "count":
                len(vals)
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 5. PROCESS EACH FOV
# ============================================================

all_summary = []

for dataset in DATASETS:

    name = dataset["name"]
    local_path = dataset["local"]
    defect_path = dataset["defects"]

    print(
        "\n"
        + "="*90
    )

    print(
        "Processing:",
        name
    )

    print(
        "="*90
    )

    if not os.path.exists(
        local_path
    ):
        raise FileNotFoundError(
            f"Local-order CSV not found:\n{local_path}"
        )

    if not os.path.exists(
        defect_path
    ):
        raise FileNotFoundError(
            f"Merged-defect CSV not found:\n{defect_path}"
        )

    out_dir = os.path.join(
        OUTPUT_FOLDER,
        name
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load local-order points
    # --------------------------------------------------------

    local_df = pd.read_csv(
        local_path
    )

    needed_local = [
        "X",
        "Y",
        "local_S2D"
    ]

    missing = [
        c
        for c in needed_local
        if c not in local_df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing local-order columns {missing}"
        )

    for c in needed_local:

        local_df[c] = pd.to_numeric(
            local_df[c],
            errors="coerce"
        )

    local_df = local_df.dropna(
        subset=needed_local
    ).copy()

    # --------------------------------------------------------
    # Load merged defects
    # --------------------------------------------------------

    defect_df = pd.read_csv(
        defect_path
    )

    needed_def = [
        "x_px",
        "y_px"
    ]

    missing = [
        c
        for c in needed_def
        if c not in defect_df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing defect columns {missing}"
        )

    for c in needed_def:

        defect_df[c] = pd.to_numeric(
            defect_df[c],
            errors="coerce"
        )

    defect_df = defect_df.dropna(
        subset=needed_def
    ).copy()

    if len(defect_df) < 1:
        raise RuntimeError(
            f"{name}: no merged defects available."
        )

    # --------------------------------------------------------
    # Build KD-tree of merged defects
    # --------------------------------------------------------

    defect_coords = defect_df[
        ["x_px", "y_px"]
    ].to_numpy()

    local_coords = local_df[
        ["X", "Y"]
    ].to_numpy()

    defect_tree = cKDTree(
        defect_coords
    )

    # --------------------------------------------------------
    # Distance to nearest defect
    # --------------------------------------------------------

    nearest_distance_px, nearest_index = defect_tree.query(
        local_coords,
        k=1
    )

    nearest_distance_um = (
        nearest_distance_px
        * PIXEL_SIZE_UM
    )

    local_df[
        "nearest_defect_distance_px"
    ] = nearest_distance_px

    local_df[
        "nearest_defect_distance_um"
    ] = nearest_distance_um

    local_df[
        "nearest_defect_index"
    ] = nearest_index

    # --------------------------------------------------------
    # Local defect density around each orientation point
    # --------------------------------------------------------

    nearby_lists = defect_tree.query_ball_point(
        local_coords,
        r=LOCAL_DEFECT_RADIUS_PX
    )

    local_defect_count = np.array(
        [
            len(ids)
            for ids in nearby_lists
        ],
        dtype=int
    )

    local_area_um2 = (
        np.pi
        * LOCAL_DEFECT_RADIUS_UM**2
    )

    local_defect_density = (
        local_defect_count
        / local_area_um2
    )

    local_df[
        "local_defect_count"
    ] = local_defect_count

    local_df[
        "local_defect_density_um_minus2"
    ] = local_defect_density

    # --------------------------------------------------------
    # Spearman coupling
    # --------------------------------------------------------

    rho_distance, p_distance = spearmanr(
        local_df[
            "local_S2D"
        ],
        local_df[
            "nearest_defect_distance_um"
        ],
        nan_policy="omit"
    )

    rho_density, p_density = spearmanr(
        local_df[
            "local_S2D"
        ],
        local_df[
            "local_defect_density_um_minus2"
        ],
        nan_policy="omit"
    )

    # --------------------------------------------------------
    # Save pointwise table
    # --------------------------------------------------------

    local_df.to_csv(
        os.path.join(
            out_dir,
            "order_defect_pointwise_table.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Binned diagnostic trends
    # --------------------------------------------------------

    bins_distance = binned_mean(
        local_df[
            "nearest_defect_distance_um"
        ],
        local_df[
            "local_S2D"
        ],
        n_bins=N_BINS
    )

    bins_density = binned_mean(
        local_df[
            "local_defect_density_um_minus2"
        ],
        local_df[
            "local_S2D"
        ],
        n_bins=N_BINS
    )

    bins_distance.to_csv(
        os.path.join(
            out_dir,
            "binned_order_vs_distance.csv"
        ),
        index=False
    )

    bins_density.to_csv(
        os.path.join(
            out_dir,
            "binned_order_vs_local_defect_density.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Plot 1: scatter local S2D vs nearest defect distance
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        local_df[
            "nearest_defect_distance_um"
        ],
        local_df[
            "local_S2D"
        ],
        s=10,
        alpha=0.35
    )

    plt.xlabel(
        "Nearest-defect distance (µm)"
    )

    plt.ylabel(
        "Local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        f"$\\rho$ = {rho_distance:.3f}, "
        f"p = {p_distance:.2e}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "01_local_order_vs_nearest_defect_distance_scatter.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: binned local order vs nearest defect distance
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    if len(
        bins_distance
    ) > 0:

        plt.errorbar(
            bins_distance[
                "x_center"
            ],
            bins_distance[
                "mean_y"
            ],
            yerr=bins_distance[
                "sem_y"
            ],
            marker="o",
            capsize=3
        )

    plt.xlabel(
        "Nearest-defect distance (µm)"
    )

    plt.ylabel(
        "Mean local $S_{2D}$"
    )

    plt.title(
        f"{name}\n"
        "Binned orientational order vs defect distance"
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
    # Plot 3: scatter local S2D vs local defect density
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        local_df[
            "local_defect_density_um_minus2"
        ],
        local_df[
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
        f"$\\rho$ = {rho_density:.3f}, "
        f"p = {p_density:.2e}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "03_local_order_vs_local_defect_density_scatter.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 4: binned local order vs local defect density
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    if len(
        bins_density
    ) > 0:

        plt.errorbar(
            bins_density[
                "x_center"
            ],
            bins_density[
                "mean_y"
            ],
            yerr=bins_density[
                "sem_y"
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
        "Binned orientational order vs local defect density"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "04_binned_local_order_vs_local_defect_density.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Per-FOV summary
    # --------------------------------------------------------

    summary = {
        "FOV":
            name,

        "n_local_order_points":
            len(
                local_df
            ),

        "n_merged_defects":
            len(
                defect_df
            ),

        "local_defect_radius_px":
            LOCAL_DEFECT_RADIUS_PX,

        "local_defect_radius_um":
            LOCAL_DEFECT_RADIUS_UM,

        "rho_Slocal_vs_nearest_defect_distance":
            rho_distance,

        "p_Slocal_vs_nearest_defect_distance":
            p_distance,

        "rho_Slocal_vs_local_defect_density":
            rho_density,

        "p_Slocal_vs_local_defect_density":
            p_density,

        "mean_nearest_defect_distance_um":
            np.mean(
                nearest_distance_um
            ),

        "median_nearest_defect_distance_um":
            np.median(
                nearest_distance_um
            ),

        "mean_local_S2D":
            np.mean(
                local_df[
                    "local_S2D"
                ]
            ),

        "median_local_S2D":
            np.median(
                local_df[
                    "local_S2D"
                ]
            )
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

    all_summary.append(
        summary
    )

    # terminal summary
    print(
        "rho(Slocal, nearest defect distance) = "
        f"{rho_distance:.4f}"
    )

    print(
        "p = "
        f"{p_distance:.3e}"
    )

    print(
        "rho(Slocal, local defect density) = "
        f"{rho_density:.4f}"
    )

    print(
        "p = "
        f"{p_density:.3e}"
    )

# ============================================================
# 6. COMBINED 4-FOV SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    all_summary
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP5_ALL_4FOV_summary.csv"
    ),
    index=False
)

# ============================================================
# 7. GROUP MEAN ± SD ± SEM
# ============================================================

metrics = [
    "rho_Slocal_vs_nearest_defect_distance",
    "rho_Slocal_vs_local_defect_density",
    "mean_nearest_defect_distance_um",
    "median_nearest_defect_distance_um",
    "mean_local_S2D",
    "median_local_S2D"
]

group_rows = []

for metric in metrics:

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
        "STEP5_30_70_GROUP_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 8. GROUP PLOTS
# ============================================================

# Coupling strength per FOV
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
        "rho_Slocal_vs_nearest_defect_distance"
    ],
    width,
    label="$\\rho(S_{local},d_{defect})$"
)

plt.bar(
    x + width/2,
    summary_df[
        "rho_Slocal_vs_local_defect_density"
    ],
    width,
    label="$\\rho(S_{local},\\rho_{defect})$"
)

plt.axhline(
    0,
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
    "Spearman $\\rho$"
)

plt.title(
    "30:70 + NaOH: order–defect coupling across 4 FOVs"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_4FOV_order_defect_coupling.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 9. FINAL TERMINAL OUTPUT
# ============================================================

print(
    "\n"
    + "="*95
)

print(
    "STEP 5 COMPLETE — 30:70 + NaOH, 4 FOVs"
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
    "\nINTERPRETATION GUIDE:"
)

print(
    "Positive rho(Slocal, distance) means local order tends to increase "
    "farther away from defect candidates."
)

print(
    "Negative rho(Slocal, local defect density) means local order tends to "
    "decrease in regions with a higher local defect concentration."
)

print(
    "Use image-level effect sizes across FOVs for manuscript interpretation; "
    "do not treat thousands of grid points as independent biological/material replicates."
)
