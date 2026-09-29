
# ============================================================
# STEP 4 — 30:70 + NaOH, 4 FOVs
# PATH-BASED BATCH ANALYSIS
#
# PURPOSE:
#   Convert RAW winding-derived ±1/2 detections from Step 3
#   into merged/unique defect candidates and calculate:
#
#   - merged +1/2 count
#   - merged -1/2 count
#   - total merged defect count
#   - defect density (µm^-2)
#   - characteristic spacing ~ 1/sqrt(rho)
#   - mean nearest-neighbor distance
#   - median nearest-neighbor distance
#   - descriptive unlike-sign NN fraction
#
# IMPORTANT:
#   These are image-derived topological defect CANDIDATES,
#   not direct molecular-defect measurements.
#
# EDIT ONLY SECTION 1 PATHS.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

# Step 3 raw defect files
DEFECTS_1 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV1\raw_half_defect_candidates.csv"
DEFECTS_2 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV2\raw_half_defect_candidates.csv"
DEFECTS_3 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV3\raw_half_defect_candidates.csv"
DEFECTS_4 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV4\raw_half_defect_candidates.csv"

# Valid-region masks from Step 1
MASK_1 = r"C:\Users\Prayas Singh\Desktop\3070\FOV1\02_valid_region_mask.png"
MASK_2 = r"C:\Users\Prayas Singh\Desktop\3070\FOV2\02_valid_region_mask.png"
MASK_3 = r"C:\Users\Prayas Singh\Desktop\3070\FOV3\02_valid_region_mask.png"
MASK_4 = r"C:\Users\Prayas Singh\Desktop\3070\FOV4\02_valid_region_mask.png"

# Final output folder
OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\3070\STEP4_RESULTS"

# ============================================================
# 2. PHYSICAL CALIBRATION + ANALYSIS SETTINGS
# ============================================================

# 10 µm scale bar = 157 px
PIXEL_SIZE_UM = 10.0 / 157.0
PIXEL_AREA_UM2 = PIXEL_SIZE_UM ** 2

# Old matched workflow used 35 px.
# We define it physically and convert back to pixels.
MERGE_RADIUS_UM = 35.0 * PIXEL_SIZE_UM
MERGE_RADIUS_PX = MERGE_RADIUS_UM / PIXEL_SIZE_UM

print(f"Pixel size = {PIXEL_SIZE_UM:.6f} µm/px")
print(f"Merge radius = {MERGE_RADIUS_UM:.3f} µm = {MERGE_RADIUS_PX:.1f} px")

# ============================================================
# 3. DATASET TABLE
# ============================================================

DATASETS = [
    {
        "name": "30_70_NaOH_FOV1",
        "defects": DEFECTS_1,
        "mask": MASK_1,
    },
    {
        "name": "30_70_NaOH_FOV2",
        "defects": DEFECTS_2,
        "mask": MASK_2,
    },
    {
        "name": "30_70_NaOH_FOV3",
        "defects": DEFECTS_3,
        "mask": MASK_3,
    },
    {
        "name": "30_70_NaOH_FOV4",
        "defects": DEFECTS_4,
        "mask": MASK_4,
    },
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 4. UNION-FIND FOR SAME-SIGN MERGING
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def load_valid_area(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    valid = mask > 127
    valid_area_px = int(np.sum(valid))
    valid_area_um2 = valid_area_px * PIXEL_AREA_UM2
    return valid_area_px, valid_area_um2


def standardize_defect_columns(df):
    """
    Accept either:
      x_px, y_px, defect_type
    or close variants if necessary.
    """

    required = ["x_px", "y_px", "defect_type"]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Defect CSV must contain columns: "
            "'x_px', 'y_px', 'defect_type'. "
            f"Missing: {missing}"
        )

    df = df.copy()

    df["x_px"] = pd.to_numeric(
        df["x_px"],
        errors="coerce"
    )

    df["y_px"] = pd.to_numeric(
        df["y_px"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["x_px", "y_px", "defect_type"]
    )

    return df


def merge_one_sign(df_sign, merge_radius_px):
    """
    Merge nearby SAME-SIGN detections using connected components.
    Returns one centroid per connected cluster.
    """

    if len(df_sign) == 0:
        return pd.DataFrame(
            columns=[
                "x_px",
                "y_px",
                "defect_type",
                "cluster_size"
            ]
        )

    pts = df_sign[
        ["x_px", "y_px"]
    ].to_numpy()

    n = len(pts)

    uf = UnionFind(n)

    if n > 1:
        tree = cKDTree(pts)

        close_pairs = tree.query_pairs(
            r=merge_radius_px
        )

        for a, b in close_pairs:
            uf.union(a, b)

    groups = {}

    for i in range(n):
        root = uf.find(i)

        groups.setdefault(
            root,
            []
        ).append(i)

    out = []

    sign_label = df_sign[
        "defect_type"
    ].iloc[0]

    for ids in groups.values():

        cluster_pts = pts[
            ids
        ]

        out.append({
            "x_px":
                np.mean(
                    cluster_pts[:, 0]
                ),

            "y_px":
                np.mean(
                    cluster_pts[:, 1]
                ),

            "defect_type":
                sign_label,

            "cluster_size":
                len(ids)
        })

    return pd.DataFrame(out)


def merge_same_sign_defects(df, merge_radius_px):
    """
    Merge +1/2 only with +1/2
    and -1/2 only with -1/2.
    """

    pieces = []

    for sign in [
        "+1/2",
        "-1/2"
    ]:

        sub = df[
            df["defect_type"] == sign
        ].copy()

        merged = merge_one_sign(
            sub,
            merge_radius_px
        )

        pieces.append(
            merged
        )

    if len(pieces) == 0:
        return pd.DataFrame()

    merged_all = pd.concat(
        pieces,
        ignore_index=True
    )

    if len(merged_all) > 0:

        merged_all[
            "x_um"
        ] = (
            merged_all["x_px"]
            * PIXEL_SIZE_UM
        )

        merged_all[
            "y_um"
        ] = (
            merged_all["y_px"]
            * PIXEL_SIZE_UM
        )

    return merged_all


def nearest_neighbor_statistics(merged_df):
    """
    Returns:
      mean NN distance
      median NN distance
      unlike-sign NN fraction
      pointwise NN table
    """

    if len(merged_df) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
            pd.DataFrame()
        )

    coords = merged_df[
        ["x_px", "y_px"]
    ].to_numpy()

    signs = merged_df[
        "defect_type"
    ].to_numpy()

    tree = cKDTree(
        coords
    )

    distances, indices = tree.query(
        coords,
        k=2
    )

    nn_px = distances[:, 1]
    nn_idx = indices[:, 1]

    nn_um = (
        nn_px
        * PIXEL_SIZE_UM
    )

    nn_sign = signs[
        nn_idx
    ]

    unlike = (
        nn_sign
        != signs
    )

    table = merged_df[
        [
            "x_px",
            "y_px",
            "x_um",
            "y_um",
            "defect_type",
            "cluster_size"
        ]
    ].copy()

    table[
        "nearest_neighbor_index"
    ] = nn_idx

    table[
        "nearest_neighbor_distance_px"
    ] = nn_px

    table[
        "nearest_neighbor_distance_um"
    ] = nn_um

    table[
        "nearest_neighbor_sign"
    ] = nn_sign

    table[
        "nearest_neighbor_unlike_sign"
    ] = unlike

    return (
        float(
            np.mean(nn_um)
        ),

        float(
            np.median(nn_um)
        ),

        float(
            np.mean(unlike)
        ),

        table
    )


def plot_merged_defects(merged_df, name, out_path):
    plt.figure(
        figsize=(9, 7)
    )

    if len(merged_df) > 0:

        plus = merged_df[
            merged_df["defect_type"]
            == "+1/2"
        ]

        minus = merged_df[
            merged_df["defect_type"]
            == "-1/2"
        ]

        if len(plus) > 0:
            plt.scatter(
                plus["x_um"],
                plus["y_um"],
                marker="^",
                s=60,
                label="+1/2"
            )

        if len(minus) > 0:
            plt.scatter(
                minus["x_um"],
                minus["y_um"],
                marker="v",
                s=60,
                label="-1/2"
            )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "X (µm)"
    )

    plt.ylabel(
        "Y (µm)"
    )

    plt.title(
        f"{name}\n"
        "Merged winding-derived defect candidates"
    )

    if len(merged_df) > 0:
        plt.legend()

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 6. PROCESS ALL FOUR FOVs
# ============================================================

all_summary = []

for dataset in DATASETS:

    name = dataset["name"]
    defect_path = dataset["defects"]
    mask_path = dataset["mask"]

    print(
        "\n"
        + "="*85
    )

    print(
        "Processing:",
        name
    )

    print(
        "="*85
    )

    if not os.path.exists(
        defect_path
    ):
        raise FileNotFoundError(
            f"Defect CSV not found:\n{defect_path}"
        )

    if not os.path.exists(
        mask_path
    ):
        raise FileNotFoundError(
            f"Mask not found:\n{mask_path}"
        )

    fov_out = os.path.join(
        OUTPUT_FOLDER,
        name
    )

    os.makedirs(
        fov_out,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load raw defects
    # --------------------------------------------------------

    raw = pd.read_csv(
        defect_path
    )

    raw = standardize_defect_columns(
        raw
    )

    raw_plus = int(
        np.sum(
            raw["defect_type"]
            == "+1/2"
        )
    )

    raw_minus = int(
        np.sum(
            raw["defect_type"]
            == "-1/2"
        )
    )

    raw_total = len(
        raw
    )

    # --------------------------------------------------------
    # Merge same-sign detections
    # --------------------------------------------------------

    merged = merge_same_sign_defects(
        raw,
        MERGE_RADIUS_PX
    )

    merged.to_csv(
        os.path.join(
            fov_out,
            "merged_unique_defect_candidates.csv"
        ),
        index=False
    )

    merged_plus = int(
        np.sum(
            merged["defect_type"]
            == "+1/2"
        )
    ) if len(merged) else 0

    merged_minus = int(
        np.sum(
            merged["defect_type"]
            == "-1/2"
        )
    ) if len(merged) else 0

    merged_total = len(
        merged
    )

    # --------------------------------------------------------
    # Valid area
    # --------------------------------------------------------

    valid_area_px, valid_area_um2 = load_valid_area(
        mask_path
    )

    # --------------------------------------------------------
    # Density and spacing
    # --------------------------------------------------------

    if valid_area_um2 > 0:

        defect_density_um2 = (
            merged_total
            / valid_area_um2
        )

    else:
        defect_density_um2 = np.nan

    if (
        np.isfinite(defect_density_um2)
        and defect_density_um2 > 0
    ):

        characteristic_spacing_um = (
            1.0
            / np.sqrt(
                defect_density_um2
            )
        )

    else:
        characteristic_spacing_um = np.nan

    # --------------------------------------------------------
    # Nearest-neighbor statistics
    # --------------------------------------------------------

    (
        mean_nn_um,
        median_nn_um,
        unlike_nn_fraction,
        nn_table
    ) = nearest_neighbor_statistics(
        merged
    )

    nn_table.to_csv(
        os.path.join(
            fov_out,
            "nearest_neighbor_table.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Merge-cluster QC
    # --------------------------------------------------------

    if len(merged) > 0:

        mean_raw_per_merged = float(
            merged[
                "cluster_size"
            ].mean()
        )

        max_cluster_size = int(
            merged[
                "cluster_size"
            ].max()
        )

    else:

        mean_raw_per_merged = np.nan
        max_cluster_size = np.nan

    # --------------------------------------------------------
    # Save map
    # --------------------------------------------------------

    plot_merged_defects(
        merged,
        name,
        os.path.join(
            fov_out,
            "01_merged_defect_map.png"
        )
    )

    # --------------------------------------------------------
    # Per-FOV summary
    # --------------------------------------------------------

    summary = {
        "FOV":
            name,

        "raw_plus_half":
            raw_plus,

        "raw_minus_half":
            raw_minus,

        "raw_total":
            raw_total,

        "merged_plus_half":
            merged_plus,

        "merged_minus_half":
            merged_minus,

        "merged_total":
            merged_total,

        "merge_radius_px":
            MERGE_RADIUS_PX,

        "merge_radius_um":
            MERGE_RADIUS_UM,

        "valid_area_px":
            valid_area_px,

        "valid_area_um2":
            valid_area_um2,

        "defect_density_um_minus2":
            defect_density_um2,

        "characteristic_spacing_um":
            characteristic_spacing_um,

        "mean_NN_distance_um":
            mean_nn_um,

        "median_NN_distance_um":
            median_nn_um,

        "unlike_sign_NN_fraction_descriptive":
            unlike_nn_fraction,

        "mean_raw_detections_per_merged_defect":
            mean_raw_per_merged,

        "max_merge_cluster_size":
            max_cluster_size
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            fov_out,
            "STEP4_FOV_summary.csv"
        ),
        index=False
    )

    all_summary.append(
        summary
    )

    # terminal output
    print(
        f"Raw defects: {raw_total} "
        f"(+1/2={raw_plus}, -1/2={raw_minus})"
    )

    print(
        f"Merged unique candidates: {merged_total} "
        f"(+1/2={merged_plus}, -1/2={merged_minus})"
    )

    print(
        f"Density = {defect_density_um2:.6f} µm^-2"
    )

    print(
        f"Characteristic spacing = {characteristic_spacing_um:.3f} µm"
    )

    print(
        f"Mean NN = {mean_nn_um:.3f} µm"
    )

    print(
        f"Median NN = {median_nn_um:.3f} µm"
    )

    print(
        f"Unlike-sign NN fraction = {unlike_nn_fraction:.3f}"
    )

# ============================================================
# 7. FINAL 4-FOV SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(
    all_summary
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP4_ALL_4FOV_summary.csv"
    ),
    index=False
)

# ============================================================
# 8. GROUP MEAN ± SD ± SEM
# ============================================================

group_metrics = [
    "merged_total",
    "merged_plus_half",
    "merged_minus_half",
    "defect_density_um_minus2",
    "characteristic_spacing_um",
    "mean_NN_distance_um",
    "median_NN_distance_um",
    "unlike_sign_NN_fraction_descriptive",
    "mean_raw_detections_per_merged_defect"
]

group_rows = []

for metric in group_metrics:

    vals = pd.to_numeric(
        summary_df[metric],
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
                vals.std(ddof=1)
                if len(vals) > 1
                else np.nan
            ),

        "SEM":
            (
                vals.sem(ddof=1)
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
        "STEP4_30_70_GROUP_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 9. FINAL COMPARISON PLOTS
# ============================================================

# defect density
plt.figure(
    figsize=(7, 5)
)

plt.bar(
    summary_df["FOV"],
    summary_df[
        "defect_density_um_minus2"
    ]
)

plt.ylabel(
    "Defect density (µm$^{-2}$)"
)

plt.title(
    "30:70 + NaOH: defect density across 4 FOVs"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_4FOV_defect_density.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# NN distance
plt.figure(
    figsize=(7, 5)
)

x = np.arange(
    len(summary_df)
)

width = 0.35

plt.bar(
    x - width/2,
    summary_df[
        "mean_NN_distance_um"
    ],
    width,
    label="Mean NN"
)

plt.bar(
    x + width/2,
    summary_df[
        "median_NN_distance_um"
    ],
    width,
    label="Median NN"
)

plt.xticks(
    x,
    summary_df["FOV"],
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Nearest-neighbor distance (µm)"
)

plt.title(
    "30:70 + NaOH: defect nearest-neighbor spacing"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_4FOV_NN_distance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 10. FINAL TERMINAL SUMMARY
# ============================================================

print(
    "\n"
    + "="*95
)

print(
    "STEP 4 COMPLETE — 30:70 + NaOH, 4 FOVs"
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
    "\nNOTE:"
)

print(
    "The unlike-sign NN fraction reported here is DESCRIPTIVE only."
)

print(
    "Statistical sign-shuffle testing should be performed later "
    "in the dedicated defect-interaction step."
)
