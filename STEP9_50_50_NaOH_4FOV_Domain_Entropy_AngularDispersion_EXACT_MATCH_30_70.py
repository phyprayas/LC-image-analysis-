
# ============================================================
# STEP 9 — 50:50 + NaOH | 4 FOVs
# ORIENTATION ENTROPY + ANGULAR DISPERSION + DOMAIN AREA
# EXACTLY MATCHED TO THE 30:70 WORKFLOW
#
# INPUT PER FOV
#   Step 3:
#       filtered_vectors.csv
#
# WHAT THIS SCRIPT CALCULATES
#   1. Normalized orientation entropy H_N
#   2. Nematic resultant R = |<exp(i 2 theta)>|
#   3. Angular dispersion sigma_theta
#   4. Orientation domains from 4-neighbour connectivity
#   5. Domain count, mean area, median area,
#      largest area, area-weighted mean area
#   6. Fraction of valid grid retained in domains
#   7. Threshold sensitivity at 10°, 15°, 20°
#
# MATCHED SETTINGS TO 30:70
#   Orientation histogram:
#       18 bins over 0–180°  -> 10° bins
#
#   Domain definition:
#       4-neighbour connected grid cells
#       neighbouring nematic angular difference <= threshold
#
#   Primary threshold:
#       15°
#
#   Sensitivity thresholds:
#       10°, 15°, 20°
#
#   Minimum domain size:
#       4 grid cells
#
#   Grid spacing:
#       20 px
#
#   Calibration:
#       10 µm / 157 px
#
# IMPORTANT
#   Only vectors already retained after coherency + buffered
#   texture-mask filtering in Step 3 are used.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

FOV1_NAME = "50_50_NaOH_FOV1"
FOV1_VECTORS = r"C:\Users\Prayas Singh\Desktop\5050\STEP3_50_50_RESULTS\50_50_NaOH_FOV1\filtered_vectors.csv"

FOV2_NAME = "50_50_NaOH_FOV2"
FOV2_VECTORS = r"C:\Users\Prayas Singh\Desktop\5050\STEP3_50_50_RESULTS\50_50_NaOH_FOV2\filtered_vectors.csv"

FOV3_NAME = "50_50_NaOH_FOV3"
FOV3_VECTORS = r"C:\Users\Prayas Singh\Desktop\5050\STEP3_50_50_RESULTS\50_50_NaOH_FOV3\filtered_vectors.csv"

FOV4_NAME = "50_50_NaOH_FOV4"
FOV4_VECTORS = r"C:\Users\Prayas Singh\Desktop\5050\STEP3_50_50_RESULTS\50_50_NaOH_FOV4\filtered_vectors.csv"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\5050\STEP9_50_50_DOMAIN_ENTROPY_DISPERSION"

DATASETS = [
    {"name": FOV1_NAME, "vectors": FOV1_VECTORS},
    {"name": FOV2_NAME, "vectors": FOV2_VECTORS},
    {"name": FOV3_NAME, "vectors": FOV3_VECTORS},
    {"name": FOV4_NAME, "vectors": FOV4_VECTORS},
]

# ============================================================
# 2. SETTINGS — SAME AS 30:70
# ============================================================

PIXEL_SIZE_UM = 10.0 / 157.0

GRID_SPACING_PX = 20.0
GRID_SPACING_UM = GRID_SPACING_PX * PIXEL_SIZE_UM
GRID_CELL_AREA_UM2 = GRID_SPACING_UM ** 2

ORIENTATION_BINS = 18

PRIMARY_DOMAIN_THRESHOLD_DEG = 15.0
SENSITIVITY_THRESHOLDS_DEG = [10.0, 15.0, 20.0]

MIN_DOMAIN_CELLS = 4

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. HELPERS
# ============================================================

def load_vectors(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Filtered vector file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {"X", "Y", "Orientation_deg"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    for c in ["X", "Y", "Orientation_deg"]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna(
        subset=["X", "Y", "Orientation_deg"]
    ).copy()

    # nematic orientation in [0, 180)
    df["theta_deg"] = (
        df["Orientation_deg"] % 180.0
    )

    df["theta_rad"] = np.deg2rad(
        df["theta_deg"]
    )

    return df


def normalized_orientation_entropy(theta_deg, n_bins=18):
    hist, edges = np.histogram(
        theta_deg,
        bins=n_bins,
        range=(0.0, 180.0)
    )

    p = hist.astype(float)

    if p.sum() == 0:
        return np.nan, hist, edges

    p /= p.sum()

    nz = p > 0

    H = -np.sum(
        p[nz] * np.log(p[nz])
    )

    H_norm = H / np.log(n_bins)

    return H_norm, hist, edges


def nematic_resultant(theta_rad):
    z = np.mean(
        np.exp(
            1j * 2.0 * theta_rad
        )
    )

    R = np.abs(z)

    return R


def angular_dispersion_deg(R):
    if not np.isfinite(R):
        return np.nan

    # avoid log(0)
    R = np.clip(
        R,
        1e-12,
        1.0
    )

    sigma_rad = 0.5 * np.sqrt(
        -2.0 * np.log(R)
    )

    return np.rad2deg(
        sigma_rad
    )


def nematic_angle_difference_deg(a_deg, b_deg):
    """
    Minimal nematic angular difference in degrees, range [0, 90].
    """
    d = abs(
        a_deg - b_deg
    ) % 180.0

    if d > 90.0:
        d = 180.0 - d

    return d


def build_grid(df):
    xvals = np.sort(
        df["X"].unique()
    )

    yvals = np.sort(
        df["Y"].unique()
    )

    x_to_i = {
        x: i for i, x in enumerate(xvals)
    }

    y_to_i = {
        y: i for i, y in enumerate(yvals)
    }

    theta_grid = np.full(
        (
            len(yvals),
            len(xvals)
        ),
        np.nan,
        dtype=float
    )

    for _, row in df.iterrows():
        ix = x_to_i[row["X"]]
        iy = y_to_i[row["Y"]]

        theta_grid[
            iy,
            ix
        ] = row["theta_deg"]

    return xvals, yvals, theta_grid


def identify_domains(theta_grid, threshold_deg, min_cells=4):
    """
    4-neighbour connected components.
    A neighbour is joined if nematic orientation difference
    <= threshold_deg.
    """

    ny, nx = theta_grid.shape

    visited = np.zeros(
        (ny, nx),
        dtype=bool
    )

    domain_id = np.full(
        (ny, nx),
        -1,
        dtype=int
    )

    domains = []

    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    next_id = 0

    for iy in range(ny):
        for ix in range(nx):

            if visited[iy, ix]:
                continue

            if not np.isfinite(
                theta_grid[iy, ix]
            ):
                visited[iy, ix] = True
                continue

            queue = deque()
            queue.append(
                (iy, ix)
            )

            visited[iy, ix] = True

            cells = []

            while queue:

                cy, cx = queue.popleft()
                cells.append(
                    (cy, cx)
                )

                theta_c = theta_grid[
                    cy,
                    cx
                ]

                for dy, dx in neighbors:

                    nyy = cy + dy
                    nxx = cx + dx

                    if (
                        nyy < 0
                        or nyy >= ny
                        or nxx < 0
                        or nxx >= nx
                    ):
                        continue

                    if visited[
                        nyy,
                        nxx
                    ]:
                        continue

                    theta_n = theta_grid[
                        nyy,
                        nxx
                    ]

                    if not np.isfinite(
                        theta_n
                    ):
                        continue

                    dtheta = nematic_angle_difference_deg(
                        theta_c,
                        theta_n
                    )

                    if dtheta <= threshold_deg:

                        visited[
                            nyy,
                            nxx
                        ] = True

                        queue.append(
                            (nyy, nxx)
                        )

            if len(cells) >= min_cells:

                for cy, cx in cells:
                    domain_id[
                        cy,
                        cx
                    ] = next_id

                domains.append(
                    cells
                )

                next_id += 1

    return domain_id, domains


def domain_statistics(domains, n_valid_grid_cells):

    if len(domains) == 0:
        return {
            "n_domains": 0,
            "mean_domain_area_um2": np.nan,
            "median_domain_area_um2": np.nan,
            "largest_domain_area_um2": np.nan,
            "area_weighted_mean_domain_area_um2": np.nan,
            "fraction_valid_grid_in_retained_domains": 0.0
        }, np.array([])

    areas = np.array(
        [
            len(cells) * GRID_CELL_AREA_UM2
            for cells in domains
        ],
        dtype=float
    )

    weighted_mean = (
        np.sum(
            areas * areas
        )
        / np.sum(areas)
    )

    retained_cells = int(
        sum(
            len(cells)
            for cells in domains
        )
    )

    fraction = (
        retained_cells
        / n_valid_grid_cells
        if n_valid_grid_cells > 0
        else np.nan
    )

    stats = {
        "n_domains":
            len(domains),

        "mean_domain_area_um2":
            np.mean(areas),

        "median_domain_area_um2":
            np.median(areas),

        "largest_domain_area_um2":
            np.max(areas),

        "area_weighted_mean_domain_area_um2":
            weighted_mean,

        "fraction_valid_grid_in_retained_domains":
            fraction
    }

    return stats, areas


# ============================================================
# 4. PROCESS EACH FOV
# ============================================================

summary_rows = []
sensitivity_rows = []

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

    df = load_vectors(
        item["vectors"]
    )

    theta_deg = df[
        "theta_deg"
    ].to_numpy()

    theta_rad = df[
        "theta_rad"
    ].to_numpy()

    # --------------------------------------------------------
    # A. Orientation entropy
    # --------------------------------------------------------

    H_norm, hist, hist_edges = normalized_orientation_entropy(
        theta_deg,
        ORIENTATION_BINS
    )

    # --------------------------------------------------------
    # B. Nematic resultant and angular dispersion
    # --------------------------------------------------------

    R = nematic_resultant(
        theta_rad
    )

    sigma_theta_deg = angular_dispersion_deg(
        R
    )

    # --------------------------------------------------------
    # C. Orientation histogram figure
    # --------------------------------------------------------

    hist_centers = 0.5 * (
        hist_edges[:-1]
        + hist_edges[1:]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        hist_centers,
        hist,
        width=(
            hist_edges[1]
            - hist_edges[0]
        )
    )

    plt.xlabel(
        "Orientation angle (deg)"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        f"{name}\n"
        f"Orientation histogram, "
        f"$H_N$ = {H_norm:.3f}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "01_orientation_histogram.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # D. Grid and primary domain analysis
    # --------------------------------------------------------

    xvals, yvals, theta_grid = build_grid(
        df
    )

    n_valid_grid_cells = int(
        np.sum(
            np.isfinite(
                theta_grid
            )
        )
    )

    domain_id, domains = identify_domains(
        theta_grid,
        threshold_deg=PRIMARY_DOMAIN_THRESHOLD_DEG,
        min_cells=MIN_DOMAIN_CELLS
    )

    dom_stats, areas = domain_statistics(
        domains,
        n_valid_grid_cells
    )

    # --------------------------------------------------------
    # E. Save domain areas
    # --------------------------------------------------------

    pd.DataFrame({
        "domain_id":
            np.arange(
                1,
                len(areas) + 1
            ),

        "area_um2":
            areas
    }).to_csv(
        os.path.join(
            out_dir,
            "orientation_domain_areas.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # F. Domain map
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    masked_domains = np.ma.masked_where(
        domain_id < 0,
        domain_id
    )

    plt.imshow(
        masked_domains,
        origin="upper",
        interpolation="nearest"
    )

    plt.colorbar(
        label="Orientation domain ID"
    )

    plt.title(
        f"{name}\n"
        f"Orientation domains "
        f"(Δθ ≤ {PRIMARY_DOMAIN_THRESHOLD_DEG:.0f}°, "
        f"minimum {MIN_DOMAIN_CELLS} cells)"
    )

    plt.xlabel(
        "Grid x-index"
    )

    plt.ylabel(
        "Grid y-index"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "02_orientation_domain_map.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # G. Domain-area distribution
    # --------------------------------------------------------

    if len(areas) > 0:

        plt.figure(
            figsize=(7, 5)
        )

        plt.hist(
            areas,
            bins=min(
                20,
                max(
                    5,
                    int(np.sqrt(len(areas)))
                )
            )
        )

        plt.xlabel(
            "Orientation-domain area (µm²)"
        )

        plt.ylabel(
            "Count"
        )

        plt.title(
            f"{name}\n"
            "Orientation-domain area distribution"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                out_dir,
                "03_domain_area_distribution.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------------
    # H. Threshold sensitivity
    # --------------------------------------------------------

    for threshold in SENSITIVITY_THRESHOLDS_DEG:

        dom_id_t, domains_t = identify_domains(
            theta_grid,
            threshold_deg=threshold,
            min_cells=MIN_DOMAIN_CELLS
        )

        stats_t, areas_t = domain_statistics(
            domains_t,
            n_valid_grid_cells
        )

        sensitivity_rows.append({
            "FOV":
                name,

            "threshold_deg":
                threshold,

            "n_domains":
                stats_t[
                    "n_domains"
                ],

            "mean_domain_area_um2":
                stats_t[
                    "mean_domain_area_um2"
                ],

            "median_domain_area_um2":
                stats_t[
                    "median_domain_area_um2"
                ],

            "largest_domain_area_um2":
                stats_t[
                    "largest_domain_area_um2"
                ],

            "area_weighted_mean_domain_area_um2":
                stats_t[
                    "area_weighted_mean_domain_area_um2"
                ],

            "fraction_valid_grid_in_retained_domains":
                stats_t[
                    "fraction_valid_grid_in_retained_domains"
                ]
        })

    # --------------------------------------------------------
    # I. FOV summary
    # --------------------------------------------------------

    summary = {

        "FOV":
            name,

        "n_valid_orientation_vectors":
            len(df),

        "normalized_orientation_entropy":
            H_norm,

        "nematic_resultant_R":
            R,

        "angular_dispersion_deg":
            sigma_theta_deg,

        "domain_threshold_deg":
            PRIMARY_DOMAIN_THRESHOLD_DEG,

        "minimum_domain_cells":
            MIN_DOMAIN_CELLS,

        "grid_spacing_px":
            GRID_SPACING_PX,

        "grid_spacing_um":
            GRID_SPACING_UM,

        "grid_cell_area_um2":
            GRID_CELL_AREA_UM2,

        "n_domains":
            dom_stats[
                "n_domains"
            ],

        "mean_domain_area_um2":
            dom_stats[
                "mean_domain_area_um2"
            ],

        "median_domain_area_um2":
            dom_stats[
                "median_domain_area_um2"
            ],

        "largest_domain_area_um2":
            dom_stats[
                "largest_domain_area_um2"
            ],

        "area_weighted_mean_domain_area_um2":
            dom_stats[
                "area_weighted_mean_domain_area_um2"
            ],

        "fraction_valid_grid_in_retained_domains":
            dom_stats[
                "fraction_valid_grid_in_retained_domains"
            ]
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            out_dir,
            "STEP9_FOV_summary.csv"
        ),
        index=False
    )

    summary_rows.append(
        summary
    )

# ============================================================
# 5. SAVE ALL-FOV SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP9_ALL_4FOV_summary.csv"
    ),
    index=False
)

sensitivity_df = pd.DataFrame(
    sensitivity_rows
)

sensitivity_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP9_domain_threshold_sensitivity_10_15_20deg.csv"
    ),
    index=False
)

# ============================================================
# 6. GROUP MEAN ± SD ± SEM
# ============================================================

group_metrics = [

    "normalized_orientation_entropy",

    "nematic_resultant_R",

    "angular_dispersion_deg",

    "n_domains",

    "mean_domain_area_um2",

    "median_domain_area_um2",

    "largest_domain_area_um2",

    "area_weighted_mean_domain_area_um2",

    "fraction_valid_grid_in_retained_domains"
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
        "STEP9_50_50_GROUP_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 7. GROUP SUMMARY FIGURES
# ============================================================

# Entropy across FOVs
plt.figure(
    figsize=(8, 5)
)

plt.bar(
    summary_df["FOV"],
    summary_df[
        "normalized_orientation_entropy"
    ]
)

plt.ylabel(
    "Normalized orientation entropy"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.title(
    "50:50 + NaOH — orientation entropy"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "04_4FOV_orientation_entropy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Angular dispersion
plt.figure(
    figsize=(8, 5)
)

plt.bar(
    summary_df["FOV"],
    summary_df[
        "angular_dispersion_deg"
    ]
)

plt.ylabel(
    "Angular dispersion (deg)"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.title(
    "50:50 + NaOH — angular dispersion"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "05_4FOV_angular_dispersion.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Mean domain area
plt.figure(
    figsize=(8, 5)
)

plt.bar(
    summary_df["FOV"],
    summary_df[
        "mean_domain_area_um2"
    ]
)

plt.ylabel(
    "Mean orientation-domain area (µm²)"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.title(
    "50:50 + NaOH — orientation-domain area"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "06_4FOV_mean_domain_area.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 8. TERMINAL REPORT
# ============================================================

print("\n" + "="*95)
print("STEP 9 COMPLETE — 50:50 + NaOH, matched to 30:70")
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

print("\nPrimary manuscript parameters:")
print("  normalized orientation entropy")
print("  angular dispersion")
print("  mean orientation-domain area")
print("  area-weighted mean domain area")
print("  number of orientation domains")

print("\nSensitivity test saved for thresholds:")
print("  10°, 15°, 20°")

print("\nOutput folder:")
print(
    OUTPUT_FOLDER
)
