
# ============================================================
# STEP 9 — 30:70 + NaOH, 4 FOVs
# PATH-BASED DOMAIN AREA + ORIENTATION ENTROPY
# + ANGULAR DISPERSION ANALYSIS
#
# INPUT:
#   Step 3 filtered_vectors_with_local_S2D.csv for each FOV
#
# OUTPUTS PER FOV:
#   - orientation histogram
#   - normalized orientation entropy
#   - global nematic resultant R (= global S2D)
#   - nematic angular dispersion sigma_theta (deg)
#   - orientation-defined domains
#   - domain count
#   - mean/median/largest domain area (µm²)
#   - area-weighted mean domain area
#   - domain area distribution
#
# DOMAIN DEFINITION:
#   neighboring OrientationJ grid points are connected if:
#     1) both are retained after coherency filtering
#     2) nematic angular difference <= DOMAIN_TOL_DEG
#
# IMPORTANT:
#   Edit ONLY SECTION 1 PATHS.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

FOV1 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV1\filtered_vectors_with_local_S2D.csv"
FOV2 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV2\filtered_vectors_with_local_S2D.csv"
FOV3 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV3\filtered_vectors_with_local_S2D.csv"
FOV4 = r"C:\Users\Prayas Singh\Desktop\3070\STEP3_RESULTS\30_70_NaOH_FOV4\filtered_vectors_with_local_S2D.csv"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\3070\STEP9_DOMAIN_ENTROPY_DISPERSION"

# ============================================================
# 2. SETTINGS
# ============================================================

# 10 µm scale bar = 157 px
PIXEL_SIZE_UM = 10.0 / 157.0

# OrientationJ grid spacing
GRID_SPACING_PX = 20.0
GRID_SPACING_UM = GRID_SPACING_PX * PIXEL_SIZE_UM

# Area represented by one retained grid node
GRID_CELL_AREA_UM2 = GRID_SPACING_UM ** 2

# Main domain threshold
DOMAIN_TOL_DEG = 15.0

# Minimum connected grid cells to call a domain
MIN_DOMAIN_CELLS = 4

# Sensitivity thresholds
SENSITIVITY_TOLS = [10.0, 15.0, 20.0]

# Orientation entropy histogram
N_ORIENTATION_BINS = 18   # 10° bins over 0–180°

DATASETS = [
    {"name": "30_70_NaOH_FOV1", "path": FOV1},
    {"name": "30_70_NaOH_FOV2", "path": FOV2},
    {"name": "30_70_NaOH_FOV3", "path": FOV3},
    {"name": "30_70_NaOH_FOV4", "path": FOV4},
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. HELPERS
# ============================================================

def nematic_angle_diff_deg(a, b):
    """
    Smallest nematic angular difference in degrees, range [0, 90].
    """
    d = np.deg2rad(a - b)
    wrapped = 0.5 * np.arctan2(
        np.sin(2*d),
        np.cos(2*d)
    )
    return np.abs(
        np.rad2deg(wrapped)
    )


def orientation_entropy(theta_deg, n_bins=18):
    """
    Shannon entropy of nematic orientation histogram over [0,180).
    Returns raw H and normalized H/log(Nbins).
    """
    theta = np.mod(theta_deg, 180.0)

    edges = np.linspace(
        0.0,
        180.0,
        n_bins + 1
    )

    counts, _ = np.histogram(
        theta,
        bins=edges
    )

    p = counts / np.sum(counts)

    p_nonzero = p[p > 0]

    H = -np.sum(
        p_nonzero * np.log(p_nonzero)
    )

    H_norm = H / np.log(n_bins)

    return H, H_norm, edges, counts


def nematic_resultant_and_dispersion(theta_deg):
    """
    R = |<exp(i 2 theta)>|
    angular dispersion sigma_theta = 0.5*sqrt(-2 ln R)
    returned in degrees.

    Note:
    R is mathematically the same global nematic order metric S2D.
    """
    theta = np.deg2rad(theta_deg)

    z = np.mean(
        np.exp(
            1j * 2.0 * theta
        )
    )

    R = np.abs(z)

    # numerical safety
    R_safe = np.clip(
        R,
        1e-12,
        1.0
    )

    sigma_rad = 0.5 * np.sqrt(
        -2.0 * np.log(
            R_safe
        )
    )

    sigma_deg = np.rad2deg(
        sigma_rad
    )

    return R, sigma_deg


def build_grid(df):
    """
    Build orientation array on exact OrientationJ X/Y grid.
    """
    xs = np.sort(
        df["X"].unique()
    )

    ys = np.sort(
        df["Y"].unique()
    )

    orientation = df.pivot(
        index="Y",
        columns="X",
        values="Orientation"
    ).reindex(
        index=ys,
        columns=xs
    )

    arr = orientation.to_numpy(
        dtype=float
    )

    valid = np.isfinite(arr)

    return xs, ys, arr, valid


def connected_orientation_domains(
    theta_grid,
    valid_grid,
    tolerance_deg,
    min_cells
):
    """
    4-neighbor connected-components segmentation.

    Two neighboring valid grid cells are linked when their
    nematic orientation difference <= tolerance_deg.
    """

    ny, nx = theta_grid.shape

    visited = np.zeros(
        (ny, nx),
        dtype=bool
    )

    domain_id_map = np.full(
        (ny, nx),
        -1,
        dtype=int
    )

    domains = []

    neighbor_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    next_domain_id = 0

    for iy in range(ny):

        for ix in range(nx):

            if (
                not valid_grid[iy, ix]
                or visited[iy, ix]
            ):
                continue

            # flood fill
            q = deque()
            q.append(
                (iy, ix)
            )

            visited[
                iy,
                ix
            ] = True

            cells = [
                (iy, ix)
            ]

            while q:

                cy, cx = q.popleft()

                theta_current = theta_grid[
                    cy,
                    cx
                ]

                for dy, dx in neighbor_offsets:

                    nyy = cy + dy
                    nxx = cx + dx

                    if (
                        nyy < 0
                        or nyy >= ny
                        or nxx < 0
                        or nxx >= nx
                    ):
                        continue

                    if (
                        visited[nyy, nxx]
                        or not valid_grid[nyy, nxx]
                    ):
                        continue

                    theta_neighbor = theta_grid[
                        nyy,
                        nxx
                    ]

                    dtheta = nematic_angle_diff_deg(
                        theta_current,
                        theta_neighbor
                    )

                    if dtheta <= tolerance_deg:

                        visited[
                            nyy,
                            nxx
                        ] = True

                        q.append(
                            (nyy, nxx)
                        )

                        cells.append(
                            (nyy, nxx)
                        )

            # retain only sufficiently large components
            if len(cells) >= min_cells:

                for cy, cx in cells:

                    domain_id_map[
                        cy,
                        cx
                    ] = next_domain_id

                domains.append(
                    cells
                )

                next_domain_id += 1

    return domains, domain_id_map


def domain_metrics(
    domains,
    theta_grid
):
    """
    Convert connected components to domain-size table.
    """

    rows = []

    for i, cells in enumerate(domains):

        n_cells = len(cells)

        area_um2 = (
            n_cells
            * GRID_CELL_AREA_UM2
        )

        theta_vals = np.array(
            [
                theta_grid[y, x]
                for y, x in cells
            ],
            dtype=float
        )

        # local nematic resultant inside domain
        theta_rad = np.deg2rad(
            theta_vals
        )

        z = np.mean(
            np.exp(
                1j * 2.0 * theta_rad
            )
        )

        local_R = np.abs(
            z
        )

        preferred_angle = (
            0.5
            * np.rad2deg(
                np.angle(z)
            )
        ) % 180.0

        rows.append({
            "domain_id":
                i,

            "n_grid_cells":
                n_cells,

            "domain_area_um2":
                area_um2,

            "domain_local_R":
                local_R,

            "domain_preferred_orientation_deg":
                preferred_angle
        })

    return pd.DataFrame(
        rows
    )


def weighted_mean(values, weights):
    values = np.asarray(
        values,
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    if (
        len(values) == 0
        or np.sum(weights) <= 0
    ):
        return np.nan

    return np.sum(
        values * weights
    ) / np.sum(
        weights
    )


def analyze_domains_for_tolerance(
    theta_grid,
    valid_grid,
    tolerance_deg
):
    domains, domain_map = connected_orientation_domains(
        theta_grid,
        valid_grid,
        tolerance_deg=tolerance_deg,
        min_cells=MIN_DOMAIN_CELLS
    )

    dtable = domain_metrics(
        domains,
        theta_grid
    )

    if len(dtable) == 0:

        summary = {
            "domain_tolerance_deg":
                tolerance_deg,

            "n_domains":
                0,

            "mean_domain_area_um2":
                np.nan,

            "median_domain_area_um2":
                np.nan,

            "largest_domain_area_um2":
                np.nan,

            "area_weighted_mean_domain_area_um2":
                np.nan,

            "fraction_valid_grid_in_retained_domains":
                0.0
        }

        return summary, dtable, domain_map

    total_valid_cells = np.sum(
        valid_grid
    )

    retained_cells = np.sum(
        dtable[
            "n_grid_cells"
        ]
    )

    summary = {

        "domain_tolerance_deg":
            tolerance_deg,

        "n_domains":
            len(dtable),

        "mean_domain_area_um2":
            dtable[
                "domain_area_um2"
            ].mean(),

        "median_domain_area_um2":
            dtable[
                "domain_area_um2"
            ].median(),

        "largest_domain_area_um2":
            dtable[
                "domain_area_um2"
            ].max(),

        "area_weighted_mean_domain_area_um2":
            weighted_mean(
                dtable[
                    "domain_area_um2"
                ],
                dtable[
                    "domain_area_um2"
                ]
            ),

        "fraction_valid_grid_in_retained_domains":
            (
                retained_cells
                / total_valid_cells
                if total_valid_cells > 0
                else np.nan
            )
    }

    return summary, dtable, domain_map


# ============================================================
# 4. PROCESS EACH FOV
# ============================================================

all_main_summaries = []
all_sensitivity = []

for dataset in DATASETS:

    name = dataset["name"]
    path = dataset["path"]

    print(
        "\n"
        + "="*95
    )

    print(
        "Processing:",
        name
    )

    print(
        "="*95
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"Input CSV not found:\n{path}"
        )

    out_dir = os.path.join(
        OUTPUT_FOLDER,
        name
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    df = pd.read_csv(
        path
    )

    required = [
        "X",
        "Y",
        "Orientation"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}"
        )

    for c in required:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    ).copy()

    theta_deg = np.mod(
        df[
            "Orientation"
        ].to_numpy(),
        180.0
    )

    # --------------------------------------------------------
    # ORIENTATION ENTROPY
    # --------------------------------------------------------

    (
        H,
        H_norm,
        hist_edges,
        hist_counts
    ) = orientation_entropy(
        theta_deg,
        n_bins=N_ORIENTATION_BINS
    )

    hist_centers = 0.5 * (
        hist_edges[:-1]
        + hist_edges[1:]
    )

    hist_df = pd.DataFrame({
        "orientation_bin_center_deg":
            hist_centers,

        "count":
            hist_counts,

        "probability":
            hist_counts
            / np.sum(hist_counts)
    })

    hist_df.to_csv(
        os.path.join(
            out_dir,
            "orientation_histogram.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # ANGULAR DISPERSION
    # --------------------------------------------------------

    R_global, sigma_theta_deg = nematic_resultant_and_dispersion(
        theta_deg
    )

    # --------------------------------------------------------
    # GRID + DOMAIN ANALYSIS
    # --------------------------------------------------------

    (
        xs,
        ys,
        theta_grid,
        valid_grid
    ) = build_grid(
        df
    )

    main_summary = None
    main_domain_table = None
    main_domain_map = None

    for tol in SENSITIVITY_TOLS:

        (
            dsummary,
            dtable,
            dmap
        ) = analyze_domains_for_tolerance(
            theta_grid,
            valid_grid,
            tolerance_deg=tol
        )

        dsummary[
            "FOV"
        ] = name

        all_sensitivity.append(
            dsummary
        )

        if np.isclose(
            tol,
            DOMAIN_TOL_DEG
        ):

            main_summary = dsummary
            main_domain_table = dtable
            main_domain_map = dmap

    # --------------------------------------------------------
    # SAVE MAIN DOMAIN TABLE
    # --------------------------------------------------------

    main_domain_table.to_csv(
        os.path.join(
            out_dir,
            "orientation_domain_table.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Plot 1 — orientation histogram
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        hist_centers,
        hist_counts,
        width=(
            180.0
            / N_ORIENTATION_BINS
            * 0.9
        )
    )

    plt.xlabel(
        "Orientation (deg)"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        f"{name}\n"
        f"Orientation distribution | "
        f"Normalized entropy = {H_norm:.3f}"
    )

    plt.xlim(
        0,
        180
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
    # Plot 2 — orientation-domain map
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    display_map = np.array(
        main_domain_map,
        dtype=float
    )

    display_map[
        display_map < 0
    ] = np.nan

    plt.imshow(
        display_map,
        origin="upper",
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(
        label="Orientation-domain ID"
    )

    plt.xlabel(
        "OrientationJ grid column"
    )

    plt.ylabel(
        "OrientationJ grid row"
    )

    plt.title(
        f"{name}\n"
        f"Orientation-defined domains "
        f"(Δθ ≤ {DOMAIN_TOL_DEG:.0f}°, "
        f"minimum {MIN_DOMAIN_CELLS} cells)"
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
    # Plot 3 — domain-area distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    if len(
        main_domain_table
    ) > 0:

        plt.hist(
            main_domain_table[
                "domain_area_um2"
            ],
            bins="auto",
            alpha=0.8
        )

    plt.xlabel(
        "Orientation-domain area (µm$^2$)"
    )

    plt.ylabel(
        "Domain count"
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
    # COMBINED SUMMARY
    # --------------------------------------------------------

    final_summary = {

        "FOV":
            name,

        "n_orientation_points":
            len(df),

        "orientation_entropy_H":
            H,

        "normalized_orientation_entropy":
            H_norm,

        "global_nematic_resultant_R":
            R_global,

        "angular_dispersion_deg":
            sigma_theta_deg,

        "domain_tolerance_deg":
            DOMAIN_TOL_DEG,

        "minimum_domain_cells":
            MIN_DOMAIN_CELLS,

        "grid_spacing_um":
            GRID_SPACING_UM,

        "grid_cell_area_um2":
            GRID_CELL_AREA_UM2,

        "n_domains":
            main_summary[
                "n_domains"
            ],

        "mean_domain_area_um2":
            main_summary[
                "mean_domain_area_um2"
            ],

        "median_domain_area_um2":
            main_summary[
                "median_domain_area_um2"
            ],

        "largest_domain_area_um2":
            main_summary[
                "largest_domain_area_um2"
            ],

        "area_weighted_mean_domain_area_um2":
            main_summary[
                "area_weighted_mean_domain_area_um2"
            ],

        "fraction_valid_grid_in_retained_domains":
            main_summary[
                "fraction_valid_grid_in_retained_domains"
            ]
    }

    pd.DataFrame(
        [final_summary]
    ).to_csv(
        os.path.join(
            out_dir,
            "STEP9_FOV_summary.csv"
        ),
        index=False
    )

    all_main_summaries.append(
        final_summary
    )

    print(
        pd.DataFrame(
            [final_summary]
        ).to_string(
            index=False
        )
    )


# ============================================================
# 5. COMBINED 4-FOV SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    all_main_summaries
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP9_ALL_4FOV_summary.csv"
    ),
    index=False
)

sensitivity_df = pd.DataFrame(
    all_sensitivity
)

sensitivity_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP9_DOMAIN_THRESHOLD_SENSITIVITY_10_15_20deg.csv"
    ),
    index=False
)

# ============================================================
# 6. GROUP MEAN ± SD ± SEM
# ============================================================

group_metrics = [
    "normalized_orientation_entropy",
    "global_nematic_resultant_R",
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
        "STEP9_30_70_GROUP_mean_SD_SEM.csv"
    ),
    index=False
)

# ============================================================
# 7. GROUP PLOTS
# ============================================================

# Entropy
plt.figure(
    figsize=(7, 5)
)

plt.bar(
    summary_df[
        "FOV"
    ],
    summary_df[
        "normalized_orientation_entropy"
    ]
)

plt.ylabel(
    "Normalized orientation entropy"
)

plt.ylim(
    0,
    1
)

plt.title(
    "30:70 + NaOH: orientation entropy across 4 FOVs"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_4FOV_orientation_entropy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Angular dispersion
plt.figure(
    figsize=(7, 5)
)

plt.bar(
    summary_df[
        "FOV"
    ],
    summary_df[
        "angular_dispersion_deg"
    ]
)

plt.ylabel(
    "Nematic angular dispersion (deg)"
)

plt.title(
    "30:70 + NaOH: angular dispersion across 4 FOVs"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_4FOV_angular_dispersion.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Domain area
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
        "mean_domain_area_um2"
    ],
    width,
    label="Mean"
)

plt.bar(
    x + width/2,
    summary_df[
        "median_domain_area_um2"
    ],
    width,
    label="Median"
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
    "Orientation-domain area (µm$^2$)"
)

plt.title(
    "30:70 + NaOH: orientation-domain area"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_4FOV_domain_area.png"
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
    "STEP 9 COMPLETE — 30:70 + NaOH, 4 FOVs"
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
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "Normalized orientation entropy close to 0 = narrow orientation distribution."
)

print(
    "Normalized orientation entropy close to 1 = broad/multidirectional orientation distribution."
)

print(
    "Angular dispersion is a circular-statistical width derived from the nematic resultant."
)

print(
    "Orientation-domain area is threshold-defined and should be interpreted together with the 10°/15°/20° sensitivity table."
)

print(
    "The primary domain result uses Δθ <= 15° and minimum 4 connected OrientationJ grid cells."
)
