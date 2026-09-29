
# ============================================================
# TEXTURE STUDY — STEP 5 (FIXED VERSION)
# CORE–BOUNDARY–EXTERIOR ANALYSIS FOR T1 / T2 / T3
# JUPYTER / VS CODE FRIENDLY
#
# FIX:
# Original POM image path is now entered manually.
# The code no longer requires a 'source_image' column
# inside TEXTURE_ROI_MASTER.csv.
#
# PURPOSE
# -------
# For each manually selected texture ROI:
#   1) divide the texture into CORE, BOUNDARY and EXTERIOR zones
#   2) compare POM intensity
#   3) compare |grad Psi|
#   4) compare local nematic order S_local
#
# QUANTITATIVE SOURCE
# -------------------
# - original full POM image
# - original retained OrientationJ vectors
# - previously calculated whole-field gradient CSV
#
# IMPORTANT
# ---------
# Multiple ROIs from ONE FOV are spatial subsamples,
# not independent material replicates.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

# ============================================================
# 1. USER INPUT — CHANGE THESE PATHS
# ============================================================

ROI_MASTER_CSV = r"D:\New results\texture_specific_study\STEP1_ROI_SELECTION\TEXTURE_ROI_MASTER.csv"

# ORIGINAL POM image WITHOUT blue circles / annotations
ORIGINAL_IMAGE_PATH = r"D:\New results\POM Images\30_70_FOV2.png"

# SAME OrientationJ export for 30_70_FOV2
OJ_VECTOR_CSV = r"D:\New results\YOUR_ORIENTATIONJ_FOLDER\30_70_FOV2.csv"

# SAME gradient numerical CSV from Psi Step 2
GRADIENT_CSV = r"D:\New results\psi map results\New folder\step 2\30_70_FOV2\YOUR_GRADIENT_VALUES.csv"

OUTPUT_DIR = r"D:\New results\texture_specific_study\STEP5_CORE_BOUNDARY_EXTERIOR"

# fixed calibration
UM_PER_PX = 10.0 / 157.0

# same vector filtering as earlier study
COHERENCE_MIN = 0.30

# local order radius, consistent with earlier 30:70 analysis
SLOCAL_RADIUS_PX = 40.0

# zone definitions
BOUNDARY_WIDTH_UM = 1.0
EXTERIOR_WIDTH_UM = 2.0

# minimum number of OJ vectors desired in a zone
MIN_VECTORS_PER_ZONE = 3

DPI = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. HELPERS
# ============================================================

def pick_col(df, candidates, required=True):
    low = {str(c).strip().lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]

    for c in df.columns:
        cl = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c

    if required:
        raise KeyError(
            f"Could not find one of {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def nematic_s2d(angles_rad):
    angles_rad = np.asarray(angles_rad, dtype=float)
    angles_rad = angles_rad[np.isfinite(angles_rad)]

    if len(angles_rad) == 0:
        return np.nan

    return np.abs(
        np.mean(
            np.exp(1j * 2.0 * angles_rad)
        )
    )


def local_s2d_for_vectors(x, y, psi_rad, radius_px):
    pts = np.column_stack([x, y]).astype(float)
    tree = cKDTree(pts)

    out = np.full(len(pts), np.nan, dtype=float)

    for i, p in enumerate(pts):
        idx = tree.query_ball_point(
            p,
            r=radius_px
        )

        if len(idx) < 2:
            continue

        out[i] = nematic_s2d(
            psi_rad[idx]
        )

    return out


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.mean(values) if len(values) else np.nan


def safe_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.std(values, ddof=1) if len(values) > 1 else np.nan


def safe_subtract(a, b):
    if np.isfinite(a) and np.isfinite(b):
        return a - b
    return np.nan


# ============================================================
# 3. LOAD ROI MASTER
# ============================================================

if not os.path.isfile(ROI_MASTER_CSV):
    raise FileNotFoundError(
        f"ROI master not found:\n{ROI_MASTER_CSV}"
    )

roi_master = pd.read_csv(ROI_MASTER_CSV)

roi_id_col = pick_col(roi_master, ["ROI_ID", "roi_id"])
texture_col = pick_col(roi_master, ["Texture", "texture_code"])
texture_name_col = pick_col(roi_master, ["Texture_name", "texture_name"])

# Easy Step-1 format uses x_px and y_px.
# Older version may use bbox_x0_px / bbox_y0_px.
x0_col = pick_col(
    roi_master,
    ["x_px", "bbox_x0_px", "bbox_x0"]
)

y0_col = pick_col(
    roi_master,
    ["y_px", "bbox_y0_px", "bbox_y0"]
)

mask_col = pick_col(
    roi_master,
    ["mask_path", "crop_mask_png", "full_mask_png"]
)

width_col = pick_col(
    roi_master,
    ["width_px", "crop_width_px"],
    required=False
)

height_col = pick_col(
    roi_master,
    ["height_px", "crop_height_px"],
    required=False
)

print("\nROI master loaded:")
print(
    roi_master[
        [roi_id_col, texture_col, texture_name_col]
    ]
)

# ============================================================
# 4. LOAD ORIGINAL POM IMAGE
# ============================================================

if not os.path.isfile(ORIGINAL_IMAGE_PATH):
    raise FileNotFoundError(
        f"Original POM image not found:\n{ORIGINAL_IMAGE_PATH}"
    )

full_rgb = np.asarray(
    Image.open(ORIGINAL_IMAGE_PATH).convert("RGB")
)

full_gray = np.asarray(
    Image.open(ORIGINAL_IMAGE_PATH).convert("L")
).astype(float)

H, W = full_gray.shape

print("\nOriginal image loaded:")
print(ORIGINAL_IMAGE_PATH)
print("Image size:", W, "x", H)

# ============================================================
# 5. LOAD ORIGINAL ORIENTATIONJ VECTORS
# ============================================================

if not os.path.isfile(OJ_VECTOR_CSV):
    raise FileNotFoundError(
        f"OrientationJ CSV not found:\n{OJ_VECTOR_CSV}"
    )

oj = pd.read_csv(OJ_VECTOR_CSV)
oj.columns = [str(c).strip() for c in oj.columns]

x_col = pick_col(oj, ["X", "x", "x_px"])
y_col = pick_col(oj, ["Y", "y", "y_px"])
ori_col = pick_col(oj, ["Orientation", "orientation", "angle"])
coh_col = pick_col(
    oj,
    ["Coherency", "Coherence", "coherency", "coherence"],
    required=False
)

oj["X_px"] = pd.to_numeric(
    oj[x_col],
    errors="coerce"
)

oj["Y_px"] = pd.to_numeric(
    oj[y_col],
    errors="coerce"
)

oj["Orientation_raw"] = pd.to_numeric(
    oj[ori_col],
    errors="coerce"
)

if coh_col is not None:
    oj["Coherence"] = pd.to_numeric(
        oj[coh_col],
        errors="coerce"
    )
else:
    oj["Coherence"] = 1.0

oj = oj.dropna(
    subset=[
        "X_px",
        "Y_px",
        "Orientation_raw"
    ]
).copy()

oj = oj[
    oj["Coherence"] >= COHERENCE_MIN
].copy()

max_abs_ori = np.nanmax(
    np.abs(
        oj["Orientation_raw"].to_numpy(float)
    )
)

if max_abs_ori <= (2 * np.pi + 0.5):
    psi = oj["Orientation_raw"].to_numpy(float)
else:
    psi = np.deg2rad(
        oj["Orientation_raw"].to_numpy(float)
    )

oj["Psi_rad"] = np.mod(
    psi,
    np.pi
)

# calculate local S2D once for all OJ vectors
oj["S_local"] = local_s2d_for_vectors(
    oj["X_px"].to_numpy(float),
    oj["Y_px"].to_numpy(float),
    oj["Psi_rad"].to_numpy(float),
    SLOCAL_RADIUS_PX
)

print(
    f"\nRetained OrientationJ vectors after C >= {COHERENCE_MIN}:",
    len(oj)
)

# ============================================================
# 6. LOAD WHOLE-FIELD GRADIENT DATA
# ============================================================

if not os.path.isfile(GRADIENT_CSV):
    raise FileNotFoundError(
        f"Gradient CSV not found:\n{GRADIENT_CSV}"
    )

grad = pd.read_csv(GRADIENT_CSV)
grad.columns = [str(c).strip() for c in grad.columns]

grad_val_col = pick_col(
    grad,
    [
        "gradPsi",
        "grad_psi",
        "gradient",
        "Gradient_magnitude",
        "orientation_gradient",
        "gradPsi_rad_per_um",
        "grad_psi_rad_per_um"
    ]
)

gx_px_col = pick_col(
    grad,
    ["x_px", "X_px", "x", "X"],
    required=False
)

gy_px_col = pick_col(
    grad,
    ["y_px", "Y_px", "y", "Y"],
    required=False
)

gx_um_col = pick_col(
    grad,
    ["x_um", "X_um"],
    required=False
)

gy_um_col = pick_col(
    grad,
    ["y_um", "Y_um"],
    required=False
)

if gx_px_col is not None and gy_px_col is not None:

    grad["GX_px"] = pd.to_numeric(
        grad[gx_px_col],
        errors="coerce"
    )

    grad["GY_px"] = pd.to_numeric(
        grad[gy_px_col],
        errors="coerce"
    )

elif gx_um_col is not None and gy_um_col is not None:

    grad["GX_px"] = (
        pd.to_numeric(
            grad[gx_um_col],
            errors="coerce"
        )
        / UM_PER_PX
    )

    grad["GY_px"] = (
        pd.to_numeric(
            grad[gy_um_col],
            errors="coerce"
        )
        / UM_PER_PX
    )

else:
    raise KeyError(
        "Gradient CSV requires x/y coordinates in pixels or micrometers.\n"
        f"Available columns: {list(grad.columns)}"
    )

grad["gradPsi"] = pd.to_numeric(
    grad[grad_val_col],
    errors="coerce"
)

grad = grad.dropna(
    subset=[
        "GX_px",
        "GY_px",
        "gradPsi"
    ]
).copy()

grad_tree = cKDTree(
    grad[
        ["GX_px", "GY_px"]
    ].to_numpy(float)
)

print(
    "Gradient points loaded:",
    len(grad)
)

# ============================================================
# 7. ZONE WIDTHS
# ============================================================

boundary_width_px = max(
    1,
    int(
        round(
            BOUNDARY_WIDTH_UM / UM_PER_PX
        )
    )
)

exterior_width_px = max(
    1,
    int(
        round(
            EXTERIOR_WIDTH_UM / UM_PER_PX
        )
    )
)

print(
    "\nBoundary width:",
    boundary_width_px,
    "px =",
    BOUNDARY_WIDTH_UM,
    "µm"
)

print(
    "Exterior width:",
    exterior_width_px,
    "px =",
    EXTERIOR_WIDTH_UM,
    "µm"
)

# ============================================================
# 8. PROCESS EACH ROI
# ============================================================

zone_rows = []
per_roi_rows = []

# vector coordinates rounded once
x_vec_all = np.rint(
    oj["X_px"].to_numpy(float)
).astype(int)

y_vec_all = np.rint(
    oj["Y_px"].to_numpy(float)
).astype(int)

valid_vec_all = (
    (x_vec_all >= 0)
    &
    (x_vec_all < W)
    &
    (y_vec_all >= 0)
    &
    (y_vec_all < H)
)

oj_valid = oj.loc[
    valid_vec_all
].copy()

xv_all = x_vec_all[
    valid_vec_all
]

yv_all = y_vec_all[
    valid_vec_all
]

for _, roi in roi_master.iterrows():

    roi_id = str(
        roi[roi_id_col]
    )

    texture = str(
        roi[texture_col]
    )

    texture_name = str(
        roi[texture_name_col]
    )

    x0 = int(
        round(
            float(
                roi[x0_col]
            )
        )
    )

    y0 = int(
        round(
            float(
                roi[y0_col]
            )
        )
    )

    mask_path = str(
        roi[mask_col]
    )

    print(
        "\nProcessing:",
        roi_id
    )

    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"Mask not found for {roi_id}:\n{mask_path}"
        )

    mask_img = np.asarray(
        Image.open(mask_path).convert("L")
    )

    mask_bool = mask_img > 0

    # --------------------------------------------------------
    # Determine if mask is crop-sized or full-image-sized
    # --------------------------------------------------------

    if mask_bool.shape == (H, W):

        full_mask = mask_bool.copy()

    else:

        h_crop, w_crop = mask_bool.shape

        # If width/height columns exist, use them only as QC.
        if width_col is not None:
            expected_w = int(
                round(
                    float(
                        roi[width_col]
                    )
                )
            )
        else:
            expected_w = w_crop

        if height_col is not None:
            expected_h = int(
                round(
                    float(
                        roi[height_col]
                    )
                )
            )
        else:
            expected_h = h_crop

        # warning only; actual mask dimensions are authoritative
        if expected_w != w_crop or expected_h != h_crop:
            print(
                f"  Warning: CSV ROI size ({expected_w}x{expected_h}) "
                f"!= mask size ({w_crop}x{h_crop}). "
                f"Using mask dimensions."
            )

        full_mask = np.zeros(
            (H, W),
            dtype=bool
        )

        x1 = min(
            W,
            x0 + w_crop
        )

        y1 = min(
            H,
            y0 + h_crop
        )

        valid_w = x1 - x0
        valid_h = y1 - y0

        if valid_w <= 0 or valid_h <= 0:
            print(
                "  ROI lies outside original image. Skipped."
            )
            continue

        full_mask[
            y0:y1,
            x0:x1
        ] = mask_bool[
            :valid_h,
            :valid_w
        ]

    if np.sum(full_mask) == 0:
        print(
            "  Empty ROI mask. Skipped."
        )
        continue

    # ========================================================
    # 9. CORE / BOUNDARY / EXTERIOR MASKS
    # ========================================================

    # distance inside ROI to ROI edge
    inside_dist = distance_transform_edt(
        full_mask
    )

    # distance outside ROI to nearest ROI pixel
    outside_dist = distance_transform_edt(
        ~full_mask
    )

    boundary_mask = (
        full_mask
        &
        (
            inside_dist
            <= boundary_width_px
        )
    )

    core_mask = (
        full_mask
        &
        (
            inside_dist
            > boundary_width_px
        )
    )

    exterior_mask = (
        (~full_mask)
        &
        (
            outside_dist > 0
        )
        &
        (
            outside_dist
            <= exterior_width_px
        )
    )

    zone_masks = {
        "Core": core_mask,
        "Boundary": boundary_mask,
        "Exterior": exterior_mask
    }

    # ========================================================
    # 10. SAMPLE OJ VECTORS + GRADIENT + INTENSITY
    # ========================================================

    zone_cache = {}

    for zone_name, zone_mask in zone_masks.items():

        in_zone = zone_mask[
            yv_all,
            xv_all
        ]

        zdf = oj_valid.loc[
            in_zone
        ].copy()

        if len(zdf) > 0:

            pts = zdf[
                ["X_px", "Y_px"]
            ].to_numpy(float)

            dist_grad, idx_grad = grad_tree.query(
                pts,
                k=1
            )

            zdf["gradPsi"] = grad.iloc[
                idx_grad
            ]["gradPsi"].to_numpy(float)

            zdf["grad_match_distance_px"] = dist_grad

            xi = np.rint(
                zdf["X_px"].to_numpy(float)
            ).astype(int)

            yi = np.rint(
                zdf["Y_px"].to_numpy(float)
            ).astype(int)

            zdf["Intensity"] = full_gray[
                yi,
                xi
            ]

        zone_cache[
            zone_name
        ] = zdf

        mean_intensity = safe_mean(
            zdf["Intensity"]
            if len(zdf)
            else []
        )

        mean_grad = safe_mean(
            zdf["gradPsi"]
            if len(zdf)
            else []
        )

        mean_slocal = safe_mean(
            zdf["S_local"]
            if len(zdf)
            else []
        )

        zone_rows.append({

            "ROI_ID":
                roi_id,

            "Texture":
                texture,

            "Texture_name":
                texture_name,

            "Zone":
                zone_name,

            "N_vectors":
                len(zdf),

            "Enough_vectors":
                len(zdf) >= MIN_VECTORS_PER_ZONE,

            "Mean_intensity":
                mean_intensity,

            "SD_intensity":
                safe_sd(
                    zdf["Intensity"]
                    if len(zdf)
                    else []
                ),

            "Mean_gradPsi":
                mean_grad,

            "SD_gradPsi":
                safe_sd(
                    zdf["gradPsi"]
                    if len(zdf)
                    else []
                ),

            "Mean_Slocal":
                mean_slocal,

            "SD_Slocal":
                safe_sd(
                    zdf["S_local"]
                    if len(zdf)
                    else []
                ),

            "Zone_area_um2":
                np.sum(
                    zone_mask
                )
                * UM_PER_PX**2
        })

    # ========================================================
    # 11. PER-ROI CONTRASTS
    # ========================================================

    core_df = zone_cache["Core"]
    boundary_df = zone_cache["Boundary"]
    exterior_df = zone_cache["Exterior"]

    core_I = safe_mean(
        core_df["Intensity"]
        if len(core_df)
        else []
    )

    boundary_I = safe_mean(
        boundary_df["Intensity"]
        if len(boundary_df)
        else []
    )

    exterior_I = safe_mean(
        exterior_df["Intensity"]
        if len(exterior_df)
        else []
    )

    core_G = safe_mean(
        core_df["gradPsi"]
        if len(core_df)
        else []
    )

    boundary_G = safe_mean(
        boundary_df["gradPsi"]
        if len(boundary_df)
        else []
    )

    exterior_G = safe_mean(
        exterior_df["gradPsi"]
        if len(exterior_df)
        else []
    )

    core_S = safe_mean(
        core_df["S_local"]
        if len(core_df)
        else []
    )

    boundary_S = safe_mean(
        boundary_df["S_local"]
        if len(boundary_df)
        else []
    )

    exterior_S = safe_mean(
        exterior_df["S_local"]
        if len(exterior_df)
        else []
    )

    per_roi_rows.append({

        "ROI_ID":
            roi_id,

        "Texture":
            texture,

        "Texture_name":
            texture_name,

        "Core_N":
            len(core_df),

        "Boundary_N":
            len(boundary_df),

        "Exterior_N":
            len(exterior_df),

        "Core_intensity":
            core_I,

        "Boundary_intensity":
            boundary_I,

        "Exterior_intensity":
            exterior_I,

        "Boundary_minus_core_intensity":
            safe_subtract(
                boundary_I,
                core_I
            ),

        "Boundary_minus_exterior_intensity":
            safe_subtract(
                boundary_I,
                exterior_I
            ),

        "Core_gradPsi":
            core_G,

        "Boundary_gradPsi":
            boundary_G,

        "Exterior_gradPsi":
            exterior_G,

        "Boundary_minus_core_gradPsi":
            safe_subtract(
                boundary_G,
                core_G
            ),

        "Boundary_minus_exterior_gradPsi":
            safe_subtract(
                boundary_G,
                exterior_G
            ),

        "Core_Slocal":
            core_S,

        "Boundary_Slocal":
            boundary_S,

        "Exterior_Slocal":
            exterior_S,

        "Boundary_minus_core_Slocal":
            safe_subtract(
                boundary_S,
                core_S
            ),

        "Boundary_minus_exterior_Slocal":
            safe_subtract(
                boundary_S,
                exterior_S
            )
    })

    # ========================================================
    # 12. ZONE OVERLAY
    # ========================================================

    ys_roi, xs_roi = np.where(
        full_mask
    )

    pad = exterior_width_px + 10

    xx0 = max(
        0,
        int(xs_roi.min()) - pad
    )

    xx1 = min(
        W,
        int(xs_roi.max()) + pad + 1
    )

    yy0 = max(
        0,
        int(ys_roi.min()) - pad
    )

    yy1 = min(
        H,
        int(ys_roi.max()) + pad + 1
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(
        full_rgb[
            yy0:yy1,
            xx0:xx1
        ]
    )

    # contours intentionally use default matplotlib colors
    for zone_name, zone_mask in zone_masks.items():

        local_mask = zone_mask[
            yy0:yy1,
            xx0:xx1
        ]

        if np.any(local_mask):
            try:
                ax.contour(
                    local_mask.astype(float),
                    levels=[0.5],
                    linewidths=1.5
                )
            except Exception:
                pass

    ax.set_title(
        f"{roi_id}\nCore / Boundary / Exterior"
    )

    ax.axis(
        "off"
    )

    roi_out = os.path.join(
        OUTPUT_DIR,
        roi_id
    )

    os.makedirs(
        roi_out,
        exist_ok=True
    )

    fig.savefig(
        os.path.join(
            roi_out,
            f"{roi_id}_ZONE_OVERLAY.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

# ============================================================
# 13. SAVE TABLES
# ============================================================

zone_summary = pd.DataFrame(
    zone_rows
)

zone_csv = os.path.join(
    OUTPUT_DIR,
    "STEP5_ZONE_LEVEL_SUMMARY.csv"
)

zone_summary.to_csv(
    zone_csv,
    index=False
)

per_roi = pd.DataFrame(
    per_roi_rows
)

per_roi_csv = os.path.join(
    OUTPUT_DIR,
    "STEP5_CORE_BOUNDARY_EXTERIOR_PER_ROI.csv"
)

per_roi.to_csv(
    per_roi_csv,
    index=False
)

# ============================================================
# 14. TEXTURE-CLASS CONTRAST SUMMARY
# ============================================================

contrast_metrics = [
    "Boundary_minus_core_intensity",
    "Boundary_minus_exterior_intensity",
    "Boundary_minus_core_gradPsi",
    "Boundary_minus_exterior_gradPsi",
    "Boundary_minus_core_Slocal",
    "Boundary_minus_exterior_Slocal"
]

texture_rows = []

for texture in [
    "T1",
    "T2",
    "T3"
]:

    sub = per_roi[
        per_roi["Texture"] == texture
    ]

    for metric in contrast_metrics:

        vals = pd.to_numeric(
            sub[metric],
            errors="coerce"
        ).dropna().to_numpy(float)

        texture_rows.append({

            "Texture":
                texture,

            "Metric":
                metric,

            "N_ROI":
                len(vals),

            "Mean":
                np.mean(vals)
                if len(vals)
                else np.nan,

            "SD":
                np.std(
                    vals,
                    ddof=1
                )
                if len(vals) > 1
                else np.nan
        })

texture_summary = pd.DataFrame(
    texture_rows
)

texture_summary_csv = os.path.join(
    OUTPUT_DIR,
    "STEP5_TEXTURE_CLASS_ZONE_CONTRAST_SUMMARY.csv"
)

texture_summary.to_csv(
    texture_summary_csv,
    index=False
)

# ============================================================
# 15. PLOTS
# ============================================================

plot_specs = [
    (
        "Mean_intensity",
        "POM intensity (a.u.)"
    ),
    (
        "Mean_gradPsi",
        r"$|\nabla\Psi|$ (rad $\mu$m$^{-1}$)"
    ),
    (
        "Mean_Slocal",
        r"Local orientational order $S_{local}$"
    )
]

zone_order = [
    "Core",
    "Boundary",
    "Exterior"
]

for metric, ylabel in plot_specs:

    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )

    xbase = np.arange(
        len(zone_order)
    )

    offsets = {
        "T1": -0.18,
        "T2": 0.0,
        "T3": 0.18
    }

    for texture in [
        "T1",
        "T2",
        "T3"
    ]:

        means = []
        sds = []

        for zone in zone_order:

            vals = zone_summary.loc[
                (
                    zone_summary["Texture"] == texture
                )
                &
                (
                    zone_summary["Zone"] == zone
                ),
                metric
            ].dropna().to_numpy(float)

            means.append(
                np.mean(vals)
                if len(vals)
                else np.nan
            )

            sds.append(
                np.std(
                    vals,
                    ddof=1
                )
                if len(vals) > 1
                else 0
            )

        ax.errorbar(
            xbase + offsets[texture],
            means,
            yerr=sds,
            marker="o",
            linewidth=1.4,
            capsize=4,
            label=texture
        )

    ax.set_xticks(
        xbase
    )

    ax.set_xticklabels(
        zone_order
    )

    ax.set_xlabel(
        "Texture zone"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.legend(
        frameon=False,
        title="Texture"
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"STEP5_{metric}_CORE_BOUNDARY_EXTERIOR.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n======================================")
print("STEP 5 COMPLETE")
print("======================================")

print("\nZone-level summary:")
print(zone_csv)

print("\nPer-ROI contrasts:")
print(per_roi_csv)

print("\nTexture-class contrast summary:")
print(texture_summary_csv)

print("\nInterpretation guide:")

print(
    "Boundary_minus_core_gradPsi > 0  -> "
    "boundary has stronger orientation distortion than core."
)

print(
    "Boundary_minus_exterior_gradPsi > 0  -> "
    "boundary is more distorted than surrounding texture."
)

print(
    "Boundary_minus_core_Slocal < 0  -> "
    "local orientational order is lower at boundary than core."
)

print(
    "Boundary_minus_exterior_Slocal < 0  -> "
    "boundary is less ordered than surrounding texture."
)
