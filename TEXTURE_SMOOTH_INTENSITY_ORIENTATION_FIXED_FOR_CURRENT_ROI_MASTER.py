
# ============================================================
# FIXED CODE
# T1 / T2 / T3 — SMOOTH INTENSITY + LOCAL OPTICAL ORIENTATION
# Uses the ACTUAL ROI master columns:
#
# ['ROI_ID', 'Texture', 'Texture_name',
#  'x_px', 'y_px', 'width_px', 'height_px',
#  'area_px', 'area_um2',
#  'crop_path', 'mask_path', 'masked_crop_path']
#
# No polygon vertices are required.
# No source_image column is required.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline

# ============================================================
# 1. USER INPUT
# ============================================================

ROI_MASTER_CSV = r"D:\New results\texture_specific_study\STEP1_ROI_SELECTION\TEXTURE_ROI_MASTER.csv"

OUTPUT_ROOT = r"D:\New results\texture_specific_study\STEP8_SMOOTH_INTENSITY_AND_ORIENTATION"

# Correct calibration from your study:
UM_PER_PX = 10.0 / 157.0

# smoothing settings
IMAGE_SMOOTH_SIGMA = 1.0
ORIENTATION_SIGMA = 1.5
SMOOTH_CURVE_POINTS = 300

DPI = 500

TEXTURE_LABELS = {
    "T1": "Orange/brown circular",
    "T2": "Green curved",
    "T3": "Yellow-green four-brush"
}

# ============================================================
# 2. OUTPUT FOLDERS
# ============================================================

PANEL_DIR = os.path.join(
    OUTPUT_ROOT,
    "PANELS"
)

PROFILE_DIR = os.path.join(
    OUTPUT_ROOT,
    "PROFILES"
)

DATA_DIR = os.path.join(
    OUTPUT_ROOT,
    "DATA"
)

for folder in [
    OUTPUT_ROOT,
    PANEL_DIR,
    PROFILE_DIR,
    DATA_DIR
]:
    os.makedirs(
        folder,
        exist_ok=True
    )

# ============================================================
# 3. HELPERS
# ============================================================

def load_rgb(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    return np.asarray(
        Image.open(path).convert("RGB")
    )


def load_mask(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Mask not found:\n{path}"
        )

    mask = np.asarray(
        Image.open(path).convert("L")
    )

    return mask > 0


def rgb_to_gray(rgb):

    rgb = rgb.astype(
        float
    )

    return (
        0.2989 * rgb[..., 0]
        +
        0.5870 * rgb[..., 1]
        +
        0.1140 * rgb[..., 2]
    )


def local_optical_orientation(
    gray,
    sigma=1.5
):

    # image-gradient structure tensor
    gy, gx = np.gradient(
        gray
    )

    Jxx = gaussian_filter(
        gx * gx,
        sigma=sigma
    )

    Jyy = gaussian_filter(
        gy * gy,
        sigma=sigma
    )

    Jxy = gaussian_filter(
        gx * gy,
        sigma=sigma
    )

    psi = 0.5 * np.arctan2(
        2.0 * Jxy,
        Jxx - Jyy
    )

    # nematic symmetry: 0 <= Psi < pi
    psi = np.mod(
        psi,
        np.pi
    )

    return psi


def radial_profile(
    gray,
    mask
):

    ys, xs = np.where(
        mask
    )

    if len(xs) == 0:

        raise ValueError(
            "Empty mask."
        )

    # centroid of selected texture
    cx = np.mean(
        xs
    )

    cy = np.mean(
        ys
    )

    yy, xx = np.indices(
        gray.shape
    )

    radius_px = np.sqrt(
        (xx - cx)**2
        +
        (yy - cy)**2
    )

    radius_um = (
        radius_px
        *
        UM_PER_PX
    )

    r = radius_um[
        mask
    ]

    intensity = gray[
        mask
    ]

    # ~0.2 µm radial bins
    dr = 0.20

    bins = np.arange(
        0,
        np.max(r) + dr,
        dr
    )

    r_mid = []
    mean_i = []
    sd_i = []
    n_pix = []

    for lo, hi in zip(
        bins[:-1],
        bins[1:]
    ):

        sel = (
            (r >= lo)
            &
            (r < hi)
        )

        n = np.sum(
            sel
        )

        if n < 3:
            continue

        r_mid.append(
            0.5 * (
                lo + hi
            )
        )

        mean_i.append(
            np.mean(
                intensity[sel]
            )
        )

        sd_i.append(
            np.std(
                intensity[sel],
                ddof=1
            )
            if n > 1
            else 0
        )

        n_pix.append(
            n
        )

    return (
        np.asarray(r_mid),
        np.asarray(mean_i),
        np.asarray(sd_i),
        np.asarray(n_pix),
        cx,
        cy
    )


def smooth_curve(
    x,
    y,
    n_points=300
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if len(x) < 4:
        return x, y

    order = np.argsort(
        x
    )

    x = x[
        order
    ]

    y = y[
        order
    ]

    x_unique, idx = np.unique(
        x,
        return_index=True
    )

    y_unique = y[
        idx
    ]

    if len(
        x_unique
    ) < 4:

        return (
            x_unique,
            y_unique
        )

    try:

        xs = np.linspace(
            np.min(
                x_unique
            ),
            np.max(
                x_unique
            ),
            n_points
        )

        spline = make_interp_spline(
            x_unique,
            y_unique,
            k=3
        )

        ys = spline(
            xs
        )

        return xs, ys

    except Exception:

        return (
            x_unique,
            y_unique
        )


# ============================================================
# 4. LOAD ROI MASTER
# ============================================================

if not os.path.isfile(
    ROI_MASTER_CSV
):

    raise FileNotFoundError(
        f"ROI master not found:\n{ROI_MASTER_CSV}"
    )

roi_master = pd.read_csv(
    ROI_MASTER_CSV
)

print(
    "\nAvailable columns:"
)

print(
    list(
        roi_master.columns
    )
)

required = [
    "ROI_ID",
    "Texture",
    "Texture_name",
    "crop_path",
    "mask_path"
]

missing = [
    c
    for c in required
    if c not in roi_master.columns
]

if missing:

    raise KeyError(
        f"Missing columns: {missing}"
    )


# ============================================================
# 5. PROCESS EVERY ROI
# ============================================================

summary_rows = []

texture_profiles = {
    "T1": [],
    "T2": [],
    "T3": []
}

for _, row in roi_master.iterrows():

    roi_id = str(
        row[
            "ROI_ID"
        ]
    )

    texture = str(
        row[
            "Texture"
        ]
    )

    texture_name = str(
        row[
            "Texture_name"
        ]
    )

    crop_path = str(
        row[
            "crop_path"
        ]
    )

    mask_path = str(
        row[
            "mask_path"
        ]
    )

    print(
        "\nProcessing:",
        roi_id
    )

    rgb = load_rgb(
        crop_path
    )

    mask = load_mask(
        mask_path
    )

    # If mask is full image sized, crop cannot match.
    # Easy Step-1 normally saves crop-sized masks.
    if mask.shape != rgb.shape[:2]:

        raise ValueError(
            f"Crop/mask size mismatch for {roi_id}\n"
            f"crop = {rgb.shape[:2]}\n"
            f"mask = {mask.shape}"
        )

    gray = rgb_to_gray(
        rgb
    )

    gray_smoothed = gaussian_filter(
        gray,
        sigma=IMAGE_SMOOTH_SIGMA
    )

    psi = local_optical_orientation(
        gray_smoothed,
        sigma=ORIENTATION_SIGMA
    )

    # mask outside selected texture
    intensity_map = gray_smoothed.copy()

    intensity_map[
        ~mask
    ] = np.nan

    psi_map = psi.copy()

    psi_map[
        ~mask
    ] = np.nan


    # ========================================================
    # RADIAL INTENSITY
    # ========================================================

    (
        r_um,
        mean_i,
        sd_i,
        n_pix,
        cx,
        cy
    ) = radial_profile(
        gray_smoothed,
        mask
    )

    r_smooth, i_smooth = smooth_curve(
        r_um,
        mean_i,
        SMOOTH_CURVE_POINTS
    )

    # save raw radial data
    raw_profile_df = pd.DataFrame({
        "r_um":
            r_um,

        "mean_intensity":
            mean_i,

        "sd_intensity":
            sd_i,

        "N_pixels":
            n_pix
    })

    raw_profile_csv = os.path.join(
        DATA_DIR,
        f"{roi_id}_RADIAL_INTENSITY_RAW.csv"
    )

    raw_profile_df.to_csv(
        raw_profile_csv,
        index=False
    )

    # save smoothed radial data
    smooth_profile_df = pd.DataFrame({
        "r_um":
            r_smooth,

        "smoothed_intensity":
            i_smooth
    })

    smooth_profile_csv = os.path.join(
        DATA_DIR,
        f"{roi_id}_RADIAL_INTENSITY_SMOOTH.csv"
    )

    smooth_profile_df.to_csv(
        smooth_profile_csv,
        index=False
    )

    if texture in texture_profiles:

        texture_profiles[
            texture
        ].append({
            "ROI_ID":
                roi_id,

            "r":
                r_um,

            "mean":
                mean_i,

            "sd":
                sd_i
        })


    # ========================================================
    # SUMMARY VALUES
    # ========================================================

    vals_I = intensity_map[
        mask
    ]

    vals_Psi = psi_map[
        mask
    ]

    summary_rows.append({

        "ROI_ID":
            roi_id,

        "Texture":
            texture,

        "Texture_name":
            texture_name,

        "Mean_intensity":
            np.nanmean(
                vals_I
            ),

        "SD_intensity":
            np.nanstd(
                vals_I,
                ddof=1
            ),

        "Mean_Psi_rad":
            np.nanmean(
                vals_Psi
            ),

        "SD_Psi_rad":
            np.nanstd(
                vals_Psi,
                ddof=1
            ),

        "N_mask_pixels":
            int(
                np.sum(
                    mask
                )
            )
    })


    # ========================================================
    # MANUSCRIPT-STYLE 4 PANEL
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            15,
            3.8
        )
    )

    # A. raw crop
    axes[
        0
    ].imshow(
        rgb
    )

    axes[
        0
    ].set_title(
        f"{roi_id}\nRaw texture"
    )

    axes[
        0
    ].axis(
        "off"
    )


    # B. intensity image
    im1 = axes[
        1
    ].imshow(
        intensity_map,
        cmap="inferno"
    )

    axes[
        1
    ].set_title(
        "Intensity"
    )

    axes[
        1
    ].axis(
        "off"
    )

    cb1 = plt.colorbar(
        im1,
        ax=axes[
            1
        ],
        fraction=0.046,
        pad=0.04
    )

    cb1.set_label(
        "Intensity (a.u.)"
    )


    # C. local optical orientation
    im2 = axes[
        2
    ].imshow(
        psi_map,
        cmap="hsv",
        vmin=0,
        vmax=np.pi
    )

    axes[
        2
    ].set_title(
        "Local optical orientation"
    )

    axes[
        2
    ].axis(
        "off"
    )

    cb2 = plt.colorbar(
        im2,
        ax=axes[
            2
        ],
        fraction=0.046,
        pad=0.04
    )

    cb2.set_label(
        r"$\Psi$ (rad)"
    )

    cb2.set_ticks(
        [
            0,
            np.pi / 2,
            np.pi
        ]
    )

    cb2.set_ticklabels(
        [
            "0",
            r"$\pi/2$",
            r"$\pi$"
        ]
    )


    # D. radial intensity profile
    axes[
        3
    ].plot(
        r_um,
        mean_i,
        "o",
        markersize=3,
        alpha=0.65,
        label="Raw binned"
    )

    axes[
        3
    ].plot(
        r_smooth,
        i_smooth,
        linewidth=2.2,
        label="Smooth"
    )

    axes[
        3
    ].fill_between(
        r_um,
        mean_i - sd_i,
        mean_i + sd_i,
        alpha=0.14
    )

    axes[
        3
    ].set_xlabel(
        r"Radius, $r$ ($\mu$m)"
    )

    axes[
        3
    ].set_ylabel(
        "Mean intensity (a.u.)"
    )

    axes[
        3
    ].set_title(
        "Radial intensity"
    )

    axes[
        3
    ].tick_params(
        direction="in",
        top=True,
        right=True
    )

    axes[
        3
    ].legend(
        frameon=False,
        fontsize=8
    )

    plt.tight_layout()

    panel_png = os.path.join(
        PANEL_DIR,
        f"{roi_id}_INTENSITY_ORIENTATION_PANEL.png"
    )

    panel_pdf = os.path.join(
        PANEL_DIR,
        f"{roi_id}_INTENSITY_ORIENTATION_PANEL.pdf"
    )

    fig.savefig(
        panel_png,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        panel_pdf,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(
        fig
    )


# ============================================================
# 6. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary_csv = os.path.join(
    DATA_DIR,
    "ROI_INTENSITY_ORIENTATION_SUMMARY.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# 7. COMBINE T1 / T2 / T3 RADIAL PROFILES
# ============================================================

def combine_profiles(
    profile_list
):

    if len(
        profile_list
    ) == 0:

        return None

    common_max = min(
        np.max(
            p[
                "r"
            ]
        )
        for p in profile_list
    )

    common_r = np.arange(
        0,
        common_max + 0.1,
        0.20
    )

    curves = []

    for p in profile_list:

        rr = p[
            "r"
        ]

        ii = p[
            "mean"
        ]

        interp = np.interp(
            common_r,
            rr,
            ii
        )

        curves.append(
            interp
        )

    arr = np.vstack(
        curves
    )

    mean_curve = np.mean(
        arr,
        axis=0
    )

    if arr.shape[
        0
    ] > 1:

        sd_curve = np.std(
            arr,
            axis=0,
            ddof=1
        )

    else:

        sd_curve = np.zeros_like(
            mean_curve
        )

    r_smooth, i_smooth = smooth_curve(
        common_r,
        mean_curve,
        SMOOTH_CURVE_POINTS
    )

    return (
        common_r,
        mean_curve,
        sd_curve,
        r_smooth,
        i_smooth,
        arr.shape[
            0
        ]
    )


# ============================================================
# 8. SINGLE T1/T2/T3 SMOOTH PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        5.8
    )
)

combined_rows = []

for texture in [
    "T1",
    "T2",
    "T3"
]:

    result = combine_profiles(
        texture_profiles[
            texture
        ]
    )

    if result is None:
        continue

    (
        r_raw,
        mean_curve,
        sd_curve,
        r_smooth,
        i_smooth,
        n_roi
    ) = result

    line, = ax.plot(
        r_smooth,
        i_smooth,
        linewidth=2.5,
        label=(
            f"{texture} – "
            f"{TEXTURE_LABELS[texture]}"
        )
    )

    ax.plot(
        r_raw,
        mean_curve,
        "o",
        markersize=3,
        color=line.get_color(),
        alpha=0.55
    )

    if n_roi > 1:

        ax.fill_between(
            r_raw,
            mean_curve - sd_curve,
            mean_curve + sd_curve,
            color=line.get_color(),
            alpha=0.16
        )

    for rr, mm, ss in zip(
        r_raw,
        mean_curve,
        sd_curve
    ):

        combined_rows.append({

            "Texture":
                texture,

            "r_um":
                rr,

            "Mean_intensity":
                mm,

            "SD_intensity":
                ss,

            "N_ROI":
                n_roi
        })


ax.set_xlabel(
    r"Radius, $r$ ($\mu$m)",
    fontsize=12
)

ax.set_ylabel(
    "Mean POM intensity (a.u.)",
    fontsize=12
)

ax.set_title(
    "Smoothed radial intensity profiles of three optical textures",
    fontsize=13
)

ax.tick_params(
    direction="in",
    top=True,
    right=True
)

ax.legend(
    frameon=False,
    fontsize=9
)

plt.tight_layout()

overlay_png = os.path.join(
    PROFILE_DIR,
    "T1_T2_T3_SMOOTH_RADIAL_INTENSITY.png"
)

overlay_pdf = os.path.join(
    PROFILE_DIR,
    "T1_T2_T3_SMOOTH_RADIAL_INTENSITY.pdf"
)

fig.savefig(
    overlay_png,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    overlay_pdf,
    bbox_inches="tight"
)

plt.show()

plt.close(
    fig
)


# ============================================================
# 9. SAVE COMBINED DATA
# ============================================================

combined_df = pd.DataFrame(
    combined_rows
)

combined_csv = os.path.join(
    DATA_DIR,
    "T1_T2_T3_COMBINED_RADIAL_INTENSITY.csv"
)

combined_df.to_csv(
    combined_csv,
    index=False
)


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print(
    "\n========================================"
)

print(
    "DONE"
)

print(
    "========================================"
)

print(
    "\nPer-ROI panels:"
)

print(
    PANEL_DIR
)

print(
    "\nCombined smooth intensity plot:"
)

print(
    overlay_png
)

print(
    "\nSummary CSV:"
)

print(
    summary_csv
)

print(
    "\nCombined radial data:"
)

print(
    combined_csv
)

print(
    "\nIMPORTANT:"
)

print(
    "The orientation-color panel is an image-derived local "
    "optical orientation map. It is NOT a direct polarization-state "
    "or Stokes-parameter measurement."
)
