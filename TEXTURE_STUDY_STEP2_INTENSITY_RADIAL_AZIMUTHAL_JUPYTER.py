
# ============================================================
# TEXTURE STUDY — STEP 2
# INTENSITY + RADIAL + AZIMUTHAL + HARMONIC ANALYSIS
# Uses outputs from STEP 1
#
# Calculates for each ROI:
#   - mean / median / SD intensity
#   - radial intensity I(r)
#   - peak radius r_peak
#   - FWHM of radial peak
#   - azimuthal intensity I(phi)
#   - 2-fold anisotropy A2/A0
#   - 4-fold anisotropy A4/A0
#   - preferred phases phi2 and phi4
#
# IMPORTANT:
# T1/T3 are approximately circular, so radial metrics are
# especially meaningful there.
# For T2 (curved/elongated), radial metrics are descriptive only.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# ============================================================
# 1. USER INPUT
# ============================================================

STEP1_DIR = r"D:\New results\texture_specific_study\STEP1_ROI_SELECTION"

MASTER_CSV = os.path.join(
    STEP1_DIR,
    "TEXTURE_ROI_MASTER.csv"
)

OUTPUT_DIR = r"D:\New results\texture_specific_study\STEP2_INTENSITY_ANALYSIS"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

UM_PER_PX = 10.0 / 157.0

RADIAL_BIN_UM = 0.20
AZIMUTH_BIN_DEG = 5.0

DPI = 300


# ============================================================
# 2. HELPERS
# ============================================================

def load_gray(path):
    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return img.astype(float)


def load_mask(path):
    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not read mask:\n{path}"
        )

    return mask > 0


def weighted_centroid(mask):

    y, x = np.where(mask)

    if len(x) == 0:
        return np.nan, np.nan

    return np.mean(x), np.mean(y)


def radial_profile(
    intensity,
    mask,
    cx,
    cy,
    bin_um=0.20
):

    yy, xx = np.indices(
        intensity.shape
    )

    r_px = np.sqrt(
        (xx - cx)**2
        +
        (yy - cy)**2
    )

    r_um = r_px * UM_PER_PX

    values_r = r_um[
        mask
    ]

    values_i = intensity[
        mask
    ]

    rmax = np.nanmax(
        values_r
    )

    bins = np.arange(
        0,
        rmax + bin_um,
        bin_um
    )

    centers = (
        bins[:-1]
        +
        bins[1:]
    ) / 2

    mean_i = []
    nvals = []

    for lo, hi in zip(
        bins[:-1],
        bins[1:]
    ):

        sel = (
            (values_r >= lo)
            &
            (values_r < hi)
        )

        n = np.sum(sel)

        nvals.append(
            n
        )

        if n > 0:

            mean_i.append(
                np.mean(
                    values_i[sel]
                )
            )

        else:

            mean_i.append(
                np.nan
            )

    return pd.DataFrame({
        "r_um": centers,
        "mean_intensity": mean_i,
        "N_pixels": nvals
    })


def estimate_peak_and_fwhm(
    profile
):

    df = profile.dropna(
        subset=[
            "mean_intensity"
        ]
    ).copy()

    if len(df) < 3:
        return (
            np.nan,
            np.nan,
            np.nan
        )

    r = df[
        "r_um"
    ].to_numpy()

    y = df[
        "mean_intensity"
    ].to_numpy()

    peak_index = np.argmax(
        y
    )

    r_peak = r[
        peak_index
    ]

    i_peak = y[
        peak_index
    ]

    baseline = np.nanmin(
        y
    )

    half_level = baseline + (
        i_peak - baseline
    ) / 2.0

    above = y >= half_level

    if not np.any(
        above
    ):

        return (
            r_peak,
            i_peak,
            np.nan
        )

    idx = np.where(
        above
    )[0]

    # use contiguous peak region around maximum
    left = peak_index

    while (
        left > 0
        and
        above[left - 1]
    ):

        left -= 1

    right = peak_index

    while (
        right < len(y) - 1
        and
        above[right + 1]
    ):

        right += 1

    fwhm = (
        r[right]
        -
        r[left]
    )

    return (
        r_peak,
        i_peak,
        fwhm
    )


def azimuthal_profile(
    intensity,
    mask,
    cx,
    cy,
    bin_deg=5.0
):

    yy, xx = np.indices(
        intensity.shape
    )

    phi = np.arctan2(
        yy - cy,
        xx - cx
    )

    phi_deg = (
        np.degrees(phi)
        +
        360
    ) % 360

    values_phi = phi_deg[
        mask
    ]

    values_i = intensity[
        mask
    ]

    bins = np.arange(
        0,
        360 + bin_deg,
        bin_deg
    )

    centers = (
        bins[:-1]
        +
        bins[1:]
    ) / 2

    mean_i = []
    nvals = []

    for lo, hi in zip(
        bins[:-1],
        bins[1:]
    ):

        sel = (
            (values_phi >= lo)
            &
            (values_phi < hi)
        )

        n = np.sum(
            sel
        )

        nvals.append(
            n
        )

        if n > 0:

            mean_i.append(
                np.mean(
                    values_i[sel]
                )
            )

        else:

            mean_i.append(
                np.nan
            )

    return pd.DataFrame({
        "phi_deg": centers,
        "mean_intensity": mean_i,
        "N_pixels": nvals
    })


def harmonic_metrics(
    az_profile
):

    df = az_profile.dropna(
        subset=[
            "mean_intensity"
        ]
    ).copy()

    if len(df) < 12:

        return {
            "A0": np.nan,
            "A2": np.nan,
            "A4": np.nan,
            "A2_over_A0": np.nan,
            "A4_over_A0": np.nan,
            "phi2_deg": np.nan,
            "phi4_deg": np.nan
        }

    phi = np.deg2rad(
        df[
            "phi_deg"
        ].to_numpy()
    )

    I = df[
        "mean_intensity"
    ].to_numpy()

    # Mean term
    A0 = np.mean(
        I
    )

    # complex Fourier coefficients
    c2 = np.mean(
        I
        *
        np.exp(
            -1j * 2 * phi
        )
    )

    c4 = np.mean(
        I
        *
        np.exp(
            -1j * 4 * phi
        )
    )

    # factor 2 converts complex coefficient to cosine amplitude
    A2 = 2 * np.abs(
        c2
    )

    A4 = 2 * np.abs(
        c4
    )

    # preferred axes
    phi2 = (
        -np.angle(c2) / 2
    )

    phi4 = (
        -np.angle(c4) / 4
    )

    phi2_deg = (
        np.degrees(phi2)
        %
        180
    )

    phi4_deg = (
        np.degrees(phi4)
        %
        90
    )

    return {
        "A0": A0,
        "A2": A2,
        "A4": A4,
        "A2_over_A0": (
            A2 / A0
            if A0 != 0
            else np.nan
        ),
        "A4_over_A0": (
            A4 / A0
            if A0 != 0
            else np.nan
        ),
        "phi2_deg": phi2_deg,
        "phi4_deg": phi4_deg
    }


# ============================================================
# 3. LOAD MASTER TABLE
# ============================================================

if not os.path.isfile(
    MASTER_CSV
):

    raise FileNotFoundError(
        f"Master CSV not found:\n{MASTER_CSV}"
    )

master = pd.read_csv(
    MASTER_CSV
)

print(
    "Loaded",
    len(master),
    "ROI(s)"
)

print(
    master[
        [
            "ROI_ID",
            "Texture",
            "Texture_name"
        ]
    ]
)


# ============================================================
# 4. PROCESS EACH ROI
# ============================================================

summary_rows = []

for _, row in master.iterrows():

    roi_id = row[
        "ROI_ID"
    ]

    texture_code = row[
        "Texture"
    ]

    texture_name = row[
        "Texture_name"
    ]

    crop_path = row[
        "crop_path"
    ]

    mask_path = row[
        "mask_path"
    ]

    print(
        "\nProcessing:",
        roi_id
    )

    intensity = load_gray(
        crop_path
    )

    mask = load_mask(
        mask_path
    )

    # shape safety
    if intensity.shape != mask.shape:

        raise ValueError(
            f"Image/mask shape mismatch for {roi_id}: "
            f"{intensity.shape} vs {mask.shape}"
        )

    cx, cy = weighted_centroid(
        mask
    )

    pix = intensity[
        mask
    ]

    mean_I = np.mean(
        pix
    )

    median_I = np.median(
        pix
    )

    sd_I = np.std(
        pix,
        ddof=1
    ) if len(pix) > 1 else np.nan

    min_I = np.min(
        pix
    )

    max_I = np.max(
        pix
    )

    integrated_I = np.sum(
        pix
    )

    # ----------------------------------------
    # radial profile
    # ----------------------------------------

    radial = radial_profile(
        intensity,
        mask,
        cx,
        cy,
        RADIAL_BIN_UM
    )

    r_peak, i_peak, fwhm = estimate_peak_and_fwhm(
        radial
    )

    # ----------------------------------------
    # azimuthal profile
    # ----------------------------------------

    az = azimuthal_profile(
        intensity,
        mask,
        cx,
        cy,
        AZIMUTH_BIN_DEG
    )

    harm = harmonic_metrics(
        az
    )

    # ----------------------------------------
    # output folder
    # ----------------------------------------

    roi_out = os.path.join(
        OUTPUT_DIR,
        roi_id
    )

    os.makedirs(
        roi_out,
        exist_ok=True
    )

    radial_csv = os.path.join(
        roi_out,
        f"{roi_id}_RADIAL_INTENSITY.csv"
    )

    az_csv = os.path.join(
        roi_out,
        f"{roi_id}_AZIMUTHAL_INTENSITY.csv"
    )

    radial.to_csv(
        radial_csv,
        index=False
    )

    az.to_csv(
        az_csv,
        index=False
    )

    # ----------------------------------------
    # radial plot
    # ----------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 4.5)
    )

    ax.plot(
        radial["r_um"],
        radial["mean_intensity"],
        marker="o",
        markersize=3,
        linewidth=1.5
    )

    if np.isfinite(
        r_peak
    ):

        ax.axvline(
            r_peak,
            linestyle="--",
            linewidth=1,
            label=f"r_peak = {r_peak:.2f} µm"
        )

    ax.set_xlabel(
        "Radius, r (µm)"
    )

    ax.set_ylabel(
        "Mean POM intensity (a.u.)"
    )

    ax.set_title(
        f"{roi_id}\nRadial intensity"
    )

    ax.legend(
        frameon=False
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            roi_out,
            f"{roi_id}_RADIAL_INTENSITY.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # ----------------------------------------
    # azimuthal plot
    # ----------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 4.5)
    )

    ax.plot(
        az["phi_deg"],
        az["mean_intensity"],
        linewidth=1.5
    )

    ax.set_xlabel(
        "Azimuthal angle, φ (degrees)"
    )

    ax.set_ylabel(
        "Mean POM intensity (a.u.)"
    )

    ax.set_title(
        f"{roi_id}\nAzimuthal intensity"
    )

    ax.set_xlim(
        0,
        360
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            roi_out,
            f"{roi_id}_AZIMUTHAL_INTENSITY.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # ----------------------------------------
    # summary
    # ----------------------------------------

    summary_rows.append({

        "ROI_ID":
            roi_id,

        "Texture":
            texture_code,

        "Texture_name":
            texture_name,

        "Mean_intensity":
            mean_I,

        "Median_intensity":
            median_I,

        "SD_intensity":
            sd_I,

        "Min_intensity":
            min_I,

        "Max_intensity":
            max_I,

        "Integrated_intensity":
            integrated_I,

        "r_peak_um":
            r_peak,

        "I_peak":
            i_peak,

        "FWHM_um":
            fwhm,

        "A0":
            harm["A0"],

        "A2":
            harm["A2"],

        "A4":
            harm["A4"],

        "A2_over_A0":
            harm["A2_over_A0"],

        "A4_over_A0":
            harm["A4_over_A0"],

        "phi2_deg":
            harm["phi2_deg"],

        "phi4_deg":
            harm["phi4_deg"]
    })


# ============================================================
# 5. SAVE ROI SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "STEP2_TEXTURE_INTENSITY_SUMMARY_PER_ROI.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)

print(
    "\n======================================="
)

print(
    "STEP 2 COMPLETE"
)

print(
    "======================================="
)

print(
    summary
)

print(
    "\nSaved:"
)

print(
    summary_csv
)


# ============================================================
# 6. TEXTURE-CLASS SUMMARY
# ============================================================

metrics = [
    "Mean_intensity",
    "r_peak_um",
    "FWHM_um",
    "A2_over_A0",
    "A4_over_A0"
]

texture_summary_rows = []

for texture in [
    "T1",
    "T2",
    "T3"
]:

    sub = summary[
        summary["Texture"] == texture
    ]

    for metric in metrics:

        values = pd.to_numeric(
            sub[metric],
            errors="coerce"
        ).dropna().to_numpy()

        texture_summary_rows.append({

            "Texture":
                texture,

            "Metric":
                metric,

            "N_ROI":
                len(values),

            "Mean":
                np.mean(values)
                if len(values)
                else np.nan,

            "SD":
                np.std(
                    values,
                    ddof=1
                )
                if len(values) > 1
                else np.nan
        })


texture_summary = pd.DataFrame(
    texture_summary_rows
)

texture_summary_csv = os.path.join(
    OUTPUT_DIR,
    "STEP2_TEXTURE_CLASS_SUMMARY.csv"
)

texture_summary.to_csv(
    texture_summary_csv,
    index=False
)

print(
    "\nTexture-class summary:"
)

print(
    texture_summary
)
