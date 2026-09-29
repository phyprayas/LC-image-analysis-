
# ============================================================
# TEXTURE STUDY — STEP 4
# ORIENTATION–INTENSITY + DISTORTION–INTENSITY COUPLING
# For T1 / T2 / T3 ROIs from ONE representative FOV
#
# QUANTITATIVE SOURCE:
#   - Original retained OrientationJ vectors (NOT dense Psi pixels)
#   - POM intensity sampled at those vector positions
#   - |grad Psi| sampled from the previously calculated gradient map
#
# OUTPUTS PER ROI:
#   1) matched vector/intensity/gradient CSV
#   2) binned I(Psi)
#   3) I vs sin^2(2Psi)
#   4) I vs |gradPsi|
#   5) harmonic orientation-intensity model
#
# SUMMARY:
#   - Spearman/Pearson coupling values
#   - harmonic model R^2
#   - T1/T2/T3 comparison tables/plots
#
# IMPORTANT:
# Multiple ROIs from one FOV are spatial subsamples, not
# independent material replicates.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree

# ============================================================
# 1. USER INPUT — CHANGE ONLY THESE PATHS
# ============================================================

# STEP 1 ROI master
ROI_MASTER_CSV = r"D:\New results\texture_specific_study\STEP1_ROI_SELECTION\TEXTURE_ROI_MASTER.csv"

# ORIGINAL OrientationJ vector CSV for the SAME full image/FOV
# Example: 30_70_FOV2 OrientationJ export
OJ_VECTOR_CSV = r"D:\New results\YOUR_ORIENTATIONJ_FOLDER\30_70_FOV2.csv"

# Previously calculated whole-field gradient CSV for SAME FOV
# It should contain x/y position + |gradPsi| values.
GRADIENT_CSV = r"D:\New results\psi map results\New folder\step 2\30_70_FOV2\YOUR_GRADIENT_VALUES.csv"

# Output folder
OUTPUT_DIR = r"D:\New results\texture_specific_study\STEP4_ORIENTATION_INTENSITY_COUPLING"

# Fixed calibration
UM_PER_PX = 10.0 / 157.0

# Same quantitative OrientationJ coherence threshold used earlier
COHERENCE_MIN = 0.30

# Psi binning
PSI_BIN_DEG = 10.0

# Minimum matched points needed for correlation/model
MIN_POINTS = 8

DPI = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. COLUMN DETECTION HELPERS
# ============================================================

def pick_col(df, candidates, required=True):
    low = {str(c).strip().lower(): c for c in df.columns}

    for cand in candidates:
        key = cand.lower()
        if key in low:
            return low[key]

    # relaxed contains-match
    for c in df.columns:
        cl = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c

    if required:
        raise KeyError(
            f"Could not find any of columns {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) < MIN_POINTS:
        return np.nan, np.nan

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan

    try:
        if method == "pearson":
            r, p = pearsonr(x, y)
        else:
            r, p = spearmanr(x, y)

        return float(r), float(p)

    except Exception:
        return np.nan, np.nan


# ============================================================
# 3. LOAD ROI MASTER
# ============================================================

if not os.path.isfile(ROI_MASTER_CSV):
    raise FileNotFoundError(
        f"ROI master not found:\n{ROI_MASTER_CSV}"
    )

roi_master = pd.read_csv(ROI_MASTER_CSV)

# Supports the easy Step-1 format
roi_id_col = pick_col(roi_master, ["ROI_ID", "roi_id"])
texture_col = pick_col(roi_master, ["Texture", "texture_code"])
texture_name_col = pick_col(roi_master, ["Texture_name", "texture_name"])

x0_col = pick_col(roi_master, ["x_px", "bbox_x0_px", "bbox_x0"])
y0_col = pick_col(roi_master, ["y_px", "bbox_y0_px", "bbox_y0"])

crop_col = pick_col(roi_master, ["crop_path", "crop_rgb_png"])
mask_col = pick_col(roi_master, ["mask_path", "crop_mask_png", "full_mask_png"])

print("\nLoaded ROI master:")
print(
    roi_master[
        [roi_id_col, texture_col, texture_name_col]
    ]
)


# ============================================================
# 4. LOAD ORIGINAL ORIENTATIONJ VECTORS
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

oj["X_px"] = pd.to_numeric(oj[x_col], errors="coerce")
oj["Y_px"] = pd.to_numeric(oj[y_col], errors="coerce")
oj["Orientation_raw"] = pd.to_numeric(oj[ori_col], errors="coerce")

if coh_col is not None:
    oj["Coherence"] = pd.to_numeric(
        oj[coh_col],
        errors="coerce"
    )
else:
    oj["Coherence"] = 1.0

oj = oj.dropna(
    subset=["X_px", "Y_px", "Orientation_raw"]
).copy()

oj = oj[
    oj["Coherence"] >= COHERENCE_MIN
].copy()

# OrientationJ usually exports degrees.
# If values look radian-like, handle automatically.
max_abs_ori = np.nanmax(
    np.abs(oj["Orientation_raw"].to_numpy())
)

if max_abs_ori <= (2 * np.pi + 0.5):
    psi_rad = oj["Orientation_raw"].to_numpy(float)
else:
    psi_rad = np.deg2rad(
        oj["Orientation_raw"].to_numpy(float)
    )

# Nematic orientation: 0 <= Psi < pi
oj["Psi_rad"] = np.mod(psi_rad, np.pi)
oj["Psi_deg"] = np.degrees(oj["Psi_rad"])

print(
    f"\nOrientationJ retained vectors after C >= {COHERENCE_MIN}:",
    len(oj)
)


# ============================================================
# 5. LOAD WHOLE-FIELD GRADIENT DATA
# ============================================================

if not os.path.isfile(GRADIENT_CSV):
    raise FileNotFoundError(
        f"Gradient CSV not found:\n{GRADIENT_CSV}"
    )

grad = pd.read_csv(GRADIENT_CSV)
grad.columns = [str(c).strip() for c in grad.columns]

# detect gradient value
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

# detect coordinates
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
        "Gradient CSV needs x/y coordinates in pixels or micrometers.\n"
        f"Available columns: {list(grad.columns)}"
    )

grad["gradPsi"] = pd.to_numeric(
    grad[grad_val_col],
    errors="coerce"
)

grad = grad.dropna(
    subset=["GX_px", "GY_px", "gradPsi"]
).copy()

grad_tree = cKDTree(
    grad[["GX_px", "GY_px"]].to_numpy(float)
)

print(
    "Gradient points loaded:",
    len(grad)
)


# ============================================================
# 6. HARMONIC ORIENTATION–INTENSITY MODEL
# ============================================================

def fit_harmonic_intensity_model(psi_rad, intensity):

    psi_rad = np.asarray(
        psi_rad,
        dtype=float
    )

    intensity = np.asarray(
        intensity,
        dtype=float
    )

    keep = (
        np.isfinite(psi_rad)
        &
        np.isfinite(intensity)
    )

    psi_rad = psi_rad[keep]
    intensity = intensity[keep]

    if len(intensity) < MIN_POINTS:
        return {
            "harmonic_R2": np.nan,
            "B0": np.nan,
            "B2cos": np.nan,
            "B2sin": np.nan,
            "B4cos": np.nan,
            "B4sin": np.nan,
            "Amp2": np.nan,
            "Amp4": np.nan
        }

    X = np.column_stack([
        np.ones(len(psi_rad)),
        np.cos(2 * psi_rad),
        np.sin(2 * psi_rad),
        np.cos(4 * psi_rad),
        np.sin(4 * psi_rad)
    ])

    beta, _, _, _ = np.linalg.lstsq(
        X,
        intensity,
        rcond=None
    )

    pred = X @ beta

    ss_res = np.sum(
        (intensity - pred)**2
    )

    ss_tot = np.sum(
        (intensity - np.mean(intensity))**2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    amp2 = np.sqrt(
        beta[1]**2 + beta[2]**2
    )

    amp4 = np.sqrt(
        beta[3]**2 + beta[4]**2
    )

    return {
        "harmonic_R2": r2,
        "B0": beta[0],
        "B2cos": beta[1],
        "B2sin": beta[2],
        "B4cos": beta[3],
        "B4sin": beta[4],
        "Amp2": amp2,
        "Amp4": amp4
    }


# ============================================================
# 7. BINNED I(Psi)
# ============================================================

def make_binned_i_psi(df):

    edges = np.arange(
        0,
        180 + PSI_BIN_DEG,
        PSI_BIN_DEG
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    rows = []

    psi = df["Psi_deg"].to_numpy(float)
    intensity = df["Intensity"].to_numpy(float)

    for lo, hi, center in zip(
        edges[:-1],
        edges[1:],
        centers
    ):

        sel = (
            (psi >= lo)
            &
            (psi < hi)
        )

        vals = intensity[sel]

        rows.append({
            "Psi_center_deg": center,
            "N": len(vals),
            "Mean_intensity": np.mean(vals) if len(vals) else np.nan,
            "SD_intensity": (
                np.std(vals, ddof=1)
                if len(vals) > 1
                else np.nan
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. PROCESS EVERY ROI
# ============================================================

summary_rows = []

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
            float(roi[x0_col])
        )
    )

    y0 = int(
        round(
            float(roi[y0_col])
        )
    )

    crop_path = str(
        roi[crop_col]
    )

    mask_path = str(
        roi[mask_col]
    )

    print(
        "\nProcessing:",
        roi_id
    )

    # --------------------------------------------------------
    # load local crop and mask
    # --------------------------------------------------------

    crop_rgb = np.asarray(
        Image.open(crop_path).convert("RGB")
    )

    crop_gray = np.asarray(
        Image.open(crop_path).convert("L")
    ).astype(float)

    mask_img = np.asarray(
        Image.open(mask_path).convert("L")
    )

    # If mask path is full-image rather than crop-sized,
    # crop it to the crop dimensions using ROI origin.
    if mask_img.shape != crop_gray.shape:

        h_crop, w_crop = crop_gray.shape

        if (
            mask_img.shape[0] >= y0 + h_crop
            and
            mask_img.shape[1] >= x0 + w_crop
        ):
            mask_img = mask_img[
                y0:y0+h_crop,
                x0:x0+w_crop
            ]
        else:
            raise ValueError(
                f"Mask/crop size mismatch for {roi_id}\n"
                f"crop={crop_gray.shape}, mask={mask_img.shape}"
            )

    roi_mask = mask_img > 0

    h_crop, w_crop = crop_gray.shape

    # --------------------------------------------------------
    # OrientationJ vectors inside bounding crop
    # --------------------------------------------------------

    oj_roi = oj[
        (oj["X_px"] >= x0)
        &
        (oj["X_px"] < x0 + w_crop)
        &
        (oj["Y_px"] >= y0)
        &
        (oj["Y_px"] < y0 + h_crop)
    ].copy()

    if len(oj_roi) == 0:
        print("No OrientationJ vectors inside ROI.")
        continue

    # local image coordinates
    oj_roi["local_x"] = oj_roi["X_px"] - x0
    oj_roi["local_y"] = oj_roi["Y_px"] - y0

    xi = np.rint(
        oj_roi["local_x"].to_numpy()
    ).astype(int)

    yi = np.rint(
        oj_roi["local_y"].to_numpy()
    ).astype(int)

    valid_xy = (
        (xi >= 0)
        &
        (xi < w_crop)
        &
        (yi >= 0)
        &
        (yi < h_crop)
    )

    oj_roi = oj_roi.loc[
        valid_xy
    ].copy()

    xi = xi[valid_xy]
    yi = yi[valid_xy]

    # keep only actual ellipse/ROI mask
    in_roi = roi_mask[
        yi,
        xi
    ]

    oj_roi = oj_roi.loc[
        in_roi
    ].copy()

    xi = xi[in_roi]
    yi = yi[in_roi]

    if len(oj_roi) < MIN_POINTS:

        print(
            f"Only {len(oj_roi)} valid OrientationJ vectors. "
            f"Need at least {MIN_POINTS}."
        )

    # --------------------------------------------------------
    # sample POM intensity
    # --------------------------------------------------------

    oj_roi["Intensity"] = crop_gray[
        yi,
        xi
    ]

    # --------------------------------------------------------
    # sample whole-field gradPsi by nearest coordinate
    # --------------------------------------------------------

    full_xy = oj_roi[
        ["X_px", "Y_px"]
    ].to_numpy(float)

    dist_px, idx_grad = grad_tree.query(
        full_xy,
        k=1
    )

    oj_roi["gradPsi"] = grad.iloc[
        idx_grad
    ]["gradPsi"].to_numpy(float)

    oj_roi["grad_match_distance_px"] = dist_px

    # --------------------------------------------------------
    # Derived optical orientation term
    # --------------------------------------------------------

    oj_roi["sin2_2Psi"] = (
        np.sin(
            2 * oj_roi["Psi_rad"].to_numpy(float)
        )**2
    )

    # --------------------------------------------------------
    # correlations
    # --------------------------------------------------------

    pearson_sin, p_pearson_sin = safe_corr(
        oj_roi["Intensity"],
        oj_roi["sin2_2Psi"],
        method="pearson"
    )

    spearman_sin, p_spearman_sin = safe_corr(
        oj_roi["Intensity"],
        oj_roi["sin2_2Psi"],
        method="spearman"
    )

    pearson_grad, p_pearson_grad = safe_corr(
        oj_roi["Intensity"],
        oj_roi["gradPsi"],
        method="pearson"
    )

    spearman_grad, p_spearman_grad = safe_corr(
        oj_roi["Intensity"],
        oj_roi["gradPsi"],
        method="spearman"
    )

    # --------------------------------------------------------
    # harmonic model
    # --------------------------------------------------------

    harmonic = fit_harmonic_intensity_model(
        oj_roi["Psi_rad"],
        oj_roi["Intensity"]
    )

    # --------------------------------------------------------
    # output folder
    # --------------------------------------------------------

    roi_out = os.path.join(
        OUTPUT_DIR,
        roi_id
    )

    os.makedirs(
        roi_out,
        exist_ok=True
    )

    # save matched vector table
    matched_csv = os.path.join(
        roi_out,
        f"{roi_id}_ORIENTATION_INTENSITY_MATCHED.csv"
    )

    oj_roi.to_csv(
        matched_csv,
        index=False
    )

    # --------------------------------------------------------
    # binned I(Psi)
    # --------------------------------------------------------

    binned = make_binned_i_psi(
        oj_roi
    )

    binned_csv = os.path.join(
        roi_out,
        f"{roi_id}_BINNED_I_VS_PSI.csv"
    )

    binned.to_csv(
        binned_csv,
        index=False
    )

    # ========================================================
    # PLOT 1 — I(Psi)
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.2, 4.8)
    )

    ax.scatter(
        oj_roi["Psi_deg"],
        oj_roi["Intensity"],
        s=28,
        alpha=0.55,
        edgecolors="none"
    )

    ax.plot(
        binned["Psi_center_deg"],
        binned["Mean_intensity"],
        marker="o",
        linewidth=1.7
    )

    ax.set_xlim(
        0,
        180
    )

    ax.set_xlabel(
        r"Local texture orientation $\Psi$ (degrees)"
    )

    ax.set_ylabel(
        "POM intensity (a.u.)"
    )

    ax.set_title(
        f"{roi_id}: intensity vs orientation"
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
            f"{roi_id}_I_VS_PSI.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # ========================================================
    # PLOT 2 — I vs sin^2(2Psi)
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.2, 4.8)
    )

    ax.scatter(
        oj_roi["sin2_2Psi"],
        oj_roi["Intensity"],
        s=32,
        alpha=0.65
    )

    ax.set_xlabel(
        r"$\sin^2(2\Psi)$"
    )

    ax.set_ylabel(
        "POM intensity (a.u.)"
    )

    ax.set_title(
        f"{roi_id}\n"
        f"Spearman rho = {spearman_sin:.3f}"
        if np.isfinite(spearman_sin)
        else f"{roi_id}\nSpearman rho = NaN"
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
            f"{roi_id}_I_VS_SIN2_2PSI.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # ========================================================
    # PLOT 3 — I vs |gradPsi|
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.2, 4.8)
    )

    ax.scatter(
        oj_roi["gradPsi"],
        oj_roi["Intensity"],
        s=32,
        alpha=0.65
    )

    ax.set_xlabel(
        r"$|\nabla\Psi|$ (rad $\mu$m$^{-1}$)"
    )

    ax.set_ylabel(
        "POM intensity (a.u.)"
    )

    ax.set_title(
        f"{roi_id}\n"
        f"Spearman rho = {spearman_grad:.3f}"
        if np.isfinite(spearman_grad)
        else f"{roi_id}\nSpearman rho = NaN"
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
            f"{roi_id}_I_VS_GRADPSI.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # --------------------------------------------------------
    # summary row
    # --------------------------------------------------------

    summary_rows.append({

        "ROI_ID":
            roi_id,

        "Texture":
            texture,

        "Texture_name":
            texture_name,

        "N_vectors":
            len(oj_roi),

        "Mean_intensity":
            np.nanmean(
                oj_roi["Intensity"]
            ),

        "Mean_Psi_deg":
            np.nanmean(
                oj_roi["Psi_deg"]
            ),

        "Mean_gradPsi":
            np.nanmean(
                oj_roi["gradPsi"]
            ),

        "Mean_gradient_match_distance_px":
            np.nanmean(
                oj_roi["grad_match_distance_px"]
            ),

        "Pearson_I_vs_sin2_2Psi":
            pearson_sin,

        "Pearson_I_vs_sin2_2Psi_p":
            p_pearson_sin,

        "Spearman_I_vs_sin2_2Psi":
            spearman_sin,

        "Spearman_I_vs_sin2_2Psi_p":
            p_spearman_sin,

        "Pearson_I_vs_gradPsi":
            pearson_grad,

        "Pearson_I_vs_gradPsi_p":
            p_pearson_grad,

        "Spearman_I_vs_gradPsi":
            spearman_grad,

        "Spearman_I_vs_gradPsi_p":
            p_spearman_grad,

        "Harmonic_I_Psi_R2":
            harmonic["harmonic_R2"],

        "Orientation_Amp2":
            harmonic["Amp2"],

        "Orientation_Amp4":
            harmonic["Amp4"],

        "Orientation_Amp2_over_meanI":
            (
                harmonic["Amp2"]
                /
                np.nanmean(
                    oj_roi["Intensity"]
                )
                if np.nanmean(
                    oj_roi["Intensity"]
                ) != 0
                else np.nan
            ),

        "Orientation_Amp4_over_meanI":
            (
                harmonic["Amp4"]
                /
                np.nanmean(
                    oj_roi["Intensity"]
                )
                if np.nanmean(
                    oj_roi["Intensity"]
                ) != 0
                else np.nan
            )
    })


# ============================================================
# 9. SAVE PER-ROI SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "STEP4_ORIENTATION_INTENSITY_COUPLING_PER_ROI.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)

print(
    "\n======================================"
)

print(
    "STEP 4 PER-ROI SUMMARY"
)

print(
    "======================================"
)

print(
    summary
)


# ============================================================
# 10. TEXTURE-CLASS SUMMARY
# ============================================================

metrics = [
    "Spearman_I_vs_sin2_2Psi",
    "Spearman_I_vs_gradPsi",
    "Harmonic_I_Psi_R2",
    "Orientation_Amp2_over_meanI",
    "Orientation_Amp4_over_meanI",
    "Mean_gradPsi"
]

texture_rows = []

for texture in ["T1", "T2", "T3"]:

    sub = summary[
        summary["Texture"] == texture
    ]

    for metric in metrics:

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
    "STEP4_TEXTURE_CLASS_COUPLING_SUMMARY.csv"
)

texture_summary.to_csv(
    texture_summary_csv,
    index=False
)


# ============================================================
# 11. T1/T2/T3 COMPARISON PLOTS
# ============================================================

plot_metrics = [
    (
        "Spearman_I_vs_sin2_2Psi",
        r"Spearman $\rho$: $I$ vs $\sin^2(2\Psi)$"
    ),
    (
        "Spearman_I_vs_gradPsi",
        r"Spearman $\rho$: $I$ vs $|\nabla\Psi|$"
    ),
    (
        "Harmonic_I_Psi_R2",
        r"Orientation–intensity harmonic model $R^2$"
    )
]

for metric, ylabel in plot_metrics:

    fig, ax = plt.subplots(
        figsize=(6.3, 5.0)
    )

    for xpos, texture in enumerate(
        ["T1", "T2", "T3"]
    ):

        vals = summary.loc[
            summary["Texture"] == texture,
            metric
        ].dropna().to_numpy(float)

        if len(vals) == 0:
            continue

        jitter = (
            np.array([0.0])
            if len(vals) == 1
            else np.linspace(
                -0.07,
                0.07,
                len(vals)
            )
        )

        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=65,
            edgecolors="black",
            linewidths=0.7
        )

        mean = np.mean(vals)

        sd = (
            np.std(
                vals,
                ddof=1
            )
            if len(vals) > 1
            else 0
        )

        ax.errorbar(
            xpos,
            mean,
            yerr=sd,
            fmt="_",
            markersize=24,
            markeredgewidth=2,
            capsize=5
        )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        ["T1", "T2", "T3"]
    )

    ax.set_xlabel(
        "Texture class"
    )

    ax.set_ylabel(
        ylabel
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
            f"STEP4_COMPARE_{metric}.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print(
    "\n======================================"
)

print(
    "STEP 4 COMPLETE"
)

print(
    "======================================"
)

print(
    "\nPer-ROI summary:"
)

print(
    summary_csv
)

print(
    "\nTexture-class summary:"
)

print(
    texture_summary_csv
)

print(
    "\nInterpretation guide:"
)

print(
    "1. I vs sin^2(2Psi): tests whether optical intensity "
    "co-varies with local texture orientation."
)

print(
    "2. I vs |gradPsi|: tests whether bright/dark regions "
    "are associated with stronger orientation distortion."
)

print(
    "3. Harmonic model R2: measures how much of the local "
    "intensity variation is described by nematic-compatible "
    "2-fold/4-fold orientation harmonics."
)
