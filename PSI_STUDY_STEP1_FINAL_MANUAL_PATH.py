# ============================================================
# PSI STUDY — STEP 1
# SINGLE-FOV DENSE Ψ(x,y) MAP
# Manual path version
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# 1. USER SETTINGS — EDIT ONLY THIS SECTION
# ============================================================
INPUT_CSV = r"D:\\YOUR_PATH\\filtered_vectors_with_local_S2D.csv"
OUTPUT_FOLDER = r"D:\\YOUR_PATH\\PSI_STUDY_STEP1"
SAMPLE_NAME = "30:70 NaOH FOV1"

# ============================================================
# 2. FIXED SETTINGS FOR YOUR STUDY
# ============================================================
UM_PER_PX = 10.0 / 157.0
GRID_NX = 700
GRID_NY = 520
SMOOTH_SIGMA = 1.0
COHERENCY_THRESHOLD = 0.30
Q_AMPLITUDE_THRESHOLD = 0.15
FIG_DPI = 600
SCALEBAR_UM = 10.0

# ============================================================
# 3. HELPERS
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_col(df, aliases, required=False, label="column"):
    for c in aliases:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find {label}. Tried: {aliases}. Available: {list(df.columns)}"
        )
    return None


def to_boolean(series):
    if series.dtype == bool:
        return series.to_numpy()
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() > 0:
        return num.fillna(0).to_numpy() > 0
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "yes", "1", "valid"]).to_numpy()


def make_nematic_colormap():
    colors = [
        (1.00, 0.00, 0.00),
        (1.00, 0.50, 0.00),
        (1.00, 1.00, 0.00),
        (0.00, 1.00, 0.00),
        (0.00, 1.00, 1.00),
        (0.00, 0.00, 1.00),
        (0.55, 0.00, 1.00),
        (1.00, 0.00, 1.00),
        (1.00, 0.00, 0.45),
        (1.00, 0.00, 0.00),
    ]
    return LinearSegmentedColormap.from_list("nematic_psi", colors, N=256)

# ============================================================
# 4. LOAD + STANDARDIZE VECTOR DATA
# ============================================================
def load_orientation_data(path):
    df = pd.read_csv(path)
    print("Available columns:")
    print(list(df.columns))

    x_um_col = find_col(df, ["x_um", "X_um"])
    y_um_col = find_col(df, ["y_um", "Y_um"])

    if x_um_col is not None and y_um_col is not None:
        x_um = pd.to_numeric(df[x_um_col], errors="coerce").to_numpy(float)
        y_um = pd.to_numeric(df[y_um_col], errors="coerce").to_numpy(float)
        coordinate_source = "existing x_um/y_um"
    else:
        x_col = find_col(df, ["X", "x", "X_px", "x_px"], True, "X coordinate")
        y_col = find_col(df, ["Y", "y", "Y_px", "y_px"], True, "Y coordinate")
        x_px = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
        y_px = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
        x_um = x_px * UM_PER_PX
        y_um = y_px * UM_PER_PX
        coordinate_source = f"pixel coordinates converted using {UM_PER_PX:.9f} µm/px"

    rad_col = find_col(df, ["theta_rad", "Theta_rad", "orientation_rad", "Orientation_rad"])
    deg_col = find_col(df, ["Orientation_deg", "orientation_deg", "Orientation", "orientation", "theta_deg", "Theta_deg", "Angle", "angle"])

    if rad_col is not None:
        theta = pd.to_numeric(df[rad_col], errors="coerce").to_numpy(float)
        orientation_source = rad_col
    elif deg_col is not None:
        theta_deg = pd.to_numeric(df[deg_col], errors="coerce").to_numpy(float)
        theta = np.deg2rad(theta_deg)
        orientation_source = deg_col
    else:
        raise KeyError(f"No orientation-angle column found. Available: {list(df.columns)}")

    psi = np.mod(theta, np.pi)
    valid = np.isfinite(x_um) & np.isfinite(y_um) & np.isfinite(psi)

    valid_col = find_col(df, ["valid_texture", "valid", "is_valid", "texture_valid"])
    if valid_col is not None:
        valid &= to_boolean(df[valid_col])

    coherency_col = find_col(df, ["Coherency", "coherency", "Coherence", "coherence"])
    coherency_used = False
    if coherency_col is not None:
        coh = pd.to_numeric(df[coherency_col], errors="coerce").to_numpy(float)
        valid &= np.isfinite(coh) & (coh >= COHERENCY_THRESHOLD)
        coherency_used = True

    vectors = pd.DataFrame({
        "x_um": x_um[valid],
        "y_um": y_um[valid],
        "Psi_rad": psi[valid],
        "Psi_deg": np.rad2deg(psi[valid]),
    })

    if len(vectors) < 20:
        raise ValueError(f"Only {len(vectors)} valid vectors remain after filtering.")

    meta = {
        "n_input_rows": len(df),
        "n_valid_vectors": len(vectors),
        "coordinate_source": coordinate_source,
        "orientation_source": orientation_source,
        "coherency_filter_applied": coherency_used,
    }
    return vectors, meta

# ============================================================
# 5. NEMATIC-SAFE DENSE Ψ RECONSTRUCTION
# ============================================================
def reconstruct_dense_psi(vectors):
    tmp = vectors[["x_um", "y_um", "Psi_rad"]].copy()
    tmp["Qx"] = np.cos(2.0 * tmp["Psi_rad"])
    tmp["Qy"] = np.sin(2.0 * tmp["Psi_rad"])
    tmp = tmp.groupby(["x_um", "y_um"], as_index=False)[["Qx", "Qy"]].mean()

    x = tmp["x_um"].to_numpy(float)
    y = tmp["y_um"].to_numpy(float)
    qx = tmp["Qx"].to_numpy(float)
    qy = tmp["Qy"].to_numpy(float)

    xi = np.linspace(x.min(), x.max(), GRID_NX)
    yi = np.linspace(y.min(), y.max(), GRID_NY)
    Xg, Yg = np.meshgrid(xi, yi)

    Qx_grid = griddata((x, y), qx, (Xg, Yg), method="linear")
    Qy_grid = griddata((x, y), qy, (Xg, Yg), method="linear")

    try:
        tri = Delaunay(np.column_stack([x, y]))
        query = np.column_stack([Xg.ravel(), Yg.ravel()])
        hull_mask = (tri.find_simplex(query) >= 0).reshape(Xg.shape)
    except Exception as e:
        print("Warning: Delaunay hull failed; using finite interpolation mask.")
        print(e)
        hull_mask = np.isfinite(Qx_grid) & np.isfinite(Qy_grid)

    finite = np.isfinite(Qx_grid) & np.isfinite(Qy_grid) & hull_mask
    weights = finite.astype(float)
    qx0 = np.where(finite, Qx_grid, 0.0)
    qy0 = np.where(finite, Qy_grid, 0.0)

    smooth_w = gaussian_filter(weights, sigma=SMOOTH_SIGMA)
    smooth_qx = gaussian_filter(qx0, sigma=SMOOTH_SIGMA)
    smooth_qy = gaussian_filter(qy0, sigma=SMOOTH_SIGMA)

    with np.errstate(divide="ignore", invalid="ignore"):
        smooth_qx = smooth_qx / smooth_w
        smooth_qy = smooth_qy / smooth_w

    q_amp = np.sqrt(smooth_qx**2 + smooth_qy**2)
    psi_dense = 0.5 * np.arctan2(smooth_qy, smooth_qx)
    psi_dense = np.mod(psi_dense, np.pi)

    valid_dense = (
        hull_mask
        & np.isfinite(psi_dense)
        & np.isfinite(q_amp)
        & (q_amp >= Q_AMPLITUDE_THRESHOLD)
        & (smooth_w > 1e-8)
    )

    psi_dense[~valid_dense] = np.nan
    q_amp[~valid_dense] = np.nan
    return Xg, Yg, psi_dense, q_amp, valid_dense

# ============================================================
# 6. PLOT
# ============================================================
def add_scale_bar(ax, Xg, Yg):
    xmin, xmax = np.nanmin(Xg), np.nanmax(Xg)
    ymin, ymax = np.nanmin(Yg), np.nanmax(Yg)
    xr = xmax - xmin
    yr = ymax - ymin
    x0 = xmin + 0.07 * xr
    x1 = x0 + SCALEBAR_UM
    y0 = ymax - 0.08 * yr

    ax.plot([x0, x1], [y0, y0], color="white", linewidth=4, solid_capstyle="butt")
    ax.text((x0+x1)/2, y0 - 0.03*yr, f"{SCALEBAR_UM:g} µm",
            color="white", fontsize=10, fontweight="bold",
            ha="center", va="top")


def save_psi_plot(Xg, Yg, psi_dense):
    cmap = make_nematic_colormap().copy()
    cmap.set_bad("black")

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    im = ax.imshow(
        np.ma.masked_invalid(psi_dense),
        origin="upper",
        extent=[np.nanmin(Xg), np.nanmax(Xg), np.nanmax(Yg), np.nanmin(Yg)],
        cmap=cmap,
        vmin=0,
        vmax=np.pi,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_title(rf"{SAMPLE_NAME} — $\Psi(x,y)$", fontsize=14)
    ax.set_xlabel(r"X ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Y ($\mu$m)", fontsize=12)
    ax.tick_params(direction="in", top=True, right=True)
    add_scale_bar(ax, Xg, Yg)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Psi$", rotation=270, labelpad=18, fontsize=13)
    cbar.set_ticks([0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_FOLDER, "Psi_map.png")
    pdf_path = os.path.join(OUTPUT_FOLDER, "Psi_map.pdf")
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return png_path, pdf_path

# ============================================================
# 7. RUN STEP 1
# ============================================================
ensure_dir(OUTPUT_FOLDER)

if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}\nReplace INPUT_CSV with your real CSV path."
    )

vectors, meta = load_orientation_data(INPUT_CSV)
Xg, Yg, psi_dense, q_amp, valid_dense = reconstruct_dense_psi(vectors)
png_path, pdf_path = save_psi_plot(Xg, Yg, psi_dense)

vectors.to_csv(
    os.path.join(OUTPUT_FOLDER, "Psi_input_vectors_standardized.csv"),
    index=False,
)

pd.DataFrame({
    "x_um": Xg.ravel(),
    "y_um": Yg.ravel(),
    "Psi_rad": psi_dense.ravel(),
    "Psi_deg": np.rad2deg(psi_dense).ravel(),
    "Q_amplitude": q_amp.ravel(),
    "valid_dense_map": valid_dense.ravel(),
}).to_csv(
    os.path.join(OUTPUT_FOLDER, "Psi_dense_values.csv"),
    index=False,
)

psi_measured = vectors["Psi_rad"].to_numpy(float)
Qmean = np.mean(np.exp(1j * 2.0 * psi_measured))
S2D_global = np.abs(Qmean)
mean_axis_rad = np.mod(0.5 * np.angle(Qmean), np.pi)

pd.DataFrame([{
    "Sample": SAMPLE_NAME,
    "Input_file": INPUT_CSV,
    "n_input_rows": meta["n_input_rows"],
    "n_valid_vectors": meta["n_valid_vectors"],
    "coordinate_source": meta["coordinate_source"],
    "orientation_source": meta["orientation_source"],
    "coherency_filter_applied": meta["coherency_filter_applied"],
    "coherency_threshold": COHERENCY_THRESHOLD,
    "UM_PER_PX_if_needed": UM_PER_PX,
    "Global_S2D_from_original_vectors": S2D_global,
    "Mean_axis_Psi_deg": np.rad2deg(mean_axis_rad),
    "Dense_map_valid_fraction": np.mean(valid_dense),
}]).to_csv(
    os.path.join(OUTPUT_FOLDER, "Psi_step1_summary.csv"),
    index=False,
)

print("\nPSI STUDY — STEP 1 COMPLETE")
print("Saved in:", OUTPUT_FOLDER)
print("Important: quantitative statistics are based on original retained vectors, not dense interpolated pixels.")
