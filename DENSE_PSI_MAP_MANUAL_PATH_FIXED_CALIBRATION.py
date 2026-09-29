
# ============================================================
# DENSE PUBLICATION-STYLE PSI MAP
# MANUAL PATH + FIXED STUDY CALIBRATION
#
# Calibration used in this study:
# 10 µm = 157 px  ->  UM_PER_PX = 10/157
#
# If CSV contains x_um/y_um, those are used directly.
# Otherwise X/Y are assumed to be pixels and converted automatically.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# ============================================================
# 1. MANUAL INPUT PATHS
# ============================================================

INPUT_FILES = [
    r"D:\New results\3070\step 3 orientation order correlation\step 3b\30_70_NaOH_FOV1\filtered_vectors_with_local_S2D.csv",
    # add remaining files manually:
    # r"D:\...\30_70_NaOH_FOV2\filtered_vectors_with_local_S2D.csv",
    # r"D:\...\50_50_NaOH_FOV1\filtered_vectors.csv",
]

OUTPUT_DIR = r"D:\New results\Psi_map_output"

# ============================================================
# 2. FIXED CALIBRATION FOR YOUR STUDY
# ============================================================

UM_PER_PX = 10.0 / 157.0   # = 0.0636942675 µm/px

# ============================================================
# 3. MAP SETTINGS
# ============================================================

DPI = 600
GRID_NX = 700
GRID_NY = 700
SMOOTH_SIGMA = 1.2
MIN_VALID_POINTS = 50
BACKGROUND_COLOR = "black"
FIGSIZE = (10, 8)

# ============================================================
# 4. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_col(df, aliases):
    for c in aliases:
        if c in df.columns:
            return c
    return None

def psi_cmap():
    colors = [
        (1.00, 0.00, 0.00),
        (1.00, 0.50, 0.00),
        (1.00, 1.00, 0.00),
        (0.00, 1.00, 0.00),
        (0.00, 1.00, 1.00),
        (0.00, 0.00, 1.00),
        (0.60, 0.00, 1.00),
        (1.00, 0.00, 1.00),
        (1.00, 0.00, 0.50),
        (1.00, 0.00, 0.00),
    ]
    return LinearSegmentedColormap.from_list("psi_map", colors, N=256)

def load_vectors(csv_path):
    df = pd.read_csv(csv_path)

    print("\nReading:", csv_path)
    print("Available columns:", list(df.columns))

    # coordinate detection
    x_um_col = find_col(df, ["x_um", "X_um"])
    y_um_col = find_col(df, ["y_um", "Y_um"])

    if x_um_col is not None and y_um_col is not None:
        x = pd.to_numeric(df[x_um_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[y_um_col], errors="coerce").to_numpy(float)
        print("Using existing x_um/y_um.")
    else:
        x_px_col = find_col(df, ["X", "x", "x_px", "X_px"])
        y_px_col = find_col(df, ["Y", "y", "y_px", "Y_px"])

        if x_px_col is None or y_px_col is None:
            raise KeyError(
                "Could not find coordinate columns.\n"
                f"Available columns: {list(df.columns)}"
            )

        x_px = pd.to_numeric(df[x_px_col], errors="coerce").to_numpy(float)
        y_px = pd.to_numeric(df[y_px_col], errors="coerce").to_numpy(float)

        x = x_px * UM_PER_PX
        y = y_px * UM_PER_PX

        print(f"Converted X/Y px -> µm using {UM_PER_PX:.9f} µm/px")

    # orientation
    rad_col = find_col(df, ["theta_rad", "Theta_rad", "orientation_rad"])
    deg_col = find_col(
        df,
        ["Orientation_deg", "orientation_deg", "Orientation",
         "orientation", "theta_deg", "angle_deg", "Angle"]
    )

    if rad_col is not None:
        theta = pd.to_numeric(df[rad_col], errors="coerce").to_numpy(float)
    elif deg_col is not None:
        theta_deg = pd.to_numeric(df[deg_col], errors="coerce").to_numpy(float)
        theta = np.deg2rad(theta_deg)
    else:
        raise KeyError(
            "Could not find an orientation column.\n"
            f"Available columns: {list(df.columns)}"
        )

    # optional valid_texture filtering
    valid = np.ones(len(df), dtype=bool)
    valid_col = find_col(df, ["valid_texture", "valid", "is_valid", "mask_valid"])

    if valid_col is not None:
        raw = df[valid_col]
        if raw.dtype == bool:
            valid &= raw.to_numpy()
        else:
            num = pd.to_numeric(raw, errors="coerce")
            valid &= num.fillna(0).to_numpy() > 0

    valid &= np.isfinite(x) & np.isfinite(y) & np.isfinite(theta)

    x = x[valid]
    y = y[valid]
    theta = theta[valid]

    if len(x) < MIN_VALID_POINTS:
        raise ValueError(f"Only {len(x)} valid vectors found.")

    # apolar nematic orientation: 0 <= Psi < pi
    theta = np.mod(theta, np.pi)

    return x, y, theta

def dense_psi(x, y, theta):
    xi = np.linspace(x.min(), x.max(), GRID_NX)
    yi = np.linspace(y.min(), y.max(), GRID_NY)
    Xg, Yg = np.meshgrid(xi, yi)

    # nematic-safe interpolation through doubled angle
    C = np.cos(2.0 * theta)
    S = np.sin(2.0 * theta)

    Cg = griddata((x, y), C, (Xg, Yg), method="linear")
    Sg = griddata((x, y), S, (Xg, Yg), method="linear")

    Cn = griddata((x, y), C, (Xg, Yg), method="nearest")
    Sn = griddata((x, y), S, (Xg, Yg), method="nearest")

    Cg = np.where(np.isfinite(Cg), Cg, Cn)
    Sg = np.where(np.isfinite(Sg), Sg, Sn)

    Cg = gaussian_filter(Cg, sigma=SMOOTH_SIGMA)
    Sg = gaussian_filter(Sg, sigma=SMOOTH_SIGMA)

    psi = 0.5 * np.arctan2(Sg, Cg)
    psi = np.mod(psi, np.pi)

    confidence = np.sqrt(Cg**2 + Sg**2)
    valid = confidence > 0.15

    return Xg, Yg, psi, valid

def add_scale_bar(ax, Xg, Yg, length_um=10):
    xmin, xmax = Xg.min(), Xg.max()
    ymin, ymax = Yg.min(), Yg.max()

    xr = xmax - xmin
    yr = ymax - ymin

    x0 = xmin + 0.06*xr
    y0 = ymin + 0.07*yr

    ax.plot(
        [x0, x0 + length_um],
        [y0, y0],
        color="white",
        lw=4,
        solid_capstyle="butt"
    )

    ax.text(
        x0 + length_um/2,
        y0 + 0.025*yr,
        f"{length_um} µm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

def make_map(csv_path, out_dir):
    x, y, theta = load_vectors(csv_path)
    Xg, Yg, psi, valid = dense_psi(x, y, theta)

    cmap = psi_cmap().copy()
    cmap.set_bad(BACKGROUND_COLOR)

    psi_plot = np.ma.array(psi, mask=~valid)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    im = ax.imshow(
        psi_plot,
        origin="lower",
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        cmap=cmap,
        vmin=0,
        vmax=np.pi,
        aspect="equal"
    )

    ax.set_title(r"$\Psi$ Map", fontsize=28, fontweight="bold")
    ax.set_xlabel(r"X ($\mu$m)", fontsize=16)
    ax.set_ylabel(r"Y ($\mu$m)", fontsize=16)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)

    ax.tick_params(direction="in", top=True, right=True, labelsize=12)

    add_scale_bar(ax, Xg, Yg, length_um=10)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Psi$", fontsize=20, rotation=270, labelpad=22)
    cb.set_ticks([0, np.pi/2, np.pi])
    cb.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
    cb.ax.tick_params(labelsize=15)

    plt.tight_layout()

    parent = os.path.basename(os.path.dirname(csv_path))
    base = os.path.splitext(os.path.basename(csv_path))[0]
    prefix = parent + "_" + base

    png = os.path.join(out_dir, prefix + "_dense_Psi_map.png")
    pdf = os.path.join(out_dir, prefix + "_dense_Psi_map.pdf")

    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # save interpolated numerical map
    pd.DataFrame({
        "x_um": Xg.ravel(),
        "y_um": Yg.ravel(),
        "Psi_rad": psi.ravel(),
        "Psi_deg": np.rad2deg(psi).ravel(),
        "valid_map": valid.ravel()
    }).to_csv(
        os.path.join(out_dir, prefix + "_dense_Psi_values.csv"),
        index=False
    )

    return png

def main():
    ensure_dir(OUTPUT_DIR)

    print(f"Study calibration: 10 µm / 157 px = {UM_PER_PX:.9f} µm/px")

    for i, path in enumerate(INPUT_FILES, 1):
        print("\n" + "="*70)
        print(f"{i}/{len(INPUT_FILES)}")
        print(path)

        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        make_map(path, OUTPUT_DIR)

    print("\nAll dense Psi maps completed.")
    print("Saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
