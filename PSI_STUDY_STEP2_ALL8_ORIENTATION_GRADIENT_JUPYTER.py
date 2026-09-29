
# ============================================================
# PSI STUDY — STEP 2
# ORIENTATION-GRADIENT / DISTORTION ANALYSIS FOR ALL 8 FOVs
# JUPYTER / VS CODE FRIENDLY
#
# INPUT:
#   The 8 dense CSV files created in PSI Step 1:
#   *_Psi_dense_values.csv
#
# OUTPUT per FOV:
#   1) |grad Psi| map
#   2) distortion proxy = 0.5 |grad Psi|^2 map
#   3) |grad Psi| histogram
#   4) numerical gradient CSV
#
# GROUP OUTPUTS:
#   5) STEP2_ALL_8_SUMMARY.csv
#   6) STEP2_COMPOSITION_SUMMARY.csv
#   7) mean-gradient raw-FOV comparison PNG/PDF
#
# IMPORTANT PHYSICS:
#   We do NOT differentiate Psi directly because Psi has nematic
#   0/pi symmetry and branch-cut problems.
#
#   Instead:
#       Qx = cos(2 Psi)
#       Qy = sin(2 Psi)
#
#   Then:
#       |grad Psi| =
#       0.5 * sqrt[
#           (dQx/dx)^2 + (dQx/dy)^2
#         + (dQy/dx)^2 + (dQy/dy)^2
#       ]
#
#   The quantity 0.5|grad Psi|^2 is only an
#   ORIENTATION-DISTORTION PROXY.
#   It is NOT a direct Frank elastic-energy density unless an
#   elastic constant K is independently known.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CELL 1 — USER INPUT
# Replace all 8 paths with your real Step-1 dense CSV files.
# ============================================================

FILES = [
    ("30_70_FOV1", r"D:\YOUR_PATH\30_70_FOV1_Psi_dense_values.csv"),
    ("30_70_FOV2", r"D:\YOUR_PATH\30_70_FOV2_Psi_dense_values.csv"),
    ("30_70_FOV3", r"D:\YOUR_PATH\30_70_FOV3_Psi_dense_values.csv"),
    ("30_70_FOV4", r"D:\YOUR_PATH\30_70_FOV4_Psi_dense_values.csv"),

    ("50_50_FOV1", r"D:\YOUR_PATH\50_50_FOV1_Psi_dense_values.csv"),
    ("50_50_FOV2", r"D:\YOUR_PATH\50_50_FOV2_Psi_dense_values.csv"),
    ("50_50_FOV3", r"D:\YOUR_PATH\50_50_FOV3_Psi_dense_values.csv"),
    ("50_50_FOV4", r"D:\YOUR_PATH\50_50_FOV4_Psi_dense_values.csv"),
]

OUTPUT_DIR = r"D:\YOUR_PATH\PSI_STUDY_STEP2"

# ------------------------------------------------------------
# Analysis settings
# ------------------------------------------------------------

# Exclude weak Q regions from quantitative gradient analysis.
# This removes places where interpolated nematic orientation is poorly defined.
USE_Q_AMPLITUDE_MASK = True
Q_AMPLITUDE_THRESHOLD = 0.15

# Remove this many pixels from each outer edge.
# This avoids artificial derivatives from rectangular interpolation boundaries.
EDGE_EXCLUDE_PIXELS = 3

# For plotting only: upper color limit = this percentile.
# Extreme isolated pixels will not dominate the colormap.
MAP_VMAX_PERCENTILE = 99.0

DPI = 600

# Display maps inside Jupyter as they are created.
SHOW_PLOTS_IN_NOTEBOOK = True


# ============================================================
# CELL 2 — LOAD DENSE STEP-1 MAP
# ============================================================

def load_dense_step1_csv(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["x_um", "y_um", "Psi_rad"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns {missing}\n"
            f"File: {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Make coordinates numeric
    df["x_um"] = pd.to_numeric(df["x_um"], errors="coerce")
    df["y_um"] = pd.to_numeric(df["y_um"], errors="coerce")
    df["Psi_rad"] = pd.to_numeric(df["Psi_rad"], errors="coerce")

    if "Q_amplitude" in df.columns:
        df["Q_amplitude"] = pd.to_numeric(
            df["Q_amplitude"],
            errors="coerce"
        )

    df = df.dropna(subset=["x_um", "y_um"])

    x = np.sort(df["x_um"].dropna().unique())
    y = np.sort(df["y_um"].dropna().unique())

    if len(x) < 5 or len(y) < 5:
        raise ValueError(
            f"Grid is too small in:\n{path}\n"
            f"Unique x = {len(x)}, unique y = {len(y)}"
        )

    # Pivot is safer than assuming row ordering
    psi_pivot = df.pivot_table(
        index="y_um",
        columns="x_um",
        values="Psi_rad",
        aggfunc="first"
    ).reindex(index=y, columns=x)

    psi = psi_pivot.to_numpy(dtype=float)

    qamp = None

    if "Q_amplitude" in df.columns:
        qamp_pivot = df.pivot_table(
            index="y_um",
            columns="x_um",
            values="Q_amplitude",
            aggfunc="first"
        ).reindex(index=y, columns=x)

        qamp = qamp_pivot.to_numpy(dtype=float)

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))

    if dx <= 0 or dy <= 0:
        raise ValueError(
            f"Invalid physical grid spacing in {path}: dx={dx}, dy={dy}"
        )

    return x, y, psi, qamp, dx, dy


# ============================================================
# CELL 3 — NEMATIC-SAFE GRADIENT
# ============================================================

def compute_nematic_gradient(psi, qamp, dx, dy):
    """
    Compute |grad Psi| from Q = exp(i 2 Psi), not from Psi directly.
    """

    valid = np.isfinite(psi)

    if USE_Q_AMPLITUDE_MASK and qamp is not None:
        valid &= (
            np.isfinite(qamp)
            & (qamp >= Q_AMPLITUDE_THRESHOLD)
        )

    # Q field
    qx = np.full_like(psi, np.nan, dtype=float)
    qy = np.full_like(psi, np.nan, dtype=float)

    qx[valid] = np.cos(2.0 * psi[valid])
    qy[valid] = np.sin(2.0 * psi[valid])

    # Because Step-1 dense maps are rectangular and normally finite,
    # most values will already be valid.
    # For derivative continuity, fill any isolated NaNs locally.
    def nearest_neighbor_fill_simple(a, iterations=12):
        out = a.copy()

        for _ in range(iterations):
            bad = ~np.isfinite(out)

            if not bad.any():
                break

            neighbors = np.stack(
                [
                    np.roll(out, 1, axis=0),
                    np.roll(out, -1, axis=0),
                    np.roll(out, 1, axis=1),
                    np.roll(out, -1, axis=1)
                ],
                axis=0
            )

            finite_count = np.sum(
                np.isfinite(neighbors),
                axis=0
            )

            summed = np.nansum(
                neighbors,
                axis=0
            )

            replacement = np.full_like(
                out,
                np.nan,
                dtype=float
            )

            good_count = finite_count > 0

            replacement[good_count] = (
                summed[good_count]
                / finite_count[good_count]
            )

            use = bad & np.isfinite(replacement)

            out[use] = replacement[use]

        return out

    qx_f = nearest_neighbor_fill_simple(qx)
    qy_f = nearest_neighbor_fill_simple(qy)

    if np.isnan(qx_f).all() or np.isnan(qy_f).all():
        raise RuntimeError(
            "Q-field could not be reconstructed after masking."
        )

    # If any NaN remains after local filling, use global finite mean only
    # for derivative calculation. Final quantitative map is re-masked.
    if np.isnan(qx_f).any():
        qx_f = np.where(
            np.isfinite(qx_f),
            qx_f,
            np.nanmean(qx_f)
        )

    if np.isnan(qy_f).any():
        qy_f = np.where(
            np.isfinite(qy_f),
            qy_f,
            np.nanmean(qy_f)
        )

    # np.gradient order: axis0 = y, axis1 = x
    dqx_dy, dqx_dx = np.gradient(
        qx_f,
        dy,
        dx,
        edge_order=2
    )

    dqy_dy, dqy_dx = np.gradient(
        qy_f,
        dy,
        dx,
        edge_order=2
    )

    grad_psi = 0.5 * np.sqrt(
        dqx_dx**2
        + dqx_dy**2
        + dqy_dx**2
        + dqy_dy**2
    )

    distortion_proxy = 0.5 * grad_psi**2

    # --------------------------------------------------------
    # Strict valid mask:
    # require current point + immediate 4-neighbours to be valid
    # so gradients across masked holes are not interpreted physically.
    # --------------------------------------------------------

    core = valid.copy()

    core[1:-1, 1:-1] &= (
        valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )

    # remove outer edge
    n = EDGE_EXCLUDE_PIXELS

    if n > 0:
        core[:n, :] = False
        core[-n:, :] = False
        core[:, :n] = False
        core[:, -n:] = False

    grad_psi[~core] = np.nan
    distortion_proxy[~core] = np.nan

    return grad_psi, distortion_proxy, core


# ============================================================
# CELL 4 — PLOTTING
# ============================================================

def robust_vmax(arr):
    values = arr[np.isfinite(arr)]

    if len(values) == 0:
        return None

    return float(
        np.percentile(
            values,
            MAP_VMAX_PERCENTILE
        )
    )


def plot_map(
    x,
    y,
    data,
    title,
    cbar_label,
    png_path,
    pdf_path
):
    vmax = robust_vmax(data)

    fig, ax = plt.subplots(
        figsize=(8.4, 6.2)
    )

    im = ax.imshow(
        np.ma.masked_invalid(data),
        origin="upper",
        extent=[
            x.min(),
            x.max(),
            y.max(),
            y.min()
        ],
        aspect="equal",
        interpolation="nearest",
        vmin=0,
        vmax=vmax
    )

    ax.set_title(
        title,
        fontsize=14
    )

    ax.set_xlabel(
        r"X ($\mu$m)",
        fontsize=12
    )

    ax.set_ylabel(
        r"Y ($\mu$m)",
        fontsize=12
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cb.set_label(
        cbar_label,
        fontsize=11
    )

    plt.tight_layout()

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()

    plt.close(fig)


def plot_histogram(
    values,
    title,
    png_path
):
    fig, ax = plt.subplots(
        figsize=(6.4, 4.8)
    )

    ax.hist(
        values,
        bins=60
    )

    ax.set_xlabel(
        r"$|\nabla\Psi|$ (rad $\mu$m$^{-1}$)"
    )

    ax.set_ylabel(
        "Pixel count"
    )

    ax.set_title(
        title
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()

    plt.close(fig)


# ============================================================
# CELL 5 — ANALYZE ONE FOV
# ============================================================

def analyze_one_fov(
    label,
    path
):
    print("\n" + "="*80)
    print("Processing:", label)
    print("Input:", path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    x, y, psi, qamp, dx, dy = load_dense_step1_csv(
        path
    )

    grad, distortion, valid = compute_nematic_gradient(
        psi,
        qamp,
        dx,
        dy
    )

    grad_values = grad[
        np.isfinite(grad)
    ]

    distortion_values = distortion[
        np.isfinite(distortion)
    ]

    if len(grad_values) == 0:
        raise RuntimeError(
            f"No valid gradient values for {label}"
        )

    pretty = (
        label
        .replace("30_70", "30:70")
        .replace("50_50", "50:50")
        .replace("_", " ")
    )

    grad_png = os.path.join(
        OUTPUT_DIR,
        f"{label}_gradPsi_map.png"
    )

    grad_pdf = os.path.join(
        OUTPUT_DIR,
        f"{label}_gradPsi_map.pdf"
    )

    distortion_png = os.path.join(
        OUTPUT_DIR,
        f"{label}_distortion_proxy_map.png"
    )

    distortion_pdf = os.path.join(
        OUTPUT_DIR,
        f"{label}_distortion_proxy_map.pdf"
    )

    hist_png = os.path.join(
        OUTPUT_DIR,
        f"{label}_gradPsi_histogram.png"
    )

    values_csv = os.path.join(
        OUTPUT_DIR,
        f"{label}_STEP2_gradient_values.csv"
    )

    # gradient map
    plot_map(
        x,
        y,
        grad,
        rf"{pretty} — $|\nabla\Psi|$",
        r"$|\nabla\Psi|$ (rad $\mu$m$^{-1}$)",
        grad_png,
        grad_pdf
    )

    # distortion proxy map
    plot_map(
        x,
        y,
        distortion,
        rf"{pretty} — orientation-distortion proxy",
        r"$\frac{1}{2}|\nabla\Psi|^2$ (rad$^2$ $\mu$m$^{-2}$)",
        distortion_png,
        distortion_pdf
    )

    # histogram
    plot_histogram(
        grad_values,
        rf"{pretty} — $|\nabla\Psi|$ distribution",
        hist_png
    )

    # numerical output
    Xg, Yg = np.meshgrid(
        x,
        y
    )

    pd.DataFrame({
        "x_um": Xg.ravel(),
        "y_um": Yg.ravel(),
        "gradPsi_rad_per_um": grad.ravel(),
        "distortion_proxy_half_grad2": distortion.ravel(),
        "valid_gradient": valid.ravel()
    }).to_csv(
        values_csv,
        index=False
    )

    # summary metrics
    result = {
        "Label": label,
        "Input_file": path,
        "dx_um": dx,
        "dy_um": dy,
        "N_valid_gradient_pixels": len(grad_values),

        "Mean_gradPsi_rad_per_um": np.mean(grad_values),
        "Median_gradPsi_rad_per_um": np.median(grad_values),
        "SD_gradPsi_rad_per_um": np.std(grad_values, ddof=1),

        "P90_gradPsi_rad_per_um": np.percentile(
            grad_values,
            90
        ),

        "P95_gradPsi_rad_per_um": np.percentile(
            grad_values,
            95
        ),

        "Mean_distortion_proxy": np.mean(
            distortion_values
        ),

        "Median_distortion_proxy": np.median(
            distortion_values
        ),

        "Gradient_map_PNG": grad_png,
        "Distortion_map_PNG": distortion_png,
        "Histogram_PNG": hist_png,
        "Values_CSV": values_csv
    }

    print("Mean |grad Psi| =", result["Mean_gradPsi_rad_per_um"])
    print("Median |grad Psi| =", result["Median_gradPsi_rad_per_um"])
    print("P90 |grad Psi| =", result["P90_gradPsi_rad_per_um"])
    print("P95 |grad Psi| =", result["P95_gradPsi_rad_per_um"])

    return result


# ============================================================
# CELL 6 — RUN ALL 8 FOVs
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

summary_rows = []

for label, path in FILES:

    result = analyze_one_fov(
        label,
        path
    )

    summary_rows.append(
        result
    )

summary = pd.DataFrame(
    summary_rows
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "STEP2_ALL_8_SUMMARY.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\nSaved master summary:")
print(summary_path)


# ============================================================
# CELL 7 — COMPOSITION-LEVEL SUMMARY
# ============================================================

summary["Composition"] = summary["Label"].apply(
    lambda s: "30:70" if s.startswith("30_70") else "50:50"
)

metrics = [
    "Mean_gradPsi_rad_per_um",
    "Median_gradPsi_rad_per_um",
    "P90_gradPsi_rad_per_um",
    "P95_gradPsi_rad_per_um",
    "Mean_distortion_proxy"
]

composition_rows = []

for composition in ["30:70", "50:50"]:

    sub = summary[
        summary["Composition"] == composition
    ]

    for metric in metrics:

        vals = sub[
            metric
        ].dropna().to_numpy(float)

        composition_rows.append({
            "Composition": composition,
            "Metric": metric,
            "N_FOV": len(vals),
            "Mean": np.mean(vals),
            "SD": (
                np.std(vals, ddof=1)
                if len(vals) > 1
                else np.nan
            ),
            "SEM": (
                np.std(vals, ddof=1)
                / np.sqrt(len(vals))
                if len(vals) > 1
                else np.nan
            )
        })

composition_summary = pd.DataFrame(
    composition_rows
)

composition_summary_path = os.path.join(
    OUTPUT_DIR,
    "STEP2_COMPOSITION_SUMMARY.csv"
)

composition_summary.to_csv(
    composition_summary_path,
    index=False
)

print("\nSaved composition summary:")
print(composition_summary_path)


# ============================================================
# CELL 8 — RAW-FOV COMPARISON PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.4, 5.0)
)

for xpos, composition in enumerate(
    ["30:70", "50:50"]
):

    vals = summary.loc[
        summary["Composition"] == composition,
        "Mean_gradPsi_rad_per_um"
    ].to_numpy(float)

    # deterministic small horizontal spacing
    jitter = np.linspace(
        -0.06,
        0.06,
        len(vals)
    )

    ax.scatter(
        np.full(len(vals), xpos) + jitter,
        vals,
        s=65,
        edgecolors="black",
        linewidths=0.7
    )

    mean = np.mean(
        vals
    )

    sd = np.std(
        vals,
        ddof=1
    )

    ax.errorbar(
        xpos,
        mean,
        yerr=sd,
        fmt="_",
        markersize=24,
        markeredgewidth=2,
        capsize=5,
        elinewidth=1.5
    )

ax.set_xticks(
    [0, 1]
)

ax.set_xticklabels(
    ["30:70", "50:50"]
)

ax.set_xlabel(
    "CPC:glycerol composition"
)

ax.set_ylabel(
    r"Mean $|\nabla\Psi|$ (rad $\mu$m$^{-1}$)"
)

ax.tick_params(
    direction="in",
    top=True,
    right=True
)

for spine in ax.spines.values():
    spine.set_linewidth(
        1.1
    )

plt.tight_layout()

comparison_png = os.path.join(
    OUTPUT_DIR,
    "STEP2_mean_gradPsi_30_70_vs_50_50.png"
)

comparison_pdf = os.path.join(
    OUTPUT_DIR,
    "STEP2_mean_gradPsi_30_70_vs_50_50.pdf"
)

fig.savefig(
    comparison_png,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    comparison_pdf,
    bbox_inches="tight"
)

if SHOW_PLOTS_IN_NOTEBOOK:
    plt.show()

plt.close(fig)

print("\nSTEP 2 COMPLETE")
print("Output folder:")
print(OUTPUT_DIR)
