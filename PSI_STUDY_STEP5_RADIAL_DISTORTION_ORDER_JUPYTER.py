
# ============================================================
# PSI STUDY — STEP 5
# DEFECT-CENTERED RADIAL DISTORTION + ORDER RECOVERY
# JUPYTER / VS CODE FRIENDLY
#
# PURPOSE
# -------
# For reliable winding defects from PSI Step 4:
#   1) calculate radial |grad Psi|(r)
#   2) calculate radial local S2D(r) if local S2D exists
#   3) compare core region vs far-field region
#   4) save per-defect radial profiles
#   5) generate composition-level 30:70 vs 50:50 summaries
#
# INPUT PER FOV
# -------------
# A) original filtered orientation-vector CSV
#    e.g. filtered_vectors_with_local_S2D.csv / filtered_vectors.csv
#
# B) PSI Step 2 gradient CSV
#    e.g. *_STEP2_gradient_values.csv
#
# C) PSI Step 4 reliable-defects CSV
#    e.g. *_STEP4_reliable_defects_only.csv
#
# IMPORTANT
# ---------
# - |grad Psi| comes from the nematic-safe Step-2 calculation.
# - local S2D is taken from the ORIGINAL vector data when available.
# - If local S2D is absent, gradient analysis still runs.
# - This step complements your previous recovery-length analysis;
#   it does not replace the existing lambda_S result.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT — EDIT THESE PATHS
# ============================================================

DATASETS = [
    (
        "30_70_FOV1",
        r"D:\YOUR_PATH\30_70_FOV1\filtered_vectors_with_local_S2D.csv",
        r"D:\YOUR_PATH\STEP2\30_70_FOV1_STEP2_gradient_values.csv",
        r"D:\YOUR_PATH\STEP4\30_70_FOV1\30_70_FOV1_STEP4_reliable_defects_only.csv"
    ),
    (
        "30_70_FOV2",
        r"D:\YOUR_PATH\30_70_FOV2\filtered_vectors_with_local_S2D.csv",
        r"D:\YOUR_PATH\STEP2\30_70_FOV2_STEP2_gradient_values.csv",
        r"D:\YOUR_PATH\STEP4\30_70_FOV2\30_70_FOV2_STEP4_reliable_defects_only.csv"
    ),
    (
        "30_70_FOV3",
        r"D:\YOUR_PATH\30_70_FOV3\filtered_vectors_with_local_S2D.csv",
        r"D:\YOUR_PATH\STEP2\30_70_FOV3_STEP2_gradient_values.csv",
        r"D:\YOUR_PATH\STEP4\30_70_FOV3\30_70_FOV3_STEP4_reliable_defects_only.csv"
    ),
    (
        "30_70_FOV4",
        r"D:\YOUR_PATH\30_70_FOV4\filtered_vectors_with_local_S2D.csv",
        r"D:\YOUR_PATH\STEP2\30_70_FOV4_STEP2_gradient_values.csv",
        r"D:\YOUR_PATH\STEP4\30_70_FOV4\30_70_FOV4_STEP4_reliable_defects_only.csv"
    ),

    (
        "50_50_FOV1",
        r"D:\YOUR_PATH\50_50_FOV1\filtered_vectors.csv",
        r"D:\YOUR_PATH\STEP2\50_50_FOV1_STEP2_gradient_values.csv",
        r"D:\YOUR_PATH\STEP4\50_50_FOV1\50_50_FOV1_STEP4_reliable_defects_only.csv"
    ),
    (
        "50_50_FOV2",
        r"D:\YOUR_PATH\50_50_FOV2\filtered_vectors.csv",
        r"D:\YOUR_PATH\STEP2\50_50_FOV2_STEP2_gradient_values.csv",
        r"D:\YOUR_PATH\STEP4\50_50_FOV2\50_50_FOV2_STEP4_reliable_defects_only.csv"
    ),
    (
        "50_50_FOV3",
        r"D:\YOUR_PATH\50_50_FOV3\filtered_vectors.csv",
        r"D:\YOUR_PATH\STEP2\50_50_FOV3_STEP2_gradient_values.csv",
        r"D:\YOUR_PATH\STEP4\50_50_FOV3\50_50_FOV3_STEP4_reliable_defects_only.csv"
    ),
    (
        "50_50_FOV4",
        r"D:\YOUR_PATH\50_50_FOV4\filtered_vectors.csv",
        r"D:\YOUR_PATH\STEP2\50_50_FOV4_STEP2_gradient_values.csv",
        r"D:\YOUR_PATH\STEP4\50_50_FOV4\50_50_FOV4_STEP4_reliable_defects_only.csv"
    ),
]

OUTPUT_DIR = r"D:\YOUR_PATH\PSI_STUDY_STEP5"

# ============================================================
# 2. SETTINGS
# ============================================================

UM_PER_PX = 10.0 / 157.0

# Radial profile
RADIAL_BIN_UM = 0.5
MAX_RADIUS_UM = 8.0

# Core and far-field windows
CORE_MAX_UM = 2.0
FAR_MIN_UM = 5.0
FAR_MAX_UM = 8.0

# Minimum points in a radial bin
MIN_POINTS_PER_BIN = 5

DPI = 600
SHOW_PLOTS_IN_NOTEBOOK = False

# ============================================================
# 3. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_csv_robust(path):
    for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def find_col(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

# ============================================================
# 4. LOAD ORIGINAL VECTORS
# ============================================================

def load_vectors(path):
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]

    # coords
    x_um_col = find_col(df, ["x_um"])
    y_um_col = find_col(df, ["y_um"])

    if x_um_col is not None and y_um_col is not None:
        x = pd.to_numeric(df[x_um_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[y_um_col], errors="coerce").to_numpy(float)
    else:
        x_col = find_col(df, ["X", "x", "X_px", "x_px"])
        y_col = find_col(df, ["Y", "y", "Y_px", "y_px"])

        if x_col is None or y_col is None:
            raise KeyError("Could not find x/y coordinates in vector CSV.")

        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float) * UM_PER_PX
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float) * UM_PER_PX

    # local S2D aliases
    s_col = find_col(
        df,
        [
            "S_local",
            "Slocal",
            "local_S2D",
            "S2D_local",
            "local_order",
            "Local_S2D",
            "S_local_mean"
        ]
    )

    if s_col is not None:
        s_local = pd.to_numeric(df[s_col], errors="coerce").to_numpy(float)
    else:
        s_local = np.full(len(df), np.nan)

    valid = np.isfinite(x) & np.isfinite(y)

    out = pd.DataFrame({
        "x_um": x[valid],
        "y_um": y[valid],
        "S_local": s_local[valid]
    })

    return out, s_col

# ============================================================
# 5. LOAD STEP-2 GRADIENT MAP
# ============================================================

def load_gradient(path):
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["x_um", "y_um", "gradPsi_rad_per_um"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing {missing} in Step-2 gradient CSV:\n{path}\n"
            f"Columns: {list(df.columns)}"
        )

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "x_um",
            "y_um",
            "gradPsi_rad_per_um"
        ]
    ).copy()

    return df

# ============================================================
# 6. LOAD RELIABLE DEFECTS FROM STEP 4
# ============================================================

def load_reliable_defects(path):
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]

    if len(df) == 0:
        return df

    required = ["defect_id", "x_um", "y_um", "s_fit", "R2"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing {missing} in Step-4 reliable defect CSV:\n{path}\n"
            f"Columns: {list(df.columns)}"
        )

    for c in ["defect_id", "x_um", "y_um", "s_fit", "R2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "defect_id",
            "x_um",
            "y_um"
        ]
    ).copy()

    return df

# ============================================================
# 7. RADIAL BINNING
# ============================================================

def radial_profile(x, y, values, xc, yc):
    r = np.sqrt(
        (x - xc)**2
        + (y - yc)**2
    )

    valid = (
        np.isfinite(r)
        & np.isfinite(values)
        & (r <= MAX_RADIUS_UM)
    )

    r = r[valid]
    values = values[valid]

    edges = np.arange(
        0,
        MAX_RADIUS_UM + RADIAL_BIN_UM,
        RADIAL_BIN_UM
    )

    rows = []

    for i in range(len(edges)-1):
        r0 = edges[i]
        r1 = edges[i+1]

        use = (
            (r >= r0)
            & (r < r1)
        )

        vals = values[use]

        if len(vals) >= MIN_POINTS_PER_BIN:
            rows.append({
                "r_center_um": 0.5*(r0+r1),
                "r_min_um": r0,
                "r_max_um": r1,
                "N": len(vals),
                "Mean": np.mean(vals),
                "SD": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                "SEM": (
                    np.std(vals, ddof=1)/np.sqrt(len(vals))
                    if len(vals) > 1
                    else np.nan
                ),
                "Median": np.median(vals)
            })

    return pd.DataFrame(rows)

# ============================================================
# 8. CORE VS FAR-FIELD
# ============================================================

def region_stats(x, y, values, xc, yc):
    r = np.sqrt(
        (x-xc)**2
        + (y-yc)**2
    )

    core = (
        np.isfinite(values)
        & (r <= CORE_MAX_UM)
    )

    far = (
        np.isfinite(values)
        & (r >= FAR_MIN_UM)
        & (r <= FAR_MAX_UM)
    )

    core_vals = values[core]
    far_vals = values[far]

    return {
        "core_N": len(core_vals),
        "core_mean": np.mean(core_vals) if len(core_vals) else np.nan,
        "core_median": np.median(core_vals) if len(core_vals) else np.nan,
        "far_N": len(far_vals),
        "far_mean": np.mean(far_vals) if len(far_vals) else np.nan,
        "far_median": np.median(far_vals) if len(far_vals) else np.nan,
        "core_minus_far": (
            np.mean(core_vals) - np.mean(far_vals)
            if len(core_vals) and len(far_vals)
            else np.nan
        ),
        "core_over_far": (
            np.mean(core_vals) / np.mean(far_vals)
            if len(core_vals) and len(far_vals) and np.mean(far_vals) != 0
            else np.nan
        )
    }

# ============================================================
# 9. PER-DEFECT PLOT
# ============================================================

def plot_radial_profiles(
    label,
    defect_id,
    grad_profile,
    s_profile,
    output_path
):
    fig, ax1 = plt.subplots(
        figsize=(7.2, 5.0)
    )

    if len(grad_profile) > 0:
        ax1.plot(
            grad_profile["r_center_um"],
            grad_profile["Mean"],
            marker="o",
            linewidth=1.8,
            label=r"$|\nabla\Psi|$"
        )

        ax1.fill_between(
            grad_profile["r_center_um"],
            grad_profile["Mean"] - grad_profile["SEM"].fillna(0),
            grad_profile["Mean"] + grad_profile["SEM"].fillna(0),
            alpha=0.2
        )

    ax1.set_xlabel(
        r"Distance from defect center, $r$ ($\mu$m)"
    )

    ax1.set_ylabel(
        r"$|\nabla\Psi|$ (rad $\mu$m$^{-1}$)"
    )

    ax1.axvline(
        CORE_MAX_UM,
        linestyle="--",
        linewidth=1
    )

    ax1.axvspan(
        FAR_MIN_UM,
        FAR_MAX_UM,
        alpha=0.08
    )

    # local order on secondary y-axis if available
    if len(s_profile) > 0:
        ax2 = ax1.twinx()

        ax2.plot(
            s_profile["r_center_um"],
            s_profile["Mean"],
            marker="s",
            linewidth=1.5,
            label=r"$S_{\mathrm{local}}$"
        )

        ax2.set_ylabel(
            r"$S_{\mathrm{local}}$"
        )

        ax2.set_ylim(
            0,
            1
        )

    ax1.set_title(
        f"{label} — defect {int(defect_id)}"
    )

    ax1.tick_params(
        direction="in",
        top=True
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()

    plt.close(fig)

# ============================================================
# 10. ANALYZE ONE FOV
# ============================================================

def analyze_fov(
    label,
    vector_path,
    gradient_path,
    reliable_defect_path
):
    print("\n" + "="*80)
    print("PROCESSING:", label)

    for p in [vector_path, gradient_path, reliable_defect_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    vectors, s_col = load_vectors(
        vector_path
    )

    gradient = load_gradient(
        gradient_path
    )

    defects = load_reliable_defects(
        reliable_defect_path
    )

    fov_out = os.path.join(
        OUTPUT_DIR,
        label
    )

    ensure_dir(
        fov_out
    )

    if len(defects) == 0:
        print("No reliable Step-4 defects found.")

        return {
            "Label": label,
            "N_reliable_defects": 0,
            "Slocal_column_found": s_col,
            "Mean_core_grad": np.nan,
            "Mean_far_grad": np.nan,
            "Mean_core_minus_far_grad": np.nan,
            "Mean_core_over_far_grad": np.nan,
            "Mean_core_Slocal": np.nan,
            "Mean_far_Slocal": np.nan,
            "Mean_core_minus_far_Slocal": np.nan
        }

    all_summary = []

    for _, defect in defects.iterrows():

        defect_id = int(
            defect["defect_id"]
        )

        xc = float(
            defect["x_um"]
        )

        yc = float(
            defect["y_um"]
        )

        # ----------------------------
        # Gradient profile
        # ----------------------------

        gx = gradient["x_um"].to_numpy(float)
        gy = gradient["y_um"].to_numpy(float)
        gv = gradient["gradPsi_rad_per_um"].to_numpy(float)

        grad_profile = radial_profile(
            gx,
            gy,
            gv,
            xc,
            yc
        )

        grad_region = region_stats(
            gx,
            gy,
            gv,
            xc,
            yc
        )

        # ----------------------------
        # S_local profile
        # ----------------------------

        sx = vectors["x_um"].to_numpy(float)
        sy = vectors["y_um"].to_numpy(float)
        sv = vectors["S_local"].to_numpy(float)

        if np.isfinite(sv).sum() > 0:

            s_profile = radial_profile(
                sx,
                sy,
                sv,
                xc,
                yc
            )

            s_region = region_stats(
                sx,
                sy,
                sv,
                xc,
                yc
            )

        else:

            s_profile = pd.DataFrame()

            s_region = {
                "core_N": 0,
                "core_mean": np.nan,
                "core_median": np.nan,
                "far_N": 0,
                "far_mean": np.nan,
                "far_median": np.nan,
                "core_minus_far": np.nan,
                "core_over_far": np.nan
            }

        # ----------------------------
        # Save per-defect profiles
        # ----------------------------

        grad_profile.to_csv(
            os.path.join(
                fov_out,
                f"{label}_defect_{defect_id:03d}_gradPsi_radial_profile.csv"
            ),
            index=False
        )

        if len(s_profile) > 0:
            s_profile.to_csv(
                os.path.join(
                    fov_out,
                    f"{label}_defect_{defect_id:03d}_Slocal_radial_profile.csv"
                ),
                index=False
            )

        plot_radial_profiles(
            label,
            defect_id,
            grad_profile,
            s_profile,
            os.path.join(
                fov_out,
                f"{label}_defect_{defect_id:03d}_radial_grad_and_order.png"
            )
        )

        all_summary.append({
            "Label": label,
            "defect_id": defect_id,
            "x_um": xc,
            "y_um": yc,
            "s_fit": defect.get("s_fit", np.nan),
            "R2": defect.get("R2", np.nan),

            "grad_core_N": grad_region["core_N"],
            "grad_core_mean": grad_region["core_mean"],
            "grad_far_N": grad_region["far_N"],
            "grad_far_mean": grad_region["far_mean"],
            "grad_core_minus_far": grad_region["core_minus_far"],
            "grad_core_over_far": grad_region["core_over_far"],

            "Slocal_core_N": s_region["core_N"],
            "Slocal_core_mean": s_region["core_mean"],
            "Slocal_far_N": s_region["far_N"],
            "Slocal_far_mean": s_region["far_mean"],
            "Slocal_core_minus_far": s_region["core_minus_far"],
            "Slocal_core_over_far": s_region["core_over_far"],
        })

    defect_summary = pd.DataFrame(
        all_summary
    )

    defect_summary_path = os.path.join(
        fov_out,
        f"{label}_STEP5_defect_radial_summary.csv"
    )

    defect_summary.to_csv(
        defect_summary_path,
        index=False
    )

    return {
        "Label": label,
        "N_reliable_defects": len(defect_summary),
        "Slocal_column_found": s_col,

        "Mean_core_grad": defect_summary["grad_core_mean"].mean(),
        "Mean_far_grad": defect_summary["grad_far_mean"].mean(),
        "Mean_core_minus_far_grad": defect_summary["grad_core_minus_far"].mean(),
        "Mean_core_over_far_grad": defect_summary["grad_core_over_far"].mean(),

        "Mean_core_Slocal": defect_summary["Slocal_core_mean"].mean(),
        "Mean_far_Slocal": defect_summary["Slocal_far_mean"].mean(),
        "Mean_core_minus_far_Slocal": defect_summary["Slocal_core_minus_far"].mean(),

        "Defect_summary_CSV": defect_summary_path
    }

# ============================================================
# 11. RUN ALL 8
# ============================================================

ensure_dir(
    OUTPUT_DIR
)

master_rows = []

for (
    label,
    vector_path,
    gradient_path,
    reliable_defect_path
) in DATASETS:

    result = analyze_fov(
        label,
        vector_path,
        gradient_path,
        reliable_defect_path
    )

    master_rows.append(
        result
    )

master = pd.DataFrame(
    master_rows
)

master["Composition"] = master["Label"].apply(
    lambda s:
    "30:70"
    if str(s).startswith("30_70")
    else "50:50"
)

master_path = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP5_ALL_FOV_SUMMARY.csv"
)

master.to_csv(
    master_path,
    index=False
)

# ============================================================
# 12. COMPOSITION SUMMARY
# ============================================================

metrics = [
    "Mean_core_grad",
    "Mean_far_grad",
    "Mean_core_minus_far_grad",
    "Mean_core_over_far_grad",
    "Mean_core_Slocal",
    "Mean_far_Slocal",
    "Mean_core_minus_far_Slocal"
]

rows = []

for composition in ["30:70", "50:50"]:

    sub = master[
        master["Composition"] == composition
    ]

    for metric in metrics:

        vals = sub[
            metric
        ].dropna().to_numpy(float)

        rows.append({
            "Composition": composition,
            "Metric": metric,
            "N_FOV": len(vals),
            "Mean": np.mean(vals) if len(vals) else np.nan,
            "SD": (
                np.std(vals, ddof=1)
                if len(vals) > 1
                else np.nan
            ),
            "SEM": (
                np.std(vals, ddof=1)/np.sqrt(len(vals))
                if len(vals) > 1
                else np.nan
            )
        })

composition_summary = pd.DataFrame(
    rows
)

composition_summary_path = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP5_COMPOSITION_SUMMARY.csv"
)

composition_summary.to_csv(
    composition_summary_path,
    index=False
)

# ============================================================
# 13. 30:70 vs 50:50 CORE/FAR GRADIENT PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.7, 5.0)
)

x_positions = {
    ("30:70", "core"): 0.0,
    ("30:70", "far"): 0.35,
    ("50:50", "core"): 1.0,
    ("50:50", "far"): 1.35
}

for composition in ["30:70", "50:50"]:

    sub = master[
        master["Composition"] == composition
    ]

    for region, metric in [
        ("core", "Mean_core_grad"),
        ("far", "Mean_far_grad")
    ]:

        vals = sub[
            metric
        ].dropna().to_numpy(float)

        if len(vals) == 0:
            continue

        xpos = x_positions[
            (composition, region)
        ]

        jitter = np.linspace(
            -0.035,
            0.035,
            len(vals)
        )

        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=60,
            edgecolors="black",
            linewidths=0.7
        )

        mean = np.mean(vals)

        sd = (
            np.std(vals, ddof=1)
            if len(vals) > 1
            else 0
        )

        ax.errorbar(
            xpos,
            mean,
            yerr=sd,
            fmt="_",
            markersize=22,
            markeredgewidth=2,
            capsize=4
        )

ax.set_xticks(
    [0.175, 1.175]
)

ax.set_xticklabels(
    ["30:70", "50:50"]
)

ax.set_ylabel(
    r"Mean $|\nabla\Psi|$ (rad $\mu$m$^{-1}$)"
)

ax.set_xlabel(
    "CPC:glycerol composition"
)

ax.set_title(
    "Defect-core vs far-field orientation distortion"
)

ax.tick_params(
    direction="in",
    top=True,
    right=True
)

plt.tight_layout()

png = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP5_core_vs_far_gradient_30_70_vs_50_50.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP5_core_vs_far_gradient_30_70_vs_50_50.pdf"
)

fig.savefig(
    png,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    pdf,
    bbox_inches="tight"
)

if SHOW_PLOTS_IN_NOTEBOOK:
    plt.show()

plt.close(fig)

print("\n" + "="*80)
print("PSI STEP 5 COMPLETE")
print("="*80)
print("Master summary:")
print(master_path)
print("Composition summary:")
print(composition_summary_path)
