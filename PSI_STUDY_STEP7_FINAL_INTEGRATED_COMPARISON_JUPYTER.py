
# ============================================================
# PSI STUDY — STEP 7
# FINAL INTEGRATED 30:70 vs 50:50 COMPARISON
# JUPYTER / VS CODE FRIENDLY
#
# PURPOSE
# -------
# Merge the main outputs from PSI Steps 1–6 into one final
# per-FOV master table and create publication-ready comparison plots.
#
# INPUT SUMMARY FILES
# -------------------
# Step 1: PSI_STEP1_ALL_8_SUMMARY.csv
# Step 2: STEP2_ALL_8_SUMMARY.csv
# Step 4: PSI_STEP4_ALL_FOV_WINDING_SUMMARY.csv
# Step 5: PSI_STEP5_ALL_FOV_SUMMARY.csv
# Step 6: PSI_STEP6_ALL_FOV_INTENSITY_SUMMARY.csv
#
# NOTE:
# Step 3 is an intermediate azimuthal-data preparation step.
# Its key output is used by Step 4, so it does not need to be
# merged separately here.
#
# OUTPUT
# ------
# 1) PSI_STEP7_MASTER_PER_FOV.csv
# 2) PSI_STEP7_COMPOSITION_SUMMARY.csv
# 3) publication-ready comparison plots
# 4) one compact metrics table for manuscript drafting
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT — EDIT THESE PATHS
# ============================================================

STEP1_SUMMARY = r"D:\YOUR_PATH\STEP1\PSI_STEP1_ALL_8_SUMMARY.csv"
STEP2_SUMMARY = r"D:\YOUR_PATH\STEP2\STEP2_ALL_8_SUMMARY.csv"
STEP4_SUMMARY = r"D:\YOUR_PATH\STEP4\PSI_STEP4_ALL_FOV_WINDING_SUMMARY.csv"
STEP5_SUMMARY = r"D:\YOUR_PATH\STEP5\PSI_STEP5_ALL_FOV_SUMMARY.csv"
STEP6_SUMMARY = r"D:\YOUR_PATH\STEP6\PSI_STEP6_ALL_FOV_INTENSITY_SUMMARY.csv"

OUTPUT_DIR = r"D:\YOUR_PATH\PSI_STUDY_STEP7_FINAL"

DPI = 600
SHOW_PLOTS_IN_NOTEBOOK = True

# ============================================================
# 2. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def standardize_label(df):
    """
    Standardize the FOV identifier to one common column: Label.
    """
    possible = ["Label", "label", "FOV", "Sample"]

    found = None
    for c in possible:
        if c in df.columns:
            found = c
            break

    if found is None:
        raise KeyError(
            f"No Label/FOV/Sample column found.\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out["Label"] = out[found].astype(str)

    # Normalize some common styles
    out["Label"] = (
        out["Label"]
        .str.replace("30:70", "30_70", regex=False)
        .str.replace("50:50", "50_50", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    # if FOV column only says FOV1 etc., composition may already exist separately
    if not out["Label"].str.contains("30_70|50_50", regex=True).all():
        if "Composition" in out.columns:
            comp = (
                out["Composition"].astype(str)
                .str.replace("30:70", "30_70", regex=False)
                .str.replace("50:50", "50_50", regex=False)
            )
            out["Label"] = comp + "_" + out["Label"]

    return out

def composition_from_label(label):
    s = str(label)
    if s.startswith("30_70"):
        return "30:70"
    elif s.startswith("50_50"):
        return "50:50"
    return "Unknown"

def safe_numeric(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)

# ============================================================
# 3. LOAD ALL SUMMARIES
# ============================================================

ensure_dir(OUTPUT_DIR)

s1 = standardize_label(read_csv(STEP1_SUMMARY))
s2 = standardize_label(read_csv(STEP2_SUMMARY))
s4 = standardize_label(read_csv(STEP4_SUMMARY))
s5 = standardize_label(read_csv(STEP5_SUMMARY))
s6 = standardize_label(read_csv(STEP6_SUMMARY))

# ============================================================
# 4. SELECT ONLY RELEVANT COLUMNS FROM EACH STEP
# ============================================================

# Step 1
keep1 = ["Label"]
for c in [
    "Global_S2D",
    "Global_S2D_from_vectors",
    "Global_S2D_from_filtered_vectors",
    "Mean_axis_Psi_deg",
    "Mean_axis_Psi_deg",
]:
    if c in s1.columns and c not in keep1:
        keep1.append(c)

s1_small = s1[keep1].copy()

# unify Step-1 S2D name
for c in [
    "Global_S2D",
    "Global_S2D_from_vectors",
    "Global_S2D_from_filtered_vectors"
]:
    if c in s1_small.columns:
        s1_small["Psi_Global_S2D"] = pd.to_numeric(
            s1_small[c],
            errors="coerce"
        )
        break

if "Mean_axis_Psi_deg" in s1_small.columns:
    s1_small["Psi_Mean_axis_deg"] = pd.to_numeric(
        s1_small["Mean_axis_Psi_deg"],
        errors="coerce"
    )

s1_small = s1_small[
    [c for c in ["Label", "Psi_Global_S2D", "Psi_Mean_axis_deg"] if c in s1_small.columns]
]

# Step 2
cols2 = ["Label"]
map2 = {
    "Mean_gradPsi_rad_per_um": "Mean_gradPsi",
    "Median_gradPsi_rad_per_um": "Median_gradPsi",
    "P90_gradPsi_rad_per_um": "P90_gradPsi",
    "P95_gradPsi_rad_per_um": "P95_gradPsi",
    "Mean_distortion_proxy": "Mean_distortion_proxy",
}

s2_small = s2[["Label"] + [c for c in map2 if c in s2.columns]].copy()
s2_small = s2_small.rename(columns=map2)

# Step 4
map4 = {
    "N_candidates_fit_attempted": "N_winding_candidates",
    "N_reliable_winding_fits": "N_reliable_winding",
    "Fraction_reliable_winding_fits": "Fraction_reliable_winding",
    "Mean_R2_all_possible": "Mean_winding_R2",
    "Mean_abs_s_fit_reliable": "Mean_abs_s_fit_reliable",
    "Sign_agreement_fraction": "Sign_agreement_fraction",
}

s4_small = s4[["Label"] + [c for c in map4 if c in s4.columns]].copy()
s4_small = s4_small.rename(columns=map4)

# Step 5
map5 = {
    "N_reliable_defects": "N_step5_defects",
    "Mean_core_grad": "Core_gradPsi",
    "Mean_far_grad": "Far_gradPsi",
    "Mean_core_minus_far_grad": "Core_minus_far_gradPsi",
    "Mean_core_over_far_grad": "Core_over_far_gradPsi",
    "Mean_core_Slocal": "Core_Slocal",
    "Mean_far_Slocal": "Far_Slocal",
    "Mean_core_minus_far_Slocal": "Core_minus_far_Slocal",
}

s5_small = s5[["Label"] + [c for c in map5 if c in s5.columns]].copy()
s5_small = s5_small.rename(columns=map5)

# Step 6
map6 = {
    "N_reliable_defects": "N_step6_defects",
    "Mean_core_intensity": "Core_intensity",
    "Mean_far_intensity": "Far_intensity",
    "Mean_core_minus_far": "Core_minus_far_intensity",
    "Mean_core_over_far": "Core_over_far_intensity",
}

s6_small = s6[["Label"] + [c for c in map6 if c in s6.columns]].copy()
s6_small = s6_small.rename(columns=map6)

# ============================================================
# 5. MERGE ALL STEPS
# ============================================================

master = s1_small.copy()

for df in [s2_small, s4_small, s5_small, s6_small]:
    master = master.merge(
        df,
        on="Label",
        how="outer"
    )

master["Composition"] = master["Label"].apply(
    composition_from_label
)

# Sort in natural expected order
order = [
    "30_70_FOV1", "30_70_FOV2", "30_70_FOV3", "30_70_FOV4",
    "50_50_FOV1", "50_50_FOV2", "50_50_FOV3", "50_50_FOV4"
]

master["sort_order"] = master["Label"].apply(
    lambda x: order.index(x) if x in order else 999
)

master = master.sort_values(
    "sort_order"
).drop(
    columns="sort_order"
).reset_index(
    drop=True
)

master_path = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP7_MASTER_PER_FOV.csv"
)

master.to_csv(
    master_path,
    index=False
)

print("\nMASTER TABLE")
print(master)

# ============================================================
# 6. COMPOSITION SUMMARY
# ============================================================

metrics = [
    "Psi_Global_S2D",
    "Mean_gradPsi",
    "Median_gradPsi",
    "P90_gradPsi",
    "P95_gradPsi",
    "Mean_distortion_proxy",
    "Fraction_reliable_winding",
    "Mean_winding_R2",
    "Mean_abs_s_fit_reliable",
    "Sign_agreement_fraction",
    "Core_gradPsi",
    "Far_gradPsi",
    "Core_minus_far_gradPsi",
    "Core_over_far_gradPsi",
    "Core_Slocal",
    "Far_Slocal",
    "Core_minus_far_Slocal",
    "Core_intensity",
    "Far_intensity",
    "Core_minus_far_intensity",
    "Core_over_far_intensity",
]

summary_rows = []

for comp in ["30:70", "50:50"]:
    sub = master[master["Composition"] == comp]

    for metric in metrics:
        if metric not in sub.columns:
            continue

        vals = pd.to_numeric(
            sub[metric],
            errors="coerce"
        ).dropna().to_numpy(float)

        if len(vals) == 0:
            continue

        summary_rows.append({
            "Composition": comp,
            "Metric": metric,
            "N_FOV": len(vals),
            "Mean": np.mean(vals),
            "SD": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "SEM": (
                np.std(vals, ddof=1)/np.sqrt(len(vals))
                if len(vals) > 1 else np.nan
            ),
            "Median": np.median(vals),
        })

composition_summary = pd.DataFrame(
    summary_rows
)

composition_summary_path = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP7_COMPOSITION_SUMMARY.csv"
)

composition_summary.to_csv(
    composition_summary_path,
    index=False
)

# ============================================================
# 7. GENERIC RAW-FOV COMPARISON PLOT
# ============================================================

def raw_fov_plot(
    df,
    metric,
    ylabel,
    title,
    filename,
    ylim=None
):
    if metric not in df.columns:
        print("Skipping missing metric:", metric)
        return

    fig, ax = plt.subplots(
        figsize=(6.4, 5.0)
    )

    for xpos, comp in enumerate(["30:70", "50:50"]):
        vals = pd.to_numeric(
            df.loc[
                df["Composition"] == comp,
                metric
            ],
            errors="coerce"
        ).dropna().to_numpy(float)

        if len(vals) == 0:
            continue

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

        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0

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

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["30:70", "50:50"])
    ax.set_xlabel("CPC:glycerol composition")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    plt.tight_layout()

    png = os.path.join(
        OUTPUT_DIR,
        filename + ".png"
    )

    pdf = os.path.join(
        OUTPUT_DIR,
        filename + ".pdf"
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

# ============================================================
# 8. PUBLICATION PLOTS
# ============================================================

raw_fov_plot(
    master,
    "Mean_gradPsi",
    r"Mean $|\nabla\Psi|$ (rad $\mu$m$^{-1}$)",
    "Whole-field orientation distortion",
    "FIG_STEP7_mean_gradPsi"
)

raw_fov_plot(
    master,
    "Fraction_reliable_winding",
    "Fraction of reliable winding fits",
    "Defect winding validation",
    "FIG_STEP7_reliable_winding_fraction",
    ylim=(0, 1.05)
)

raw_fov_plot(
    master,
    "Core_minus_far_gradPsi",
    r"Core - far $|\nabla\Psi|$ (rad $\mu$m$^{-1}$)",
    "Defect-localized orientation distortion",
    "FIG_STEP7_core_minus_far_gradPsi"
)

if "Core_minus_far_Slocal" in master.columns:
    raw_fov_plot(
        master,
        "Core_minus_far_Slocal",
        r"Core - far $S_{\mathrm{local}}$",
        "Defect-associated local-order difference",
        "FIG_STEP7_core_minus_far_Slocal"
    )

if "Core_minus_far_intensity" in master.columns:
    raw_fov_plot(
        master,
        "Core_minus_far_intensity",
        "Core - far POM intensity (a.u.)",
        "Defect-centered optical intensity difference",
        "FIG_STEP7_core_minus_far_intensity"
    )

# ============================================================
# 9. CORE VS FAR GRADIENT PLOT
# ============================================================

if (
    "Core_gradPsi" in master.columns
    and "Far_gradPsi" in master.columns
):

    fig, ax = plt.subplots(
        figsize=(6.8, 5.0)
    )

    xmap = {
        ("30:70", "core"): 0.0,
        ("30:70", "far"): 0.30,
        ("50:50", "core"): 1.0,
        ("50:50", "far"): 1.30
    }

    for comp in ["30:70", "50:50"]:

        for region, metric in [
            ("core", "Core_gradPsi"),
            ("far", "Far_gradPsi")
        ]:

            vals = pd.to_numeric(
                master.loc[
                    master["Composition"] == comp,
                    metric
                ],
                errors="coerce"
            ).dropna().to_numpy(float)

            if len(vals) == 0:
                continue

            xpos = xmap[(comp, region)]

            jitter = np.linspace(
                -0.03,
                0.03,
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
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0

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
        [0.15, 1.15]
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
        "Defect core vs far-field distortion"
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
            "FIG_STEP7_core_vs_far_gradPsi.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "FIG_STEP7_core_vs_far_gradPsi.pdf"
        ),
        bbox_inches="tight"
    )

    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()

    plt.close(fig)

# ============================================================
# 10. COMPACT MANUSCRIPT METRICS TABLE
# ============================================================

priority_metrics = [
    "Psi_Global_S2D",
    "Mean_gradPsi",
    "Fraction_reliable_winding",
    "Mean_winding_R2",
    "Mean_abs_s_fit_reliable",
    "Core_gradPsi",
    "Far_gradPsi",
    "Core_minus_far_gradPsi",
    "Core_Slocal",
    "Far_Slocal",
    "Core_minus_far_Slocal",
    "Core_intensity",
    "Far_intensity",
    "Core_minus_far_intensity",
]

compact_rows = []

for metric in priority_metrics:

    if metric not in master.columns:
        continue

    row = {
        "Metric": metric
    }

    for comp in ["30:70", "50:50"]:

        vals = pd.to_numeric(
            master.loc[
                master["Composition"] == comp,
                metric
            ],
            errors="coerce"
        ).dropna().to_numpy(float)

        if len(vals) > 0:
            mean = np.mean(vals)
            sd = np.std(vals, ddof=1) if len(vals) > 1 else np.nan

            row[
                comp + "_mean"
            ] = mean

            row[
                comp + "_SD"
            ] = sd

            row[
                comp + "_N_FOV"
            ] = len(vals)

        else:
            row[
                comp + "_mean"
            ] = np.nan

            row[
                comp + "_SD"
            ] = np.nan

            row[
                comp + "_N_FOV"
            ] = 0

    compact_rows.append(
        row
    )

compact = pd.DataFrame(
    compact_rows
)

compact_path = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP7_COMPACT_MANUSCRIPT_METRICS.csv"
)

compact.to_csv(
    compact_path,
    index=False
)

print("\n" + "="*80)
print("PSI STEP 7 COMPLETE")
print("="*80)
print("Master per-FOV table:")
print(master_path)
print("Composition summary:")
print(composition_summary_path)
print("Compact manuscript metrics:")
print(compact_path)
