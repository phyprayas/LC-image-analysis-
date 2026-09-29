
# ============================================================
# STEP 10 — FINAL 30:70 vs 50:50 COMPARISON
# DIRECT STEP-FOLDER VERSION
#
# This version uses the exact folder structure shown by the user.
# It does NOT scan the whole parent folder blindly.
# It looks only inside the known Step 3/4/5/6/7/9 folders.
#
# YOU ONLY NEED TO EDIT:
#   ROOT_30_70
#   ROOT_50_50
#
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. EDIT ONLY THESE TWO ROOT PATHS
# ============================================================

ROOT_30_70 = r"D:\New results\3070"
ROOT_50_50 = r"D:\New results\5070"

OUTPUT_FOLDER = r"D:\New results\FINAL_30_70_vs_50_50_COMPARISON"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. STEP FOLDER DISCOVERY BY STEP NUMBER
# ============================================================
#
# Instead of requiring the exact spelling of each folder,
# this version finds folders by the beginning of their names:
#
#   step 3 ...
#   step 4 ...
#   step 5 ...
#   step 6 ...
#   step 7 ...
#   step 9 ...
#
# This avoids errors from typos such as:
#   "orineitation", "orientation", "marged", "merged", etc.
#
# ============================================================

STEP_NUMBERS = [3, 4, 5, 6, 7, 9]


def normalize_name(s):
    return (
        str(s)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_step_folder(root, step_number):
    if not os.path.exists(root):
        raise FileNotFoundError(
            f"Root folder not found:\n{root}"
        )

    entries = []

    for name in os.listdir(root):
        full = os.path.join(root, name)

        if os.path.isdir(full):
            entries.append((name, full))

    # strict prefix patterns first
    prefixes = [
        f"step {step_number} ",
        f"step{step_number} ",
        f"step {step_number}",
        f"step{step_number}"
    ]

    matches = []

    for name, full in entries:
        norm = normalize_name(name)

        if any(norm.startswith(p) for p in prefixes):
            matches.append((name, full))

    if len(matches) == 1:
        print(f"FOUND Step {step_number} folder: {matches[0][1]}")
        return matches[0][1]

    if len(matches) > 1:
        print(f"\nMultiple Step {step_number} folders found:")
        for name, full in matches:
            print("  ", full)

        # choose shortest name as safest default
        matches = sorted(
            matches,
            key=lambda x: len(x[0])
        )

        print(
            f"Using: {matches[0][1]}"
        )

        return matches[0][1]

    # fallback: any folder containing both "step" and the number
    loose = []

    for name, full in entries:
        norm = normalize_name(name)

        if (
            "step" in norm
            and str(step_number) in norm
        ):
            loose.append((name, full))

    if len(loose) == 1:
        print(
            f"FOUND Step {step_number} folder by loose match: "
            f"{loose[0][1]}"
        )
        return loose[0][1]

    available = "\n".join(
        f"  {name}"
        for name, _ in entries
    )

    raise FileNotFoundError(
        f"Could not identify Step {step_number} folder under:\n"
        f"{root}\n\n"
        f"Available folders:\n{available}"
    )
# ============================================================
# 3. EXPECTED COLUMN SIGNATURES
# ============================================================

STEP_SIGNATURES = {
    3: [
        ["global_S2D", "Global_S2D", "S2D_global", "global_s2d"],
        ["mean_local_S2D", "Mean_local_S2D", "local_S2D_mean", "mean_local_s2d"],
        ["xi_um", "correlation_length_um", "C2_xi_um", "xi_micrometers"],
    ],
    4: [
        ["merged_defects_total", "merged_total", "n_merged_defects", "total_merged_defects"],
        ["density_total_per_um2", "defect_density_per_um2", "defect_density_um2",
         "defect_density_um_minus2", "rho_d"],
        ["mean_NN_all_um", "mean_NN_um", "mean_nearest_neighbor_um",
         "mean_defect_NN_um", "mean_NN_distance_um"],
    ],
    5: [
        ["rho_Slocal_nearest_defect_distance",
         "rho_Slocal_defect_distance",
         "rho_Slocal_distance",
         "rho_Slocal_vs_nearest_defect_distance"],
        ["rho_Slocal_local_defect_density",
         "rho_Slocal_defect_density",
         "rho_Slocal_density",
         "rho_Slocal_vs_local_defect_density"],
    ],
    6: [
        ["observed_unlike_sign_NN_fraction",
         "unlike_sign_NN_fraction",
         "observed_unlike_fraction"],
        ["shuffle_null_mean", "null_mean", "shuffle_mean"],
    ],
    7: [
        ["lambda_S_um", "lambdaS_um", "lambda_um", "recovery_length_um"],
    ],
    9: [
        ["normalized_orientation_entropy",
         "normalized_entropy",
         "orientation_entropy_normalized"],
        ["angular_dispersion_deg",
         "angular_dispersion",
         "sigma_theta_deg"],
        ["n_domains", "number_orientation_domains", "domain_count"],
    ],
}

# ============================================================
# 4. FILE SEARCH HELPERS
# ============================================================

def all_csv_files(folder):
    found = []
    for base, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                found.append(os.path.join(base, f))
    return found


def content_score(path, step):
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return -999, []

    cols = set(df.columns)
    score = 0
    matched = []

    if "FOV" in cols:
        score += 20

    for alias_group in STEP_SIGNATURES[step]:
        hit = next((a for a in alias_group if a in cols), None)
        if hit is not None:
            score += 30
            matched.append(hit)

    name = os.path.basename(path).lower()

    if "all_4fov_summary" in name:
        score += 20
    elif "summary" in name:
        score += 8

    if "mean_sd_sem" in name:
        score -= 20
    if "pooled" in name:
        score -= 10

    return score, matched


def select_step_file(root, step):

    folder = find_step_folder(
        root,
        step
    )

    files = all_csv_files(folder)

    if not files:
        raise FileNotFoundError(
            f"No CSV files found inside Step {step} folder:\n{folder}"
        )

    ranked = []

    for p in files:
        score, matched = content_score(p, step)
        if score > 0:
            ranked.append((score, p, matched))

    if not ranked:
        raise FileNotFoundError(
            f"No CSV in Step {step} folder matched expected metrics:\n{folder}"
        )

    ranked.sort(key=lambda x: x[0], reverse=True)

    print(f"\nStep {step} candidates in {folder}:")
    for score, p, matched in ranked[:10]:
        print(f"  score={score:>3} matched={matched} -> {p}")

    for score, p, matched in ranked:
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue

        if "FOV" in df.columns:
            print(f"SELECTED Step {step}: {p}")
            return p

    raise FileNotFoundError(
        f"Matching Step {step} CSVs found, but none had an FOV column."
    )


def collect_step_files(root):
    out = {}

    for step in STEP_NUMBERS:
        out[step] = select_step_file(
            root,
            step
        )

    return out

# ============================================================
# 5. COLUMN HELPERS
# ============================================================

def normalize_fov_name(series):
    s = series.astype(str)
    extracted = s.str.extract(r"(FOV\s*[_-]?\s*\d+)", expand=False)
    extracted = extracted.fillna(s)
    return (
        extracted
        .str.replace(r"[\s_-]+", "", regex=True)
        .str.upper()
    )


def pick_column(df, aliases, required=True, label=""):
    for col in aliases:
        if col in df.columns:
            return df[col]

    if required:
        raise KeyError(
            f"\n{label}\n"
            f"None of these columns found:\n{aliases}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return pd.Series(np.nan, index=df.index)


def make_metric_frame(df, mapping, label):
    if "FOV" not in df.columns:
        raise ValueError(
            f"{label}: FOV column missing.\n"
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "FOV_key": normalize_fov_name(df["FOV"])
    })

    for name, spec in mapping.items():
        out[name] = pick_column(
            df,
            spec["aliases"],
            required=spec.get("required", True),
            label=f"{label} -> {name}"
        )

    return out

# ============================================================
# 6. LOAD COMPOSITION
# ============================================================

def load_composition(composition, step_files):

    s3 = pd.read_csv(step_files[3])
    s4 = pd.read_csv(step_files[4])
    s5 = pd.read_csv(step_files[5])
    s6 = pd.read_csv(step_files[6])
    s7 = pd.read_csv(step_files[7])
    s9 = pd.read_csv(step_files[9])

    a = make_metric_frame(
        s3,
        {
            "global_S2D": {
                "aliases": ["global_S2D", "Global_S2D", "S2D_global", "global_s2d"]
            },
            "mean_local_S2D": {
                "aliases": ["mean_local_S2D", "Mean_local_S2D", "local_S2D_mean", "mean_local_s2d"]
            },
            "xi_um": {
                "aliases": ["xi_um", "correlation_length_um", "C2_xi_um", "xi_micrometers"]
            },
            "corr_fit_R2": {
                "aliases": ["corr_fit_R2", "C2_fit_R2", "fit_R2", "correlation_fit_R2"],
                "required": False
            },
        },
        f"{composition} Step 3"
    )

    b = make_metric_frame(
        s4,
        {
            "merged_defects_total": {
                "aliases": ["merged_defects_total", "merged_total", "n_merged_defects", "total_merged_defects"]
            },
            "density_total_per_um2": {
                "aliases": ["density_total_per_um2", "defect_density_per_um2",
                            "defect_density_um2", "defect_density_um_minus2", "rho_d"]
            },
            "mean_NN_all_um": {
                "aliases": ["mean_NN_all_um", "mean_NN_um", "mean_nearest_neighbor_um",
                            "mean_defect_NN_um", "mean_NN_distance_um"]
            },
        },
        f"{composition} Step 4"
    )

    c = make_metric_frame(
        s5,
        {
            "rho_Slocal_nearest_defect_distance": {
                "aliases": ["rho_Slocal_nearest_defect_distance",
                            "rho_Slocal_defect_distance",
                            "rho_Slocal_distance",
                            "rho_Slocal_vs_nearest_defect_distance"]
            },
            "rho_Slocal_local_defect_density": {
                "aliases": ["rho_Slocal_local_defect_density",
                            "rho_Slocal_defect_density",
                            "rho_Slocal_density",
                            "rho_Slocal_vs_local_defect_density"]
            },
        },
        f"{composition} Step 5"
    )

    d = make_metric_frame(
        s6,
        {
            "observed_unlike_sign_NN_fraction": {
                "aliases": ["observed_unlike_sign_NN_fraction",
                            "unlike_sign_NN_fraction", "observed_unlike_fraction"]
            },
            "shuffle_null_mean": {
                "aliases": ["shuffle_null_mean", "null_mean", "shuffle_mean"]
            },
            "empirical_one_sided_p": {
                "aliases": ["empirical_one_sided_p", "empirical_upper_p",
                            "p_one_sided_observed_ge_null", "p_value"],
                "required": False
            },
        },
        f"{composition} Step 6"
    )

    e = make_metric_frame(
        s7,
        {
            "lambda_S_um": {
                "aliases": ["lambda_S_um", "lambdaS_um", "lambda_um", "recovery_length_um"]
            },
            "R2_step7": {
                "aliases": ["R2", "fit_R2", "recovery_fit_R2"],
                "required": False
            },
            "S_inf": {
                "aliases": ["S_inf", "Sinf", "S_infinity"],
                "required": False
            },
            "reliable_fit": {
                "aliases": ["reliable_fit", "reliable", "fit_reliable"],
                "required": False
            },
        },
        f"{composition} Step 7"
    )

    f = make_metric_frame(
        s9,
        {
            "normalized_orientation_entropy": {
                "aliases": ["normalized_orientation_entropy", "normalized_entropy",
                            "orientation_entropy_normalized"]
            },
            "angular_dispersion_deg": {
                "aliases": ["angular_dispersion_deg", "angular_dispersion", "sigma_theta_deg"]
            },
            "n_domains": {
                "aliases": ["n_domains", "number_orientation_domains", "domain_count"]
            },
            "mean_domain_area_um2": {
                "aliases": ["mean_domain_area_um2", "mean_orientation_domain_area_um2"]
            },
            "median_domain_area_um2": {
                "aliases": ["median_domain_area_um2", "median_orientation_domain_area_um2"],
                "required": False
            },
            "largest_domain_area_um2": {
                "aliases": ["largest_domain_area_um2", "max_domain_area_um2"]
            },
            "area_weighted_mean_domain_area_um2": {
                "aliases": ["area_weighted_mean_domain_area_um2", "weighted_mean_domain_area_um2"]
            },
            "fraction_valid_grid_in_retained_domains": {
                "aliases": ["fraction_valid_grid_in_retained_domains",
                            "fraction_grid_in_domains", "retained_domain_fraction"]
            },
        },
        f"{composition} Step 9"
    )

    merged = (
        a.merge(b, on="FOV_key", how="outer")
         .merge(c, on="FOV_key", how="outer")
         .merge(d, on="FOV_key", how="outer")
         .merge(e, on="FOV_key", how="outer")
         .merge(f, on="FOV_key", how="outer")
    )

    merged["Composition"] = composition
    return merged

# ============================================================
# 7. SELECT FILES
# ============================================================

FILES_30_70 = collect_step_files(
    ROOT_30_70
)

FILES_50_50 = collect_step_files(
    ROOT_50_50
)

selected = []

for comp, files in [
    ("30:70", FILES_30_70),
    ("50:50", FILES_50_50),
]:
    for step, path in files.items():
        selected.append({
            "Composition": comp,
            "Step": step,
            "Selected_file": path
        })

pd.DataFrame(selected).to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP10_SELECTED_INPUT_FILES.csv"
    ),
    index=False
)

# ============================================================
# 8. BUILD MASTER TABLE
# ============================================================

df3070 = load_composition(
    "30:70",
    FILES_30_70
)

df5050 = load_composition(
    "50:50",
    FILES_50_50
)

master = pd.concat(
    [df3070, df5050],
    ignore_index=True
)

master.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP10_MASTER_PER_FOV_30_70_vs_50_50.csv"
    ),
    index=False
)

# ============================================================
# 9. SUMMARY STATISTICS
# ============================================================

METRICS = [
    "global_S2D",
    "mean_local_S2D",
    "xi_um",
    "density_total_per_um2",
    "mean_NN_all_um",
    "rho_Slocal_nearest_defect_distance",
    "rho_Slocal_local_defect_density",
    "observed_unlike_sign_NN_fraction",
    "lambda_S_um",
    "normalized_orientation_entropy",
    "angular_dispersion_deg",
    "n_domains",
    "mean_domain_area_um2",
    "median_domain_area_um2",
    "largest_domain_area_um2",
    "area_weighted_mean_domain_area_um2",
    "fraction_valid_grid_in_retained_domains",
]

def cohen_d(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)

    pooled = np.sqrt(
        ((len(x)-1)*sx2 + (len(y)-1)*sy2)
        /
        (len(x)+len(y)-2)
    )

    if pooled == 0:
        return np.nan

    return (np.mean(y) - np.mean(x)) / pooled


rows = []

for metric in METRICS:

    x = pd.to_numeric(
        df3070[metric],
        errors="coerce"
    ).dropna()

    y = pd.to_numeric(
        df5050[metric],
        errors="coerce"
    ).dropna()

    rows.append({
        "Metric": metric,
        "30_70_N": len(x),
        "30_70_mean": x.mean() if len(x) else np.nan,
        "30_70_SD": x.std(ddof=1) if len(x) > 1 else np.nan,
        "30_70_SEM": x.sem(ddof=1) if len(x) > 1 else np.nan,
        "50_50_N": len(y),
        "50_50_mean": y.mean() if len(y) else np.nan,
        "50_50_SD": y.std(ddof=1) if len(y) > 1 else np.nan,
        "50_50_SEM": y.sem(ddof=1) if len(y) > 1 else np.nan,
        "difference_50_50_minus_30_70":
            y.mean() - x.mean() if len(x) and len(y) else np.nan,
        "cohen_d_50_50_minus_30_70":
            cohen_d(x, y),
    })

summary_df = pd.DataFrame(rows)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP10_COMPOSITION_MEAN_SD_SEM_EFFECTSIZE.csv"
    ),
    index=False
)

# ============================================================
# 10. MAIN COMPARISON FIGURES
# ============================================================

MAIN_METRICS = {
    "global_S2D": "Global $S_{2D}$",
    "mean_local_S2D": "Mean local $S_{2D}$",
    "normalized_orientation_entropy": "Normalized orientation entropy",
    "mean_domain_area_um2": "Mean domain area (µm²)",
    "xi_um": "Correlation length $\\xi$ (µm)",
    "density_total_per_um2": "Defect density (µm$^{-2}$)",
    "rho_Slocal_nearest_defect_distance": r"$\rho(S_{local},d_{defect})$",
    "rho_Slocal_local_defect_density": r"$\rho(S_{local},\rho_{defect})$",
    "observed_unlike_sign_NN_fraction": "Unlike-sign NN fraction",
    "lambda_S_um": "Order recovery length $\\lambda_S$ (µm)",
}

for metric, ylabel in MAIN_METRICS.items():

    x = pd.to_numeric(df3070[metric], errors="coerce").dropna()
    y = pd.to_numeric(df5050[metric], errors="coerce").dropna()

    if len(x) == 0 and len(y) == 0:
        continue

    means = [
        x.mean() if len(x) else np.nan,
        y.mean() if len(y) else np.nan
    ]

    sds = [
        x.std(ddof=1) if len(x) > 1 else 0,
        y.std(ddof=1) if len(y) > 1 else 0
    ]

    plt.figure(figsize=(6, 5))

    plt.bar(
        ["30:70 + NaOH", "50:50 + NaOH"],
        means,
        yerr=sds,
        capsize=5
    )

    for i, vals in enumerate([x, y]):

        if len(vals):
            jitter = np.linspace(
                -0.06,
                0.06,
                len(vals)
            )

            plt.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=35,
                zorder=3
            )

    plt.ylabel(ylabel)
    plt.title(f"NaOH-treated composition comparison\n{ylabel}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"COMPARE_{metric}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 11. FINAL REPORT
# ============================================================

print("\n" + "="*95)
print("STEP 10 COMPLETE — DIRECT STEP-FOLDER VERSION")
print("="*95)

print("\nSelected input files:")
print(
    pd.DataFrame(selected).to_string(index=False)
)

print("\nSaved:")
print("  STEP10_SELECTED_INPUT_FILES.csv")
print("  STEP10_MASTER_PER_FOV_30_70_vs_50_50.csv")
print("  STEP10_COMPOSITION_MEAN_SD_SEM_EFFECTSIZE.csv")

print("\nCheck STEP10_SELECTED_INPUT_FILES.csv once before using the final figures.")
