
# ============================================================
# PSI STUDY — STEP 8
# OBJECTIVE REPRESENTATIVE-FOV SELECTION + FIGURE GUIDE
# JUPYTER / VS CODE FRIENDLY
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT
# ============================================================

STEP7_MASTER = r"D:\YOUR_PATH\PSI_STUDY_STEP7_FINAL\PSI_STEP7_MASTER_PER_FOV.csv"
OUTPUT_DIR = r"D:\YOUR_PATH\PSI_STUDY_STEP8_REPRESENTATIVE_FOV"

DPI = 600
SHOW_PLOTS_IN_NOTEBOOK = True

# ============================================================
# 2. METRICS FOR REPRESENTATIVE-FOV SELECTION
# ============================================================

CANDIDATE_METRICS = [
    "Psi_Global_S2D",
    "Mean_gradPsi",
    "Median_gradPsi",
    "P90_gradPsi",
    "Fraction_reliable_winding",
    "Mean_winding_R2",
    "Core_minus_far_gradPsi",
    "Core_minus_far_Slocal",
]

MIN_VALID_METRICS = 3

# ============================================================
# 3. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def composition_from_label(label):
    s = str(label)
    if s.startswith("30_70"):
        return "30:70"
    if s.startswith("50_50"):
        return "50:50"
    return "Unknown"

def safe_zscore(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    sd = np.nanstd(values, ddof=1)

    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(values, dtype=float)

    return (values - mean) / sd

# ============================================================
# 4. LOAD MASTER DATA
# ============================================================

ensure_dir(OUTPUT_DIR)

if not os.path.isfile(STEP7_MASTER):
    raise FileNotFoundError(
        f"Step-7 master file not found:\n{STEP7_MASTER}"
    )

df = pd.read_csv(STEP7_MASTER)
df.columns = [str(c).strip() for c in df.columns]

if "Label" not in df.columns:
    raise KeyError(
        f"Column 'Label' not found.\nAvailable columns: {list(df.columns)}"
    )

if "Composition" not in df.columns:
    df["Composition"] = df["Label"].apply(composition_from_label)

available_metrics = [
    m for m in CANDIDATE_METRICS
    if m in df.columns
]

if len(available_metrics) < MIN_VALID_METRICS:
    raise ValueError(
        f"Only {len(available_metrics)} representative metrics are available.\n"
        f"Found: {available_metrics}\n"
        f"Need at least {MIN_VALID_METRICS}."
    )

print("Metrics used for representative-FOV ranking:")
for metric in available_metrics:
    print(" -", metric)

# ============================================================
# 5. CALCULATE REPRESENTATIVE DISTANCE
# ============================================================

ranking_rows = []

for composition in ["30:70", "50:50"]:

    sub = df[
        df["Composition"] == composition
    ].copy()

    if len(sub) == 0:
        continue

    z_columns = []

    for metric in available_metrics:
        sub[metric] = pd.to_numeric(
            sub[metric],
            errors="coerce"
        )

        values = sub[metric].to_numpy(float)

        if np.isfinite(values).sum() == 0:
            continue

        zname = "Z_" + metric
        sub[zname] = safe_zscore(values)
        z_columns.append(zname)

    if len(z_columns) < MIN_VALID_METRICS:
        raise ValueError(
            f"{composition}: only {len(z_columns)} usable metrics."
        )

    distances = []
    n_used = []

    for _, row in sub.iterrows():

        zvals = pd.to_numeric(
            row[z_columns],
            errors="coerce"
        ).to_numpy(float)

        valid = np.isfinite(zvals)

        if valid.sum() < MIN_VALID_METRICS:
            distance = np.nan
        else:
            distance = np.sqrt(
                np.mean(
                    zvals[valid]**2
                )
            )

        distances.append(distance)
        n_used.append(valid.sum())

    sub["Representative_distance"] = distances
    sub["N_metrics_used"] = n_used

    sub = sub.sort_values(
        "Representative_distance",
        ascending=True
    ).reset_index(drop=True)

    sub["Representative_rank"] = np.arange(
        1,
        len(sub) + 1
    )

    for _, row in sub.iterrows():

        out = {
            "Composition": composition,
            "Label": row["Label"],
            "Representative_rank": row["Representative_rank"],
            "Representative_distance": row["Representative_distance"],
            "N_metrics_used": row["N_metrics_used"]
        }

        for metric in available_metrics:
            out[metric] = row.get(metric, np.nan)

        ranking_rows.append(out)

ranking = pd.DataFrame(ranking_rows)

ranking_path = os.path.join(
    OUTPUT_DIR,
    "STEP8_REPRESENTATIVE_FOV_RANKING.csv"
)

ranking.to_csv(
    ranking_path,
    index=False
)

selected = (
    ranking.sort_values(
        ["Composition", "Representative_rank"]
    )
    .groupby(
        "Composition",
        as_index=False
    )
    .first()
)

selected_path = os.path.join(
    OUTPUT_DIR,
    "STEP8_SELECTED_REPRESENTATIVE_FOVs.csv"
)

selected.to_csv(
    selected_path,
    index=False
)

print("\nSelected representative FOVs:")
print(
    selected[
        [
            "Composition",
            "Label",
            "Representative_distance"
        ]
    ]
)

# ============================================================
# 6. RANKING PLOTS
# ============================================================

for composition in ["30:70", "50:50"]:

    sub = ranking[
        ranking["Composition"] == composition
    ].copy()

    if len(sub) == 0:
        continue

    fig, ax = plt.subplots(
        figsize=(6.7, 4.8)
    )

    ax.scatter(
        np.arange(len(sub)),
        sub["Representative_distance"],
        s=70,
        edgecolors="black",
        linewidths=0.7
    )

    ax.set_xticks(
        np.arange(len(sub))
    )

    ax.set_xticklabels(
        sub["Label"],
        rotation=25,
        ha="right"
    )

    ax.set_ylabel(
        "Multivariate distance from composition centroid"
    )

    ax.set_xlabel(
        "Field of view"
    )

    ax.set_title(
        f"{composition}: representative-FOV ranking"
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

    plt.tight_layout()

    fname = composition.replace(":", "_")

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"STEP8_{fname}_representative_FOV_ranking.png"
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"STEP8_{fname}_representative_FOV_ranking.pdf"
        ),
        bbox_inches="tight"
    )

    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()

    plt.close(fig)

# ============================================================
# 7. MANUSCRIPT FIGURE GUIDE
# ============================================================

selected_lookup = dict(
    zip(
        selected["Composition"],
        selected["Label"]
    )
)

guide = pd.DataFrame([
    {
        "Figure_panel": "Raw POM",
        "30_70_source": selected_lookup.get("30:70", ""),
        "50_50_source": selected_lookup.get("50:50", ""),
        "Source_step": "Original image",
        "Purpose": "Representative optical texture"
    },
    {
        "Figure_panel": "Psi(x,y) map",
        "30_70_source": selected_lookup.get("30:70", ""),
        "50_50_source": selected_lookup.get("50:50", ""),
        "Source_step": "Psi Step 1",
        "Purpose": "Spatial texture-orientation organization"
    },
    {
        "Figure_panel": "|grad Psi| map",
        "30_70_source": selected_lookup.get("30:70", ""),
        "50_50_source": selected_lookup.get("50:50", ""),
        "Source_step": "Psi Step 2",
        "Purpose": "Orientation-distortion landscape"
    },
    {
        "Figure_panel": "Winding-fit example",
        "30_70_source": "Reliable defect from representative FOV",
        "50_50_source": "Reliable defect from representative FOV",
        "Source_step": "Psi Step 4",
        "Purpose": "Winding-derived defect validation"
    },
    {
        "Figure_panel": "Core vs far-field distortion",
        "30_70_source": "All FOVs",
        "50_50_source": "All FOVs",
        "Source_step": "Psi Step 5/7",
        "Purpose": "Defect-localized distortion comparison"
    },
    {
        "Figure_panel": "Optional intensity profile",
        "30_70_source": "Representative reliable defect",
        "50_50_source": "Representative reliable defect",
        "Source_step": "Psi Step 6",
        "Purpose": "Optical-texture support"
    },
])

guide_path = os.path.join(
    OUTPUT_DIR,
    "STEP8_MANUSCRIPT_FIGURE_GUIDE.csv"
)

guide.to_csv(
    guide_path,
    index=False
)

print("\nSTEP 8 COMPLETE")
print("Ranking:", ranking_path)
print("Selected FOVs:", selected_path)
print("Figure guide:", guide_path)
