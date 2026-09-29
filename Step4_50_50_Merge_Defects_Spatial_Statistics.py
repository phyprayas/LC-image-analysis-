
# ============================================================
# STEP 4 — MERGE DEFECT CANDIDATES + SPATIAL STATISTICS
# For 50:50 CPCL:Glycerol + NaOH
#
# INPUT:
# raw_defect_candidates.csv from Step 3
#
# SAME SETTINGS AS 30:70 ANALYSIS:
# merge radius = 35 px
#
# OUTPUT:
# - merged_unique_defects.csv
# - merged_defect_summary.csv
# - nearest_neighbor_distances.csv
# - figures
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ============================================================
# 1. SETTINGS
# ============================================================

MERGE_RADIUS_PX = 35.0

# Use same calibration only if same magnification/resolution was used
PIXEL_SIZE_UM = 10.0 / 157.0

# ============================================================
# 2. SELECT INPUT + OUTPUT
# ============================================================

root = Tk()
root.withdraw()

defect_csv = filedialog.askopenfilename(
    title="Select Step 3 raw_defect_candidates.csv",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not defect_csv:
    raise SystemExit("No defect CSV selected.")

output_dir = filedialog.askdirectory(
    title="Select output folder for STEP 4"
)

if not output_dir:
    raise SystemExit("No output folder selected.")

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 3. LOAD DEFECTS
# ============================================================

df = pd.read_csv(defect_csv)

required = ["X", "Y", "charge"]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

for c in required:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=required).copy()

if len(df) == 0:
    raise RuntimeError("No valid defect candidates found.")

raw_n = len(df)
raw_plus = int(np.sum(df["charge"] > 0))
raw_minus = int(np.sum(df["charge"] < 0))

# ============================================================
# 4. SAME-SIGN SPATIAL CLUSTERING
# ============================================================

def cluster_same_sign(sub_df, radius):
    """
    Simple connected clustering:
    two candidates belong to same cluster if connected through
    same-sign points within merge radius.
    """
    pts = sub_df[["X", "Y"]].values.astype(float)
    n = len(pts)

    if n == 0:
        return []

    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        cluster = []
        stack = [i]
        visited[i] = True

        while stack:
            k = stack.pop()
            cluster.append(k)

            dx = pts[:, 0] - pts[k, 0]
            dy = pts[:, 1] - pts[k, 1]
            dist = np.sqrt(dx*dx + dy*dy)

            neighbors = np.where(
                (dist <= radius) &
                (~visited)
            )[0]

            for nb in neighbors:
                visited[nb] = True
                stack.append(nb)

        clusters.append(cluster)

    return clusters


merged_rows = []

cluster_id = 0

for sign_value in [0.5, -0.5]:

    sub = df[np.sign(df["charge"]) == np.sign(sign_value)].copy()
    sub = sub.reset_index(drop=True)

    clusters = cluster_same_sign(sub, MERGE_RADIUS_PX)

    for cluster in clusters:
        cluster_id += 1

        part = sub.iloc[cluster]

        x_mean = part["X"].mean()
        y_mean = part["Y"].mean()

        merged_rows.append({
            "cluster_id": cluster_id,
            "X": x_mean,
            "Y": y_mean,
            "charge": sign_value,
            "type": "+1/2" if sign_value > 0 else "-1/2",
            "n_raw_candidates": len(part),
            "x_min": part["X"].min(),
            "x_max": part["X"].max(),
            "y_min": part["Y"].min(),
            "y_max": part["Y"].max()
        })

merged = pd.DataFrame(merged_rows)

merged_n = len(merged)
merged_plus = int(np.sum(merged["charge"] > 0))
merged_minus = int(np.sum(merged["charge"] < 0))

# ============================================================
# 5. NEAREST-NEIGHBOR DISTANCES
# ============================================================

pts = merged[["X", "Y"]].values.astype(float)

nn_rows = []

if len(pts) >= 2:

    for i in range(len(pts)):
        dx = pts[:, 0] - pts[i, 0]
        dy = pts[:, 1] - pts[i, 1]

        d = np.sqrt(dx*dx + dy*dy)
        d[i] = np.inf

        j = int(np.argmin(d))

        nn_rows.append({
            "cluster_id": int(merged.iloc[i]["cluster_id"]),
            "charge": float(merged.iloc[i]["charge"]),
            "nearest_cluster_id": int(merged.iloc[j]["cluster_id"]),
            "nearest_charge": float(merged.iloc[j]["charge"]),
            "nearest_distance_px": float(d[j]),
            "nearest_distance_um": float(d[j] * PIXEL_SIZE_UM),
            "unlike_sign_neighbor": bool(
                np.sign(merged.iloc[i]["charge"]) !=
                np.sign(merged.iloc[j]["charge"])
            )
        })

nn_df = pd.DataFrame(nn_rows)

mean_nn_px = float(nn_df["nearest_distance_px"].mean()) if len(nn_df) else np.nan
median_nn_px = float(nn_df["nearest_distance_px"].median()) if len(nn_df) else np.nan

mean_nn_um = mean_nn_px * PIXEL_SIZE_UM if np.isfinite(mean_nn_px) else np.nan
median_nn_um = median_nn_px * PIXEL_SIZE_UM if np.isfinite(median_nn_px) else np.nan

unlike_nn_fraction = (
    float(nn_df["unlike_sign_neighbor"].mean())
    if len(nn_df)
    else np.nan
)

# ============================================================
# 6. DEFECT DENSITY
# ============================================================
# Because Step 4 only receives defect CSV, the exact valid-LC area
# is not automatically known here. Use the Step 3 valid area from
# orientation_summary.csv if supplied manually below.
#
# You can either:
# A) enter VALID_LC_AREA_PX2 from Step 3, or
# B) leave as None and density will remain NaN.
# ============================================================

VALID_LC_AREA_PX2 = 2427526
# ^ Replace if your Step 3 summary gives a different exact value.

if VALID_LC_AREA_PX2 is not None and VALID_LC_AREA_PX2 > 0:
    defect_density_px2 = merged_n / VALID_LC_AREA_PX2

    characteristic_spacing_px = (
        1.0 / np.sqrt(defect_density_px2)
        if defect_density_px2 > 0
        else np.nan
    )

    characteristic_spacing_um = (
        characteristic_spacing_px * PIXEL_SIZE_UM
        if np.isfinite(characteristic_spacing_px)
        else np.nan
    )

    defect_density_um2 = (
        defect_density_px2 / (PIXEL_SIZE_UM**2)
    )

else:
    defect_density_px2 = np.nan
    characteristic_spacing_px = np.nan
    characteristic_spacing_um = np.nan
    defect_density_um2 = np.nan

# ============================================================
# 7. SAVE TABLES
# ============================================================

merged_csv = os.path.join(
    output_dir,
    "merged_unique_defects.csv"
)
merged.to_csv(merged_csv, index=False)

nn_csv = os.path.join(
    output_dir,
    "nearest_neighbor_distances.csv"
)
nn_df.to_csv(nn_csv, index=False)

summary = pd.DataFrame({
    "Parameter": [
        "merge_radius_px",
        "raw_total_candidates",
        "raw_plus_half",
        "raw_minus_half",
        "merged_total_defects",
        "merged_plus_half",
        "merged_minus_half",
        "mean_raw_candidates_per_merged_defect",
        "max_raw_candidates_in_one_cluster",
        "valid_LC_area_px2",
        "merged_defect_density_per_px2",
        "merged_defect_density_per_um2",
        "characteristic_spacing_px",
        "characteristic_spacing_um",
        "mean_nearest_neighbor_px",
        "median_nearest_neighbor_px",
        "mean_nearest_neighbor_um",
        "median_nearest_neighbor_um",
        "unlike_sign_nearest_neighbor_fraction"
    ],
    "Value": [
        MERGE_RADIUS_PX,
        raw_n,
        raw_plus,
        raw_minus,
        merged_n,
        merged_plus,
        merged_minus,
        float(merged["n_raw_candidates"].mean()) if len(merged) else np.nan,
        int(merged["n_raw_candidates"].max()) if len(merged) else np.nan,
        VALID_LC_AREA_PX2,
        defect_density_px2,
        defect_density_um2,
        characteristic_spacing_px,
        characteristic_spacing_um,
        mean_nn_px,
        median_nn_px,
        mean_nn_um,
        median_nn_um,
        unlike_nn_fraction
    ]
})

summary_csv = os.path.join(
    output_dir,
    "merged_defect_summary.csv"
)
summary.to_csv(summary_csv, index=False)

# ============================================================
# 8. FIGURE 1 — MERGED DEFECT MAP
# ============================================================

plt.figure(figsize=(10, 7))

p = merged[merged["charge"] > 0]
m = merged[merged["charge"] < 0]

if len(p):
    plt.scatter(
        p["X"], p["Y"],
        marker="^",
        s=65,
        label="+1/2"
    )

if len(m):
    plt.scatter(
        m["X"], m["Y"],
        marker="v",
        s=65,
        label="-1/2"
    )

plt.gca().invert_yaxis()
plt.axis("equal")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.title(f"Merged unique defect candidates (R = {MERGE_RADIUS_PX:.0f} px)")
plt.legend()
plt.tight_layout()

fig1 = os.path.join(
    output_dir,
    "01_merged_unique_defect_map.png"
)

plt.savefig(fig1, dpi=300)
plt.close()

# ============================================================
# 9. FIGURE 2 — NEAREST-NEIGHBOR DISTANCE DISTRIBUTION
# ============================================================

if len(nn_df):

    plt.figure(figsize=(8, 5))

    plt.hist(
        nn_df["nearest_distance_px"],
        bins=15
    )

    plt.axvline(
        mean_nn_px,
        linestyle="--",
        label=f"Mean = {mean_nn_px:.1f} px"
    )

    plt.axvline(
        median_nn_px,
        linestyle=":",
        label=f"Median = {median_nn_px:.1f} px"
    )

    plt.xlabel("Nearest-neighbor distance (px)")
    plt.ylabel("Count")
    plt.title("Nearest-neighbor distribution of merged defects")
    plt.legend()
    plt.tight_layout()

    fig2 = os.path.join(
        output_dir,
        "02_nearest_neighbor_distribution.png"
    )

    plt.savefig(fig2, dpi=300)
    plt.close()

# ============================================================
# 10. FIGURE 3 — CLUSTER SIZE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    merged["n_raw_candidates"],
    bins=np.arange(
        0.5,
        merged["n_raw_candidates"].max() + 1.5,
        1
    )
)

plt.xlabel("Raw candidates merged into one unique defect")
plt.ylabel("Count")
plt.title("Duplicate-candidate clustering")
plt.tight_layout()

fig3 = os.path.join(
    output_dir,
    "03_cluster_size_distribution.png"
)

plt.savefig(fig3, dpi=300)
plt.close()

# ============================================================
# 11. READY-TO-PASTE TEXT
# ============================================================

text = f"""
STEP 4 — Merged defect analysis

Using the same-sign spatial clustering rule and a merge radius of
{MERGE_RADIUS_PX:.0f} px, {raw_n} raw winding-derived defect candidates
were reduced to {merged_n} merged unique defect candidates.

Merged counts:
+1/2 = {merged_plus}
-1/2 = {merged_minus}
Total = {merged_n}

The mean nearest-neighbor distance was {mean_nn_px:.2f} px
({mean_nn_um:.2f} µm), and the median nearest-neighbor distance was
{median_nn_px:.2f} px ({median_nn_um:.2f} µm).

Using a valid LC area of {VALID_LC_AREA_PX2} px², the merged defect density
was {defect_density_px2:.6e} px⁻²
({defect_density_um2:.4f} µm⁻²), corresponding to a density-derived
characteristic spacing of {characteristic_spacing_px:.2f} px
({characteristic_spacing_um:.2f} µm).

The fraction of defects whose nearest merged neighbor had the opposite
topological sign was {unlike_nn_fraction:.3f}.

Interpretation note:
These values describe winding-derived defect candidates. Sign-resolved
interaction conclusions should be tested separately using the Step 11
pair-correlation / shuffle-null analysis.
"""

txt_path = os.path.join(
    output_dir,
    "ready_to_paste_step4_summary.txt"
)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(text)

# ============================================================
# 12. PRINT
# ============================================================

print("\n" + "="*80)
print("STEP 4 COMPLETE")
print("="*80)

print(f"Raw total candidates               : {raw_n}")
print(f"Merged unique defects              : {merged_n}")
print(f"Merged +1/2                        : {merged_plus}")
print(f"Merged -1/2                        : {merged_minus}")
print(f"Mean NN distance                   : {mean_nn_px:.2f} px")
print(f"Median NN distance                 : {median_nn_px:.2f} px")
print(f"Defect density                     : {defect_density_px2:.6e} px^-2")
print(f"Characteristic spacing             : {characteristic_spacing_px:.2f} px")
print(f"Unlike-sign NN fraction            : {unlike_nn_fraction:.3f}")

print("\nSaved to:")
print(output_dir)

print("\nNext:")
print("Upload merged_defect_summary.csv and 01_merged_unique_defect_map.png")
