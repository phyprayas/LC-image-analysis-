
# ============================================================
# POOLED COLOR CLUSTERING — 30:70 vs 50:50 + NaOH
#
# Purpose:
# Build ONE common CIELAB color model from BOTH samples, then apply
# the same fixed cluster centers to both images.
#
# This solves the "independent KMeans labels are not directly comparable"
# problem.
#
# INPUTS:
# - 30:70 raw POM image
# - 30:70 valid LC mask
# - 30:70 local_order_map.csv
# - 30:70 merged_unique_defects.csv
# - 50:50 raw POM image
# - 50:50 valid LC mask
# - 50:50 local_order_map.csv
# - 50:50 merged_unique_defects.csv
#
# OUTPUT:
# - pooled_cluster_centers.csv
# - per-sample class maps
# - pooled_color_comparison.csv
# - pooled_color_topology_comparison.csv
# - figures
#
# IMPORTANT:
# Optical color classes are descriptive optical-texture populations,
# not automatically distinct thermodynamic phases.
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

# ============================================================
# 1. PATHS — EDIT THESE
# ============================================================

# ---------- 30:70 ----------
IMG_30 = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\G3\YOUR_30_70_RAW_IMAGE.jpg"
MASK_30 = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\G3\YOUR_30_70_VALID_MASK.png"
ORDER_30 = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\G3\YOUR_30_70_local_order_map.csv"
DEFECTS_30 = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\G3\YOUR_30_70_merged_unique_defects.csv"

# ---------- 50:50 ----------
IMG_50 = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\G4\GO_Drop_2mg_9658n.jpg"
MASK_50 = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\G4\STEP1_Crystal_Mask\03_valid_LC_mask.png"
ORDER_50 = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\G4\STEP3_Orientation_Results\local_order_map.csv"
DEFECTS_50 = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\G4\STEP4_Merged_Defects\merged_unique_defects.csv"

OUTPUT_DIR = r"C:\Users\owner\Desktop\Pyhton results-20260920T124158Z-1-001\Pyhton results\POOLED_COLOR_30_70_vs_50_50"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. SETTINGS
# ============================================================

N_CLASSES = 3
RANDOM_STATE = 42
MAX_TRAIN_PER_SAMPLE = 100000
PREDICT_CHUNK = 250000
MIN_LAB_L = 25

# ============================================================
# 3. HELPERS
# ============================================================

def check_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found:\n{path}")

def load_sample(image_path, mask_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    if img_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions differ:\n"
            f"Image = {img_bgr.shape[:2]}\nMask = {mask.shape[:2]}"
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    valid = mask >= 128
    L = lab[:, :, 0]

    foreground = valid & (L >= MIN_LAB_L)

    pixels = lab[foreground].reshape(-1, 3).astype(np.float32)

    return img_bgr, img_rgb, lab, foreground, pixels

def assign_cluster_names(centers):
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8).reshape(-1,1,3)
    centers_bgr = cv2.cvtColor(
        centers_u8,
        cv2.COLOR_LAB2BGR
    ).reshape(-1,3)

    centers_hsv = cv2.cvtColor(
        centers_bgr.reshape(-1,1,3),
        cv2.COLOR_BGR2HSV
    ).reshape(-1,3)

    rows = []
    for cid in range(len(centers)):
        Lc, ac, bc = centers[cid]
        Hc, Sc, Vc = centers_hsv[cid]
        bgr = centers_bgr[cid]
        rgb = bgr[::-1]

        rows.append({
            "color_cluster_id": cid,
            "L": float(Lc),
            "a": float(ac),
            "b": float(bc),
            "H": float(Hc),
            "S": float(Sc),
            "V": float(Vc),
            "R": int(rgb[0]),
            "G": int(rgb[1]),
            "B_rgb": int(rgb[2]),
        })

    df = pd.DataFrame(rows)

    bright_id = int(df["L"].idxmax())
    dim_id = int(df["L"].idxmin())
    mid_id = [i for i in range(len(df)) if i not in [bright_id, dim_id]][0]

    name_map = {
        bright_id: "bright_yellow",
        dim_id: "dim_yellow_olive",
        mid_id: "green_cyan"
    }

    df["color_class"] = df["color_cluster_id"].map(name_map)

    return df, name_map

def predict_map(lab, foreground, model):
    pixels = lab[foreground].reshape(-1,3).astype(np.float32)
    labels_flat = np.empty(len(pixels), dtype=np.int16)

    for start in range(0, len(pixels), PREDICT_CHUNK):
        end = min(start + PREDICT_CHUNK, len(pixels))
        labels_flat[start:end] = model.predict(pixels[start:end])

    cmap = np.full(foreground.shape, -1, dtype=np.int16)
    cmap[foreground] = labels_flat

    return cmap

def add_local_order(sample_name, class_map, name_map, order_csv):
    df = pd.read_csv(order_csv)

    for c in ["X","Y","local_S2D"]:
        if c not in df.columns:
            raise ValueError(
                f"{sample_name}: missing {c} in {order_csv}"
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["X","Y","local_S2D"]).copy()

    x = np.rint(df["X"].values).astype(int)
    y = np.rint(df["Y"].values).astype(int)

    H, W = class_map.shape

    inside = (
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H)
    )

    cid = np.full(len(df), -1, dtype=int)
    ii = np.where(inside)[0]
    cid[ii] = class_map[y[ii], x[ii]]

    df["color_cluster_id"] = cid
    df["color_class"] = [
        name_map.get(v, "invalid")
        for v in cid
    ]

    return df[df["color_cluster_id"] >= 0].copy()

def add_defects(sample_name, class_map, name_map, defect_csv):
    df = pd.read_csv(defect_csv)

    for c in ["X","Y","charge"]:
        if c not in df.columns:
            raise ValueError(
                f"{sample_name}: missing {c} in {defect_csv}"
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["X","Y","charge"]).copy()

    x = np.rint(df["X"].values).astype(int)
    y = np.rint(df["Y"].values).astype(int)

    H, W = class_map.shape

    inside = (
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H)
    )

    cid = np.full(len(df), -1, dtype=int)
    ii = np.where(inside)[0]
    cid[ii] = class_map[y[ii], x[ii]]

    df["color_cluster_id"] = cid
    df["color_class"] = [
        name_map.get(v, "invalid")
        for v in cid
    ]

    return df[df["color_cluster_id"] >= 0].copy()

# ============================================================
# 4. CHECK PATHS
# ============================================================

for p, label in [
    (IMG_30, "IMG_30"),
    (MASK_30, "MASK_30"),
    (ORDER_30, "ORDER_30"),
    (DEFECTS_30, "DEFECTS_30"),
    (IMG_50, "IMG_50"),
    (MASK_50, "MASK_50"),
    (ORDER_50, "ORDER_50"),
    (DEFECTS_50, "DEFECTS_50"),
]:
    check_file(p, label)

# ============================================================
# 5. LOAD BOTH SAMPLES
# ============================================================

img30_bgr, img30_rgb, lab30, fg30, pix30 = load_sample(IMG_30, MASK_30)
img50_bgr, img50_rgb, lab50, fg50, pix50 = load_sample(IMG_50, MASK_50)

print("30:70 valid pixels:", len(pix30))
print("50:50 valid pixels:", len(pix50))

# ============================================================
# 6. BUILD BALANCED POOLED TRAINING SET
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

def subsample(pixels, nmax):
    if len(pixels) <= nmax:
        return pixels
    idx = rng.choice(len(pixels), size=nmax, replace=False)
    return pixels[idx]

train30 = subsample(pix30, MAX_TRAIN_PER_SAMPLE)
train50 = subsample(pix50, MAX_TRAIN_PER_SAMPLE)

# IMPORTANT:
# equal-ish contribution from both compositions
train_pool = np.vstack([train30, train50])

print("Pooled training pixels:", len(train_pool))

# ============================================================
# 7. FIT ONE COMMON MODEL
# ============================================================

model = MiniBatchKMeans(
    n_clusters=N_CLASSES,
    random_state=RANDOM_STATE,
    batch_size=4096,
    n_init=10,
    max_iter=200
)

model.fit(train_pool)

centers = model.cluster_centers_
centers_df, name_map = assign_cluster_names(centers)

centers_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pooled_cluster_centers.csv"
    ),
    index=False
)

print("\nPooled class centers:")
print(centers_df.to_string(index=False))

# ============================================================
# 8. APPLY SAME CENTERS TO BOTH SAMPLES
# ============================================================

map30 = predict_map(lab30, fg30, model)
map50 = predict_map(lab50, fg50, model)

np.save(
    os.path.join(OUTPUT_DIR, "30_70_color_class_map.npy"),
    map30
)

np.save(
    os.path.join(OUTPUT_DIR, "50_50_color_class_map.npy"),
    map50
)

# ============================================================
# 9. ORDER + DEFECT ASSIGNMENT
# ============================================================

order30 = add_local_order(
    "30:70", map30, name_map, ORDER_30
)

order50 = add_local_order(
    "50:50", map50, name_map, ORDER_50
)

def30 = add_defects(
    "30:70", map30, name_map, DEFECTS_30
)

def50 = add_defects(
    "50:50", map50, name_map, DEFECTS_50
)

order30.to_csv(
    os.path.join(OUTPUT_DIR, "30_70_local_order_with_pooled_color.csv"),
    index=False
)

order50.to_csv(
    os.path.join(OUTPUT_DIR, "50_50_local_order_with_pooled_color.csv"),
    index=False
)

def30.to_csv(
    os.path.join(OUTPUT_DIR, "30_70_defects_with_pooled_color.csv"),
    index=False
)

def50.to_csv(
    os.path.join(OUTPUT_DIR, "50_50_defects_with_pooled_color.csv"),
    index=False
)

# ============================================================
# 10. BUILD COMPARISON TABLES
# ============================================================

PIXEL_SIZE_UM = 10.0 / 157.0

def summarize_sample(sample_name, class_map, order_df, defect_df):

    total_valid = np.sum(class_map >= 0)
    rows = []

    for cid, cname in name_map.items():

        area_px = int(np.sum(class_map == cid))
        area_fraction = area_px / total_valid

        osub = order_df[
            order_df["color_cluster_id"] == cid
        ]

        dsub = defect_df[
            defect_df["color_cluster_id"] == cid
        ]

        ndef = len(dsub)
        nplus = int(np.sum(dsub["charge"] > 0))
        nminus = int(np.sum(dsub["charge"] < 0))

        density_px2 = (
            ndef / area_px
            if area_px > 0 else np.nan
        )

        density_um2 = (
            density_px2 / (PIXEL_SIZE_UM**2)
            if np.isfinite(density_px2)
            else np.nan
        )

        spacing_px = (
            1.0 / np.sqrt(density_px2)
            if density_px2 > 0
            else np.nan
        )

        rows.append({
            "Sample": sample_name,
            "color_cluster_id": cid,
            "color_class": cname,

            "area_px": area_px,
            "area_fraction": area_fraction,

            "mean_local_S2D":
                osub["local_S2D"].mean()
                if len(osub) else np.nan,

            "median_local_S2D":
                osub["local_S2D"].median()
                if len(osub) else np.nan,

            "n_order_points": len(osub),

            "total_merged_defects": ndef,
            "plus_half_count": nplus,
            "minus_half_count": nminus,

            "defect_density_per_px2": density_px2,
            "defect_density_per_um2": density_um2,

            "characteristic_spacing_px": spacing_px,
            "characteristic_spacing_um":
                spacing_px * PIXEL_SIZE_UM
                if np.isfinite(spacing_px)
                else np.nan
        })

    return pd.DataFrame(rows)

sum30 = summarize_sample(
    "30:70 + NaOH",
    map30,
    order30,
    def30
)

sum50 = summarize_sample(
    "50:50 + NaOH",
    map50,
    order50,
    def50
)

pooled_compare = pd.concat(
    [sum30, sum50],
    ignore_index=True
)

pooled_compare.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pooled_color_order_topology_comparison.csv"
    ),
    index=False
)

# ============================================================
# 11. VISUAL CLASS MAPS
# ============================================================

display_colors = {
    "green_cyan": (100, 220, 180),
    "bright_yellow": (255, 240, 80),
    "dim_yellow_olive": (170, 140, 60)
}

def make_vis(class_map):
    vis = np.zeros((*class_map.shape,3), dtype=np.uint8)

    for cid, cname in name_map.items():
        vis[class_map == cid] = display_colors[cname]

    return vis

vis30 = make_vis(map30)
vis50 = make_vis(map50)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "30_70_pooled_color_map.png"
    ),
    cv2.cvtColor(vis30, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "50_50_pooled_color_map.png"
    ),
    cv2.cvtColor(vis50, cv2.COLOR_RGB2BGR)
)

# ============================================================
# 12. PLOTS
# ============================================================

classes = [
    "bright_yellow",
    "green_cyan",
    "dim_yellow_olive"
]

# area fraction
x = np.arange(len(classes))
w = 0.35

vals30 = [
    sum30.loc[
        sum30["color_class"] == c,
        "area_fraction"
    ].iloc[0]
    for c in classes
]

vals50 = [
    sum50.loc[
        sum50["color_class"] == c,
        "area_fraction"
    ].iloc[0]
    for c in classes
]

plt.figure(figsize=(9,5))
plt.bar(x-w/2, vals30, w, label="30:70 + NaOH")
plt.bar(x+w/2, vals50, w, label="50:50 + NaOH")
plt.xticks(x, classes, rotation=15)
plt.ylabel("Area fraction")
plt.title("Pooled optical-class area fractions")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_pooled_color_area_fraction_comparison.png"
    ),
    dpi=300
)
plt.close()

# local order
vals30 = [
    sum30.loc[
        sum30["color_class"] == c,
        "mean_local_S2D"
    ].iloc[0]
    for c in classes
]

vals50 = [
    sum50.loc[
        sum50["color_class"] == c,
        "mean_local_S2D"
    ].iloc[0]
    for c in classes
]

plt.figure(figsize=(9,5))
plt.bar(x-w/2, vals30, w, label="30:70 + NaOH")
plt.bar(x+w/2, vals50, w, label="50:50 + NaOH")
plt.xticks(x, classes, rotation=15)
plt.ylabel("Mean local $S_{2D}$")
plt.title("Pooled color-specific orientational order")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_pooled_color_local_S2D_comparison.png"
    ),
    dpi=300
)
plt.close()

# defect density
vals30 = [
    sum30.loc[
        sum30["color_class"] == c,
        "defect_density_per_px2"
    ].iloc[0]
    for c in classes
]

vals50 = [
    sum50.loc[
        sum50["color_class"] == c,
        "defect_density_per_px2"
    ].iloc[0]
    for c in classes
]

plt.figure(figsize=(9,5))
plt.bar(x-w/2, vals30, w, label="30:70 + NaOH")
plt.bar(x+w/2, vals50, w, label="50:50 + NaOH")
plt.xticks(x, classes, rotation=15)
plt.ylabel("Defect density (px$^{-2}$)")
plt.title("Pooled color-specific defect density")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_pooled_color_defect_density_comparison.png"
    ),
    dpi=300
)
plt.close()

# ============================================================
# 13. TEXT SUMMARY
# ============================================================

text = """
Pooled color clustering comparison

A single CIELAB clustering model was trained using balanced pixel samples
from both the 30:70 and 50:50 NaOH-treated compositions. The same fixed
cluster centers were then applied to both images, allowing direct comparison
of optical-texture area fractions, color-specific local order, and
color-specific defect density without composition-dependent relabeling.

Use pooled_color_order_topology_comparison.csv for the quantitative results.

Interpretation note:
The pooled classes are shared optical-texture populations. They should not
be interpreted as thermodynamic phases without independent structural
evidence.
"""

with open(
    os.path.join(
        OUTPUT_DIR,
        "ready_to_paste_pooled_color_method.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(text)

# ============================================================
# 14. PRINT
# ============================================================

print("\nPOOLED COLOR ANALYSIS COMPLETE")
print("\nSaved to:")
print(OUTPUT_DIR)

print("\nMain files:")
print("pooled_cluster_centers.csv")
print("pooled_color_order_topology_comparison.csv")
print("30_70_pooled_color_map.png")
print("50_50_pooled_color_map.png")
print("01_pooled_color_area_fraction_comparison.png")
print("02_pooled_color_local_S2D_comparison.png")
print("03_pooled_color_defect_density_comparison.png")
