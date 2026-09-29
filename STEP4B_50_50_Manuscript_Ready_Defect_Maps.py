
# ============================================================
# STEP 4B — MANUSCRIPT-READY DEFECT MAPS
# 50:50 + NaOH | 4 FOVs
#
# PURPOSE
#   Create clean manuscript figures using:
#     - ORIGINAL POM image as background
#     - VALID merged defect positions from Step 4
#
# IMPORTANT
#   Quantitative analysis does NOT change.
#   Buffered masks remain used internally for defect filtering.
#   This script only changes the final visualization.
# ============================================================

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

DATASETS = [
    {
        "name": "50_50_NaOH_FOV1",
        "image": r"C:\Users\Prayas Singh\Desktop\5050\50_50_NaOH_FOV1.jpg",
        "defects": r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV1\merged_defects.csv",
        "exclusion": r"C:\Users\Prayas Singh\Desktop\5050\STEP1B_BUFFERED_TEXTURE_MASKS\50_50_NaOH_FOV1\04_total_exclusion_mask_buffered.png",
    },
    {
        "name": "50_50_NaOH_FOV2",
        "image": r"C:\Users\Prayas Singh\Desktop\5050\50_50_NaOH_FOV2.jpg",
        "defects": r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV2\merged_defects.csv",
        "exclusion": r"C:\Users\Prayas Singh\Desktop\5050\STEP1B_BUFFERED_TEXTURE_MASKS\50_50_NaOH_FOV2\04_total_exclusion_mask_buffered.png",
    },
    {
        "name": "50_50_NaOH_FOV3",
        "image": r"C:\Users\Prayas Singh\Desktop\5050\50_50_NaOH_FOV3.jpg",
        "defects": r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV3\merged_defects.csv",
        "exclusion": r"C:\Users\Prayas Singh\Desktop\5050\STEP1B_BUFFERED_TEXTURE_MASKS\50_50_NaOH_FOV3\04_total_exclusion_mask_buffered.png",
    },
    {
        "name": "50_50_NaOH_FOV4",
        "image": r"C:\Users\Prayas Singh\Desktop\5050\50_50_NaOH_FOV4.jpg",
        "defects": r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV4\merged_defects.csv",
        "exclusion": r"C:\Users\Prayas Singh\Desktop\5050\STEP1B_BUFFERED_TEXTURE_MASKS\50_50_NaOH_FOV4\04_total_exclusion_mask_buffered.png",
    },
]

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\5050\STEP4B_MANUSCRIPT_DEFECT_MAPS"

# ============================================================
# 2. DISPLAY SETTINGS
# ============================================================

SHOW_ONLY_HALF_DEFECTS = True
MARKER_SIZE = 70
EDGE_WIDTH = 1.3
CONTOUR_LINEWIDTH = 0.8
SAVE_DPI = 400

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. HELPERS
# ============================================================

def load_defects(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Defect file not found:\n{path}")

    df = pd.read_csv(path)

    required = {"x_px", "y_px", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    if SHOW_ONLY_HALF_DEFECTS:
        df = df[df["class"].isin(["+1/2", "-1/2"])].copy()

    return df


def add_defect_markers(ax, df):
    styles = {
        "+1/2": ("o", "+1/2"),
        "-1/2": ("s", "-1/2"),
        "+1": ("^", "+1"),
        "-1": ("v", "-1"),
    }

    for cls, grp in df.groupby("class"):
        marker, label = styles.get(cls, ("o", cls))
        ax.scatter(
            grp["x_px"],
            grp["y_px"],
            s=MARKER_SIZE,
            marker=marker,
            label=label,
            edgecolors="black",
            linewidths=EDGE_WIDTH
        )

# ============================================================
# 4. PROCESS
# ============================================================

for item in DATASETS:

    name = item["name"]
    image_path = item["image"]
    defect_path = item["defects"]
    exclusion_path = item["exclusion"]

    print(f"Processing {name}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image:\n{image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    defects = load_defects(defect_path)

    exclusion = cv2.imread(exclusion_path, cv2.IMREAD_GRAYSCALE)
    if exclusion is None:
        raise RuntimeError(f"Could not read exclusion mask:\n{exclusion_path}")

    if exclusion.shape != (h, w):
        raise ValueError(
            f"Mask/image size mismatch for {name}: "
            f"image={(h, w)}, mask={exclusion.shape}"
        )

    # --------------------------------------------------------
    # FIGURE A — recommended for main manuscript
    # Original POM + defect markers only
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(img_rgb)
    add_defect_markers(ax, defects)

    ax.set_title(f"{name} — Winding-derived defect candidates")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    if len(defects) > 0:
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"{name}_A_POM_plus_defects.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------------
    # FIGURE B — supplementary/QC
    # Original POM + defect markers + thin exclusion contour
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(img_rgb)
    add_defect_markers(ax, defects)

    ax.contour(
        exclusion > 0,
        levels=[0.5],
        linewidths=CONTOUR_LINEWIDTH,
        linestyles="--"
    )

    ax.set_title(f"{name} — Defect candidates with excluded-region boundary")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    if len(defects) > 0:
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{name}_B_POM_plus_defects_with_exclusion_contour.png"
        ),
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------------
    # FIGURE C — optional clean schematic-style map
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 7.5))
    add_defect_markers(ax, defects)

    ax.set_title(f"{name} — Spatial distribution of defect candidates")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    if len(defects) > 0:
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"{name}_C_clean_defect_map.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )
    plt.close()

print("\n" + "="*90)
print("STEP 4B COMPLETE — manuscript-ready defect maps")
print("="*90)
print("\nMAIN manuscript recommendation:")
print("  *_A_POM_plus_defects.png")
print("\nSUPPLEMENTARY/QC recommendation:")
print("  *_B_POM_plus_defects_with_exclusion_contour.png")
print("\nOptional schematic map:")
print("  *_C_clean_defect_map.png")
print("\nQuantitative defect analysis remains unchanged.")
