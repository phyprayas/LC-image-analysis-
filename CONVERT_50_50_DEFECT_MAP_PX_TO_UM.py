
# ============================================================
# 50:50 DEFECT MAP IN MICROMETERS (µm)
# Convert x,y defect coordinates from pixels to micrometers
# to match the 30:70 defect-map units.
#
# Calibration used throughout the study:
#       10 µm = 157 px
#       UM_PER_PX = 10/157 = 0.0636942675 µm/px
#
# INPUT:
#   Step-4 merged-defect CSV for one 50:50 FOV
#
# REQUIRED COLUMNS:
#   x / y coordinates and defect charge.
#   The script auto-detects common column names.
#
# OUTPUT:
#   - clean defect map with axes in µm
#   - converted defect CSV with x_um and y_um
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. EDIT PATHS
# ============================================================

INPUT_DEFECT_CSV = r"D:\New results\5070\step 4 marged defects\FOV1\merged_defects.csv"

OUTPUT_FOLDER = r"D:\New results\5070\step 4 marged defects\FOV1\publication_um"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. CALIBRATION
# ============================================================

UM_PER_PX = 10.0 / 157.0

# If you want the same full-image field of view as the 50:50 POM:
IMAGE_WIDTH_PX = 2048
IMAGE_HEIGHT_PX = 1536

IMAGE_WIDTH_UM = IMAGE_WIDTH_PX * UM_PER_PX
IMAGE_HEIGHT_UM = IMAGE_HEIGHT_PX * UM_PER_PX

# ============================================================
# 3. LOAD DEFECT DATA
# ============================================================

df = pd.read_csv(INPUT_DEFECT_CSV)

print("Available columns:")
print(list(df.columns))

# ============================================================
# 4. AUTO-DETECT COLUMNS
# ============================================================

X_ALIASES = [
    "x_px", "X_px", "x", "X",
    "merged_x_px", "centroid_x_px"
]

Y_ALIASES = [
    "y_px", "Y_px", "y", "Y",
    "merged_y_px", "centroid_y_px"
]

CHARGE_ALIASES = [
    "charge", "Charge", "q",
    "defect_charge", "topological_charge",
    "strength", "sign"
]

def find_col(df, aliases, label):
    for c in aliases:
        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find {label} column.\n"
        f"Tried: {aliases}\n"
        f"Available columns: {list(df.columns)}"
    )

x_col = find_col(df, X_ALIASES, "x-coordinate")
y_col = find_col(df, Y_ALIASES, "y-coordinate")
q_col = find_col(df, CHARGE_ALIASES, "defect charge")

# ============================================================
# 5. CONVERT PIXELS -> MICROMETERS
# ============================================================

df["x_um"] = pd.to_numeric(
    df[x_col],
    errors="coerce"
) * UM_PER_PX

df["y_um"] = pd.to_numeric(
    df[y_col],
    errors="coerce"
) * UM_PER_PX

df["charge_numeric"] = pd.to_numeric(
    df[q_col],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "x_um",
        "y_um",
        "charge_numeric"
    ]
).copy()

# ============================================================
# 6. SEPARATE +1/2 AND -1/2 DEFECTS
# ============================================================

plus = df[
    np.isclose(
        df["charge_numeric"],
        0.5,
        atol=0.15
    )
].copy()

minus = df[
    np.isclose(
        df["charge_numeric"],
        -0.5,
        atol=0.15
    )
].copy()

print("\nDetected defects:")
print(" +1/2:", len(plus))
print(" -1/2:", len(minus))
print(" total:", len(df))

# ============================================================
# 7. SAVE CONVERTED CSV
# ============================================================

converted_csv = os.path.join(
    OUTPUT_FOLDER,
    "merged_defects_coordinates_um.csv"
)

df.to_csv(
    converted_csv,
    index=False
)

# ============================================================
# 8. PLOT IN MICROMETERS
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 6.2)
)

ax.scatter(
    plus["x_um"],
    plus["y_um"],
    s=70,
    marker="o",
    edgecolors="black",
    linewidths=0.9,
    label="+1/2",
    zorder=3
)

ax.scatter(
    minus["x_um"],
    minus["y_um"],
    s=70,
    marker="s",
    edgecolors="black",
    linewidths=0.9,
    label="-1/2",
    zorder=3
)

# Match image coordinate convention:
# image origin is at top-left, so invert y-axis.
ax.set_xlim(
    0,
    IMAGE_WIDTH_UM
)

ax.set_ylim(
    IMAGE_HEIGHT_UM,
    0
)

ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.set_xlabel(
    r"$x$ ($\mu$m)",
    fontsize=12
)

ax.set_ylabel(
    r"$y$ ($\mu$m)",
    fontsize=12
)

ax.set_title(
    "50:50 + NaOH — spatial distribution of defect candidates",
    fontsize=13
)

# Outer border on all sides
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.1)

ax.tick_params(
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0,
    labelsize=10
)

ax.grid(False)

ax.legend(
    frameon=True,
    loc="upper right"
)

plt.tight_layout()

png_path = os.path.join(
    OUTPUT_FOLDER,
    "50_50_defect_map_um.png"
)

pdf_path = os.path.join(
    OUTPUT_FOLDER,
    "50_50_defect_map_um.pdf"
)

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.show()

print("\n" + "="*76)
print("DEFECT MAP CONVERTED TO MICROMETERS")
print("="*76)

print(f"\nCalibration: {UM_PER_PX:.9f} µm/px")
print(f"Image size: {IMAGE_WIDTH_UM:.3f} × {IMAGE_HEIGHT_UM:.3f} µm")

print("\nSaved:")
print(converted_csv)
print(png_path)
print(pdf_path)
