
# ============================================================
# STEP 1 — 50:50 + NaOH, 4 FOVs
# PATH-BASED ANNOTATION MASK + CALIBRATION
#
# PURPOSE
#   - Keep the original LC texture unchanged
#   - Exclude only microscope annotations:
#       * bottom-left magnification text
#       * bottom-right scale-bar/label box
#   - Save a valid-region mask and QC overlay for each FOV
#   - Use the same calibration as the 30:70 analysis:
#         10 µm = 157 px
#         pixel size = 0.0636942675 µm/px
#
# IMPORTANT
#   Edit only the paths in SECTION 1.
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

IMAGE_1 = r"C:\Users\Prayas Singh\Desktop\5050\50_50_NaOH_FOV1.jpg"
IMAGE_2 = r"C:\Users\Prayas Singh\Desktop\5050\50_50_NaOH_FOV2.jpg"
IMAGE_3 = r"C:\Users\Prayas Singh\Desktop\5050\50_50_NaOH_FOV3.jpg"
IMAGE_4 = r"C:\Users\Prayas Singh\Desktop\5050\50_50_NaOH_FOV4.jpg"

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\5050\STEP1_MASK_CALIBRATION"

DATASETS = [
    {"name": "50_50_NaOH_FOV1", "path": IMAGE_1},
    {"name": "50_50_NaOH_FOV2", "path": IMAGE_2},
    {"name": "50_50_NaOH_FOV3", "path": IMAGE_3},
    {"name": "50_50_NaOH_FOV4", "path": IMAGE_4},
]

# ============================================================
# 2. CALIBRATION — KEEP SAME AS 30:70
# ============================================================

SCALE_BAR_UM = 10.0
SCALE_BAR_PX = 157.0

PIXEL_SIZE_UM = SCALE_BAR_UM / SCALE_BAR_PX
PIXEL_AREA_UM2 = PIXEL_SIZE_UM ** 2

# ============================================================
# 3. ANNOTATION MASK SETTINGS
# ============================================================
# These are fractions of image width/height, so the same code
# remains robust if the image dimensions are unchanged.
#
# Bottom-left mask removes "Magnification: 50 x"
# Bottom-right mask removes scale bar and "10 µm" box.
#
# The LC texture outside these annotation boxes is untouched.
# ============================================================

LEFT_X1_FRAC = 0.00
LEFT_X2_FRAC = 0.23
LEFT_Y1_FRAC = 0.935
LEFT_Y2_FRAC = 1.00

RIGHT_X1_FRAC = 0.875
RIGHT_X2_FRAC = 1.00
RIGHT_Y1_FRAC = 0.915
RIGHT_Y2_FRAC = 1.00

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

summary_rows = []

# ============================================================
# 4. PROCESS EACH FOV
# ============================================================

for item in DATASETS:

    name = item["name"]
    image_path = item["path"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    h, w = img_bgr.shape[:2]

    out_dir = os.path.join(
        OUTPUT_FOLDER,
        name
    )
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # Create valid-region mask
    # 255 = valid LC texture
    #   0 = excluded annotation
    # --------------------------------------------------------

    mask = np.full(
        (h, w),
        255,
        dtype=np.uint8
    )

    # bottom-left annotation
    lx1 = int(round(LEFT_X1_FRAC * w))
    lx2 = int(round(LEFT_X2_FRAC * w))
    ly1 = int(round(LEFT_Y1_FRAC * h))
    ly2 = int(round(LEFT_Y2_FRAC * h))

    mask[
        ly1:ly2,
        lx1:lx2
    ] = 0

    # bottom-right annotation
    rx1 = int(round(RIGHT_X1_FRAC * w))
    rx2 = int(round(RIGHT_X2_FRAC * w))
    ry1 = int(round(RIGHT_Y1_FRAC * h))
    ry2 = int(round(RIGHT_Y2_FRAC * h))

    mask[
        ry1:ry2,
        rx1:rx2
    ] = 0

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    mask_path = os.path.join(
        out_dir,
        "02_valid_region_mask.png"
    )

    cv2.imwrite(
        mask_path,
        mask
    )

    # --------------------------------------------------------
    # Save analysis image without changing original texture.
    # Annotation pixels are simply set to black in this copy.
    # Use raw image itself for OrientationJ if you prefer;
    # the mask is what defines the valid region in Python.
    # --------------------------------------------------------

    masked_bgr = img_bgr.copy()
    masked_bgr[mask == 0] = 0

    cv2.imwrite(
        os.path.join(
            out_dir,
            "01_masked_analysis_copy.png"
        ),
        masked_bgr
    )

    # --------------------------------------------------------
    # QC overlay
    # --------------------------------------------------------

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(
        figsize=(12, 9)
    )

    plt.imshow(
        img_rgb
    )

    # draw excluded rectangles
    rects = [
        (lx1, ly1, lx2-lx1, ly2-ly1),
        (rx1, ry1, rx2-rx1, ry2-ry1)
    ]

    for x, y, ww, hh in rects:
        plt.gca().add_patch(
            plt.Rectangle(
                (x, y),
                ww,
                hh,
                fill=False,
                linewidth=2
            )
        )

    plt.title(
        f"{name} — annotation-mask QC\n"
        f"10 µm = 157 px; "
        f"{PIXEL_SIZE_UM:.6f} µm/px"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "03_mask_QC_overlay.png"
        ),
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    valid_pixels = int(
        np.sum(mask > 0)
    )

    total_pixels = int(
        h * w
    )

    excluded_pixels = (
        total_pixels
        - valid_pixels
    )

    valid_fraction = (
        valid_pixels
        / total_pixels
    )

    valid_area_um2 = (
        valid_pixels
        * PIXEL_AREA_UM2
    )

    summary_rows.append({
        "FOV": name,
        "width_px": w,
        "height_px": h,
        "scale_bar_um": SCALE_BAR_UM,
        "scale_bar_px": SCALE_BAR_PX,
        "pixel_size_um": PIXEL_SIZE_UM,
        "pixel_area_um2": PIXEL_AREA_UM2,
        "valid_pixels": valid_pixels,
        "excluded_annotation_pixels": excluded_pixels,
        "valid_fraction": valid_fraction,
        "valid_area_um2": valid_area_um2,
        "left_mask_x1": lx1,
        "left_mask_y1": ly1,
        "left_mask_x2": lx2,
        "left_mask_y2": ly2,
        "right_mask_x1": rx1,
        "right_mask_y1": ry1,
        "right_mask_x2": rx2,
        "right_mask_y2": ry2,
    })

# ============================================================
# 5. SAVE GROUP SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "STEP1_50_50_4FOV_mask_calibration_summary.csv"
    ),
    index=False
)

print("\n" + "="*88)
print("STEP 1 COMPLETE — 50:50 + NaOH, 4 FOVs")
print("="*88)

print(
    f"\nCalibration: "
    f"10 µm = 157 px = "
    f"{PIXEL_SIZE_UM:.8f} µm/px"
)

print(
    "\nAll original LC pixels are retained except "
    "the two microscope-annotation regions."
)

print(
    "\nBefore proceeding to OrientationJ, inspect each:"
)

print(
    "03_mask_QC_overlay.png"
)

print(
    "\nIf the rectangles cover only the magnification text and "
    "scale-bar/label box, Step 1 is accepted."
)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_FOLDER
)
