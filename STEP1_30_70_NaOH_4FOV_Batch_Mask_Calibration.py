
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

SCALE_BAR_UM = 10.0
SCALE_BAR_PX = 157.0
UM_PER_PX = SCALE_BAR_UM / SCALE_BAR_PX
PX_PER_UM = 1.0 / UM_PER_PX

root = Tk()
root.withdraw()
root.attributes("-topmost", True)

image_paths = filedialog.askopenfilenames(
    title="Select 4 raw POM images: 30:70 + NaOH",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
        ("All files", "*.*")
    ]
)

if len(image_paths) != 4:
    raise RuntimeError(f"Exactly 4 images are required. You selected {len(image_paths)}.")

output_root = filedialog.askdirectory(
    title="Select output folder for Step 1"
)

root.destroy()

if not output_root:
    raise RuntimeError("No output folder selected.")

step1_root = os.path.join(output_root, "STEP1_30_70_NaOH_4FOV")
os.makedirs(step1_root, exist_ok=True)

LEFT_X_FRAC_1, LEFT_X_FRAC_2 = 0.000, 0.210
LEFT_Y_FRAC_1, LEFT_Y_FRAC_2 = 0.935, 1.000
RIGHT_X_FRAC_1, RIGHT_X_FRAC_2 = 0.875, 1.000
RIGHT_Y_FRAC_1, RIGHT_Y_FRAC_2 = 0.900, 1.000

def frac_to_px(frac, size):
    return int(round(frac * size))

def process_image(image_path, sample_index):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image:\n{image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]
    sample_name = f"30_70_NaOH_FOV_{sample_index}"
    outdir = os.path.join(step1_root, sample_name)
    os.makedirs(outdir, exist_ok=True)

    valid_mask = np.full((H, W), 255, dtype=np.uint8)

    lx1 = frac_to_px(LEFT_X_FRAC_1, W)
    lx2 = frac_to_px(LEFT_X_FRAC_2, W)
    ly1 = frac_to_px(LEFT_Y_FRAC_1, H)
    ly2 = frac_to_px(LEFT_Y_FRAC_2, H)
    rx1 = frac_to_px(RIGHT_X_FRAC_1, W)
    rx2 = frac_to_px(RIGHT_X_FRAC_2, W)
    ry1 = frac_to_px(RIGHT_Y_FRAC_1, H)
    ry2 = frac_to_px(RIGHT_Y_FRAC_2, H)

    valid_mask[ly1:ly2, lx1:lx2] = 0
    valid_mask[ry1:ry2, rx1:rx2] = 0

    total_px = int(H * W)
    valid_px = int(np.sum(valid_mask > 0))
    excluded_px = total_px - valid_px
    valid_fraction = valid_px / total_px
    pixel_area_um2 = UM_PER_PX ** 2
    valid_area_um2 = valid_px * pixel_area_um2

    overlay = img_rgb.copy()
    invalid = (valid_mask == 0)
    red = np.array([255, 0, 0], dtype=np.uint8)
    overlay[invalid] = (0.30 * overlay[invalid] + 0.70 * red).astype(np.uint8)

    cv2.rectangle(overlay, (lx1, ly1), (lx2 - 1, ly2 - 1), (255, 255, 255), 2)
    cv2.rectangle(overlay, (rx1, ry1), (rx2 - 1, ry2 - 1), (255, 255, 255), 2)

    cv2.imwrite(os.path.join(outdir, "01_raw_image.png"), img_bgr)
    cv2.imwrite(os.path.join(outdir, "02_valid_region_mask.png"), valid_mask)
    cv2.imwrite(
        os.path.join(outdir, "03_annotation_mask_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    plt.figure(figsize=(12, 9))
    plt.imshow(overlay)
    plt.title(f"{sample_name}\nRed = excluded annotation | {UM_PER_PX:.5f} µm/px")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04_step1_diagnostic.png"), dpi=300, bbox_inches="tight")
    plt.close()

    row = {
        "sample": sample_name,
        "source_file": os.path.basename(image_path),
        "width_px": W,
        "height_px": H,
        "scale_bar_um": SCALE_BAR_UM,
        "scale_bar_px": SCALE_BAR_PX,
        "um_per_px": UM_PER_PX,
        "px_per_um": PX_PER_UM,
        "pixel_area_um2": pixel_area_um2,
        "total_area_px": total_px,
        "valid_area_px": valid_px,
        "excluded_area_px": excluded_px,
        "valid_fraction": valid_fraction,
        "valid_area_um2": valid_area_um2
    }

    pd.DataFrame([row]).to_csv(
        os.path.join(outdir, "step1_calibration_and_mask_summary.csv"),
        index=False
    )
    return row

rows = [process_image(path, i) for i, path in enumerate(image_paths, start=1)]
summary = pd.DataFrame(rows)

summary.to_csv(
    os.path.join(step1_root, "FINAL_STEP1_30_70_NaOH_4FOV_summary.csv"),
    index=False
)

with open(os.path.join(step1_root, "PHYSICAL_CALIBRATION.txt"), "w", encoding="utf-8") as f:
    f.write("30:70 + NaOH — physical calibration\n")
    f.write("=" * 50 + "\n")
    f.write(f"Scale bar = {SCALE_BAR_UM:.3f} µm\n")
    f.write(f"Scale bar length = {SCALE_BAR_PX:.3f} px\n")
    f.write(f"µm/px = {UM_PER_PX:.10f}\n")
    f.write(f"px/µm = {PX_PER_UM:.10f}\n")
    f.write(f"pixel area = {UM_PER_PX**2:.10f} µm²\n")

print("STEP 1 COMPLETE — 30:70 + NaOH, 4 FOVs")
print(f"1 px = {UM_PER_PX:.6f} µm")
print(step1_root)
