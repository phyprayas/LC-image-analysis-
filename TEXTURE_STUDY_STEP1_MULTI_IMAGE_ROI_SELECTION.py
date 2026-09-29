# ============================================================
# TEXTURE STUDY STEP 1 — MULTI-IMAGE ROI SELECTION
# T1/T2 may come from one FOV and T3 from another FOV.
# Calibration: 10 um / 157 px.
# ============================================================
import cv2
import os
import numpy as np
import pandas as pd

IMAGE_PATHS = {
    "T1": r"D:\New results\POM Images\3070\30_70_FOV4.jpg",
    "T2": r"D:\New results\POM Images\3070\30_70_FOV4.jpg",
    "T3": r"D:\New results\POM Images\5050\50_50_FOV3.jpg",
}
SAMPLE_INFO = {
    "T1": {"Sample": "QDLC1", "Composition": "30:70", "FOV": "FOV4"},
    "T2": {"Sample": "QDLC1", "Composition": "30:70", "FOV": "FOV4"},
    "T3": {"Sample": "QDLC2", "Composition": "50:50", "FOV": "FOV3"},
}
TEXTURES = {
    "T1": "orange_brown_circular",
    "T2": "green_curved",
    "T3": "yellow_green_four_brush",
}
OUTPUT_DIR = r"D:\New results\select texture\step 1"
N_ROI_PER_TEXTURE = 1
UM_PER_PX = 10.0 / 157.0
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

for texture_code, texture_name in TEXTURES.items():
    image_path = IMAGE_PATHS[texture_code]
    info = SAMPLE_INFO[texture_code]
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    print("\n========================================")
    print(f"Select {texture_code}: {texture_name}")
    print(f"Sample={info['Sample']}, Composition={info['Composition']}, FOV={info['FOV']}")
    print("Draw rectangle; ENTER/SPACE accepts; C cancels.")

    for roi_number in range(1, N_ROI_PER_TEXTURE + 1):
        x, y, w, h = cv2.selectROI(
            f"{texture_code} | {info['Sample']} | {info['FOV']}",
            image, showCrosshair=True, fromCenter=False
        )
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            print("ROI cancelled.")
            continue

        x, y, w, h = map(int, (x, y, w, h))
        crop = image[y:y+h, x:x+w].copy()

        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (max(1, w // 2 - 2), max(1, h // 2 - 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        masked = cv2.bitwise_and(crop, crop, mask=mask)

        roi_id = f"{texture_code}_{info['Sample']}_{info['FOV']}_ROI{roi_number:02d}"
        roi_dir = os.path.join(OUTPUT_DIR, roi_id)
        os.makedirs(roi_dir, exist_ok=True)

        crop_path = os.path.join(roi_dir, f"{roi_id}_crop.png")
        mask_path = os.path.join(roi_dir, f"{roi_id}_mask.png")
        masked_path = os.path.join(roi_dir, f"{roi_id}_masked.png")
        preview_path = os.path.join(roi_dir, f"{roi_id}_preview.png")

        cv2.imwrite(crop_path, crop)
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(masked_path, masked)
        cv2.imwrite(preview_path, np.hstack((crop, masked)))

        area_px = int(np.sum(mask > 0))
        results.append({
            "ROI_ID": roi_id,
            "Texture": texture_code,
            "Texture_name": texture_name,
            "Sample": info["Sample"],
            "Composition": info["Composition"],
            "FOV": info["FOV"],
            "source_image": image_path,
            "x_px": x, "y_px": y,
            "width_px": w, "height_px": h,
            "width_um": w * UM_PER_PX,
            "height_um": h * UM_PER_PX,
            "area_px": area_px,
            "area_um2": area_px * UM_PER_PX**2,
            "um_per_px": UM_PER_PX,
            "crop_path": crop_path,
            "mask_path": mask_path,
            "masked_crop_path": masked_path,
        })

        cv2.imshow(f"{roi_id}: crop | masked", np.hstack((crop, masked)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "TEXTURE_ROI_MASTER.csv")
df.to_csv(csv_path, index=False)
print("\nSTEP 1 COMPLETE")
print(df[["ROI_ID", "Texture", "Sample", "Composition", "FOV", "area_um2"]] if len(df) else df)
print("Master CSV:", csv_path)
