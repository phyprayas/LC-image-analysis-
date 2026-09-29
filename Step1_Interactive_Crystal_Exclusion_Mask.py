
# ============================================================
# STEP 1 — INTERACTIVE CRYSTAL / BRIGHT-SOLID EXCLUSION MASK
# For 50:50 CPCL:Glycerol + NaOH POM image
#
# PURPOSE
# -------
# Create an exclusion mask BEFORE OrientationJ analysis.
# The script automatically proposes bright, low-saturation objects,
# then lets you click individual candidate objects to include/exclude
# them from the final mask.
#
# IMPORTANT
# ---------
# Do NOT inpaint/delete crystals from the image.
# Keep the raw image unchanged and use the saved binary mask later
# to remove those pixels / OrientationJ grid points from analysis.
#
# OUTPUTS
# -------
# 01_candidate_mask.png
# 02_final_exclusion_mask.png
# 03_valid_LC_mask.png
# 04_mask_overlay.png
# 05_component_table.csv
# 06_mask_summary.txt
#
# CONTROLS
# --------
# Left mouse click : toggle a candidate object ON/OFF
# A                : select all candidates
# N                : deselect all candidates
# R                : reset to automatic selection
# S or ENTER       : save final masks and finish
# Q or ESC         : quit without saving
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog

# ============================================================
# 1. SETTINGS
# Keep these unchanged initially.
# ============================================================

V_MIN = 245          # HSV brightness threshold
S_MAX = 80           # HSV saturation threshold
MIN_AREA_PX = 120    # ignore tiny bright noise
OPEN_KERNEL = 3
CLOSE_KERNEL = 7
DILATE_PX = 7        # expand final exclusion slightly around object edges

# ============================================================
# 2. SELECT RAW IMAGE + OUTPUT FOLDER
# ============================================================

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select RAW 50:50 + NaOH POM image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
        ("All files", "*.*")
    ]
)

if not image_path:
    raise SystemExit("No image selected.")

output_dir = filedialog.askdirectory(
    title="Select output folder for crystal-mask results"
)

if not output_dir:
    raise SystemExit("No output folder selected.")

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 3. LOAD RAW IMAGE
# ============================================================

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not read image:\n{image_path}")

H_img, W_img = img.shape[:2]
raw = img.copy()

# ============================================================
# 4. AUTOMATIC CANDIDATE DETECTION
# Bright + low-saturation regions
# ============================================================

hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
_, sat, val = cv2.split(hsv)

candidate = ((val >= V_MIN) & (sat <= S_MAX)).astype(np.uint8) * 255

# Clean isolated pixel noise
k_open = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
k_close = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)

candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k_open)
candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close)

# Connected components
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    candidate, connectivity=8
)

# Keep components above minimum area
valid_component_ids = []
for cid in range(1, nlabels):
    area = stats[cid, cv2.CC_STAT_AREA]
    if area >= MIN_AREA_PX:
        valid_component_ids.append(cid)

# Build candidate-only mask
candidate_clean = np.zeros((H_img, W_img), dtype=np.uint8)
for cid in valid_component_ids:
    candidate_clean[labels == cid] = 255

# Save raw automatic candidate mask
cv2.imwrite(
    os.path.join(output_dir, "01_candidate_mask.png"),
    candidate_clean
)

# ============================================================
# 5. INTERACTIVE COMPONENT REVIEW
# Automatic default = all candidate components selected.
# Click any object to toggle it.
# ============================================================

selected = {cid: True for cid in valid_component_ids}
initial_selected = selected.copy()

WINDOW = "Crystal exclusion review - click object to toggle"

def create_display():
    disp = raw.copy()

    # semi-transparent red overlay for selected exclusions
    overlay = disp.copy()

    for cid in valid_component_ids:
        mask = (labels == cid)

        if selected[cid]:
            overlay[mask] = (0, 0, 255)

        # component boundary
        comp_mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        line_color = (0, 255, 0) if selected[cid] else (255, 255, 0)
        cv2.drawContours(overlay, contours, -1, line_color, 2)

        # ID number near centroid
        cx, cy = centroids[cid]
        cv2.putText(
            overlay,
            str(cid),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    disp = cv2.addWeighted(disp, 0.72, overlay, 0.28, 0)

    text1 = "CLICK = toggle | A = all | N = none | R = reset | S/ENTER = save | Q/ESC = quit"
    cv2.rectangle(disp, (5, 5), (W_img - 5, 42), (0, 0, 0), -1)
    cv2.putText(
        disp, text1, (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    nsel = sum(selected.values())
    text2 = f"Selected exclusions: {nsel}/{len(valid_component_ids)}"
    cv2.rectangle(disp, (5, 48), (410, 82), (0, 0, 0), -1)
    cv2.putText(
        disp, text2, (15, 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    return disp

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cid = int(labels[y, x])

        if cid in selected:
            selected[cid] = not selected[cid]

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1400, 900)
cv2.setMouseCallback(WINDOW, mouse_callback)

save_requested = False

while True:
    display = create_display()
    cv2.imshow(WINDOW, display)

    key = cv2.waitKey(30) & 0xFF

    # Save: S or Enter
    if key in [ord("s"), 13]:
        save_requested = True
        break

    # Select all
    elif key == ord("a"):
        for cid in selected:
            selected[cid] = True

    # Deselect all
    elif key == ord("n"):
        for cid in selected:
            selected[cid] = False

    # Reset
    elif key == ord("r"):
        selected = initial_selected.copy()

    # Quit
    elif key in [ord("q"), 27]:
        break

cv2.destroyAllWindows()

if not save_requested:
    raise SystemExit("Review closed without saving.")

# ============================================================
# 6. FINAL EXCLUSION MASK
# ============================================================

final_mask = np.zeros((H_img, W_img), dtype=np.uint8)

for cid, keep in selected.items():
    if keep:
        final_mask[labels == cid] = 255

# Dilate ONLY after manual selection to remove bright object edges
if DILATE_PX > 0:
    k = np.ones((DILATE_PX, DILATE_PX), np.uint8)
    final_mask = cv2.dilate(final_mask, k, iterations=1)

# Valid LC mask: white = usable LC texture
valid_lc_mask = cv2.bitwise_not(final_mask)

# ============================================================
# 7. OVERLAY FOR VISUAL VERIFICATION
# ============================================================

overlay = raw.copy()
red_layer = np.zeros_like(raw)
red_layer[:, :, 2] = 255

alpha = 0.40
mask_bool = final_mask > 0

overlay[mask_bool] = cv2.addWeighted(
    raw[mask_bool], 1.0 - alpha,
    red_layer[mask_bool], alpha,
    0
)

# Draw exclusion boundary
contours, _ = cv2.findContours(
    final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

final_mask_path = os.path.join(output_dir, "02_final_exclusion_mask.png")
valid_mask_path = os.path.join(output_dir, "03_valid_LC_mask.png")
overlay_path = os.path.join(output_dir, "04_mask_overlay.png")

cv2.imwrite(final_mask_path, final_mask)
cv2.imwrite(valid_mask_path, valid_lc_mask)
cv2.imwrite(overlay_path, overlay)

# ============================================================
# 9. COMPONENT TABLE
# ============================================================

rows = []

for cid in valid_component_ids:
    x = stats[cid, cv2.CC_STAT_LEFT]
    y = stats[cid, cv2.CC_STAT_TOP]
    w = stats[cid, cv2.CC_STAT_WIDTH]
    h = stats[cid, cv2.CC_STAT_HEIGHT]
    area = stats[cid, cv2.CC_STAT_AREA]
    cx, cy = centroids[cid]

    rows.append({
        "component_id": cid,
        "selected_for_exclusion": bool(selected[cid]),
        "area_px_before_dilation": int(area),
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_width": int(w),
        "bbox_height": int(h),
        "centroid_x": float(cx),
        "centroid_y": float(cy)
    })

component_df = pd.DataFrame(rows)
component_csv = os.path.join(output_dir, "05_component_table.csv")
component_df.to_csv(component_csv, index=False)

# ============================================================
# 10. SUMMARY
# ============================================================

excluded_px = int(np.count_nonzero(final_mask))
total_px = int(H_img * W_img)
valid_px = total_px - excluded_px

excluded_fraction = excluded_px / total_px
valid_fraction = valid_px / total_px

summary = f"""
STEP 1 — Crystal / bright-solid exclusion mask

Input image:
{image_path}

Image dimensions:
Width  = {W_img} px
Height = {H_img} px

Automatic candidate settings:
V_MIN       = {V_MIN}
S_MAX       = {S_MAX}
MIN_AREA_PX = {MIN_AREA_PX}
DILATE_PX   = {DILATE_PX}

Candidate connected components:
{len(valid_component_ids)}

Final selected components:
{sum(selected.values())}

Excluded area:
{excluded_px} px
{100 * excluded_fraction:.3f} %

Valid LC area:
{valid_px} px
{100 * valid_fraction:.3f} %

IMPORTANT:
Use 03_valid_LC_mask.png as the spatial validity mask in later Python analysis.
Do not calculate defect density using the full image area.
Use the valid LC area after masking.

Files:
01_candidate_mask.png
02_final_exclusion_mask.png
03_valid_LC_mask.png
04_mask_overlay.png
05_component_table.csv
"""

summary_path = os.path.join(output_dir, "06_mask_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print("\nSTEP 1 completed successfully.")
print("FIRST check: 04_mask_overlay.png")
print("Red regions should contain ONLY objects you want excluded.")
