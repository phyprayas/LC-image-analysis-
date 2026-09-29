
# ============================================================
# PSI STUDY — STEP 9
# MANUSCRIPT-READY FIGURE ASSEMBLY
# FORCE FOV2 FOR BOTH 30:70 AND 50:50
# JUPYTER / VS CODE FRIENDLY
#
# This version DOES NOT use Step-8 selected labels.
# It manually forces:
#
#   30:70 -> 30_70_FOV2
#   50:50 -> 50_50_FOV2
#
# The same FOV2 label is then used for:
#   - raw POM image
#   - Psi(x,y) map
#   - |grad Psi| map
#
# File search is recursive and supports:
#   .png .jpg .jpeg .tif .tiff .bmp
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 1. USER INPUT
# ============================================================

# ------------------------------------------------------------
# MANUALLY FORCED FOVs
# ------------------------------------------------------------
label_3070 = "30_70_FOV2"
label_5050 = "50_50_FOV2"

# ------------------------------------------------------------
# ROOT FOLDERS
# ------------------------------------------------------------

RAW_POM_FOLDER = r"D:\New results\POM Images"

STEP1_FOLDER = r"D:\New results\psi map results\New folder\step 1"

STEP2_FOLDER = r"D:\New results\psi map results\New folder\step 2"

STEP7_FOLDER = r"D:\New results\psi map results\New folder\step 7"

OUTPUT_DIR = r"D:\New results\psi map results\New folder\step 9_FOV2"

# ------------------------------------------------------------
# STEP-7 QUANTITATIVE FILES
# ------------------------------------------------------------

STEP7_MASTER = os.path.join(
    STEP7_FOLDER,
    "PSI_STEP7_MASTER_PER_FOV.csv"
)

STEP7_COMPACT = os.path.join(
    STEP7_FOLDER,
    "PSI_STEP7_COMPACT_MANUSCRIPT_METRICS.csv"
)

STEP7_MEAN_GRAD_PLOT = os.path.join(
    STEP7_FOLDER,
    "FIG_STEP7_mean_gradPsi.png"
)

STEP7_CORE_FAR_PLOT = os.path.join(
    STEP7_FOLDER,
    "FIG_STEP7_core_vs_far_gradPsi.png"
)

DPI = 600

# ============================================================
# 2. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(
        path,
        exist_ok=True
    )


def read_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            path
        )

    return np.asarray(
        Image.open(path)
    )


def auto_find_file(
    root_folder,
    label,
    required_words=None,
    exclude_words=None
):
    """
    Recursively search for image files containing the FOV label.

    Parameters
    ----------
    root_folder : str
        Root directory.

    label : str
        Example: "30_70_FOV2"

    required_words : list[str] or None
        All listed words must appear in filename.

    exclude_words : list[str] or None
        None of these words may appear in filename.
    """

    if not os.path.isdir(
        root_folder
    ):
        raise NotADirectoryError(
            f"Folder not found:\n{root_folder}"
        )

    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tif",
        "*.tiff",
        "*.bmp"
    ]

    candidates = []

    for ext in extensions:

        candidates.extend(
            glob.glob(
                os.path.join(
                    root_folder,
                    "**",
                    ext
                ),
                recursive=True
            )
        )

    label_low = label.lower()

    matches = []

    for path in candidates:

        name = os.path.basename(
            path
        ).lower()

        if label_low not in name:
            continue

        if required_words:

            if not all(
                word.lower() in name
                for word in required_words
            ):
                continue

        if exclude_words:

            if any(
                word.lower() in name
                for word in exclude_words
            ):
                continue

        matches.append(
            path
        )

    if len(matches) == 0:

        near = [
            p
            for p in candidates
            if "fov2" in os.path.basename(
                p
            ).lower()
        ]

        message = (
            f"\nCould not automatically find image.\n"
            f"Label: {label}\n"
            f"Folder: {root_folder}\n"
            f"Required words: {required_words}\n"
            f"Excluded words: {exclude_words}\n"
        )

        if near:

            message += (
                "\nSome FOV2-related files found:\n"
            )

            for p in near[:30]:
                message += (
                    "   "
                    + p
                    + "\n"
                )

        raise FileNotFoundError(
            message
        )

    # --------------------------------------------------------
    # Sort matching files by:
    # 1) shortest filename
    # 2) alphabetical
    # --------------------------------------------------------

    matches = sorted(
        matches,
        key=lambda p: (
            len(
                os.path.basename(
                    p
                )
            ),
            os.path.basename(
                p
            ).lower()
        )
    )

    chosen = matches[0]

    if len(matches) > 1:

        print(
            f"\nMultiple matches found for {label}:"
        )

        for p in matches:
            print(
                "   ",
                p
            )

        print(
            "Using:",
            chosen
        )

    return chosen


def add_panel_label(
    ax,
    text
):
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="none",
            pad=2
        )
    )


def show_image_panel(
    ax,
    image,
    title,
    label
):
    ax.imshow(
        image
    )

    ax.set_title(
        title,
        fontsize=11
    )

    ax.axis(
        "off"
    )

    add_panel_label(
        ax,
        label
    )


# ============================================================
# 3. CONFIRM FORCED FOV LABELS
# ============================================================

ensure_dir(
    OUTPUT_DIR
)

print(
    "FORCED REPRESENTATIVE FOVs"
)

print(
    "30:70 ->",
    label_3070
)

print(
    "50:50 ->",
    label_5050
)

# ============================================================
# 4. AUTO-FIND RAW POM FILES
# ============================================================
#
# Important:
# Exclude Psi / gradient maps from raw-image search.
# ============================================================

raw_3070_path = auto_find_file(
    RAW_POM_FOLDER,
    label_3070,
    required_words=None,
    exclude_words=[
        "psi",
        "grad",
        "gradient"
    ]
)

raw_5050_path = auto_find_file(
    RAW_POM_FOLDER,
    label_5050,
    required_words=None,
    exclude_words=[
        "psi",
        "grad",
        "gradient"
    ]
)

print(
    "\nRAW POM FILES SELECTED"
)

print(
    "30:70 ->",
    raw_3070_path
)

print(
    "50:50 ->",
    raw_5050_path
)

# ============================================================
# 5. AUTO-FIND PSI MAP FILES
# ============================================================

psi_3070_path = auto_find_file(
    STEP1_FOLDER,
    label_3070,
    required_words=[
        "psi"
    ]
)

psi_5050_path = auto_find_file(
    STEP1_FOLDER,
    label_5050,
    required_words=[
        "psi"
    ]
)

print(
    "\nPSI MAP FILES SELECTED"
)

print(
    "30:70 ->",
    psi_3070_path
)

print(
    "50:50 ->",
    psi_5050_path
)

# ============================================================
# 6. AUTO-FIND GRADIENT MAP FILES
# ============================================================
#
# First try filenames containing "gradpsi".
# If your filenames use "gradient" instead,
# fallback search is used automatically.
# ============================================================

def find_gradient_file(
    root_folder,
    label
):

    try:

        return auto_find_file(
            root_folder,
            label,
            required_words=[
                "gradpsi"
            ]
        )

    except FileNotFoundError:

        try:

            return auto_find_file(
                root_folder,
                label,
                required_words=[
                    "gradient"
                ]
            )

        except FileNotFoundError:

            return auto_find_file(
                root_folder,
                label,
                required_words=[
                    "grad"
                ]
            )


grad_3070_path = find_gradient_file(
    STEP2_FOLDER,
    label_3070
)

grad_5050_path = find_gradient_file(
    STEP2_FOLDER,
    label_5050
)

print(
    "\nGRADIENT MAP FILES SELECTED"
)

print(
    "30:70 ->",
    grad_3070_path
)

print(
    "50:50 ->",
    grad_5050_path
)

# ============================================================
# 7. LOAD ALL IMAGE PANELS
# ============================================================

raw_3070 = read_image(
    raw_3070_path
)

raw_5050 = read_image(
    raw_5050_path
)

psi_3070 = read_image(
    psi_3070_path
)

psi_5050 = read_image(
    psi_5050_path
)

grad_3070 = read_image(
    grad_3070_path
)

grad_5050 = read_image(
    grad_5050_path
)

# ------------------------------------------------------------
# Quantitative panels from Step 7
# ------------------------------------------------------------

mean_grad_plot = (
    read_image(
        STEP7_MEAN_GRAD_PLOT
    )
    if os.path.isfile(
        STEP7_MEAN_GRAD_PLOT
    )
    else None
)

core_far_plot = (
    read_image(
        STEP7_CORE_FAR_PLOT
    )
    if os.path.isfile(
        STEP7_CORE_FAR_PLOT
    )
    else None
)

# ============================================================
# 8. BUILD MAIN 8-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(
        16,
        8.5
    )
)

# ------------------------------------------------------------
# PANEL a
# ------------------------------------------------------------

show_image_panel(
    axes[0, 0],
    raw_3070,
    f"30:70 raw POM\n{label_3070}",
    "a"
)

# ------------------------------------------------------------
# PANEL b
# ------------------------------------------------------------

show_image_panel(
    axes[0, 1],
    raw_5050,
    f"50:50 raw POM\n{label_5050}",
    "b"
)

# ------------------------------------------------------------
# PANEL c
# ------------------------------------------------------------

show_image_panel(
    axes[0, 2],
    psi_3070,
    r"30:70 $\Psi(x,y)$",
    "c"
)

# ------------------------------------------------------------
# PANEL d
# ------------------------------------------------------------

show_image_panel(
    axes[0, 3],
    psi_5050,
    r"50:50 $\Psi(x,y)$",
    "d"
)

# ------------------------------------------------------------
# PANEL e
# ------------------------------------------------------------

show_image_panel(
    axes[1, 0],
    grad_3070,
    r"30:70 $|\nabla\Psi|$",
    "e"
)

# ------------------------------------------------------------
# PANEL f
# ------------------------------------------------------------

show_image_panel(
    axes[1, 1],
    grad_5050,
    r"50:50 $|\nabla\Psi|$",
    "f"
)

# ------------------------------------------------------------
# PANEL g
# ------------------------------------------------------------

if mean_grad_plot is not None:

    show_image_panel(
        axes[1, 2],
        mean_grad_plot,
        "Whole-field distortion comparison",
        "g"
    )

else:

    axes[1, 2].axis(
        "off"
    )

    add_panel_label(
        axes[1, 2],
        "g"
    )

    axes[1, 2].text(
        0.5,
        0.5,
        "Step-7 mean-gradient plot not found",
        transform=axes[1, 2].transAxes,
        ha="center",
        va="center"
    )

# ------------------------------------------------------------
# PANEL h
# ------------------------------------------------------------

if core_far_plot is not None:

    show_image_panel(
        axes[1, 3],
        core_far_plot,
        "Defect core vs far-field distortion",
        "h"
    )

else:

    axes[1, 3].axis(
        "off"
    )

    add_panel_label(
        axes[1, 3],
        "h"
    )

    axes[1, 3].text(
        0.5,
        0.5,
        "Step-7 core/far plot not found",
        transform=axes[1, 3].transAxes,
        ha="center",
        va="center"
    )

# ============================================================
# 9. SAVE FIGURE
# ============================================================

plt.tight_layout()

png = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP9_FOV2_MAIN_MANUSCRIPT_FIGURE.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP9_FOV2_MAIN_MANUSCRIPT_FIGURE.pdf"
)

fig.savefig(
    png,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    pdf,
    bbox_inches="tight"
)

plt.show()

plt.close(
    fig
)

# ============================================================
# 10. SAVE FILE-SELECTION RECORD
# ============================================================

selection_record = pd.DataFrame([
    {
        "Composition": "30:70",
        "Forced_FOV": label_3070,
        "Raw_POM": raw_3070_path,
        "Psi_map": psi_3070_path,
        "Gradient_map": grad_3070_path
    },
    {
        "Composition": "50:50",
        "Forced_FOV": label_5050,
        "Raw_POM": raw_5050_path,
        "Psi_map": psi_5050_path,
        "Gradient_map": grad_5050_path
    }
])

selection_csv = os.path.join(
    OUTPUT_DIR,
    "PSI_STEP9_FOV2_FILES_USED.csv"
)

selection_record.to_csv(
    selection_csv,
    index=False
)

# ============================================================
# 11. FINISH
# ============================================================

print(
    "\nSTEP 9 — FOV2 VERSION COMPLETE"
)

print(
    "\n30:70 FOV used:",
    label_3070
)

print(
    "50:50 FOV used:",
    label_5050
)

print(
    "\nPNG:",
    png
)

print(
    "PDF:",
    pdf
)

print(
    "Files-used record:",
    selection_csv
)
