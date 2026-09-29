
# ============================================================
# T1 + T2 + T3 RADIAL INTENSITY — SINGLE COMPARISON PLOT
# JUPYTER / VS CODE FRIENDLY
#
# Reads the radial-intensity CSV files generated in STEP 2.
# If only one ROI exists per texture -> plots that curve.
# If multiple ROIs exist per texture -> interpolates them onto
# a common radius axis and plots mean ± SD.
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER INPUT
# ============================================================

STEP2_DIR = r"D:\New results\texture_specific_study\STEP2_INTENSITY_ANALYSIS"

OUTPUT_DIR = r"D:\New results\texture_specific_study\STEP8_COMBINED_INTENSITY"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Normalize each curve to its own maximum?
# False = original POM intensity (recommended first)
# True  = compare profile shapes independent of brightness
NORMALIZE_INTENSITY = False

# Common radial spacing for averaging multiple ROIs
DR_UM = 0.20

DPI = 600

TEXTURES = {
    "T1": "T1 – Orange/brown circular",
    "T2": "T2 – Green curved",
    "T3": "T3 – Yellow-green four-brush"
}

# ============================================================
# 2. FIND RADIAL CSV FILES AUTOMATICALLY
# ============================================================

def find_radial_files(texture_code):

    patterns = [
        os.path.join(
            STEP2_DIR,
            "**",
            f"*{texture_code}*RADIAL_INTENSITY.csv"
        ),
        os.path.join(
            STEP2_DIR,
            "**",
            "*RADIAL_INTENSITY.csv"
        )
    ]

    files = []

    for pattern in patterns:
        found = glob.glob(
            pattern,
            recursive=True
        )

        for f in found:
            basename = os.path.basename(f)

            if texture_code.upper() in basename.upper():
                files.append(f)

    return sorted(
        list(set(files))
    )


# ============================================================
# 3. READ ONE RADIAL PROFILE
# ============================================================

def read_profile(path):

    df = pd.read_csv(path)

    if "r_um" not in df.columns:
        raise KeyError(
            f"'r_um' missing in:\n{path}\n"
            f"Columns: {list(df.columns)}"
        )

    if "mean_intensity" not in df.columns:
        raise KeyError(
            f"'mean_intensity' missing in:\n{path}\n"
            f"Columns: {list(df.columns)}"
        )

    r = pd.to_numeric(
        df["r_um"],
        errors="coerce"
    ).to_numpy(float)

    I = pd.to_numeric(
        df["mean_intensity"],
        errors="coerce"
    ).to_numpy(float)

    valid = (
        np.isfinite(r)
        &
        np.isfinite(I)
    )

    r = r[valid]
    I = I[valid]

    order = np.argsort(r)

    r = r[order]
    I = I[order]

    if NORMALIZE_INTENSITY:

        if np.nanmax(I) > 0:
            I = I / np.nanmax(I)

    return r, I


# ============================================================
# 4. COMBINE MULTIPLE ROIs OF SAME TEXTURE
# ============================================================

def combine_profiles(files):

    profiles = [
        read_profile(f)
        for f in files
    ]

    if len(profiles) == 1:

        r, I = profiles[0]

        return (
            r,
            I,
            np.zeros_like(I),
            1
        )

    # only compare radius range available to all ROIs
    common_max = min(
        np.nanmax(r)
        for r, I in profiles
    )

    common_r = np.arange(
        0,
        common_max + DR_UM / 2,
        DR_UM
    )

    interpolated = []

    for r, I in profiles:

        # avoid extrapolation
        Ii = np.interp(
            common_r,
            r,
            I
        )

        interpolated.append(
            Ii
        )

    arr = np.vstack(
        interpolated
    )

    mean_I = np.nanmean(
        arr,
        axis=0
    )

    sd_I = np.nanstd(
        arr,
        axis=0,
        ddof=1
    )

    return (
        common_r,
        mean_I,
        sd_I,
        len(files)
    )


# ============================================================
# 5. LOAD T1 / T2 / T3
# ============================================================

combined = {}

for texture_code in TEXTURES:

    files = find_radial_files(
        texture_code
    )

    print(
        f"\n{texture_code}: found {len(files)} radial profile(s)"
    )

    for f in files:
        print("   ", f)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No STEP-2 radial-intensity CSV found for {texture_code}.\n"
            f"Check STEP2_DIR:\n{STEP2_DIR}"
        )

    r, mean_I, sd_I, n = combine_profiles(
        files
    )

    combined[
        texture_code
    ] = {
        "r": r,
        "mean": mean_I,
        "sd": sd_I,
        "N": n,
        "files": files
    }


# ============================================================
# 6. SINGLE OVERLAY PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.5)
)

for texture_code, label in TEXTURES.items():

    d = combined[
        texture_code
    ]

    line, = ax.plot(
        d["r"],
        d["mean"],
        linewidth=2.0,
        label=(
            f"{label} (n={d['N']})"
            if d["N"] > 1
            else label
        )
    )

    # SD band only when more than one ROI exists
    if d["N"] > 1:

        ax.fill_between(
            d["r"],
            d["mean"] - d["sd"],
            d["mean"] + d["sd"],
            alpha=0.18,
            color=line.get_color(),
            linewidth=0
        )


ax.set_xlabel(
    r"Radius, $r$ ($\mu$m)",
    fontsize=12
)

if NORMALIZE_INTENSITY:

    ax.set_ylabel(
        "Normalized POM intensity",
        fontsize=12
    )

else:

    ax.set_ylabel(
        "Mean POM intensity (a.u.)",
        fontsize=12
    )

ax.set_title(
    "Radial intensity profiles of three optical textures",
    fontsize=13
)

ax.legend(
    frameon=False,
    fontsize=9
)

ax.tick_params(
    direction="in",
    top=True,
    right=True
)

plt.tight_layout()


# ============================================================
# 7. SAVE FIGURE
# ============================================================

suffix = (
    "NORMALIZED"
    if NORMALIZE_INTENSITY
    else "RAW"
)

png_path = os.path.join(
    OUTPUT_DIR,
    f"T1_T2_T3_RADIAL_INTENSITY_{suffix}.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    f"T1_T2_T3_RADIAL_INTENSITY_{suffix}.pdf"
)

fig.savefig(
    png_path,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.show()

plt.close(fig)


# ============================================================
# 8. SAVE COMBINED NUMERICAL DATA
# ============================================================

all_r = sorted(
    set(
        np.round(
            np.concatenate(
                [
                    combined[t]["r"]
                    for t in TEXTURES
                ]
            ),
            6
        )
    )
)

out = pd.DataFrame({
    "r_um": all_r
})

for texture_code in TEXTURES:

    d = combined[
        texture_code
    ]

    out[
        f"{texture_code}_mean_intensity"
    ] = np.interp(
        out["r_um"],
        d["r"],
        d["mean"],
        left=np.nan,
        right=np.nan
    )

    if d["N"] > 1:

        out[
            f"{texture_code}_SD"
        ] = np.interp(
            out["r_um"],
            d["r"],
            d["sd"],
            left=np.nan,
            right=np.nan
        )


csv_path = os.path.join(
    OUTPUT_DIR,
    f"T1_T2_T3_RADIAL_INTENSITY_{suffix}.csv"
)

out.to_csv(
    csv_path,
    index=False
)


print("\n====================================")
print("COMBINED INTENSITY PLOT COMPLETE")
print("====================================")

print("\nPNG:")
print(png_path)

print("\nPDF:")
print(pdf_path)

print("\nCombined numerical data:")
print(csv_path)
