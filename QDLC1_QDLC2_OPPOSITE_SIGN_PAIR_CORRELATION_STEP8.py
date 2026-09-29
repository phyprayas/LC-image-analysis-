
# ============================================================
# OPPOSITE-SIGN PAIR CORRELATION g_+-(r)
# QDLC1 (30:70) vs QDLC2 (50:50)
# Step 8 - Experimental pair-correlation + null envelope
#
# Same manuscript style as previous Order-Recovery figure:
# - QDLC1 blue
# - QDLC2 red
# - null/random envelope in gray
# - full boundary box
# - panel label (f)
# - PNG + PDF + SVG
#
# Recommended input:
# FOUR Step-8 g_+-(r) profile CSVs for each sample.
# Each CSV should contain radius and opposite-sign pair-correlation.
#
# If each CSV also contains null lower/upper bands, the code will use them.
# If not, it will still plot the experimental curves and y=1 null baseline.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# 1. USER INPUT
# ============================================================

QDLC1_FILES = [
    r"D:\New results\3070\step 8 defect pair correlation\FOV1_PAIR_CORRELATION.csv",
    r"D:\New results\3070\step 8 defect pair correlation\FOV2_PAIR_CORRELATION.csv",
    r"D:\New results\3070\step 8 defect pair correlation\FOV3_PAIR_CORRELATION.csv",
    r"D:\New results\3070\step 8 defect pair correlation\FOV4_PAIR_CORRELATION.csv",
]

QDLC2_FILES = [
    r"D:\New results\5050\step 8 defect pair correlation\FOV1_PAIR_CORRELATION.csv",
    r"D:\New results\5050\step 8 defect pair correlation\FOV2_PAIR_CORRELATION.csv",
    r"D:\New results\5050\step 8 defect pair correlation\FOV3_PAIR_CORRELATION.csv",
    r"D:\New results\5050\step 8 defect pair correlation\FOV4_PAIR_CORRELATION.csv",
]

OUTPUT_DIR = r"D:\New results\Figure_2_Defect_Analysis_Pretty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_RADIUS_UM = 20.0
DR_UM = 0.25
MIN_FOV_PER_RADIUS = 2

# Experimental composition spread
ERROR_BAND = "SD"     # "SD" or "SEM"

# Colors
QDLC1_COLOR = "#1565C0"
QDLC2_COLOR = "#E31A1C"
QDLC1_FILL = "#90CAF9"
QDLC2_FILL = "#EF9A9A"

NULL_FILL = "#BDBDBD"
NULL_LINE = "#555555"

DPI = 600

# ============================================================
# 2. FLEXIBLE COLUMN ALIASES
# ============================================================

RADIUS_ALIASES = [
    "r_center_um",
    "r_um",
    "radius_um",
    "radius",
    "r",
    "distance_um",
    "Distance_um",
]

# opposite-sign pair correlation g_+-(r)
GPM_ALIASES = [
    "g_pm",
    "g_plusminus",
    "g_plus_minus",
    "g_opposite",
    "g_opposite_sign",
    "g_unlike",
    "g_unlike_sign",
    "g+-",
    "g_+-",
    "g_pm_mean",
    "g_plusminus_mean",
    "g_opposite_mean",
    "g_unlike_mean",
    "mean_g_pm",
    "mean_g_opposite",
]

# Null/random lower envelope
NULL_LOW_ALIASES = [
    "null_low",
    "null_lower",
    "null_lower_95",
    "null_lo",
    "random_low",
    "random_lower",
    "random_lower_95",
    "null_q025",
    "null_2.5pct",
]

# Null/random upper envelope
NULL_HIGH_ALIASES = [
    "null_high",
    "null_upper",
    "null_upper_95",
    "null_hi",
    "random_high",
    "random_upper",
    "random_upper_95",
    "null_q975",
    "null_97.5pct",
]

# Null/random mean
NULL_MEAN_ALIASES = [
    "null_mean",
    "random_mean",
    "shuffle_mean",
    "null_g_mean",
    "random_g_mean",
]

# ============================================================
# 3. HELPERS
# ============================================================

def find_optional_column(df, aliases):
    for c in aliases:
        if c in df.columns:
            return c

    lower_map = {str(c).lower(): c for c in df.columns}

    for c in aliases:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def find_required_column(df, aliases, label):
    c = find_optional_column(df, aliases)

    if c is None:
        raise KeyError(
            f"\nCould not find {label} column.\n"
            f"Tried: {aliases}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return c


def load_one_profile(path, fov_name, sample_name):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nFile not found:\n{path}\n\n"
            "Replace the example filename with the actual Step-8 g_+-(r) CSV."
        )

    df = pd.read_csv(path)

    rcol = find_required_column(df, RADIUS_ALIASES, "radius")
    gcol = find_required_column(df, GPM_ALIASES, "opposite-sign g_+-(r)")

    low_col = find_optional_column(df, NULL_LOW_ALIASES)
    high_col = find_optional_column(df, NULL_HIGH_ALIASES)
    null_mean_col = find_optional_column(df, NULL_MEAN_ALIASES)

    out = pd.DataFrame({
        "r_um": pd.to_numeric(df[rcol], errors="coerce"),
        "g_pm": pd.to_numeric(df[gcol], errors="coerce"),
    })

    if low_col is not None:
        out["null_low"] = pd.to_numeric(df[low_col], errors="coerce")

    if high_col is not None:
        out["null_high"] = pd.to_numeric(df[high_col], errors="coerce")

    if null_mean_col is not None:
        out["null_mean"] = pd.to_numeric(df[null_mean_col], errors="coerce")

    out = out.dropna(subset=["r_um", "g_pm"])

    out = out[
        (out["r_um"] >= 0)
        & (out["r_um"] <= MAX_RADIUS_UM)
        & np.isfinite(out["g_pm"])
    ].copy()

    out = out.sort_values("r_um")
    out["FOV"] = fov_name
    out["Sample"] = sample_name

    print(
        f"{sample_name} {fov_name}: {len(out)} points | "
        f"r={rcol} | g_pm={gcol} | "
        f"null_low={low_col} | null_high={high_col} | null_mean={null_mean_col}"
    )

    return out


def load_sample(file_list, sample_name):

    frames = []

    for i, path in enumerate(file_list, start=1):
        frames.append(
            load_one_profile(
                path,
                f"FOV{i}",
                sample_name
            )
        )

    return frames


def interpolate_column(df, common_r, column):

    arr = np.full(common_r.shape, np.nan)

    if column not in df.columns:
        return arr

    tmp = df[["r_um", column]].dropna()

    if len(tmp) < 2:
        return arr

    r = tmp["r_um"].to_numpy(float)
    y = tmp[column].to_numpy(float)

    valid = (
        (common_r >= np.min(r))
        & (common_r <= np.max(r))
    )

    arr[valid] = np.interp(
        common_r[valid],
        r,
        y
    )

    return arr


def aggregate_sample(profile_frames):

    common_r = np.arange(
        0,
        MAX_RADIUS_UM + DR_UM/2,
        DR_UM
    )

    g_curves = []
    null_low_curves = []
    null_high_curves = []
    null_mean_curves = []

    for df in profile_frames:

        g_curves.append(
            interpolate_column(df, common_r, "g_pm")
        )

        null_low_curves.append(
            interpolate_column(df, common_r, "null_low")
        )

        null_high_curves.append(
            interpolate_column(df, common_r, "null_high")
        )

        null_mean_curves.append(
            interpolate_column(df, common_r, "null_mean")
        )

    g_arr = np.vstack(g_curves)
    low_arr = np.vstack(null_low_curves)
    high_arr = np.vstack(null_high_curves)
    mean_null_arr = np.vstack(null_mean_curves)

    rows = []

    for j, rr in enumerate(common_r):

        gvals = g_arr[:, j]
        gvals = gvals[np.isfinite(gvals)]

        if len(gvals) < MIN_FOV_PER_RADIUS:
            continue

        g_mean = np.mean(gvals)
        g_sd = np.std(gvals, ddof=1) if len(gvals) > 1 else np.nan
        g_sem = g_sd / np.sqrt(len(gvals)) if len(gvals) > 1 else np.nan

        lowvals = low_arr[:, j]
        lowvals = lowvals[np.isfinite(lowvals)]

        highvals = high_arr[:, j]
        highvals = highvals[np.isfinite(highvals)]

        nullmeans = mean_null_arr[:, j]
        nullmeans = nullmeans[np.isfinite(nullmeans)]

        rows.append({
            "r_um": rr,
            "g_mean": g_mean,
            "g_SD": g_sd,
            "g_SEM": g_sem,
            "N_FOV": len(gvals),

            # Mean of per-FOV null envelopes, if present
            "null_low_mean": np.mean(lowvals) if len(lowvals) > 0 else np.nan,
            "null_high_mean": np.mean(highvals) if len(highvals) > 0 else np.nan,
            "null_mean": np.mean(nullmeans) if len(nullmeans) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. LOAD DATA
# ============================================================

qdlc1_frames = load_sample(
    QDLC1_FILES,
    "QDLC1"
)

qdlc2_frames = load_sample(
    QDLC2_FILES,
    "QDLC2"
)

curve1 = aggregate_sample(
    qdlc1_frames
)

curve2 = aggregate_sample(
    qdlc2_frames
)

# ============================================================
# 5. SAVE NUMERICAL SUMMARY
# ============================================================

curve1.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "QDLC1_GPM_MEAN_SD.csv"
    ),
    index=False
)

curve2.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "QDLC2_GPM_MEAN_SD.csv"
    ),
    index=False
)

# ============================================================
# 6. BUILD COMMON NULL ENVELOPE
# ============================================================

# We combine the available null envelopes from both compositions
# because the reference null is random spatial organization.
# If no null envelope columns are present, only y=1 is plotted.

null_frames = []

for df in [curve1, curve2]:

    temp = df[
        [
            "r_um",
            "null_low_mean",
            "null_high_mean",
            "null_mean"
        ]
    ].copy()

    null_frames.append(
        temp
    )

all_null = pd.concat(
    null_frames,
    ignore_index=True
)

null_summary = (
    all_null
    .groupby("r_um", as_index=False)
    .agg({
        "null_low_mean": "mean",
        "null_high_mean": "mean",
        "null_mean": "mean"
    })
)

# ============================================================
# 7. PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 5.2)
)

# ------------------------------------------------------------
# NULL / RANDOM ENVELOPE
# ------------------------------------------------------------

has_null_band = (
    null_summary["null_low_mean"].notna().any()
    and
    null_summary["null_high_mean"].notna().any()
)

if has_null_band:

    rn = null_summary["r_um"].to_numpy(float)
    low = null_summary["null_low_mean"].to_numpy(float)
    high = null_summary["null_high_mean"].to_numpy(float)

    valid = (
        np.isfinite(rn)
        & np.isfinite(low)
        & np.isfinite(high)
    )

    ax.fill_between(
        rn[valid],
        low[valid],
        high[valid],
        color=NULL_FILL,
        alpha=0.45,
        linewidth=0,
        zorder=1
    )

# theoretical/random baseline
ax.axhline(
    1.0,
    color=NULL_LINE,
    linestyle="--",
    linewidth=1.5,
    zorder=2
)

# ------------------------------------------------------------
# QDLC1
# ------------------------------------------------------------

r1 = curve1["r_um"].to_numpy(float)
g1 = curve1["g_mean"].to_numpy(float)

e1 = curve1[
    "g_SD"
    if ERROR_BAND.upper() == "SD"
    else "g_SEM"
].to_numpy(float)

# optional experimental SD/SEM band
ax.fill_between(
    r1,
    g1 - e1,
    g1 + e1,
    color=QDLC1_FILL,
    alpha=0.28,
    linewidth=0,
    zorder=2
)

ax.plot(
    r1,
    g1,
    color=QDLC1_COLOR,
    linewidth=2.5,
    label="QDLC1",
    zorder=4
)

# ------------------------------------------------------------
# QDLC2
# ------------------------------------------------------------

r2 = curve2["r_um"].to_numpy(float)
g2 = curve2["g_mean"].to_numpy(float)

e2 = curve2[
    "g_SD"
    if ERROR_BAND.upper() == "SD"
    else "g_SEM"
].to_numpy(float)

ax.fill_between(
    r2,
    g2 - e2,
    g2 + e2,
    color=QDLC2_FILL,
    alpha=0.26,
    linewidth=0,
    zorder=2
)

ax.plot(
    r2,
    g2,
    color=QDLC2_COLOR,
    linewidth=2.5,
    label="QDLC2",
    zorder=4
)

# ============================================================
# 8. STYLE
# ============================================================

ax.set_xlim(
    0,
    MAX_RADIUS_UM
)

# Auto y maximum with a sensible lower floor
finite_y = np.concatenate([
    g1[np.isfinite(g1)],
    g2[np.isfinite(g2)]
])

if len(finite_y) > 0:
    y_max = max(
        2.0,
        np.nanmax(finite_y) * 1.15
    )
else:
    y_max = 4.0

ax.set_ylim(
    0,
    y_max
)

ax.set_xlabel(
    r"Radius, $r$ ($\mu$m)",
    fontsize=13
)

ax.set_ylabel(
    r"$g_{+-}(r)$",
    fontsize=14
)

# custom legend
handles = [
    Line2D(
        [0],
        [0],
        color=QDLC1_COLOR,
        lw=2.5,
        label="QDLC1"
    ),
    Line2D(
        [0],
        [0],
        color=QDLC2_COLOR,
        lw=2.5,
        label="QDLC2"
    ),
    Line2D(
        [0],
        [0],
        color=NULL_LINE,
        lw=1.5,
        linestyle="--",
        label="Null (random)"
    )
]

ax.legend(
    handles=handles,
    frameon=False,
    fontsize=11,
    loc="upper right"
)

# Full border
for side in [
    "top",
    "right",
    "bottom",
    "left"
]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.5)

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.2,
    labelsize=11
)

ax.grid(False)

# panel label + title
ax.text(
    -0.13,
    1.06,
    "(f)",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold"
)

ax.text(
    0.00,
    1.035,
    r"Opposite-sign pair correlation $g_{+-}(r)$",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold"
)

plt.tight_layout()

# ============================================================
# 9. SAVE
# ============================================================

PNG_PATH = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_OPPOSITE_SIGN_PAIR_CORRELATION.png"
)

PDF_PATH = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_OPPOSITE_SIGN_PAIR_CORRELATION.pdf"
)

SVG_PATH = os.path.join(
    OUTPUT_DIR,
    "QDLC1_QDLC2_OPPOSITE_SIGN_PAIR_CORRELATION.svg"
)

fig.savefig(
    PNG_PATH,
    dpi=DPI,
    bbox_inches="tight"
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight"
)

fig.savefig(
    SVG_PATH,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\n============================================")
print("OPPOSITE-SIGN PAIR-CORRELATION PLOT COMPLETE")
print("============================================")
print("QDLC1 = 30:70")
print("QDLC2 = 50:50")
print("Experimental band =", ERROR_BAND, "across FOVs")
print("Null envelope detected =", has_null_band)
print("\nSaved:")
print(PNG_PATH)
print(PDF_PATH)
print(SVG_PATH)
