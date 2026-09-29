
# ============================================================
# STEP 8 — 50:50 + NaOH | 4 FOVs
# SIGN-RESOLVED DEFECT PAIR CORRELATION g(r)
# MATCHED TO 30:70 WORKFLOW
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist

# ============================================================
# 1. EDIT ONLY THESE PATHS
# ============================================================

DATASETS = [
    {
        "name": "50_50_NaOH_FOV1",
        "defects": r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV1\merged_defects.csv",
        "mask": r"C:\Users\Prayas Singh\Desktop\5050\STEP1B_BUFFERED_TEXTURE_MASKS\50_50_NaOH_FOV1\05_valid_texture_mask_buffered.png",
    },
    {
        "name": "50_50_NaOH_FOV2",
        "defects": r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV2\merged_defects.csv",
        "mask": r"C:\Users\Prayas Singh\Desktop\5050\STEP1B_BUFFERED_TEXTURE_MASKS\50_50_NaOH_FOV2\05_valid_texture_mask_buffered.png",
    },
    {
        "name": "50_50_NaOH_FOV3",
        "defects": r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV3\merged_defects.csv",
        "mask": r"C:\Users\Prayas Singh\Desktop\5050\STEP1B_BUFFERED_TEXTURE_MASKS\50_50_NaOH_FOV3\05_valid_texture_mask_buffered.png",
    },
    {
        "name": "50_50_NaOH_FOV4",
        "defects": r"C:\Users\Prayas Singh\Desktop\5050\STEP4_50_50_MERGED_DEFECTS\50_50_NaOH_FOV4\merged_defects.csv",
        "mask": r"C:\Users\Prayas Singh\Desktop\5050\STEP1B_BUFFERED_TEXTURE_MASKS\50_50_NaOH_FOV4\05_valid_texture_mask_buffered.png",
    },
]

OUTPUT_FOLDER = r"C:\Users\Prayas Singh\Desktop\5050\STEP8_50_50_PAIR_CORRELATION"

# ============================================================
# 2. SETTINGS — SAME AS 30:70
# ============================================================

PIXEL_SIZE_UM = 10.0 / 157.0
BIN_WIDTH_UM = 1.0
MAX_R_UM = 30.0
N_RANDOMIZATIONS = 300
LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95
MIN_OBS_PAIRS_PER_BIN = 2
RANDOM_SEED = 42

VALID_CLASSES = ["+1/2", "-1/2"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 3. HELPERS
# ============================================================

def load_defects(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Defect file not found:\n{path}")

    df = pd.read_csv(path)
    required = {"x_um", "y_um", "class"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df = df[df["class"].isin(VALID_CLASSES)].copy()
    df["x_um"] = pd.to_numeric(df["x_um"], errors="coerce")
    df["y_um"] = pd.to_numeric(df["y_um"], errors="coerce")
    return df.dropna(subset=["x_um", "y_um"]).reset_index(drop=True)


def load_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found:\n{path}")

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask:\n{path}")

    return mask > 0


def same_sign_distances(coords):
    if len(coords) < 2:
        return np.array([])
    return pdist(coords)


def opposite_sign_distances(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.array([])
    return cdist(a, b).ravel()


def hist_counts(distances, edges):
    if len(distances) == 0:
        return np.zeros(len(edges)-1, dtype=float)
    return np.histogram(distances, bins=edges)[0].astype(float)


def sample_points_in_mask(mask, n_points, rng):
    yy, xx = np.where(mask)

    if n_points == 0:
        return np.empty((0, 2), dtype=float)

    if n_points > len(xx):
        raise RuntimeError("Requested more random points than valid mask pixels.")

    ids = rng.choice(len(xx), size=n_points, replace=False)

    return np.column_stack([
        xx[ids] * PIXEL_SIZE_UM,
        yy[ids] * PIXEL_SIZE_UM
    ])


def ratio(obs, expected):
    out = np.full_like(obs, np.nan, dtype=float)
    ok = expected > 0
    out[ok] = obs[ok] / expected[ok]
    return out


# ============================================================
# 4. PER-FOV ANALYSIS
# ============================================================

all_profiles = []

edges = np.arange(0, MAX_R_UM + BIN_WIDTH_UM, BIN_WIDTH_UM)
centers = 0.5 * (edges[:-1] + edges[1:])

for fov_i, item in enumerate(DATASETS):

    name = item["name"]
    print(f"\nProcessing {name}")

    out_dir = os.path.join(OUTPUT_FOLDER, name)
    os.makedirs(out_dir, exist_ok=True)

    defects = load_defects(item["defects"])
    valid_mask = load_mask(item["mask"])

    plus = defects.loc[
        defects["class"] == "+1/2", ["x_um", "y_um"]
    ].to_numpy()

    minus = defects.loc[
        defects["class"] == "-1/2", ["x_um", "y_um"]
    ].to_numpy()

    n_plus = len(plus)
    n_minus = len(minus)

    # Observed pair counts
    obs_pp = hist_counts(same_sign_distances(plus), edges)
    obs_mm = hist_counts(same_sign_distances(minus), edges)
    obs_pm = hist_counts(opposite_sign_distances(plus, minus), edges)

    # Random-position null preserving valid-mask geometry and sign counts
    rng = np.random.default_rng(RANDOM_SEED + fov_i)

    null_pp = np.zeros((N_RANDOMIZATIONS, len(centers)))
    null_mm = np.zeros_like(null_pp)
    null_pm = np.zeros_like(null_pp)

    for k in range(N_RANDOMIZATIONS):

        rp = sample_points_in_mask(valid_mask, n_plus, rng)
        rm = sample_points_in_mask(valid_mask, n_minus, rng)

        null_pp[k] = hist_counts(same_sign_distances(rp), edges)
        null_mm[k] = hist_counts(same_sign_distances(rm), edges)
        null_pm[k] = hist_counts(opposite_sign_distances(rp, rm), edges)

    mean_pp = null_pp.mean(axis=0)
    mean_mm = null_mm.mean(axis=0)
    mean_pm = null_pm.mean(axis=0)

    g_pp = ratio(obs_pp, mean_pp)
    g_mm = ratio(obs_mm, mean_mm)
    g_pm = ratio(obs_pm, mean_pm)

    # Null g distributions
    null_g_pp = np.divide(
        null_pp, mean_pp,
        out=np.full_like(null_pp, np.nan),
        where=mean_pp > 0
    )
    null_g_mm = np.divide(
        null_mm, mean_mm,
        out=np.full_like(null_mm, np.nan),
        where=mean_mm > 0
    )
    null_g_pm = np.divide(
        null_pm, mean_pm,
        out=np.full_like(null_pm, np.nan),
        where=mean_pm > 0
    )

    low_pp = np.nanpercentile(null_g_pp, LOW_PERCENTILE, axis=0)
    high_pp = np.nanpercentile(null_g_pp, HIGH_PERCENTILE, axis=0)

    low_mm = np.nanpercentile(null_g_mm, LOW_PERCENTILE, axis=0)
    high_mm = np.nanpercentile(null_g_mm, HIGH_PERCENTILE, axis=0)

    low_pm = np.nanpercentile(null_g_pm, LOW_PERCENTILE, axis=0)
    high_pm = np.nanpercentile(null_g_pm, HIGH_PERCENTILE, axis=0)

    sup_pp = obs_pp >= MIN_OBS_PAIRS_PER_BIN
    sup_mm = obs_mm >= MIN_OBS_PAIRS_PER_BIN
    sup_pm = obs_pm >= MIN_OBS_PAIRS_PER_BIN

    profile = pd.DataFrame({
        "r_center_um": centers,
        "observed_pairs_pp": obs_pp,
        "observed_pairs_mm": obs_mm,
        "observed_pairs_pm": obs_pm,
        "null_mean_pairs_pp": mean_pp,
        "null_mean_pairs_mm": mean_mm,
        "null_mean_pairs_pm": mean_pm,
        "g_pp": g_pp,
        "g_mm": g_mm,
        "g_pm": g_pm,
        "null5_pp": low_pp,
        "null95_pp": high_pp,
        "null5_mm": low_mm,
        "null95_mm": high_mm,
        "null5_pm": low_pm,
        "null95_pm": high_pm,
        "supported_pp": sup_pp,
        "supported_mm": sup_mm,
        "supported_pm": sup_pm,
    })

    profile.to_csv(
        os.path.join(out_dir, "defect_pair_correlation_gofr.csv"),
        index=False
    )

    group_copy = profile.copy()
    group_copy["FOV"] = name
    all_profiles.append(group_copy)

    # Figure with 5–95% null bands
    plt.figure(figsize=(9, 6))

    plt.fill_between(centers, low_pp, high_pp, alpha=0.15)
    plt.plot(
        centers[sup_pp], g_pp[sup_pp],
        marker="o", linewidth=1.7, label=r"$g_{++}(r)$"
    )

    plt.fill_between(centers, low_mm, high_mm, alpha=0.15)
    plt.plot(
        centers[sup_mm], g_mm[sup_mm],
        marker="s", linewidth=1.7, label=r"$g_{--}(r)$"
    )

    plt.fill_between(centers, low_pm, high_pm, alpha=0.15)
    plt.plot(
        centers[sup_pm], g_pm[sup_pm],
        marker="^", linewidth=1.9, label=r"$g_{+-}(r)$"
    )

    plt.axhline(1.0, linestyle="--", linewidth=1)

    plt.xlabel("Defect separation r (µm)")
    plt.ylabel("Pair correlation g(r)")
    plt.title(f"{name}\nSign-resolved defect pair correlations")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "01_defect_pair_correlation_WITH_NULL_BANDS.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Pair-count QC
    plt.figure(figsize=(9, 6))
    plt.plot(centers, obs_pp, marker="o", label="Observed ++")
    plt.plot(centers, obs_mm, marker="s", label="Observed --")
    plt.plot(centers, obs_pm, marker="^", label="Observed +-")
    plt.xlabel("Defect separation r (µm)")
    plt.ylabel("Observed pair count per bin")
    plt.title(f"{name}\nPair-count support")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "02_pair_count_support_QC.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    pd.DataFrame([{
        "FOV": name,
        "n_plus_half": n_plus,
        "n_minus_half": n_minus,
        "valid_mask_area_um2": valid_mask.sum() * PIXEL_SIZE_UM**2,
        "bin_width_um": BIN_WIDTH_UM,
        "max_r_um": MAX_R_UM,
        "n_randomizations": N_RANDOMIZATIONS,
        "min_observed_pairs_per_bin": MIN_OBS_PAIRS_PER_BIN,
        "n_supported_bins_pp": int(sup_pp.sum()),
        "n_supported_bins_mm": int(sup_mm.sum()),
        "n_supported_bins_pm": int(sup_pm.sum()),
    }]).to_csv(
        os.path.join(out_dir, "STEP8_FOV_summary.csv"),
        index=False
    )

# ============================================================
# 5. GROUP-AVERAGED g(r)
# ============================================================

all_profiles_df = pd.concat(all_profiles, ignore_index=True)

all_profiles_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "STEP8_ALL_FOV_profiles.csv"),
    index=False
)

group_rows = []

for r_value, sub in all_profiles_df.groupby("r_center_um"):

    row = {"r_center_um": r_value}

    for label in ["pp", "mm", "pm"]:

        vals = pd.to_numeric(
            sub.loc[sub[f"supported_{label}"], f"g_{label}"],
            errors="coerce"
        ).dropna()

        row[f"mean_g_{label}"] = vals.mean() if len(vals) else np.nan
        row[f"sd_g_{label}"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        row[f"sem_g_{label}"] = vals.sem(ddof=1) if len(vals) > 1 else np.nan
        row[f"n_FOV_{label}"] = len(vals)

        lowvals = pd.to_numeric(sub[f"null5_{label}"], errors="coerce").dropna()
        highvals = pd.to_numeric(sub[f"null95_{label}"], errors="coerce").dropna()

        row[f"mean_null5_{label}"] = lowvals.mean() if len(lowvals) else np.nan
        row[f"mean_null95_{label}"] = highvals.mean() if len(highvals) else np.nan

    group_rows.append(row)

group_df = pd.DataFrame(group_rows)

group_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "STEP8_4FOV_MEAN_pair_correlation.csv"),
    index=False
)

r = group_df["r_center_um"].to_numpy()

plt.figure(figsize=(9, 6))

plt.fill_between(
    r, group_df["mean_null5_pp"], group_df["mean_null95_pp"], alpha=0.15
)
plt.plot(
    r, group_df["mean_g_pp"],
    marker="o", linewidth=1.7, label=r"Mean $g_{++}(r)$"
)

plt.fill_between(
    r, group_df["mean_null5_mm"], group_df["mean_null95_mm"], alpha=0.15
)
plt.plot(
    r, group_df["mean_g_mm"],
    marker="s", linewidth=1.7, label=r"Mean $g_{--}(r)$"
)

plt.fill_between(
    r, group_df["mean_null5_pm"], group_df["mean_null95_pm"], alpha=0.15
)
plt.plot(
    r, group_df["mean_g_pm"],
    marker="^", linewidth=1.9, label=r"Mean $g_{+-}(r)$"
)

plt.axhline(1.0, linestyle="--", linewidth=1)
plt.xlabel("Defect separation r (µm)")
plt.ylabel("Pair correlation g(r)")
plt.title("50:50 + NaOH — 4-FOV mean defect pair correlations")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_4FOV_MEAN_defect_pair_correlation_WITH_NULL_BANDS.png"
    ),
    dpi=300, bbox_inches="tight"
)
plt.close()

print("\n" + "="*95)
print("STEP 8 COMPLETE — 50:50 + NaOH")
print("="*95)
print("Calculated g++(r), g--(r), and g+-(r)")
print(f"Bin width = {BIN_WIDTH_UM} µm")
print(f"Maximum r = {MAX_R_UM} µm")
print(f"Randomizations = {N_RANDOMIZATIONS}")
print(f"Null envelope = {LOW_PERCENTILE}–{HIGH_PERCENTILE}%")
print(f"Minimum observed pairs per bin = {MIN_OBS_PAIRS_PER_BIN}")
print("\nGroup plot:")
print("03_4FOV_MEAN_defect_pair_correlation_WITH_NULL_BANDS.png")
print("\nInterpretation:")
print("g(r) above the null envelope = pair enrichment at that separation.")
print("g(r) below the null envelope = pair depletion at that separation.")
print("Do not call g+-(r) enrichment alone direct thermodynamic attraction.")
