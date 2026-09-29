
# ============================================================
# STEP 11 — DEFECT–DEFECT INTERACTION ANALYSIS
#
# What this script does:
# 1. Reads merged unique defect candidates (from Step 4).
# 2. Separates +1/2 and -1/2 defects.
# 3. Computes sign-resolved pair-correlation functions:
#       g++(r), g--(r), g+-(r)
#    using Monte-Carlo normalization in the same image window.
# 4. Computes an effective interaction-like descriptor:
#       U_eff(r) = -ln g(r)
#    (report as a descriptive "potential-like" quantity only).
# 5. Computes nearest-neighbor sign statistics and a shuffle-based null model.
# 6. Saves CSV files, plots, and a compact text summary.
#
# IMPORTANT INTERPRETATION:
# - These are winding-derived defect candidates, not independently validated
#   singular cores.
# - Use the results first as a supplementary analysis unless they are
#   reproducible across multiple independent fields of view.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER SETTINGS
# ============================================================

# Replace this path with the merged unique defects CSV from Step 4.
# It should contain at least X, Y, and sign/charge information.
MERGED_DEFECTS_CSV = r"C:\Users\Prayas Singh\Desktop\STEP4_Merged_Defects\merged_unique_defects.csv"

OUTPUT_DIR = r"C:\Users\Prayas Singh\Desktop\STEP11_Defect_Interaction"

# Observation window of the analyzed image
# Set these to the actual image size used in the defect analysis
IMAGE_WIDTH_PX = 2048
IMAGE_HEIGHT_PX = 1532

# Calibration from your 50× image
PIXEL_SIZE_UM = 10.0 / 157.0

# Pair-correlation settings
BIN_WIDTH_PX = 20
R_MAX_PX = 600

# Monte-Carlo normalization
N_SIM = 300
RANDOM_SEED = 123

# Nearest-neighbor sign-shuffle null model
N_SHUFFLE = 2000

# Small epsilon for safe log in U_eff
EPS = 1e-9

# ============================================================
# 2. HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_charge(v):
    """
    Converts various sign/charge labels to +0.5 or -0.5.
    Accepts numeric or string labels such as:
    +1/2, -1/2, plus, minus, 0.5, -0.5, +, -
    """
    if pd.isna(v):
        return np.nan

    # numeric
    try:
        x = float(v)
        if x > 0:
            return 0.5
        if x < 0:
            return -0.5
        return np.nan
    except Exception:
        pass

    s = str(v).strip().lower().replace(" ", "")
    mapping_pos = ["+1/2", "1/2", "plus", "+", "positive", "0.5", "half"]
    mapping_neg = ["-1/2", "minus", "-", "negative", "-0.5"]

    if s in mapping_pos:
        return 0.5
    if s in mapping_neg:
        return -0.5

    # broader matching
    if "+1/2" in s or "plus" in s or s == "0.5":
        return 0.5
    if "-1/2" in s or "minus" in s or s == "-0.5":
        return -0.5

    return np.nan

def load_defects(csv_path):
    """
    Flexible loader for merged defect CSV.
    Required concepts: X, Y, charge/sign.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File not found:\n{csv_path}\n\n"
            "Please set MERGED_DEFECTS_CSV to your Step 4 merged-defects output."
        )

    df = pd.read_csv(csv_path)
    raw_cols = list(df.columns)
    cols = {c.strip().lower(): c for c in raw_cols}

    x_candidates = ["x", "x_px", "xpix", "x_pixel", "xcoord"]
    y_candidates = ["y", "y_px", "ypix", "y_pixel", "ycoord"]
    charge_candidates = [
        "charge", "sign", "topological_charge", "strength",
        "defect_type", "type", "class", "s"
    ]

    xcol = None
    ycol = None
    ccol = None

    for k in x_candidates:
        if k in cols:
            xcol = cols[k]
            break

    for k in y_candidates:
        if k in cols:
            ycol = cols[k]
            break

    for k in charge_candidates:
        if k in cols:
            ccol = cols[k]
            break

    if xcol is None or ycol is None or ccol is None:
        raise ValueError(
            "Could not identify required columns.\n"
            f"Found columns: {raw_cols}\n\n"
            "Expected something like X, Y, and charge/sign."
        )

    out = pd.DataFrame()
    out["x"] = pd.to_numeric(df[xcol], errors="coerce")
    out["y"] = pd.to_numeric(df[ycol], errors="coerce")
    out["charge"] = df[ccol].apply(parse_charge)

    out = out.dropna().copy()
    out = out[
        np.isin(out["charge"].values, [0.5, -0.5])
    ].copy()

    if len(out) < 3:
        raise RuntimeError("Too few valid defects after parsing.")

    return out

def pair_distances_same(points):
    """
    Unique pairwise distances within one point set.
    """
    n = len(points)
    if n < 2:
        return np.array([], dtype=float)

    dlist = []
    for i in range(n - 1):
        dx = points[i+1:, 0] - points[i, 0]
        dy = points[i+1:, 1] - points[i, 1]
        d = np.sqrt(dx * dx + dy * dy)
        dlist.append(d)

    return np.concatenate(dlist) if dlist else np.array([], dtype=float)

def pair_distances_cross(points_a, points_b):
    """
    All pairwise distances between set A and set B.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return np.array([], dtype=float)

    dlist = []
    for i in range(len(points_a)):
        dx = points_b[:, 0] - points_a[i, 0]
        dy = points_b[:, 1] - points_a[i, 1]
        d = np.sqrt(dx * dx + dy * dy)
        dlist.append(d)

    return np.concatenate(dlist) if dlist else np.array([], dtype=float)

def hist_counts(distances, bins):
    counts, _ = np.histogram(distances, bins=bins)
    return counts.astype(float)

def simulate_random_points(n, width, height, rng):
    x = rng.uniform(0, width, n)
    y = rng.uniform(0, height, n)
    return np.column_stack([x, y])

def monte_carlo_expected(n_plus, n_minus, bins, width, height, n_sim, seed):
    """
    Monte-Carlo expected pair counts under CSR in the same observation window.
    This handles finite-window effects much more robustly than simple shell-area normalization.
    """
    rng = np.random.default_rng(seed)

    sim_pp = []
    sim_mm = []
    sim_pm = []

    for _ in range(n_sim):
        p_plus = simulate_random_points(n_plus, width, height, rng)
        p_minus = simulate_random_points(n_minus, width, height, rng)

        d_pp = pair_distances_same(p_plus)
        d_mm = pair_distances_same(p_minus)
        d_pm = pair_distances_cross(p_plus, p_minus)

        sim_pp.append(hist_counts(d_pp, bins))
        sim_mm.append(hist_counts(d_mm, bins))
        sim_pm.append(hist_counts(d_pm, bins))

    sim_pp = np.array(sim_pp)
    sim_mm = np.array(sim_mm)
    sim_pm = np.array(sim_pm)

    return sim_pp, sim_mm, sim_pm

def nearest_neighbor_indices(points):
    """
    Returns nearest-neighbor index and distance for each point.
    """
    n = len(points)
    nn_idx = np.full(n, -1, dtype=int)
    nn_dist = np.full(n, np.nan, dtype=float)

    for i in range(n):
        dx = points[:, 0] - points[i, 0]
        dy = points[:, 1] - points[i, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d[i] = np.inf
        j = np.argmin(d)
        nn_idx[i] = j
        nn_dist[i] = d[j]

    return nn_idx, nn_dist

def shuffle_sign_null(points, charges, n_shuffle, seed):
    """
    Null model: keep positions fixed, randomly shuffle defect signs.
    Computes the fraction of defects whose nearest neighbor is unlike-sign.
    """
    rng = np.random.default_rng(seed)
    nn_idx, _ = nearest_neighbor_indices(points)

    unlike_fracs = []

    for _ in range(n_shuffle):
        shuffled = charges.copy()
        rng.shuffle(shuffled)

        unlike = (shuffled != shuffled[nn_idx]).astype(float)
        unlike_fracs.append(np.mean(unlike))

    return np.array(unlike_fracs)

def compute_pair_correlation(obs_counts, sim_counts):
    """
    obs_counts: shape (n_bins,)
    sim_counts: shape (n_sim, n_bins)
    Returns g(r) and null envelope.
    """
    mean_exp = sim_counts.mean(axis=0)
    q025 = np.quantile(sim_counts, 0.025, axis=0)
    q975 = np.quantile(sim_counts, 0.975, axis=0)

    g = obs_counts / np.clip(mean_exp, EPS, None)
    g_lo = q025 / np.clip(mean_exp, EPS, None)
    g_hi = q975 / np.clip(mean_exp, EPS, None)

    return g, g_lo, g_hi, mean_exp

def save_summary_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))

# ============================================================
# 3. LOAD DATA
# ============================================================

ensure_dir(OUTPUT_DIR)

df = load_defects(MERGED_DEFECTS_CSV)

points = df[["x", "y"]].values
charges = df["charge"].values

plus = df[df["charge"] > 0][["x", "y"]].values
minus = df[df["charge"] < 0][["x", "y"]].values

n_total = len(df)
n_plus = len(plus)
n_minus = len(minus)

print("Loaded merged unique defect candidates:")
print(f"Total = {n_total}")
print(f"+1/2 = {n_plus}")
print(f"-1/2 = {n_minus}")

# ============================================================
# 4. OBSERVED PAIR DISTANCES
# ============================================================

bins = np.arange(0, R_MAX_PX + BIN_WIDTH_PX, BIN_WIDTH_PX)
r_centers_px = 0.5 * (bins[:-1] + bins[1:])
r_centers_um = r_centers_px * PIXEL_SIZE_UM

d_pp = pair_distances_same(plus)
d_mm = pair_distances_same(minus)
d_pm = pair_distances_cross(plus, minus)

obs_pp = hist_counts(d_pp, bins)
obs_mm = hist_counts(d_mm, bins)
obs_pm = hist_counts(d_pm, bins)

# ============================================================
# 5. MONTE-CARLO EXPECTED COUNTS
# ============================================================

sim_pp, sim_mm, sim_pm = monte_carlo_expected(
    n_plus=n_plus,
    n_minus=n_minus,
    bins=bins,
    width=IMAGE_WIDTH_PX,
    height=IMAGE_HEIGHT_PX,
    n_sim=N_SIM,
    seed=RANDOM_SEED
)

g_pp, gpp_lo, gpp_hi, exp_pp = compute_pair_correlation(obs_pp, sim_pp)
g_mm, gmm_lo, gmm_hi, exp_mm = compute_pair_correlation(obs_mm, sim_mm)
g_pm, gpm_lo, gpm_hi, exp_pm = compute_pair_correlation(obs_pm, sim_pm)

# ============================================================
# 6. EFFECTIVE INTERACTION-LIKE DESCRIPTOR
# ============================================================

# Purely descriptive. Report as "potential-like".
u_pp = -np.log(np.clip(g_pp, EPS, None))
u_mm = -np.log(np.clip(g_mm, EPS, None))
u_pm = -np.log(np.clip(g_pm, EPS, None))

# ============================================================
# 7. NEAREST-NEIGHBOR SIGN STATISTICS
# ============================================================

nn_idx, nn_dist = nearest_neighbor_indices(points)
nn_charge = charges[nn_idx]

unlike = (charges != nn_charge).astype(float)
like = 1.0 - unlike

observed_unlike_frac = np.mean(unlike)
observed_like_frac = np.mean(like)
mean_nn_px = np.mean(nn_dist)
median_nn_px = np.median(nn_dist)
mean_nn_um = mean_nn_px * PIXEL_SIZE_UM
median_nn_um = median_nn_px * PIXEL_SIZE_UM

# By source sign
mask_p = charges > 0
mask_m = charges < 0

obs_unlike_from_plus = np.mean((nn_charge[mask_p] < 0).astype(float)) if np.sum(mask_p) > 0 else np.nan
obs_unlike_from_minus = np.mean((nn_charge[mask_m] > 0).astype(float)) if np.sum(mask_m) > 0 else np.nan

null_unlike = shuffle_sign_null(
    points=points,
    charges=charges.copy(),
    n_shuffle=N_SHUFFLE,
    seed=RANDOM_SEED + 100
)

null_mean = np.mean(null_unlike)
null_std = np.std(null_unlike, ddof=1)
null_q025 = np.quantile(null_unlike, 0.025)
null_q975 = np.quantile(null_unlike, 0.975)

# Two-sided empirical p-value
p_emp = (
    np.sum(np.abs(null_unlike - null_mean) >= np.abs(observed_unlike_frac - null_mean)) + 1
) / (len(null_unlike) + 1)

# ============================================================
# 8. SAVE CSV RESULTS
# ============================================================

pair_df = pd.DataFrame({
    "r_center_px": r_centers_px,
    "r_center_um": r_centers_um,

    "obs_pp": obs_pp,
    "exp_pp_mean": exp_pp,
    "g_pp": g_pp,
    "g_pp_null_lo95": gpp_lo,
    "g_pp_null_hi95": gpp_hi,
    "u_eff_pp": u_pp,

    "obs_mm": obs_mm,
    "exp_mm_mean": exp_mm,
    "g_mm": g_mm,
    "g_mm_null_lo95": gmm_lo,
    "g_mm_null_hi95": gmm_hi,
    "u_eff_mm": u_mm,

    "obs_pm": obs_pm,
    "exp_pm_mean": exp_pm,
    "g_pm": g_pm,
    "g_pm_null_lo95": gpm_lo,
    "g_pm_null_hi95": gpm_hi,
    "u_eff_pm": u_pm
})

pair_csv = os.path.join(OUTPUT_DIR, "sign_resolved_pair_correlation.csv")
pair_df.to_csv(pair_csv, index=False)

summary_rows = [
    ["Total defects", n_total],
    ["+1/2 defects", n_plus],
    ["-1/2 defects", n_minus],
    ["Image width (px)", IMAGE_WIDTH_PX],
    ["Image height (px)", IMAGE_HEIGHT_PX],
    ["Pixel size (um/px)", PIXEL_SIZE_UM],
    ["Mean nearest-neighbor distance (px)", mean_nn_px],
    ["Median nearest-neighbor distance (px)", median_nn_px],
    ["Mean nearest-neighbor distance (um)", mean_nn_um],
    ["Median nearest-neighbor distance (um)", median_nn_um],
    ["Observed unlike-sign NN fraction", observed_unlike_frac],
    ["Observed like-sign NN fraction", observed_like_frac],
    ["Observed unlike-sign NN fraction from +1/2", obs_unlike_from_plus],
    ["Observed unlike-sign NN fraction from -1/2", obs_unlike_from_minus],
    ["Null unlike-sign NN mean", null_mean],
    ["Null unlike-sign NN std", null_std],
    ["Null unlike-sign NN 2.5%", null_q025],
    ["Null unlike-sign NN 97.5%", null_q975],
    ["Empirical p-value for NN sign bias", p_emp],
    ["R max (px)", R_MAX_PX],
    ["Bin width (px)", BIN_WIDTH_PX],
    ["N Monte-Carlo simulations", N_SIM],
    ["N sign shuffles", N_SHUFFLE]
]

summary_df = pd.DataFrame(summary_rows, columns=["Parameter", "Value"])
summary_csv = os.path.join(OUTPUT_DIR, "defect_interaction_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# ============================================================
# 9. PLOTS
# ============================================================

# 9.1 sign-resolved pair correlation
plt.figure(figsize=(9, 6))

plt.plot(r_centers_px, g_pp, marker="o", label="g++(r)")
plt.fill_between(r_centers_px, gpp_lo, gpp_hi, alpha=0.20)

plt.plot(r_centers_px, g_mm, marker="s", label="g--(r)")
plt.fill_between(r_centers_px, gmm_lo, gmm_hi, alpha=0.20)

plt.plot(r_centers_px, g_pm, marker="^", label="g+-(r)")
plt.fill_between(r_centers_px, gpm_lo, gpm_hi, alpha=0.20)

plt.axhline(1.0, linestyle="--")
plt.xlabel("Distance r (px)")
plt.ylabel("Pair correlation g(r)")
plt.title("Sign-resolved defect pair correlations")
plt.legend()
plt.tight_layout()
plot1 = os.path.join(OUTPUT_DIR, "01_sign_resolved_pair_correlation.png")
plt.savefig(plot1, dpi=300)
plt.close()

# 9.2 effective interaction-like curves
plt.figure(figsize=(9, 6))
plt.plot(r_centers_px, u_pp, marker="o", label="Ueff, ++")
plt.plot(r_centers_px, u_mm, marker="s", label="Ueff, --")
plt.plot(r_centers_px, u_pm, marker="^", label="Ueff, +-")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Distance r (px)")
plt.ylabel("U_eff(r) = -ln g(r)")
plt.title("Effective interaction-like descriptor")
plt.legend()
plt.tight_layout()
plot2 = os.path.join(OUTPUT_DIR, "02_effective_interaction_like.png")
plt.savefig(plot2, dpi=300)
plt.close()

# 9.3 nearest-neighbor sign bias
plt.figure(figsize=(7, 5))
labels = ["Observed", "Null mean"]
vals = [observed_unlike_frac, null_mean]
errs = [0.0, null_std]
x = np.arange(len(labels))

plt.bar(x, vals, yerr=errs, capsize=5)
plt.xticks(x, labels)
plt.ylabel("Fraction with unlike-sign nearest neighbor")
plt.title("Nearest-neighbor sign bias")
plt.tight_layout()
plot3 = os.path.join(OUTPUT_DIR, "03_nearest_neighbor_sign_bias.png")
plt.savefig(plot3, dpi=300)
plt.close()

# 9.4 null distribution of unlike-sign fraction
plt.figure(figsize=(7, 5))
plt.hist(null_unlike, bins=30)
plt.axvline(observed_unlike_frac, linestyle="--", label="Observed")
plt.axvline(null_mean, linestyle=":", label="Null mean")
plt.xlabel("Unlike-sign nearest-neighbor fraction")
plt.ylabel("Count")
plt.title("Shuffle-based null distribution")
plt.legend()
plt.tight_layout()
plot4 = os.path.join(OUTPUT_DIR, "04_nn_sign_null_distribution.png")
plt.savefig(plot4, dpi=300)
plt.close()

# ============================================================
# 10. INTERPRETIVE TEXT SUMMARY
# ============================================================

def short_interpretation(g, lo, hi, name, rvals):
    """
    Finds first bin where g is clearly > or < the null baseline band.
    """
    enrich = np.where(g > hi)[0]
    deplete = np.where(g < lo)[0]

    txt = []
    if len(enrich) > 0:
        i = enrich[0]
        txt.append(
            f"{name} shows short-range enrichment around r ≈ {rvals[i]:.1f} px (g > upper null envelope)."
        )
    if len(deplete) > 0:
        i = deplete[0]
        txt.append(
            f"{name} shows short-range depletion around r ≈ {rvals[i]:.1f} px (g < lower null envelope)."
        )
    if not txt:
        txt.append(
            f"{name} does not show a strong deviation from the Monte-Carlo null envelope over the sampled distance range."
        )
    return " ".join(txt)

lines = []
lines.append(
    f"The merged defect set contained {n_total} candidates "
    f"({n_plus} positive and {n_minus} negative half-integer candidates)."
)
lines.append(
    f"The mean nearest-neighbor distance was {mean_nn_px:.1f} px "
    f"({mean_nn_um:.2f} um), and the median nearest-neighbor distance was "
    f"{median_nn_px:.1f} px ({median_nn_um:.2f} um)."
)
lines.append(
    f"The observed fraction of defects whose nearest neighbor had the opposite sign was "
    f"{observed_unlike_frac:.3f}, compared with a shuffle-based null expectation of "
    f"{null_mean:.3f} ± {null_std:.3f} (empirical p = {p_emp:.4f})."
)
lines.append(
    short_interpretation(g_pm, gpm_lo, gpm_hi, "g+-(r)", r_centers_px)
)
lines.append(
    short_interpretation(g_pp, gpp_lo, gpp_hi, "g++(r)", r_centers_px)
)
lines.append(
    short_interpretation(g_mm, gmm_lo, gmm_hi, "g--(r)", r_centers_px)
)
lines.append(
    "The curves U_eff(r) = -ln g(r) are reported only as descriptive effective interaction-like quantities and should not be interpreted as equilibrium thermodynamic interaction potentials."
)
lines.append(
    "For manuscript use, these interaction metrics should be emphasized only if the same trends are reproduced across independent fields of view and both concentrations."
)

summary_txt = os.path.join(OUTPUT_DIR, "defect_interaction_ready_to_paste.txt")
save_summary_text(summary_txt, lines)

# ============================================================
# 11. PRINT
# ============================================================

print("\n" + "="*90)
print("STEP 11 COMPLETE — DEFECT–DEFECT INTERACTION ANALYSIS")
print("="*90)

print("\nSaved files:")
print(pair_csv)
print(summary_csv)
print(summary_txt)
print(plot1)
print(plot2)
print(plot3)
print(plot4)

print("\nQuick interpretation:")
for line in lines:
    print("-", line)
