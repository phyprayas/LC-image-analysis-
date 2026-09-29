
# ============================================================
# STEP 3 — NEW 30:70 + NaOH
# EXACTLY MATCHED TO 50:50 STUDY STYLE
#
# Same analysis logic and same visual style:
# - mask applied
# - coherency threshold C >= 0.30
# - OrientationJ DX/DY quiver field
# - global S2D
# - local S2D
# - C2(r)
# - correlation length xi
# - raw +/-1/2 defect candidates
#
# IMPORTANT:
# This version is intended for direct 30:70 vs 50:50 comparison.
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# 1. PATHS
# ============================================================

ORIENTATION_CSV = r"C:\Users\owner\Desktop\30_70_NEW_ANALYSIS\OJ-Table-Vector-Field-(4).csv"

VALID_MASK_PATH = r"C:\Users\owner\Desktop\30_70_NEW_ANALYSIS\STEP1_Valid_LC_Mask\03_valid_LC_mask.png"

OUTPUT_DIR = r"C:\Users\owner\Desktop\30_70_NEW_ANALYSIS\STEP3_MATCHED_TO_50_50"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. MATCHED SETTINGS
# ============================================================

COHERENCY_THRESHOLD = 0.30
LOCAL_RADIUS_GRID = 2
MAX_CORRELATION_DISTANCE_PX = 500
MIN_PAIRS_PER_BIN = 20
DEFECT_TOLERANCE = 0.20

# ============================================================
# 3. LOAD
# ============================================================

df = pd.read_csv(ORIENTATION_CSV)

required = [
    "X","Y","DX","DY",
    "Orientation","Coherency","Energy"
]

for c in required:
    if c not in df.columns:
        raise ValueError(
            f"Missing column '{c}'. Found:\n{list(df.columns)}"
        )

for c in required:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=required).copy()

mask = cv2.imread(
    VALID_MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(VALID_MASK_PATH)

H, W = mask.shape

# ============================================================
# 4. APPLY MASK
# ============================================================

xi = np.rint(df["X"].values).astype(int)
yi = np.rint(df["Y"].values).astype(int)

inside = (
    (xi >= 0) & (xi < W) &
    (yi >= 0) & (yi < H)
)

df = df.loc[inside].copy()

xi = np.rint(df["X"].values).astype(int)
yi = np.rint(df["Y"].values).astype(int)

valid = mask[yi, xi] >= 128

n_input = len(df)
n_mask_valid = int(valid.sum())

df = df.loc[valid].copy()

# ============================================================
# 5. COHERENCY FILTER
# ============================================================

df_f = df[
    df["Coherency"] >= COHERENCY_THRESHOLD
].copy()

n_final = len(df_f)

theta = np.deg2rad(
    df_f["Orientation"].values
)

df_f["theta_rad"] = theta

# ============================================================
# 6. GLOBAL S2D
# ============================================================

c2 = np.mean(np.cos(2*theta))
s2 = np.mean(np.sin(2*theta))

global_S2D = np.sqrt(c2*c2 + s2*s2)

# ============================================================
# 7. GRID
# ============================================================

xvals = np.sort(df["X"].unique())
yvals = np.sort(df["Y"].unique())

nx = len(xvals)
ny = len(yvals)

dx_grid = np.median(np.diff(xvals))
dy_grid = np.median(np.diff(yvals))
grid_spacing = np.nanmean([dx_grid, dy_grid])

xmap = {float(x):i for i,x in enumerate(xvals)}
ymap = {float(y):i for i,y in enumerate(yvals)}

theta_grid = np.full((ny,nx), np.nan)
coh_grid = np.full((ny,nx), np.nan)

for _,r in df_f.iterrows():
    ix = xmap[float(r["X"])]
    iy = ymap[float(r["Y"])]

    theta_grid[iy,ix] = r["theta_rad"]
    coh_grid[iy,ix] = r["Coherency"]

# ============================================================
# 8. LOCAL S2D
# ============================================================

localS = np.full_like(theta_grid, np.nan)

for iy in range(ny):
    for ix in range(nx):

        if not np.isfinite(theta_grid[iy,ix]):
            continue

        y0 = max(0, iy-LOCAL_RADIUS_GRID)
        y1 = min(ny, iy+LOCAL_RADIUS_GRID+1)

        x0 = max(0, ix-LOCAL_RADIUS_GRID)
        x1 = min(nx, ix+LOCAL_RADIUS_GRID+1)

        win = theta_grid[y0:y1, x0:x1]
        vv = np.isfinite(win)

        if vv.sum() < 3:
            continue

        th = win[vv]

        cc = np.mean(np.cos(2*th))
        ss = np.mean(np.sin(2*th))

        localS[iy,ix] = np.sqrt(cc*cc + ss*ss)

local_rows = []

for iy,y in enumerate(yvals):
    for ix,x in enumerate(xvals):

        if not np.isfinite(theta_grid[iy,ix]):
            continue

        local_rows.append({
            "X":x,
            "Y":y,
            "theta_rad":theta_grid[iy,ix],
            "orientation_deg":np.rad2deg(theta_grid[iy,ix]),
            "coherency":coh_grid[iy,ix],
            "local_S2D":localS[iy,ix]
        })

local_df = pd.DataFrame(local_rows)

mean_local_S2D = float(np.nanmean(localS))
median_local_S2D = float(np.nanmedian(localS))

# ============================================================
# 9. C2(r)
# ============================================================

corr_rows = []

max_ox = int(min(nx-1, np.ceil(MAX_CORRELATION_DISTANCE_PX/dx_grid)))
max_oy = int(min(ny-1, np.ceil(MAX_CORRELATION_DISTANCE_PX/dy_grid)))

for oy in range(0, max_oy+1):
    for ox in range(-max_ox, max_ox+1):

        if oy == 0 and ox <= 0:
            continue

        rpx = np.sqrt((ox*dx_grid)**2 + (oy*dy_grid)**2)

        if rpx <= 0 or rpx > MAX_CORRELATION_DISTANCE_PX:
            continue

        vals = []

        for iy in range(ny):

            jy = iy + oy

            if jy >= ny:
                continue

            for ix in range(nx):

                jx = ix + ox

                if jx < 0 or jx >= nx:
                    continue

                t1 = theta_grid[iy,ix]
                t2 = theta_grid[jy,jx]

                if np.isfinite(t1) and np.isfinite(t2):
                    vals.append(
                        np.cos(2*(t2-t1))
                    )

        if len(vals) >= MIN_PAIRS_PER_BIN:
            corr_rows.append({
                "r_px":rpx,
                "C2":float(np.mean(vals)),
                "n_pairs":len(vals)
            })

corr_raw = pd.DataFrame(corr_rows)

corr_raw["r_bin"] = (
    np.round(corr_raw["r_px"]/grid_spacing)
    * grid_spacing
)

corr = (
    corr_raw.groupby("r_bin")
    .apply(
        lambda g: pd.Series({
            "r_px": np.average(
                g["r_px"],
                weights=g["n_pairs"]
            ),
            "C2": np.average(
                g["C2"],
                weights=g["n_pairs"]
            ),
            "n_pairs": int(g["n_pairs"].sum())
        }),
        include_groups=False
    )
    .reset_index(drop=True)
)

def corr_model(r,A,xi,Cinf):
    return A*np.exp(-r/xi)+Cinf

xfit = corr["r_px"].values
yfit = corr["C2"].values

popt, pcov = curve_fit(
    corr_model,
    xfit,
    yfit,
    p0=[
        max(yfit[0]-yfit[-1],0.1),
        50.0,
        yfit[-1]
    ],
    bounds=(
        [-2,1,-1],
        [2,2000,1]
    ),
    maxfev=20000
)

A_fit, xi_px, Cinf_fit = popt

pred = corr_model(xfit,*popt)

ss_res = np.sum((yfit-pred)**2)
ss_tot = np.sum((yfit-np.mean(yfit))**2)

R2 = 1-ss_res/ss_tot if ss_tot > 0 else np.nan

# ============================================================
# 10. RAW +/-1/2 DEFECTS
# ============================================================

def ndiff(t2,t1):
    d = t2-t1
    return 0.5*np.arctan2(
        np.sin(2*d),
        np.cos(2*d)
    )

def classify_half(s,tol=0.20):
    if abs(s-0.5) <= tol:
        return 0.5
    if abs(s+0.5) <= tol:
        return -0.5
    return np.nan

def_rows = []

for iy in range(ny-1):
    for ix in range(nx-1):

        t00 = theta_grid[iy,ix]
        t10 = theta_grid[iy,ix+1]
        t11 = theta_grid[iy+1,ix+1]
        t01 = theta_grid[iy+1,ix]

        if not np.all(np.isfinite([t00,t10,t11,t01])):
            continue

        wsum = (
            ndiff(t10,t00)
            + ndiff(t11,t10)
            + ndiff(t01,t11)
            + ndiff(t00,t01)
        )

        sraw = wsum/(2*np.pi)
        scharge = classify_half(sraw, DEFECT_TOLERANCE)

        if not np.isfinite(scharge):
            continue

        def_rows.append({
            "X":0.5*(xvals[ix]+xvals[ix+1]),
            "Y":0.5*(yvals[iy]+yvals[iy+1]),
            "charge_raw":sraw,
            "charge":scharge
        })

defects = pd.DataFrame(def_rows)

nplus = int(np.sum(defects["charge"] > 0)) if len(defects) else 0
nminus = int(np.sum(defects["charge"] < 0)) if len(defects) else 0
ntotal = len(defects)

valid_area_px = int(np.sum(mask >= 128))
raw_density = ntotal/valid_area_px

# ============================================================
# 11. SAVE CSV
# ============================================================

df_f.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "filtered_orientation_points.csv"
    ),
    index=False
)

local_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "local_order_map.csv"
    ),
    index=False
)

corr.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correlation_function.csv"
    ),
    index=False
)

defects.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "raw_defect_candidates.csv"
    ),
    index=False
)

summary = pd.DataFrame({
    "Parameter":[
        "input_orientation_points",
        "mask_valid_points_before_coherency",
        "coherency_threshold",
        "final_valid_points",
        "final_valid_fraction_of_input",
        "grid_spacing_px",
        "global_S2D",
        "mean_local_S2D",
        "median_local_S2D",
        "correlation_length_xi_px",
        "correlation_fit_R2",
        "raw_plus_half_count",
        "raw_minus_half_count",
        "raw_total_defects",
        "valid_LC_area_px",
        "raw_defect_density_per_px2"
    ],
    "Value":[
        n_input,
        n_mask_valid,
        COHERENCY_THRESHOLD,
        n_final,
        n_final/n_input,
        grid_spacing,
        global_S2D,
        mean_local_S2D,
        median_local_S2D,
        xi_px,
        R2,
        nplus,
        nminus,
        ntotal,
        valid_area_px,
        raw_density
    ]
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "orientation_summary.csv"
    ),
    index=False
)

# ============================================================
# 12. FIGURE 1 — EXACT 50:50-STYLE QUIVER FIELD
# ============================================================

plt.figure(figsize=(12,8))

plt.quiver(
    df_f["X"],
    df_f["Y"],
    df_f["DX"],
    df_f["DY"],
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.002
)

plt.xlim(0,W)
plt.ylim(H,0)

plt.xlabel("X (px)")
plt.ylabel("Y (px)")

plt.title(
    "Masked + coherency-filtered texture orientation field"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_masked_coherency_filtered_orientation_field.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 13. FIGURE 2 — LOCAL ORDER
# ============================================================

plt.figure(figsize=(10,8))

sc = plt.scatter(
    local_df["X"],
    local_df["Y"],
    c=local_df["local_S2D"],
    s=30,
    vmin=0,
    vmax=1
)

plt.colorbar(
    sc,
    label="Local S2D"
)

plt.xlim(0,W)
plt.ylim(H,0)

plt.xlabel("X (px)")
plt.ylabel("Y (px)")

plt.title(
    "Local orientational order"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_local_orientational_order.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 14. FIGURE 3 — C2(r)
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(
    corr["r_px"],
    corr["C2"],
    "o-",
    label="C2(r)"
)

xx = np.linspace(
    corr["r_px"].min(),
    corr["r_px"].max(),
    500
)

plt.plot(
    xx,
    corr_model(xx,*popt),
    "--",
    label=f"fit: xi = {xi_px:.1f} px"
)

plt.axhline(
    0,
    linestyle=":",
    linewidth=1
)

plt.xlabel("Distance r (px)")
plt.ylabel("C2(r)")

plt.title(
    "Nematic orientational correlation"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_nematic_orientational_correlation.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 15. FIGURE 4 — DEFECTS
# ============================================================

plt.figure(figsize=(10,8))

if len(defects):

    p = defects[defects["charge"] > 0]
    m = defects[defects["charge"] < 0]

    plt.scatter(
        p["X"], p["Y"],
        marker="^",
        s=70,
        label="+1/2"
    )

    plt.scatter(
        m["X"], m["Y"],
        marker="v",
        s=70,
        label="-1/2"
    )

plt.xlim(0,W)
plt.ylim(H,0)

plt.xlabel("X (px)")
plt.ylabel("Y (px)")

plt.title(
    "Raw winding-derived defect candidates"
)

if len(defects):
    plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_raw_winding_defect_candidates.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 16. PRINT
# ============================================================

print("\n" + "="*90)
print("STEP 3 COMPLETE — 30:70 MATCHED TO 50:50")
print("="*90)

print("Input points               :", n_input)
print("Mask-valid points          :", n_mask_valid)
print("Final C>=0.30 points       :", n_final)

print("\nORDER")
print("Global S2D                 :", round(global_S2D,4))
print("Mean local S2D             :", round(mean_local_S2D,4))
print("Median local S2D           :", round(median_local_S2D,4))

print("\nCORRELATION")
print("xi (px)                    :", round(xi_px,2))
print("R2                         :", round(R2,4))

print("\nRAW DEFECTS")
print("+1/2                       :", nplus)
print("-1/2                       :", nminus)
print("Total                      :", ntotal)
print("Raw density (px^-2)        :", f"{raw_density:.4e}")

print("\nSaved to:")
print(OUTPUT_DIR)
