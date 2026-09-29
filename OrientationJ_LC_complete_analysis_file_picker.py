# ============================================================
# PASSIVE NEMATIC LIQUID-CRYSTAL ANALYSIS
# OrientationJ Vector Field CSV -> S2D -> C2(r) -> xi ->
# +/-1/2 defects -> defect density -> nearest neighbor -> g(r)
#
# IMPORTANT:
# You do NOT need to manually type the CSV path.
# A file-selection window will open when you run the script.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.spatial import cKDTree, distance

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None

OUTPUT_DIR = "LC_analysis_results"
COHERENCY_THRESHOLD = 0.30
LOCAL_WINDOW = 5
HALF_DEFECT_TOL = 0.10
PIXEL_SIZE_UM = None


def choose_csv_file():
    if tk is None:
        return input("Enter full path to OrientationJ CSV: ").strip()
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select OrientationJ Vector Field CSV",
        filetypes=[("CSV / text files", "*.csv *.txt"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def load_orientationj_table(file_path):
    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin1"]
    required = {"X", "Y", "Orientation", "Coherency"}
    attempts = []
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, sep=None, engine="python")
            df.columns = (df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False))
            if required.issubset(df.columns):
                print(f"Loaded with encoding: {enc}")
                print("Columns:", df.columns.tolist())
                return df
            attempts.append(f"{enc}: columns found = {df.columns.tolist()}")
        except Exception as e:
            attempts.append(f"{enc}: {e}")
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, sep="\t")
            df.columns = (df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False))
            if required.issubset(df.columns):
                print(f"Loaded as TAB-separated with encoding: {enc}")
                print("Columns:", df.columns.tolist())
                return df
            attempts.append(f"TAB {enc}: columns found = {df.columns.tolist()}")
        except Exception as e:
            attempts.append(f"TAB {enc}: {e}")
    raise RuntimeError("Could not read the OrientationJ file.\n\n" + "\n".join(attempts[-8:]))


def nematic_diff(theta2, theta1):
    d = theta2 - theta1
    return 0.5 * np.arctan2(np.sin(2*d), np.cos(2*d))


def calculate_S2D(theta):
    theta = np.asarray(theta)
    theta = theta[np.isfinite(theta)]
    if len(theta) == 0:
        return np.nan, np.nan
    C = np.mean(np.cos(2*theta))
    S = np.mean(np.sin(2*theta))
    order = np.sqrt(C**2 + S**2)
    mean_angle = 0.5 * np.arctan2(S, C)
    return order, mean_angle


def local_S2D_map(theta, valid, window=5):
    if window % 2 == 0:
        window += 1
    ny, nx = theta.shape
    half = window // 2
    output = np.full(theta.shape, np.nan, dtype=float)
    for j in range(ny):
        for i in range(nx):
            j0, j1 = max(0, j-half), min(ny, j+half+1)
            i0, i1 = max(0, i-half), min(nx, i+half+1)
            m = valid[j0:j1, i0:i1]
            vals = theta[j0:j1, i0:i1][m]
            if len(vals) < 4:
                continue
            local_S, _ = calculate_S2D(vals)
            output[j, i] = local_S
    return output


def exp_decay(r, A, xi, C):
    return A * np.exp(-r / xi) + C


def nearest_neighbor_table(defects):
    if len(defects) < 2:
        return pd.DataFrame()
    pts = defects[["X_px", "Y_px"]].to_numpy(dtype=float)
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=2)
    rows = []
    for i in range(len(defects)):
        rows.append({
            "Defect": defects.iloc[i]["Defect"],
            "X_px": pts[i, 0],
            "Y_px": pts[i, 1],
            "NN_distance_px": d[i, 1],
            "NN_type": defects.iloc[idx[i, 1]]["Defect"]
        })
    return pd.DataFrame(rows)


def pair_correlation(points, width, height, bin_width, rmax):
    points = np.asarray(points, dtype=float)
    bins = np.arange(0, rmax + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])
    N = len(points)
    if N < 2:
        return centers, np.full(len(centers), np.nan)
    pair_distances = distance.pdist(points)
    counts, _ = np.histogram(pair_distances, bins=bins)
    area = width * height
    density = N / area
    annulus_area = np.pi * (bins[1:]**2 - bins[:-1]**2)
    expected = 0.5 * N * density * annulus_area
    with np.errstate(divide="ignore", invalid="ignore"):
        g = counts / expected
    return centers, g


CSV_PATH = choose_csv_file()
if not CSV_PATH:
    raise SystemExit("No file selected.")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File not found:\n{CSV_PATH}")

print("\nSelected file:")
print(CSV_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = load_orientationj_table(CSV_PATH)

for column in ["X", "Y", "Orientation", "Coherency"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")
df = df.dropna(subset=["X", "Y", "Orientation", "Coherency"])
print("\nValid rows:", len(df))

xs = np.sort(df["X"].unique())
ys = np.sort(df["Y"].unique())
if len(xs) < 2 or len(ys) < 2:
    raise ValueError("The table does not contain a usable 2D X-Y grid.")

dx = float(np.median(np.diff(xs)))
dy = float(np.median(np.diff(ys)))
orientation_deg = (df.pivot(index="Y", columns="X", values="Orientation").reindex(index=ys, columns=xs).to_numpy(dtype=float))
coherency = (df.pivot(index="Y", columns="X", values="Coherency").reindex(index=ys, columns=xs).to_numpy(dtype=float))
theta = np.deg2rad(orientation_deg)
valid = np.isfinite(theta) & np.isfinite(coherency) & (coherency >= COHERENCY_THRESHOLD)
ny, nx = theta.shape
width_px = xs.max() - xs.min() + dx
height_px = ys.max() - ys.min() + dy
area_px2 = width_px * height_px
print("\nGrid:")
print("Nx =", nx)
print("Ny =", ny)
print("dx =", dx, "px")
print("dy =", dy, "px")

S_global, theta_global = calculate_S2D(theta[valid])
print("\nGlobal S2D =", round(S_global, 4))
print("Global mean texture orientation =", round(np.rad2deg(theta_global), 2), "degrees")

S_local = local_S2D_map(theta, valid, window=LOCAL_WINDOW)
print("Mean local S2D =", round(np.nanmean(S_local), 4))
print("Median local S2D =", round(np.nanmedian(S_local), 4))

YY, XX = np.meshgrid(ys, xs, indexing="ij")
coords = np.column_stack([XX[valid], YY[valid]])
angles = theta[valid]
MAX_POINTS = 3000
if len(coords) > MAX_POINTS:
    rng = np.random.default_rng(12345)
    chosen = rng.choice(len(coords), MAX_POINTS, replace=False)
    coords_corr = coords[chosen]
    angles_corr = angles[chosen]
else:
    coords_corr = coords
    angles_corr = angles

ii, jj = np.triu_indices(len(angles_corr), k=1)
pair_r = np.sqrt((coords_corr[ii, 0]-coords_corr[jj, 0])**2 + (coords_corr[ii, 1]-coords_corr[jj, 1])**2)
pair_C = np.cos(2 * (angles_corr[ii] - angles_corr[jj]))
bin_width = max(dx, dy)
max_distance = min(width_px, height_px) / 3
bins = np.arange(0, max_distance + bin_width, bin_width)
r_centers = 0.5 * (bins[:-1] + bins[1:])
C2 = np.full(len(r_centers), np.nan)
C2_sem = np.full(len(r_centers), np.nan)
for i in range(len(r_centers)):
    m = (pair_r >= bins[i]) & (pair_r < bins[i+1])
    values = pair_C[m]
    if len(values) >= 20:
        C2[i] = np.mean(values)
        C2_sem[i] = np.std(values, ddof=1) / np.sqrt(len(values))

fit_mask = np.isfinite(C2) & (r_centers > 0)
xi_px = np.nan
A_fit = np.nan
C_fit = np.nan
if np.sum(fit_mask) >= 5:
    try:
        p0 = [max(0.05, C2[fit_mask][0] - S_global**2), max(2*max(dx, dy), 100), S_global**2]
        popt, _ = curve_fit(
            exp_decay,
            r_centers[fit_mask],
            C2[fit_mask],
            p0=p0,
            bounds=([0, 1, -1], [2, 10000, 1]),
            maxfev=50000
        )
        A_fit, xi_px, C_fit = popt
    except Exception as e:
        print("Correlation-length fit failed:", e)
print("Correlation length xi =", xi_px, "px")

defects = []
for j in range(ny - 1):
    for i in range(nx - 1):
        t1, t2, t3, t4 = theta[j, i], theta[j, i+1], theta[j+1, i+1], theta[j+1, i]
        c1, c2, c3, c4 = coherency[j, i], coherency[j, i+1], coherency[j+1, i+1], coherency[j+1, i]
        vals = [t1, t2, t3, t4, c1, c2, c3, c4]
        if not np.all(np.isfinite(vals)):
            continue
        mean_c = np.mean([c1, c2, c3, c4])
        if mean_c < COHERENCY_THRESHOLD:
            continue
        rotation = nematic_diff(t2, t1) + nematic_diff(t3, t2) + nematic_diff(t4, t3) + nematic_diff(t1, t4)
        charge = rotation / (2*np.pi)
        defect_type = None
        if abs(charge - 0.5) <= HALF_DEFECT_TOL:
            defect_type = "+1/2"
        elif abs(charge + 0.5) <= HALF_DEFECT_TOL:
            defect_type = "-1/2"
        if defect_type is not None:
            defects.append({
                "Defect": defect_type,
                "X_px": (xs[i] + xs[i+1]) / 2,
                "Y_px": (ys[j] + ys[j+1]) / 2,
                "Charge": charge,
                "Rotation_deg": np.rad2deg(rotation),
                "Mean_coherency": mean_c
            })

defect_df = pd.DataFrame(defects)
if len(defect_df) == 0:
    defect_df = pd.DataFrame(columns=["Defect", "X_px", "Y_px", "Charge", "Rotation_deg", "Mean_coherency"])

Nplus = int(np.sum(defect_df["Defect"] == "+1/2"))
Nminus = int(np.sum(defect_df["Defect"] == "-1/2"))
Ntotal = len(defect_df)
print("\n+1/2 candidate defects =", Nplus)
print("-1/2 candidate defects =", Nminus)
print("Total candidate defects =", Ntotal)

defect_density_px2 = Ntotal / area_px2
characteristic_spacing_px = 1/np.sqrt(defect_density_px2) if defect_density_px2 > 0 else np.nan
print("Defect density =", defect_density_px2, "per px^2")
print("Characteristic defect spacing =", characteristic_spacing_px, "px")

nn_df = nearest_neighbor_table(defect_df)
mean_nn = nn_df["NN_distance_px"].mean() if len(nn_df) > 0 else np.nan
print("Mean nearest-neighbor distance =", mean_nn, "px")

if len(defect_df) >= 2:
    points = defect_df[["X_px", "Y_px"]].to_numpy()
    r_gr, gr = pair_correlation(points, width_px, height_px, bin_width, max_distance)
else:
    r_gr = np.array([])
    gr = np.array([])

xi_um = np.nan
density_um2 = np.nan
mean_nn_um = np.nan
if PIXEL_SIZE_UM is not None:
    xi_um = xi_px * PIXEL_SIZE_UM
    area_um2 = area_px2 * PIXEL_SIZE_UM**2
    density_um2 = Ntotal / area_um2
    mean_nn_um = mean_nn * PIXEL_SIZE_UM

# Save tables
defect_df.to_csv(os.path.join(OUTPUT_DIR, "topological_defects.csv"), index=False)
nn_df.to_csv(os.path.join(OUTPUT_DIR, "nearest_neighbor.csv"), index=False)
pd.DataFrame({"r_px": r_centers, "C2": C2, "SEM": C2_sem}).to_csv(os.path.join(OUTPUT_DIR, "orientational_correlation.csv"), index=False)
if len(r_gr) > 0:
    pd.DataFrame({"r_px": r_gr, "g_r": gr}).to_csv(os.path.join(OUTPUT_DIR, "defect_pair_correlation.csv"), index=False)

extent = [xs.min(), xs.max(), ys.max(), ys.min()]

plt.figure(figsize=(10, 7))
plt.imshow(orientation_deg, extent=extent, origin="upper", aspect="equal")
plt.colorbar(label="Orientation (degree)")
plt.title("OrientationJ Texture Orientation Map")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_orientation_map.png"), dpi=300)
plt.show()

plt.figure(figsize=(10, 7))
plt.imshow(coherency, extent=extent, origin="upper", aspect="equal", vmin=0, vmax=1)
plt.colorbar(label="Coherency")
plt.title("OrientationJ Coherency Map")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_coherency_map.png"), dpi=300)
plt.show()

plt.figure(figsize=(10, 7))
plt.imshow(S_local, extent=extent, origin="upper", aspect="equal", vmin=0, vmax=1)
plt.colorbar(label="Local S2D")
plt.title(f"Local Image-Derived S2D ({LOCAL_WINDOW} x {LOCAL_WINDOW} grid window)")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_local_S2D_map.png"), dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.errorbar(r_centers, C2, yerr=C2_sem, marker="o", linestyle="none")
if np.isfinite(xi_px):
    rr = np.linspace(np.nanmin(r_centers[fit_mask]), np.nanmax(r_centers[fit_mask]), 300)
    plt.plot(rr, exp_decay(rr, A_fit, xi_px, C_fit), label=f"xi = {xi_px:.1f} px")
    plt.legend()
plt.xlabel("r (px)")
plt.ylabel("C2(r)")
plt.title("Orientational Correlation Function")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_orientational_correlation.png"), dpi=300)
plt.show()

plt.figure(figsize=(10, 7))
plt.imshow(coherency, extent=extent, origin="upper", aspect="equal", vmin=0, vmax=1)
if Nplus > 0:
    plus = defect_df[defect_df["Defect"] == "+1/2"]
    plt.scatter(plus["X_px"], plus["Y_px"], marker="o", facecolors="none", s=90, linewidths=1.5, label="+1/2")
if Nminus > 0:
    minus = defect_df[defect_df["Defect"] == "-1/2"]
    plt.scatter(minus["X_px"], minus["Y_px"], marker="x", s=70, linewidths=1.5, label="-1/2")
if Ntotal > 0:
    plt.legend()
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.title("Winding-Derived Topological Defect Candidates")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_defect_map.png"), dpi=300)
plt.show()

if len(r_gr) > 0:
    plt.figure(figsize=(7, 5))
    plt.plot(r_gr, gr, marker="o")
    plt.axhline(1, linestyle="--")
    plt.xlabel("r (px)")
    plt.ylabel("g(r)")
    plt.title("Defect Pair-Correlation Function")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_defect_pair_correlation.png"), dpi=300)
    plt.show()

summary = {
    "Selected_file": CSV_PATH,
    "Rows": len(df),
    "Grid_Nx": nx,
    "Grid_Ny": ny,
    "Grid_spacing_X_px": dx,
    "Grid_spacing_Y_px": dy,
    "Coherency_threshold": COHERENCY_THRESHOLD,
    "Valid_orientation_fraction": np.mean(valid),
    "Global_S2D": S_global,
    "Global_mean_orientation_deg": np.rad2deg(theta_global),
    "Mean_local_S2D": np.nanmean(S_local),
    "Median_local_S2D": np.nanmedian(S_local),
    "Correlation_length_xi_px": xi_px,
    "+1/2_candidate_defects": Nplus,
    "-1/2_candidate_defects": Nminus,
    "Total_candidate_defects": Ntotal,
    "Defect_density_per_px2": defect_density_px2,
    "Characteristic_defect_spacing_px": characteristic_spacing_px,
    "Mean_nearest_neighbor_px": mean_nn,
    "Pixel_size_um": PIXEL_SIZE_UM,
    "Correlation_length_um": xi_um,
    "Defect_density_per_um2": density_um2,
    "Mean_nearest_neighbor_um": mean_nn_um
}
summary_df = pd.DataFrame({"Parameter": list(summary.keys()), "Value": list(summary.values())})
summary_df.to_csv(os.path.join(OUTPUT_DIR, "analysis_summary.csv"), index=False)

print("\n" + "="*65)
print("ANALYSIS COMPLETE")
print("="*65)
print("Global S2D:", round(S_global, 4))
print("Mean local S2D:", round(np.nanmean(S_local), 4))
print("Correlation length xi:", xi_px, "px")
print("+1/2 defects:", Nplus)
print("-1/2 defects:", Nminus)
print("Total defects:", Ntotal)
print("Defect density:", defect_density_px2, "per px^2")
print("Mean nearest-neighbor:", mean_nn, "px")
print("\nResults saved in:")
print(os.path.abspath(OUTPUT_DIR))
print("\nIMPORTANT:")
print("S2D is an image-derived texture orientation order parameter.")
print("Defects are winding-derived candidates from the OrientationJ texture-orientation field.")
