# PSI STUDY STEP 6 - POM INTENSITY ANALYSIS
# Jupyter / VS Code friendly

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATASETS = [
    ("30_70_FOV1", r"D:\YOUR_PATH\30_70_FOV1.png", r"D:\YOUR_PATH\STEP4\30_70_FOV1\30_70_FOV1_STEP4_reliable_defects_only.csv"),
    ("30_70_FOV2", r"D:\YOUR_PATH\30_70_FOV2.png", r"D:\YOUR_PATH\STEP4\30_70_FOV2\30_70_FOV2_STEP4_reliable_defects_only.csv"),
    ("30_70_FOV3", r"D:\YOUR_PATH\30_70_FOV3.png", r"D:\YOUR_PATH\STEP4\30_70_FOV3\30_70_FOV3_STEP4_reliable_defects_only.csv"),
    ("30_70_FOV4", r"D:\YOUR_PATH\30_70_FOV4.png", r"D:\YOUR_PATH\STEP4\30_70_FOV4\30_70_FOV4_STEP4_reliable_defects_only.csv"),
    ("50_50_FOV1", r"D:\YOUR_PATH\50_50_FOV1.png", r"D:\YOUR_PATH\STEP4\50_50_FOV1\50_50_FOV1_STEP4_reliable_defects_only.csv"),
    ("50_50_FOV2", r"D:\YOUR_PATH\50_50_FOV2.png", r"D:\YOUR_PATH\STEP4\50_50_FOV2\50_50_FOV2_STEP4_reliable_defects_only.csv"),
    ("50_50_FOV3", r"D:\YOUR_PATH\50_50_FOV3.png", r"D:\YOUR_PATH\STEP4\50_50_FOV3\50_50_FOV3_STEP4_reliable_defects_only.csv"),
    ("50_50_FOV4", r"D:\YOUR_PATH\50_50_FOV4.png", r"D:\YOUR_PATH\STEP4\50_50_FOV4\50_50_FOV4_STEP4_reliable_defects_only.csv"),
]

OUTPUT_DIR = r"D:\YOUR_PATH\PSI_STUDY_STEP6"

UM_PER_PX = 10.0 / 157.0
PX_PER_UM = 1.0 / UM_PER_PX

MAX_RADIUS_UM = 8.0
RADIAL_BIN_UM = 0.5
N_PHI_BINS = 36
CORE_MAX_UM = 2.0
FAR_MIN_UM = 5.0
FAR_MAX_UM = 8.0
MIN_PIXELS_PER_BIN = 20
DPI = 600
SHOW_PLOTS_IN_NOTEBOOK = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_csv_robust(path):
    for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def load_gray_image(path):
    return np.asarray(Image.open(path).convert("L"), dtype=float)

def load_reliable_defects(path):
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        return df
    required = ["defect_id", "x_um", "y_um"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing {missing} in {path}. Columns: {list(df.columns)}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=required).copy()

def get_local_pixel_data(image, xc_um, yc_um):
    xc_px = xc_um * PX_PER_UM
    yc_px = yc_um * PX_PER_UM
    radius_px = int(np.ceil(MAX_RADIUS_UM * PX_PER_UM))
    h, w = image.shape
    x0 = max(0, int(np.floor(xc_px - radius_px)))
    x1 = min(w, int(np.ceil(xc_px + radius_px + 1)))
    y0 = max(0, int(np.floor(yc_px - radius_px)))
    y1 = min(h, int(np.ceil(yc_px + radius_px + 1)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx_px = xx - xc_px
    dy_px = yy - yc_px
    r_um = np.sqrt(dx_px**2 + dy_px**2) * UM_PER_PX
    phi = np.arctan2(dy_px, dx_px)
    intensity = image[y0:y1, x0:x1]
    valid = np.isfinite(intensity) & (r_um <= MAX_RADIUS_UM)
    return r_um[valid], phi[valid], intensity[valid]

def radial_intensity_profile(r_um, intensity):
    edges = np.arange(0, MAX_RADIUS_UM + RADIAL_BIN_UM, RADIAL_BIN_UM)
    rows = []
    for i in range(len(edges)-1):
        r0, r1 = edges[i], edges[i+1]
        vals = intensity[(r_um >= r0) & (r_um < r1)]
        if len(vals) >= MIN_PIXELS_PER_BIN:
            rows.append({
                "r_center_um": 0.5*(r0+r1),
                "N_pixels": len(vals),
                "Mean_intensity": np.mean(vals),
                "SD_intensity": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                "SEM_intensity": np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
                "Median_intensity": np.median(vals)
            })
    return pd.DataFrame(rows)

def azimuthal_intensity_profile(phi, r_um, intensity):
    use = (r_um >= 1.0) & (r_um <= 4.0)
    phi_use = np.mod(phi[use], 2*np.pi)
    intensity_use = intensity[use]
    edges = np.linspace(0, 2*np.pi, N_PHI_BINS + 1)
    rows = []
    for i in range(N_PHI_BINS):
        p0, p1 = edges[i], edges[i+1]
        vals = intensity_use[(phi_use >= p0) & (phi_use < p1)]
        if len(vals) >= MIN_PIXELS_PER_BIN:
            rows.append({
                "phi_center_deg": np.rad2deg(0.5*(p0+p1)),
                "N_pixels": len(vals),
                "Mean_intensity": np.mean(vals),
                "SD_intensity": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                "SEM_intensity": np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
            })
    return pd.DataFrame(rows)

def normalize_profile(df):
    if len(df) == 0:
        return df
    vals = df["Mean_intensity"].to_numpy(float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    out = df.copy()
    if vmax > vmin:
        out["Normalized_intensity"] = (vals-vmin)/(vmax-vmin)
    else:
        out["Normalized_intensity"] = 0.0
    return out

def core_far_stats(r_um, intensity):
    core = intensity[r_um <= CORE_MAX_UM]
    far = intensity[(r_um >= FAR_MIN_UM) & (r_um <= FAR_MAX_UM)]
    core_mean = np.mean(core) if len(core) else np.nan
    far_mean = np.mean(far) if len(far) else np.nan
    return {
        "core_N": len(core),
        "core_mean_intensity": core_mean,
        "far_N": len(far),
        "far_mean_intensity": far_mean,
        "core_minus_far_intensity": core_mean-far_mean if np.isfinite(core_mean) and np.isfinite(far_mean) else np.nan,
        "core_over_far_intensity": core_mean/far_mean if np.isfinite(core_mean) and np.isfinite(far_mean) and far_mean != 0 else np.nan,
    }

def plot_profile(x, y, xlabel, ylabel, title, outpath, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(x, y, marker="o", linewidth=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(direction="in", top=True, right=True)
    plt.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS_IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

def analyze_fov(label, image_path, defect_path):
    print("\nProcessing:", label)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    if not os.path.isfile(defect_path):
        raise FileNotFoundError(defect_path)

    image = load_gray_image(image_path)
    defects = load_reliable_defects(defect_path)

    out = os.path.join(OUTPUT_DIR, label)
    ensure_dir(out)

    if len(defects) == 0:
        return {"Label": label, "N_reliable_defects": 0}

    rows = []

    for _, defect in defects.iterrows():
        defect_id = int(defect["defect_id"])
        xc_um = float(defect["x_um"])
        yc_um = float(defect["y_um"])

        r_um, phi, intensity = get_local_pixel_data(image, xc_um, yc_um)

        radial = normalize_profile(radial_intensity_profile(r_um, intensity))
        azimuthal = normalize_profile(azimuthal_intensity_profile(phi, r_um, intensity))
        stats = core_far_stats(r_um, intensity)

        radial_csv = os.path.join(out, f"{label}_defect_{defect_id:03d}_radial_intensity.csv")
        az_csv = os.path.join(out, f"{label}_defect_{defect_id:03d}_azimuthal_intensity.csv")
        radial.to_csv(radial_csv, index=False)
        azimuthal.to_csv(az_csv, index=False)

        if len(radial):
            plot_profile(
                radial["r_center_um"], radial["Normalized_intensity"],
                r"Distance from defect center, $r$ ($\mu$m)",
                "Normalized POM intensity",
                f"{label} — defect {defect_id} radial intensity",
                os.path.join(out, f"{label}_defect_{defect_id:03d}_radial_intensity.png"),
                ylim=(-0.05, 1.05)
            )

        if len(azimuthal):
            plot_profile(
                azimuthal["phi_center_deg"], azimuthal["Normalized_intensity"],
                r"Azimuth $\phi$ (deg)",
                "Normalized POM intensity",
                f"{label} — defect {defect_id} azimuthal intensity",
                os.path.join(out, f"{label}_defect_{defect_id:03d}_azimuthal_intensity.png"),
                xlim=(0, 360), ylim=(-0.05, 1.05)
            )

        rows.append({
            "Label": label,
            "defect_id": defect_id,
            "x_um": xc_um,
            "y_um": yc_um,
            "s_fit": defect.get("s_fit", np.nan),
            "R2": defect.get("R2", np.nan),
            **stats
        })

    defect_summary = pd.DataFrame(rows)
    defect_summary_path = os.path.join(out, f"{label}_STEP6_intensity_summary.csv")
    defect_summary.to_csv(defect_summary_path, index=False)

    return {
        "Label": label,
        "N_reliable_defects": len(defect_summary),
        "Mean_core_intensity": defect_summary["core_mean_intensity"].mean(),
        "Mean_far_intensity": defect_summary["far_mean_intensity"].mean(),
        "Mean_core_minus_far": defect_summary["core_minus_far_intensity"].mean(),
        "Mean_core_over_far": defect_summary["core_over_far_intensity"].mean(),
        "Defect_summary_CSV": defect_summary_path
    }

ensure_dir(OUTPUT_DIR)
master_rows = []

for label, image_path, defect_path in DATASETS:
    master_rows.append(analyze_fov(label, image_path, defect_path))

master = pd.DataFrame(master_rows)
master["Composition"] = master["Label"].apply(lambda s: "30:70" if str(s).startswith("30_70") else "50:50")
master_path = os.path.join(OUTPUT_DIR, "PSI_STEP6_ALL_FOV_INTENSITY_SUMMARY.csv")
master.to_csv(master_path, index=False)

rows = []
for composition in ["30:70", "50:50"]:
    sub = master[master["Composition"] == composition]
    for metric in ["Mean_core_intensity", "Mean_far_intensity", "Mean_core_minus_far", "Mean_core_over_far"]:
        vals = sub[metric].dropna().to_numpy(float) if metric in sub.columns else np.array([])
        rows.append({
            "Composition": composition,
            "Metric": metric,
            "N_FOV": len(vals),
            "Mean": np.mean(vals) if len(vals) else np.nan,
            "SD": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "SEM": np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        })

pd.DataFrame(rows).to_csv(
    os.path.join(OUTPUT_DIR, "PSI_STEP6_COMPOSITION_SUMMARY.csv"),
    index=False
)

print("\nPSI STEP 6 COMPLETE")
print("Master summary:", master_path)
