
# ============================================================
# PRETTY MANUSCRIPT-STYLE PLOT
# Unlike-sign nearest-neighbor fraction vs shuffled null
# QDLC1 = 30:70
# QDLC2 = 50:50
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV_PATH = r"D:\New results\STEP10_FINAL_30_70_vs_50_50\STEP10_MASTER_PER_FOV_30_70_vs_50_50.csv"
OUTPUT_DIR = r"D:\New results\Figure_2_Defect_Analysis_Pretty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PANEL_LABEL = "(c)"
TITLE = "Unlike-sign nearest-neighbor fraction"
DPI = 600

OBS_BLUE = "#6FA8DC"
OBS_RED  = "#F28C8C"
EDGE_BLUE = "#2E75B6"
EDGE_RED  = "#C00000"
NULL_FACE = "#E6E6E6"
NULL_EDGE = "#7A7A7A"

def find_first_existing_column(df, aliases, label):
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find column for {label}\n"
        f"Tried: {aliases}\n"
        f"Available columns:\n{list(df.columns)}"
    )

df = pd.read_csv(CSV_PATH)

if "Composition" not in df.columns:
    raise KeyError("Column 'Composition' not found in CSV.")

observed_aliases = [
    "observed_unlike_sign_fraction",
    "observed_opposite_sign_fraction",
    "observed_unlike_sign_NN_fraction",
    "unlike_sign_NN_fraction_observed",
    "obs_unlike_sign_fraction",
    "obs_opposite_sign_fraction",
    "fraction_opposite_sign_observed",
    "observed_fraction",
    "observed_unlike_fraction",
    "unlike_sign_NN_fraction",
    "unlike_sign_NN_fraction_descriptive",
]

null_mean_aliases = [
    "shuffled_null_mean",
    "shuffle_null_mean",
    "shuffled_mean",
    "shuffled_mean_unlike_sign_fraction",
    "shuffle_mean_unlike_sign_fraction",
    "mean_shuffled_unlike_sign_fraction",
    "unlike_sign_fraction_shuffled_mean",
    "shuffled_opposite_sign_fraction_mean",
    "null_mean_opposite_sign_fraction",
    "null_mean",
    "shuffle_mean",
]

obs_col = find_first_existing_column(df, observed_aliases, "observed unlike-sign fraction")
null_col = find_first_existing_column(df, null_mean_aliases, "shuffled-null mean")

df[obs_col] = pd.to_numeric(df[obs_col], errors="coerce")
df[null_col] = pd.to_numeric(df[null_col], errors="coerce")
df = df.dropna(subset=[obs_col, null_col]).copy()

sample_map = {"30:70": "30:70", "50:50": "50:50"}
df["Sample"] = df["Composition"].astype(str).map(sample_map)
df = df[df["Sample"].isin(["30:70", "50:50"])].copy()

groups = ["30:70", "50:50"]
display_labels = ["QDLC1", "QDLC2"]
summary = []

for g in groups:
    sub = df[df["Sample"] == g]
    obs = sub[obs_col].to_numpy(float)
    nul = sub[null_col].to_numpy(float)
    summary.append({
        "Sample": g,
        "obs_mean": np.mean(obs),
        "obs_sd": np.std(obs, ddof=1) if len(obs) > 1 else 0.0,
        "null_mean": np.mean(nul),
        "null_sd": np.std(nul, ddof=1) if len(nul) > 1 else 0.0,
        "obs_raw": obs,
        "null_raw": nul
    })

fig, ax = plt.subplots(figsize=(4.6, 4.0))

centers = np.array([0.0, 1.15])
offset = 0.17
bar_w = 0.28

obs_positions = centers - offset
null_positions = centers + offset

obs_means = [s["obs_mean"] for s in summary]
obs_sds   = [s["obs_sd"] for s in summary]
null_means = [s["null_mean"] for s in summary]
null_sds   = [s["null_sd"] for s in summary]

obs_faces = [OBS_BLUE, OBS_RED]
obs_edges = [EDGE_BLUE, EDGE_RED]

for i in range(2):
    ax.bar(
        obs_positions[i], obs_means[i],
        width=bar_w, color=obs_faces[i], edgecolor=obs_edges[i],
        linewidth=1.2, alpha=0.9, zorder=2
    )

for i in range(2):
    ax.bar(
        null_positions[i], null_means[i],
        width=bar_w, color=NULL_FACE, edgecolor=NULL_EDGE,
        linewidth=1.2, hatch="////", zorder=2
    )

ax.errorbar(
    obs_positions, obs_means, yerr=obs_sds,
    fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2, zorder=4
)
ax.errorbar(
    null_positions, null_means, yerr=null_sds,
    fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2, zorder=4
)

for i, s in enumerate(summary):
    obs_raw = s["obs_raw"]
    null_raw = s["null_raw"]

    n1 = len(obs_raw)
    n2 = len(null_raw)
    jitter1 = np.linspace(-0.045, 0.045, n1) if n1 > 1 else np.array([0.0])
    jitter2 = np.linspace(-0.045, 0.045, n2) if n2 > 1 else np.array([0.0])

    x1 = np.full(n1, obs_positions[i]) + jitter1
    x2 = np.full(n2, null_positions[i]) + jitter2

    ax.scatter(x1, obs_raw, s=18, facecolors="none", edgecolors=obs_edges[i], linewidths=1.0, zorder=5)
    ax.scatter(x2, null_raw, s=18, facecolors="none", edgecolors=NULL_EDGE, linewidths=1.0, zorder=5)

ax.set_xlim(-0.45, 1.6)
ax.set_ylim(0, 1.0)
ax.set_xticks(centers)
ax.set_xticklabels(display_labels, fontsize=11)
ax.set_ylabel("Unlike-sign fraction", fontsize=12)
ax.tick_params(axis="both", direction="out", length=4, width=1.0, labelsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)

ax.text(-0.25, 1.07, PANEL_LABEL, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
ax.text(-0.02, 1.02, TITLE, transform=ax.transAxes, fontsize=12, ha="left", va="bottom")

legend_elements = [
    Patch(facecolor=OBS_BLUE, edgecolor=EDGE_BLUE, linewidth=1.0, label="Observed"),
    Patch(facecolor=NULL_FACE, edgecolor=NULL_EDGE, linewidth=1.0, hatch="////", label="Shuffled (null)")
]
ax.legend(handles=legend_elements, frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(1.02, 1.01))

plt.tight_layout()

png_path = os.path.join(OUTPUT_DIR, "pretty_unlike_sign_nn_fraction.png")
pdf_path = os.path.join(OUTPUT_DIR, "pretty_unlike_sign_nn_fraction.pdf")
svg_path = os.path.join(OUTPUT_DIR, "pretty_unlike_sign_nn_fraction.svg")
csv_summary_path = os.path.join(OUTPUT_DIR, "pretty_unlike_sign_nn_fraction_summary.csv")

fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

out_rows = []
for s in summary:
    out_rows.append({
        "Sample": "QDLC1" if s["Sample"] == "30:70" else "QDLC2",
        "Composition": s["Sample"],
        "Observed_mean": s["obs_mean"],
        "Observed_SD": s["obs_sd"],
        "Null_mean": s["null_mean"],
        "Null_SD": s["null_sd"],
        "N_obs": len(s["obs_raw"]),
        "N_null": len(s["null_raw"]),
    })
pd.DataFrame(out_rows).to_csv(csv_summary_path, index=False)

print("Saved files:")
print(png_path)
print(pdf_path)
print(svg_path)
print(csv_summary_path)
