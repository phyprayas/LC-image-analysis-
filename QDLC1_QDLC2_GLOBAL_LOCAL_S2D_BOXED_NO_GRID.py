import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"D:\New results\final output\STEP10_MASTER_PER_FOV_30_70_vs_50_50.csv"
OUTPUT_DIR = r"D:\New results\Figure_1_Order_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 600

df = pd.read_csv(CSV_PATH)
for c in ["Composition", "global_S2D", "mean_local_S2D"]:
    if c not in df.columns:
        raise KeyError(f"Missing {c}; available={list(df.columns)}")
df["Sample"] = df["Composition"].astype(str).map({"30:70":"QDLC1", "50:50":"QDLC2"})
df = df[df["Sample"].isin(["QDLC1","QDLC2"])].copy()

colors = {"G1":"#2E75B6", "G2":"#2CA02C", "L1":"#FF7F0E", "L2":"#D62728"}
positions = [0.0, 0.75, 2.25, 3.0]

def plot_group(ax, x, values, color):
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    m = np.mean(v); sd = np.std(v, ddof=1) if len(v)>1 else 0.0
    jitter = np.linspace(-0.065,0.065,len(v)) if len(v)>1 else np.array([0.0])
    ax.scatter(x+jitter, v, s=42, facecolor="white", edgecolor=color, linewidth=1.5, zorder=5)
    ax.plot([x-0.12,x+0.12],[m,m], color=color, linewidth=2.4, zorder=6)
    ax.errorbar(x,m,yerr=sd,fmt="none",ecolor=color,elinewidth=1.6,capsize=5,capthick=1.6,zorder=4)
    return m, sd

q1=df[df.Sample=="QDLC1"]; q2=df[df.Sample=="QDLC2"]
fig,ax=plt.subplots(figsize=(7.2,4.8))
plot_group(ax,positions[0],q1["global_S2D"],colors["G1"])
plot_group(ax,positions[1],q2["global_S2D"],colors["G2"])
plot_group(ax,positions[2],q1["mean_local_S2D"],colors["L1"])
plot_group(ax,positions[3],q2["mean_local_S2D"],colors["L2"])
ax.axvline((positions[1]+positions[2])/2,color="#8EC5E8",linestyle=":",linewidth=1.4,zorder=1)
ax.set_ylabel(r"$S_{2D}$",fontsize=14)
ax.set_xticks(positions); ax.set_xticklabels(["QDLC1","QDLC2","QDLC1","QDLC2"],fontsize=11)
ax.text((positions[0]+positions[1])/2,-0.10,r"Global $S_{2D}$",transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=12)
ax.text((positions[2]+positions[3])/2,-0.10,r"Local $S_{2D}$",transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=12)
ax.set_xlim(-0.45,3.45); ax.set_ylim(0,0.8)
for side in ["top","right","bottom","left"]:
    ax.spines[side].set_visible(True); ax.spines[side].set_linewidth(1.5)
ax.tick_params(axis="both",which="both",direction="in",top=True,right=True,length=5,width=1.2,labelsize=11)
ax.grid(False); ax.minorticks_off(); plt.tight_layout()
for ext in ["png","pdf","svg"]:
    kwargs={"dpi":DPI} if ext=="png" else {}
    fig.savefig(os.path.join(OUTPUT_DIR,f"QDLC1_QDLC2_GLOBAL_LOCAL_S2D.{ext}"),bbox_inches="tight",**kwargs)
plt.show()
