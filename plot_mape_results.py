# -*- coding: utf-8 -*-
"""Create the MAPE figures for the ElecViz forecasting study.

This Google Colab script plots mean MAPE and delta MAPE across three forecast
horizons, four forecasting models, and four feature configurations. It also
creates the publication figure containing paired t-test significance symbols.

"""

# Colab integration for Drive access and browser downloads.
from google.colab import files
from google.colab import drive

drive.mount('/content/drive')

# =============================================================================
# 1. DELTA MAPE PLOT WITH MINIMAL LABELS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Across-household mean MAPE by horizon, model, and feature configuration.
data = [
    ["1-Day","XGBoost",19.34,15.76,14.90,14.72],
    ["1-Day","Ridge",  28.27,27.17,22.39,22.30],
    ["1-Day","LSTM",   25.79,22.43,20.74,19.84],
    ["1-Day","CNN-LSTM",21.71,18.17,17.88,17.36],
    ["1-Week","XGBoost",19.44,15.96,15.22,14.86],
    ["1-Week","Ridge",  28.55,26.81,24.48,24.35],
    ["1-Week","LSTM",   26.43,23.43,22.57,21.22],
    ["1-Week","CNN-LSTM",21.78,18.34,19.18,17.89],
    ["1-Month","XGBoost",21.08,18.13,16.96,16.80],
    ["1-Month","Ridge",  26.73,23.14,22.17,21.90],
    ["1-Month","LSTM",   26.45,22.77,21.55,21.49],
    ["1-Month","CNN-LSTM",23.47,20.17,19.26,19.57],
]
cols = ["Horizon","Model","LP_mean","ENV_mean","BEH_mean","BE_mean"]
df = pd.DataFrame(data, columns=cols)

# Positive delta MAPE indicates improvement over load profile only.
df["Δ ENV"] = df["LP_mean"] - df["ENV_mean"]
df["Δ BEH"] = df["LP_mean"] - df["BEH_mean"]
df["Δ BE"]  = df["LP_mean"] - df["BE_mean"]

horizons = ["1-Day","1-Week","1-Month"]
models   = ["XGBoost","Ridge","LSTM","CNN-LSTM"]
palette  = {'XGBoost': '#c53334', 'Ridge': '#4961d2', 'LSTM': '#f4987a', 'CNN-LSTM': '#dddcdc'}


panes = [("Δ ENV", "Load Profile vs. Environmental"),
         ("Δ BEH", "Load Profile vs. Behavioral"),
         ("Δ BE",  "Load Profile vs. Behavioral+Environmental")]

# Configure the visual style for the first grouped-bar figure.
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.bottom": False,   
    "axes.titleweight": "semibold",
    "legend.frameon": False,
})

SHOW_TOP_LABEL = True   # Set to False to suppress numeric labels.
DECIMALS = 1            # Keep annotations compact.

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
x = np.arange(len(horizons))
bar_w = 0.22  # Width of each grouped bar.

for ax, (col, title) in zip(axes, panes):
    # Collect values used to determine the panel's y-axis limit.
    all_y = []
    # Draw one bar for each model within every horizon.
    for k, m in enumerate(models):
        y = [df[(df["Horizon"]==h)&(df["Model"]==m)][col].values[0] for h in horizons]
        all_y.extend(y)
        ax.bar(x + (k-1.5)*bar_w, y, width=bar_w,
               color=palette[m], edgecolor="none",
               label=m if title == panes[0][1] else None, zorder=3)

    # Add horizontal reference lines only.
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_title(title)
    ax.set_xticks(x, horizons)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(axis='x', rotation=0)

    # Leave space above the tallest bar for its annotation.
    top = max(all_y) if all_y else 1.0
    ax.set_ylim(0, top * 1.18)

    # Label only the largest delta within each horizon.
    if SHOW_TOP_LABEL:
        for gi, h in enumerate(horizons):
            group_vals = []
            group_xpos = []
            for k, m in enumerate(models):
                v = df[(df["Horizon"]==h)&(df["Model"]==m)][col].values[0]
                group_vals.append(v)
                group_xpos.append(x[gi] + (k-1.5)*bar_w)
            idx = int(np.argmax(group_vals))
            xi, yi = group_xpos[idx], group_vals[idx]
            ax.text(xi, yi + 0.02*top, f"{yi:.{DECIMALS}f}",
                    ha="center", va="bottom", fontsize=9, color="#333")

axes[0].set_ylabel("Δ MAPE (percentage points)")

# Use one shared legend for all three panels.
fig.legend(handles=[plt.Rectangle((0,0),1,1,color=palette[m]) for m in models],
           labels=models, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))

fig.tight_layout(rect=[0,0,1,0.95])
plt.show()

# =============================================================================
# 2. DELTA MAPE PLOT WITH LABELS ON ALL BARS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Recreate the summary data for this standalone plotting section.
data = [
    ["1-Day","XGBoost",19.34,15.76,14.90,14.72],
    ["1-Day","Ridge",  28.27,27.17,22.39,22.30],
    ["1-Day","LSTM",   25.79,22.43,20.74,19.84],
    ["1-Day","CNN-LSTM",21.71,18.17,17.88,17.36],
    ["1-Week","XGBoost",19.44,15.96,15.22,14.86],
    ["1-Week","Ridge",  28.55,26.81,24.48,24.35],
    ["1-Week","LSTM",   26.43,23.43,22.57,21.22],
    ["1-Week","CNN-LSTM",21.78,18.34,19.18,17.89],
    ["1-Month","XGBoost",21.08,18.13,16.96,16.80],
    ["1-Month","Ridge",  26.73,23.14,22.17,21.90],
    ["1-Month","LSTM",   26.45,22.77,21.55,21.49],
    ["1-Month","CNN-LSTM",23.47,20.17,19.26,19.57],
]
cols = ["Horizon","Model","LP_mean","ENV_mean","BEH_mean","BE_mean"]
df = pd.DataFrame(data, columns=cols)

# Calculate percentage-point reductions from the load-profile baseline.
df["Δ(ENV-LP)"] = df["LP_mean"] - df["ENV_mean"]
df["Δ(BEH-LP)"] = df["LP_mean"] - df["BEH_mean"]
df["Δ(BE -LP)"] = df["LP_mean"] - df["BE_mean"]

horizons = ["1-Day","1-Week","1-Month"]
models   = ["XGBoost","Ridge","LSTM","CNN-LSTM"]




model_palette = {'XGBoost': '#c53334', 'Ridge': '#4961d2', 'LSTM': '#f4987a', 'CNN-LSTM': '#dddcdc'}


comparisons = [
    ("Δ(ENV-LP)", "Load Profile vs. Environmental"),
    ("Δ(BEH-LP)", "Load Profile vs. Behavioral"),
    ("Δ(BE -LP)", "Load Profile vs. Behavioral+Environmental")
]

SHOW_TOP_LABEL = True   # Retained from the original plotting configuration.
DECIMALS = 1            # Decimal places shown above each bar.


fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
x = np.arange(len(horizons))
bar_w = 0.18

for ax, (col, title) in zip(axes, comparisons):
    # Plot model-specific bars within each horizon.
    all_y = []
    for k, m in enumerate(models):
        y = [df[(df["Horizon"]==h)&(df["Model"]==m)][col].values[0] for h in horizons]
        all_y.extend(y)
        ax.bar(x + (k-1.5)*bar_w, y, width=bar_w,
               color=model_palette[m], edgecolor="none",
               label=m if title==comparisons[0][1] else None)
        # Annotate every bar with its delta MAPE.
        for xi, yi in zip(x + (k-1.5)*bar_w, y):
            ax.text(xi, yi + 0.03*max(1, max(all_y)), f"{yi:.{DECIMALS}f}",
                    ha="center", va="bottom", fontsize=8)

    # Scale each panel using its observed maximum.
    max_y = max(all_y) if all_y else 1.0
    ax.set_ylim(0, max_y*1.15)

    ax.set_title(title)
    ax.set_xticks(x, horizons)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())  # values are in percent units
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

axes[0].set_ylabel("Δ MAPE (percentage points)")
fig.legend(handles=[plt.Rectangle((0,0),1,1, color=model_palette[m]) for m in models],
           labels=models, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0,0,1,0.95])
plt.show()

import matplotlib.pyplot as plt
from google.colab import files
# Save the active figure in the Colab runtime.
plt.savefig('delta_mape_plots_with_sig.pdf')

# Trigger the original browser download.
files.download('delta_mape_plots_with_sig.pdf')

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# Obtain the coolwarm color map for palette inspection.
cmap = sns.color_palette("coolwarm", as_cmap=True)

# Sample four fixed positions from the color map.
blue   = cmap(0.05)   # near the blue end
gray   = cmap(0.50)   # center → gray/neutral
orange = cmap(0.75)   # warm light orange
red    = cmap(0.95)   # near the red end

# Assign sampled colors to model names.
model_palette = {
    "XGBoost":   red,
    "Ridge":     blue,
    "LSTM":      orange,
    "CNN-LSTM":  gray,
}

# Convert the sampled colors to hexadecimal strings and print them.
model_palette_hex = {k: to_hex(v) for k, v in model_palette.items()}
print(model_palette_hex)

# =============================================================================
# 3. MEAN MAPE BY HORIZON WITH STANDARD-ERROR BARS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.patches import Patch

from math import sqrt
from scipy.stats import t

# Apply a consistent publication-oriented Seaborn style.
sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
#EDGE = "#4a4a4a"
#LW = 0.6
CAP = 3
DPI = 300
TITLE_FONTSIZE = 12
TICK_ROT = 30
LABEL_FONTSIZE = 12
TICK_FONTSIZE  = 12
LEGEND_FONTSIZE = 12

# Keep feature-set colors consistent across all horizon panels.
palette_feat = {
    "Load Profile": "#c53334",                 # soft blue
    "Environmental": "#4961d2",                # soft pink
    "Behavioral": "#f4987a",                   # soft yellow
    "Behavioral + Environmental": "#dddcdc",   # soft green
}

data = [
    ["1-Day","XGBoost",19.34,15.76,14.90,14.72,6.35,5.59,5.88,5.56],
    ["1-Day","Ridge",28.27,27.17,22.39,22.30,9.45,9.01,7.58,7.57],
    ["1-Day","LSTM",25.79,22.43,20.74,19.84,8.68,7.55,7.19,6.52],
    ["1-Day","CNN-LSTM",21.71,18.17,17.88,17.36,6.52,6.03,6.24,5.75],
    ["1-Week","XGBoost",19.44,15.96,15.22,14.86,6.16,6.02,5.96,5.81],
    ["1-Week","Ridge",28.55,26.81,24.48,24.35,9.47,9.71,8.66,8.73],
    ["1-Week","LSTM",26.43,23.43,22.57,21.22,8.84,8.27,8.36,7.34],
    ["1-Week","CNN-LSTM",21.78,18.34,19.18,17.89,6.36,6.57,7.37,5.83],
    ["1-Month","XGBoost",21.08,18.13,16.96,16.80,5.92,7.87,7.69,5.77],
    ["1-Month","Ridge",26.73,23.14,22.17,21.90,8.90,8.42,9.00,8.39],
    ["1-Month","LSTM",26.45,22.77,21.55,21.49,8.45,8.42,9.34,8.23],
    ["1-Month","CNN-LSTM",23.47,20.17,19.26,19.57,8.32,7.75,8.51,7.61],
]
cols = ["Horizon","Model",
        "Load Profile avg.","With Environmental avg.","With Behavior avg.","With Behavior and Environmental avg.",
        "Load Profile std","With Environmental std","With Behavior std","With Behavior and Environmental std"]
df = pd.DataFrame(data, columns=cols)

# Enforce a stable display order for horizons and models.
horizon_order = ["1-Day","1-Week","1-Month"]
model_order   = ["XGBoost","Ridge","LSTM","CNN-LSTM"]
df["Horizon"] = pd.Categorical(df["Horizon"], horizon_order, True)
df["Model"]   = pd.Categorical(df["Model"],   model_order,   True)
df = df.sort_values(["Horizon","Model"])

variants = [
    ("Load Profile", "Load Profile avg.","Load Profile std"),
    ("Environmental","With Environmental avg.","With Environmental std"),
    ("Behavioral","With Behavior avg.","With Behavior std"),
    ("Behavioral + Environmental","With Behavior and Environmental avg.","With Behavior and Environmental std"),
]

# Convert sample standard deviations to standard errors for 14 households.
n = 14
scale = 1 / sqrt(n)      # SE = SD / sqrt(n).

se_cols = {}
for name, mean_col, std_col in variants:
    se_col = std_col.replace(" std", " se")
    df[se_col] = df[std_col] * scale
    se_cols[name] = se_col


# Define grouped-bar positions and dimensions.
x = np.arange(len(model_order))
bar_w = 0.18
offsets = (np.arange(len(variants)) - 1.5) * bar_w

# Determine the shared y-axis limit from mean plus standard error.
ymax = np.ceil((
    df[[v[1] for v in variants]].to_numpy().max() +                # max mean
    df[[se_cols[v[0]] for v in variants]].to_numpy().max()         # max SE
) / 2) * 2 + 2

errkw = dict(ecolor="#3a3a3a", elinewidth=1.0, capsize=CAP, capthick=1.0)

# Draw one panel for each forecast horizon.
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, h in zip(axes, horizon_order):
    sub = df[df["Horizon"] == h].set_index("Model").loc[model_order]
    for k, (name, mean_col, std_col) in enumerate(variants):
        ax.bar(
            x + offsets[k],
            sub[mean_col].to_numpy(),
            yerr=sub[se_cols[name]].to_numpy(),   # Error bars show SE.
            width=bar_w,
            color=palette_feat[name],
            edgecolor="none", linewidth=0,
            error_kw=errkw,
            label=name if h == horizon_order[0] else None,
            zorder=3
        )
    ax.set_title(h, fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x, model_order, rotation=TICK_ROT, ha="right")
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    # Keep tick labels readable in the exported figure.
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

axes[0].set_ylabel("MAPE (%)", fontsize=LABEL_FONTSIZE)

# Add one shared legend for the four feature configurations.
handles = [Patch(facecolor=palette_feat[v[0]], edgecolor='none', label=v[0]) for v in variants]
fig.legend(handles=handles, bbox_to_anchor=(0.5, 0.98), loc="upper center", ncol=4, frameon=False, title="", fontsize=LEGEND_FONTSIZE)
#fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.subplots_adjust(top=0.86)

fig.tight_layout(rect=[0, 0.05, 1, 0.94])

plt.savefig("mape_by_horizon_subplots_SE.pdf",
            bbox_inches="tight", pad_inches=0.05)
plt.savefig("mape_by_horizon_subplots_SE.png",
            dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()

from google.colab import files
files.download("mape_by_horizon_subplots_SE.pdf")
files.download("mape_by_horizon_subplots_SE.png")   # Raster preview.



# =============================================================================
# 4. DELTA MAPE WITH PAIRED T-TEST SIGNIFICANCE SYMBOLS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Typography is tuned for a figure scaled to approximately ``\textwidth``.
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":          12,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "ytick.major.width":  0.8,
    "xtick.major.width":  0.8,
})

# Recreate the mean-MAPE table used in the publication figure.
data = [
    ["1-Day","XGBoost", 19.34,15.76,14.90,14.72],
    ["1-Day","Ridge",   28.27,27.17,22.39,22.30],
    ["1-Day","LSTM",    25.79,22.43,20.74,19.84],
    ["1-Day","CNN-LSTM",21.71,18.17,17.88,17.36],
    ["1-Week","XGBoost", 19.44,15.96,15.22,14.86],
    ["1-Week","Ridge",   28.55,26.81,24.48,24.35],
    ["1-Week","LSTM",    26.43,23.43,22.57,21.22],
    ["1-Week","CNN-LSTM",21.78,18.34,19.18,17.89],
    ["1-Month","XGBoost", 21.08,18.13,16.96,16.80],
    ["1-Month","Ridge",   26.73,23.14,22.17,21.90],
    ["1-Month","LSTM",    26.45,22.77,21.55,21.49],
    ["1-Month","CNN-LSTM",23.47,20.17,19.26,19.57],
]
cols = ["Horizon","Model","LP_mean","ENV_mean","BEH_mean","BE_mean"]
df = pd.DataFrame(data, columns=cols)
df["Δ(ENV-LP)"] = df["LP_mean"] - df["ENV_mean"]
df["Δ(BEH-LP)"] = df["LP_mean"] - df["BEH_mean"]
df["Δ(BE -LP)"] = df["LP_mean"] - df["BE_mean"]

# Symbols are obtained from paired t-tests against load profile only.
sig_map = {
    ("1-Day",  "XGBoost"):  ("***", "***", "***"),
    ("1-Day",  "Ridge"):    ("*",   "***", "***"),
    ("1-Day",  "LSTM"):     ("***", "***", "***"),
    ("1-Day",  "CNN-LSTM"): ("***", "***", "***"),
    ("1-Week", "XGBoost"):  ("***", "***", "***"),
    ("1-Week", "Ridge"):    ("**",  "***", "***"),
    ("1-Week", "LSTM"):     ("***", "***", "***"),
    ("1-Week", "CNN-LSTM"): ("***", "**",  "***"),
    ("1-Month","XGBoost"):  ("",    "**",  "***"),   # ENV: p >= 0.05.
    ("1-Month","Ridge"):    ("***", "***", "***"),
    ("1-Month","LSTM"):     ("***", "***", "***"),
    ("1-Month","CNN-LSTM"): ("***", "***", "***"),
}

horizons = ["1-Day","1-Week","1-Month"]
models   = ["XGBoost","Ridge","LSTM","CNN-LSTM"]
model_palette = {"XGBoost":"#c53334", "Ridge":"#4961d2",
                 "LSTM":"#f4987a",    "CNN-LSTM":"#b5b5b5"}

comparisons = [
    ("Δ(ENV-LP)", "Load Profile vs. Environmental",            0),
    ("Δ(BEH-LP)", "Load Profile vs. Behavioral",               1),
    ("Δ(BE -LP)", "Load Profile vs. Behavioral + Environmental", 2),
]

# Plot delta MAPE values and their significance symbols.
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
x = np.arange(len(horizons))
bar_w = 0.20

for ax, (col, title, sig_idx) in zip(axes, comparisons):
    all_y = []
    for k, m in enumerate(models):
        y = [df[(df["Horizon"] == h) & (df["Model"] == m)][col].values[0]
             for h in horizons]
        all_y.extend(y)

        ax.bar(x + (k - 1.5) * bar_w, y, width=bar_w,
               color=model_palette[m], edgecolor="white", linewidth=0.6,
               label=m if title == comparisons[0][1] else None, zorder=3)

        for hi, (xi, yi) in enumerate(zip(x + (k - 1.5) * bar_w, y)):
            ax.text(xi, yi + 0.10, f"{yi:.1f}",
                    ha="center", va="bottom",
                    fontsize=9.5, color="#1a1a1a", zorder=4)
            star = sig_map[(horizons[hi], m)][sig_idx]
            if star:
                ax.text(xi, yi + 0.55, star,
                        ha="center", va="bottom",
                        fontsize=12, fontweight="bold",
                        color="#1a1a1a", zorder=4)

    ax.set_title(title, pad=8)
    ax.set_xticks(x, horizons)
    ax.set_ylim(0, max(all_y) * 1.30)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Δ MAPE (percentage points)")

fig.legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=model_palette[m]) for m in models],
    labels=models, ncol=4, frameon=False,
    loc="upper center", bbox_to_anchor=(0.5, 1.005),
    handlelength=1.3, handleheight=1.0, columnspacing=2.2,
)

fig.text(
    0.5, 0.012,
    r"Paired $t$-test vs. Load Profile baseline:   "
    r"$^{*}\,p<0.05$        $^{**}\,p<0.01$        $^{***}\,p<0.001$",
    ha="center", va="bottom", fontsize=10.5, style="italic",
)

fig.tight_layout(rect=[0, 0.05, 1, 0.94])

plt.savefig("delta_mape_plots_with_sig.pdf",
            bbox_inches="tight", pad_inches=0.05)
plt.savefig("delta_mape_plots_with_sig.png",
            dpi=300, bbox_inches="tight", pad_inches=0.05)
# Retain the repeated saves from the original script for identical behavior.
plt.savefig("delta_mape_plots_with_sig.pdf",
            bbox_inches="tight", pad_inches=0.05)
plt.savefig("delta_mape_plots_with_sig.png",
            dpi=300, bbox_inches="tight", pad_inches=0.05)

plt.show()

from google.colab import files
files.download("delta_mape_plots_with_sig.pdf")
files.download("delta_mape_plots_with_sig.png")   # Raster preview.
