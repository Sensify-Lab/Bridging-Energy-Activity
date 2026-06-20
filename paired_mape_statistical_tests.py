"""Run paired statistical analyses of participant-level MAPE results.

The script analyzes the 1-day, 1-week, and 1-month forecasting horizons. For
each model, it compares the environmental, behavioral, and combined feature
sets with the load-profile-only baseline from the same participants.

Outputs:
    * Mean MAPE and sample standard deviation.
    * Paired t-test p-values against the load-profile baseline.
    * Wilcoxon signed-rank p-values against the same baseline.
    * Within-household Pearson correlations.

The analysis statements are retained from the original study code so the
printed numerical output remains unchanged.
"""

import numpy as np
import pandas as pd
from io import StringIO
from scipy.stats import ttest_rel, wilcoxon

# =============================================================================
# 1. ONE-DAY FORECAST HORIZON
# =============================================================================

# Raw participant-level MAPE values.
raw_csv = """participant,LP_XGB,LP_Ridge,LP_LSTM,LP_CNN,EN_XGB,EN_Ridge,EN_LSTM,EN_CNN,BE_XGB,BE_Ridge,BE_LSTM,BE_CNN,BEEN_XGB,BEEN_Ridge,BEEN_LSTM,BEEN_CNN
1,28.07,36.42,34.22,31.30,24.31,34.57,30.97,28.38,23.01,31.48,29.36,24.63,22.53,31.74,28.57,25.28
3,27.43,32.77,30.42,29.79,23.68,32.08,27.79,25.29,23.49,29.25,27.22,27.38,22.82,28.94,27.06,26.51
4,22.26,30.19,30.08,26.13,17.28,28.21,25.32,21.68,16.04,27.31,26.62,18.05,15.55,26.96,24.43,19.62
5,32.62,52.77,49.53,34.11,27.42,52.48,40.59,30.82,27.32,40.48,37.01,33.12,25.66,40.67,34.01,29.58
6,13.18,21.53,21.59,15.97,11.10,21.52,16.45,11.44,10.39,16.34,15.45,15.20,10.18,16.34,15.15,14.05
7,14.68,21.84,19.11,16.28,11.55,20.44,15.61,12.75,11.25,16.87,15.01,14.22,10.85,16.86,14.42,12.77
8,14.11,25.17,18.91,15.57,10.94,24.85,16.03,12.42,10.47,15.67,14.90,14.10,10.08,15.90,14.80,12.64
9,18.66,28.91,21.26,20.16,15.72,26.94,20.37,17.71,14.83,20.25,18.37,16.12,14.36,20.55,16.46,15.53
10,18.78,30.97,28.41,21.09,14.23,29.91,21.61,15.71,13.40,21.56,20.24,16.23,16.91,19.71,18.70,14.82
11,26.46,35.93,30.83,29.46,21.91,31.05,31.16,24.97,21.18,28.95,28.10,22.18,20.44,29.00,25.83,22.73
12,16.30,27.01,23.08,20.10,10.80,25.57,22.34,13.02,8.78,22.05,18.61,12.46,8.85,22.18,20.63,13.37
14,13.56,20.91,21.43,16.65,11.92,19.98,17.90,14.75,11.03,15.81,15.10,13.71,10.82,15.84,13.13,12.97
15,12.80,13.93,14.52,13.27,9.47,15.44,11.69,12.51,8.94,14.08,12.21,11.60,8.65,14.13,12.59,12.49
17,11.86,17.43,17.68,14.05,10.38,17.32,16.16,12.94,8.43,13.37,12.21,11.38,8.40,13.40,12.03,10.64
"""

df = pd.read_csv(StringIO(raw_csv))

# Verify the reported means and sample standard deviations.
print("Means and sample SDs (ddof = 1):\n")
verify = pd.DataFrame({
    "mean": df.drop(columns="participant").mean().round(2),
    "sd":   df.drop(columns="participant").std(ddof=1).round(2),
})
print(verify.to_string())
print()

# Build a lookup keyed by model and feature configuration.
models  = ["XGBoost", "Ridge", "LSTM", "CNN-LSTM"]
short_m = {"XGBoost":"XGB", "Ridge":"Ridge", "LSTM":"LSTM", "CNN-LSTM":"CNN"}
features = {
    "Load Profile":               "LP",
    "Environmental":              "EN",
    "Behavioral":                 "BE",
    "Behavioral + Environmental": "BEEN",
}

raw = {}
for m in models:
    for fname, fs in features.items():
        raw[(m, fname)] = df[f"{fs}_{short_m[m]}"].to_numpy()

# Convert p-values to conventional significance labels.
def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# Compare each feature configuration with its paired load-profile baseline.
print("=" * 78)
print("Paired comparisons vs. Load Profile baseline, n = 14")
print("=" * 78)

rows = []
for m in models:
    base = raw[(m, "Load Profile")]
    for fname in ["Environmental", "Behavioral", "Behavioral + Environmental"]:
        var = raw[(m, fname)]
        diff = base.mean() - var.mean()
        _, p_t = ttest_rel(base, var)
        _, p_w = wilcoxon(base, var)
        rows.append([m, fname, f"{diff:+.2f}",
                     f"{p_t:.4f}", sig(p_t),
                     f"{p_w:.4f}", sig(p_w)])

out = pd.DataFrame(rows, columns=["Model", "Variant", "Δ MAPE",
                                   "p (paired t)", "sig (t)",
                                   "p (Wilcoxon)", "sig (W)"])
print(out.to_string(index=False))

# Measure participant-level association with the load-profile baseline.
print("\n" + "=" * 78)
print("Within-household correlations across feature sets (Pearson r)")
print("=" * 78)
for m in models:
    lp = raw[(m, "Load Profile")]
    for fname in ["Environmental", "Behavioral", "Behavioral + Environmental"]:
        r = np.corrcoef(lp, raw[(m, fname)])[0, 1]
        print(f"  {m:10s}  LP vs {fname:30s}  r = {r:.3f}")

# =============================================================================
# 2. ONE-WEEK FORECAST HORIZON
# =============================================================================

import numpy as np
import pandas as pd
from io import StringIO
from scipy.stats import ttest_rel, wilcoxon

# Raw participant-level MAPE values.
raw_csv = """participant,LP_XGB,LP_Ridge,LP_LSTM,LP_CNN,EN_XGB,EN_Ridge,EN_LSTM,EN_CNN,BE_XGB,BE_Ridge,BE_LSTM,BE_CNN,BEEN_XGB,BEEN_Ridge,BEEN_LSTM,BEEN_CNN
1,27.95,36.24,34.25,31.00,23.71,33.62,30.24,25.78,23.09,32.75,31.38,26.40,22.92,32.74,28.86,24.73
3,27.63,33.09,32.23,28.77,24.55,32.75,29.07,25.92,23.84,30.48,26.82,26.65,23.42,30.41,27.61,26.65
4,22.28,29.94,29.44,25.87,16.79,28.86,26.06,21.19,14.71,26.56,24.88,18.52,14.49,26.59,23.90,17.84
5,32.26,52.63,50.59,35.80,29.05,52.91,45.06,34.24,28.75,48.43,45.99,38.76,27.70,48.77,40.24,32.03
6,13.55,22.10,21.91,15.92,12.46,22.10,20.28,13.52,10.59,18.90,18.44,13.79,10.41,18.81,16.09,13.14
7,14.72,21.89,18.29,17.30,12.25,20.68,17.47,14.16,11.51,18.48,16.48,13.55,11.27,18.20,15.90,13.23
8,14.43,25.42,19.91,15.02,11.96,24.62,16.63,13.08,11.19,17.84,15.25,14.29,10.92,17.84,14.58,14.71
9,18.44,28.81,21.01,18.97,16.08,27.10,22.26,16.23,15.22,22.28,20.23,17.83,15.12,22.36,18.82,17.12
10,19.06,34.26,30.30,22.78,14.76,31.04,26.50,16.44,13.89,24.75,23.86,19.13,13.45,23.82,22.59,17.53
11,26.11,35.59,30.45,27.19,21.96,30.51,28.72,24.09,21.02,29.84,28.05,25.75,20.63,29.74,26.95,21.73
12,16.59,27.05,26.33,19.26,9.88,25.57,22.04,13.47,8.25,24.60,20.13,12.90,8.09,23.66,19.18,11.97
14,13.84,21.05,21.12,18.65,11.96,16.75,17.98,16.91,11.12,15.96,16.37,15.83,11.34,17.77,15.09,13.24
15,13.01,14.20,15.97,13.57,9.65,15.39,13.73,11.11,9.18,14.60,12.11,13.66,8.89,14.67,13.41,14.04
17,12.35,17.38,18.21,14.85,8.40,13.40,12.03,10.64,10.73,17.26,15.95,11.50,9.58,15.59,13.90,12.59
"""

df = pd.read_csv(StringIO(raw_csv))

# Verify the reported means and sample standard deviations.
print("1-Week — means and sample SDs (ddof = 1):\n")
verify = pd.DataFrame({
    "mean": df.drop(columns="participant").mean().round(2),
    "sd":   df.drop(columns="participant").std(ddof=1).round(2),
})
print(verify.to_string())
print()

# Build a lookup keyed by model and feature configuration.
models  = ["XGBoost", "Ridge", "LSTM", "CNN-LSTM"]
short_m = {"XGBoost":"XGB", "Ridge":"Ridge", "LSTM":"LSTM", "CNN-LSTM":"CNN"}
features = {
    "Load Profile":               "LP",
    "Environmental":              "EN",
    "Behavioral":                 "BE",
    "Behavioral + Environmental": "BEEN",
}

raw = {}
for m in models:
    for fname, fs in features.items():
        raw[(m, fname)] = df[f"{fs}_{short_m[m]}"].to_numpy()

# Convert p-values to conventional significance labels.
def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# Compare each feature configuration with its paired load-profile baseline.
print("=" * 78)
print("1-Week — Paired comparisons vs. Load Profile baseline, n = 14")
print("=" * 78)

rows = []
for m in models:
    base = raw[(m, "Load Profile")]
    for fname in ["Environmental", "Behavioral", "Behavioral + Environmental"]:
        var = raw[(m, fname)]
        diff = base.mean() - var.mean()
        _, p_t = ttest_rel(base, var)
        _, p_w = wilcoxon(base, var)
        rows.append([m, fname, f"{diff:+.2f}",
                     f"{p_t:.4f}", sig(p_t),
                     f"{p_w:.4f}", sig(p_w)])

out = pd.DataFrame(rows, columns=["Model", "Variant", "Δ MAPE",
                                   "p (paired t)", "sig (t)",
                                   "p (Wilcoxon)", "sig (W)"])
print(out.to_string(index=False))

# Measure participant-level association with the load-profile baseline.
print("\n" + "=" * 78)
print("1-Week — Within-household correlations across feature sets (Pearson r)")
print("=" * 78)
for m in models:
    lp = raw[(m, "Load Profile")]
    for fname in ["Environmental", "Behavioral", "Behavioral + Environmental"]:
        r = np.corrcoef(lp, raw[(m, fname)])[0, 1]
        print(f"  {m:10s}  LP vs {fname:30s}  r = {r:.3f}")

# =============================================================================
# 3. ONE-MONTH FORECAST HORIZON
# =============================================================================

import numpy as np
import pandas as pd
from io import StringIO
from scipy.stats import ttest_rel, wilcoxon

# Raw participant-level MAPE values.
raw_csv = """participant,LP_XGB,LP_Ridge,LP_LSTM,LP_CNN,EN_XGB,EN_Ridge,EN_LSTM,EN_CNN,BE_XGB,BE_Ridge,BE_LSTM,BE_CNN,BEEN_XGB,BEEN_Ridge,BEEN_LSTM,BEEN_CNN
1,28.00,32.13,33.27,32.15,23.87,29.08,30.52,29.75,23.11,28.29,28.29,27.61,22.07,27.58,27.78,26.88
3,27.44,30.31,31.08,30.64,24.51,28.66,28.89,28.36,24.13,27.89,28.08,28.08,23.77,27.67,28.31,28.27
4,19.94,29.49,29.56,22.03,12.40,27.07,24.60,18.65,11.69,26.11,25.79,17.36,11.18,25.37,23.91,15.87
5,33.20,45.61,48.18,45.98,28.69,42.22,44.20,41.51,28.69,42.21,44.83,43.32,27.44,40.62,41.79,39.73
6,14.94,21.98,22.47,16.36,14.51,21.20,19.51,12.46,14.07,21.40,19.37,19.09,13.96,21.45,16.57,17.44
7,22.19,19.28,18.47,17.41,10.85,15.59,16.28,15.59,10.06,14.19,14.62,13.92,14.99,13.55,15.35,14.14
8,19.51,21.54,22.87,19.26,25.37,18.56,18.26,17.26,16.37,12.96,13.36,13.02,15.41,16.86,16.04,15.32
9,18.80,27.11,21.89,18.63,27.70,21.97,22.01,20.94,24.94,20.33,20.89,20.24,19.53,21.09,20.97,19.15
10,20.58,34.47,30.50,24.02,19.82,28.91,26.24,18.27,20.17,26.99,24.93,18.30,18.90,23.82,23.56,15.60
11,29.11,40.01,35.13,31.75,28.12,32.79,30.46,23.65,28.12,32.79,30.46,23.65,22.22,34.43,30.44,24.31
12,11.32,19.04,20.04,18.46,6.58,15.33,16.23,15.37,5.66,11.31,11.52,11.13,5.41,11.30,11.25,10.95
14,18.82,20.55,21.72,19.61,10.82,15.84,13.13,12.97,10.82,15.84,13.13,12.97,15.13,16.48,15.75,14.31
15,15.33,15.65,16.35,15.68,11.41,13.93,14.98,14.77,8.65,14.13,12.59,12.49,9.89,13.01,12.83,13.04
17,15.99,17.06,18.78,16.64,9.26,12.83,13.54,12.87,8.75,12.16,12.96,12.77,13.99,15.40,15.88,13.72
"""

df = pd.read_csv(StringIO(raw_csv))

# Verify the reported means and sample standard deviations.
print("1-Month — means and sample SDs (ddof = 1):\n")
verify = pd.DataFrame({
    "mean": df.drop(columns="participant").mean().round(2),
    "sd":   df.drop(columns="participant").std(ddof=1).round(2),
})
print(verify.to_string())
print()

# Build a lookup keyed by model and feature configuration.
models  = ["XGBoost", "Ridge", "LSTM", "CNN-LSTM"]
short_m = {"XGBoost":"XGB", "Ridge":"Ridge", "LSTM":"LSTM", "CNN-LSTM":"CNN"}
features = {
    "Load Profile":               "LP",
    "Environmental":              "EN",
    "Behavioral":                 "BE",
    "Behavioral + Environmental": "BEEN",
}

raw = {}
for m in models:
    for fname, fs in features.items():
        raw[(m, fname)] = df[f"{fs}_{short_m[m]}"].to_numpy()

# Convert p-values to conventional significance labels.
def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# Compare each feature configuration with its paired load-profile baseline.
print("=" * 78)
print("1-Month — Paired comparisons vs. Load Profile baseline, n = 14")
print("=" * 78)

rows = []
for m in models:
    base = raw[(m, "Load Profile")]
    for fname in ["Environmental", "Behavioral", "Behavioral + Environmental"]:
        var = raw[(m, fname)]
        diff = base.mean() - var.mean()
        _, p_t = ttest_rel(base, var)
        _, p_w = wilcoxon(base, var)
        rows.append([m, fname, f"{diff:+.2f}",
                     f"{p_t:.4f}", sig(p_t),
                     f"{p_w:.4f}", sig(p_w)])

out = pd.DataFrame(rows, columns=["Model", "Variant", "Δ MAPE",
                                   "p (paired t)", "sig (t)",
                                   "p (Wilcoxon)", "sig (W)"])
print(out.to_string(index=False))

# Measure participant-level association with the load-profile baseline.
print("\n" + "=" * 78)
print("1-Month — Within-household correlations across feature sets (Pearson r)")
print("=" * 78)
for m in models:
    lp = raw[(m, "Load Profile")]
    for fname in ["Environmental", "Behavioral", "Behavioral + Environmental"]:
        r = np.corrcoef(lp, raw[(m, fname)])[0, 1]
        print(f"  {m:10s}  LP vs {fname:30s}  r = {r:.3f}")
