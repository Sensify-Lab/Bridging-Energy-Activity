# Bridging Energy and Activity: Forecasting Residential Electricity Use with Wearable Activity Tracker Data


**Authors:** Fatimah Mohammad Alhassan and Matthew Louis Mauriello

**Venue:** In The 13th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '26), June 22-25, 2026, Banff, AB, Canada.  

**DOI:** [https://doi.org/10.1145/3744256.3812579]  


## Description

This repository contains supplementary materials for the paper. It includes analysis code, individual household forecasting results, statistical outputs, and additional figures.

The study evaluates whether wearable-derived behavioral features improve household-level electricity forecasting. Forecasting results are reported for 17 participants across three horizons:

| Horizon | Prediction target |
|---|---:|
| 1-Day | 24 hours ahead |
| 1-Week | 168 hours ahead |
| 1-Month | 720 hours ahead |

Four model families are evaluated:

- Ridge regression
- XGBoost
- LSTM
- CNN-LSTM

Four feature sets are compared:

| Feature set | Description |
|---|---|
| Load Profile | Lag and rolling-window electricity-use features |
| Environmental | Weather and calendar features |
| Behavioral | Wearable-derived activity and sleep features |
| Behavioral + Environmental | Combined behavioral, environmental, and calendar features |

## Supplementary Individual Household Results

The supplementary individual household forecasting results report per-household MAPE values across horizons, models, and feature sets.

These results are derived forecasting-error outputs. They do not include raw electricity data, raw wearable data, survey responses, demographic information, or identifiable participant information.

## Repository Files

| File | Purpose | Output |
|---|---|---|
| `pvalue_table_day.py` | 1-Day statistical analysis | Console table: means, SDs, paired tests, Wilcoxon tests, correlations |
| `pvalue_table_week.py` | 1-Week statistical analysis | Console table: means, SDs, paired tests, Wilcoxon tests, correlations |
| `pvalue_table_month.py` | 1-Month statistical analysis | Console table: means, SDs, paired tests, Wilcoxon tests, correlations |
| `mape_by_horizon_subplots.py` | MAPE comparison across horizons, models, and feature sets | `mape_by_horizon_subplots_SE.pdf`, `mape_by_horizon_subplots_SE.png` |
| `delta_mape_plots_fixed.py` | Delta-MAPE comparison against Load Profile baseline | `delta_mape_plots_with_sig.pdf`, `delta_mape_plots_with_sig.png` |

Additional figures may be included in the repository as supplementary visualizations beyond those shown in the paper.

## Statistical Analysis

For each model and horizon, the scripts compare the Load Profile baseline against:

1. Environmental
2. Behavioral
3. Behavioral + Environmental

The scripts report:

- Mean MAPE
- Sample standard deviation
- Paired t-test p-values
- Wilcoxon signed-rank p-values
- Within-household Pearson correlations

Significance labels:

| Label | Threshold |
|---|---|
| `***` | `p < 0.001` |
| `**` | `p < 0.01` |
| `*` | `p < 0.05` |
| `ns` | Not significant |

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Usage

Run the statistical scripts:

```bash
python pvalue_table_day.py
python pvalue_table_week.py
python pvalue_table_month.py
```

Generate figures:

```bash
python mape_by_horizon_subplots.py
python delta_mape_plots_fixed.py
```


## Data Availability

To encourage further research and support reproducibility, the analysis code and supplementary individual household forecasting results are available in this repository.

The full participant-level dataset is not publicly released due to IRB and participant privacy requirements. Researchers interested in accessing the full dataset may contact the authors at  sensifylab@gmail.com. 

**Approved access will require a Data Usage Agreement consistent with the study's IRB protocol and institutional requirements.**

## Citation

```ACM Reference

Fatimah Mohammad Alhassan and Matthew Louis Mauriello. 2026. Bridging Energy and Activity: Forecasting Residential Electricity Use with Wearable Activity Tracker Data. In The 13th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '26), June 22--25, 2026, Banff, AB, Canada. ACM, New York, NY, USA 12 Pages. https://doi.org/10.1145/3744256.3812579
```
```bibtex
@inproceedings{TBD,
  title     = {Bridging Energy and Activity: Forecasting Residential Electricity Use with Wearable Activity Tracker Data},
  author    = {Fatimah Mohammad Alhassan and Matthew Louis Mauriello},
  booktitle = {Proceedings of The 13th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '26)},
  year      = {2026},
  location  = {Banff, AB, Canada},
  doi       = {https://doi.org/10.1145/3744256.3812579}
}
```

## Contact

For questions, issues, or collaboration inquiries, please open a GitHub issue or contact the authors directly at sensifylab@gmail.com
