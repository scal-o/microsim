import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from pathlib import Path

OUTPUT_DIR = Path("conf/peak_sim_output")

sns.set_theme(style="whitegrid")
so.Plot.config.theme.update(sns.axes_style("whitegrid"))

# read dataframe data from test counts
data = pd.read_csv(
    OUTPUT_DIR / "unified_detector_counts.csv", sep=",", header=None
)
data = data.drop([0], axis=0)
data = data.drop([0], axis=1)

# compute mean, sd and tolerance for all detectors
stats = pd.DataFrame()
stats["mean"] = data.mean(axis=1)
stats["std"] = data.std(axis=1)
stats["tol"] = 0.1 * stats["mean"]

# t-students values from table (for n-1 = 1:15) @97.5%
t_values = [
    12.71,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
]
t_n = [i for i in range(1, 16, 1)]
prop = []
pos_n = []

for n, t in zip(t_n, t_values):
    stats[f"{n}"] = ((2 * t * stats["std"]) / stats["tol"]) ** 2
    stats = stats.fillna(0)

    stats[f"{n}"] = stats[f"{n}"] < (n + 1)

    pos_n.append(stats[f"{n}"].sum())
    prop.append((stats[f"{n}"]).sum() / stats.shape[0])

(
    so.Plot()
    .add(so.Bars(), x=np.linspace(1, 15, 15), y=pos_n)
    .add(
        so.Path(color="k", linewidth=2, linestyle="--"),
        x=np.linspace(1, 15, 15),
        y=len(stats),
        label="Total detectors",
    )
    .limit(x=(0.2, 15.8))
    .label(
        x="Number of simulations",
        y="Number of detectors",
        title="Detectors with significant results (97.5%)",
    )
    .save(OUTPUT_DIR / "stat_signif.png")
)

s_mean = stats["mean"].mean()
p = stats["mean"] > 0.05 * s_mean

filt_stats = stats[p]
filt_pos_n = [filt_stats[f"{n}"].sum() for n in t_n]

(
    so.Plot()
    .add(so.Bars(), x=np.linspace(1, 15, 15), y=filt_pos_n)
    .add(
        so.Path(color="k", linewidth=2, linestyle="--"),
        x=np.linspace(1, 15, 15),
        y=len(filt_stats),
        label="Total detectors",
    )
    .limit(x=(0.2, 15.8))
    .label(
        x="Number of simulations",
        y="Number of detectors",
        title="Detectors (5th percentile +) with significant results (97.5%)",
    )
    .save(OUTPUT_DIR / "stat_signif_reduced.png")
)


prop_adj = []
for n in t_n:
    prop_adj.append((filt_stats[f"{n}"]).sum() / filt_stats.shape[0])
