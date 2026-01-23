from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from scipy.stats import t as tdist

# set input and output directories
INPUT_DIR = Path("data")
OUTPUT_DIR = Path("results") / "plots"

# set plotting options
sns.set_theme(style="whitegrid")
so.Plot.config.theme.update(sns.axes_style("whitegrid"))

# read dataframe data from test counts
data = pd.read_csv(INPUT_DIR / "t_test_counts100.csv", sep=",", header=None)
data = data.drop([0], axis=0)
data = data.drop([0], axis=1)

# compute mean, sd and tolerance for all detectors
stats = pd.DataFrame()
stats["mean"] = data.mean(axis=1)
stats["std"] = data.std(axis=1)
stats["tol"] = 0.1 * stats["mean"]

# set test parameters
confidence_level = 0.95
alpha = 1 - confidence_level
max_sim_n = 30


def plot_det_prop(num, det_n, title, output):
    (
        so.Plot()
        .add(so.Bars(), x=np.linspace(1, max_sim_n, max_sim_n), y=100 * np.array(num) / det_n)
        .limit(x=(0.2, max_sim_n + 0.2), y=(0, 100))
        .label(
            x="Number of simulations",
            y="Detectors (%)",
            title=title,
        )
        .save(OUTPUT_DIR / output, dpi=300)
    )


# create lists
prop = []
num = []

for n in range(1, max_sim_n + 1):
    # compute t-statistic
    t_val = tdist.ppf(1 - alpha / 2, n)

    # compute number of required sims
    stats[f"{n}"] = ((t_val * stats["std"]) / stats["tol"]) ** 2
    stats = stats.fillna(0)

    # check (for each link) if the number of required sims is under the t-student sample size
    stats[f"{n}"] = stats[f"{n}"] < (n + 1)

    # append number and proportion of significant detectors
    num.append(stats[f"{n}"].sum())
    prop.append((stats[f"{n}"]).sum() / stats.shape[0])

# create plots
plot_det_prop(num, len(stats), r"Detectors within 97.5% confidence level", "stat_signif.png")

# filter links with average flow < 5% of the global average
p = stats["mean"] > (0.05 * stats["mean"].mean())
filt_stats = stats[p]
filt_num = [filt_stats[f"{n}"].sum() for n in range(1, max_sim_n + 1)]
plot_det_prop(
    filt_num,
    len(filt_stats),
    r"Meaningful detectors within 97.5% confidence level",
    "stat_signif_filt.png",
)

# prop_adj = []
# for n in range(1, max_sim_n + 1):
#     prop_adj.append((filt_stats[f"{n}"]).sum() / filt_stats.shape[0])
