from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

# set input and output directories
INPUT_DIR = Path("data")
OUTPUT_DIR = Path("results") / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

# plot settings for detectors requiring "many" replications
many_rep_threshold = 15


def plot_det_prop(num, det_n, title, output):
    x = np.arange(1, max_sim_n + 1)
    y = 100 * np.array(num) / det_n

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, y, width=0.9)

    ax.set_xlim(0.2, max_sim_n + 0.2)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel("Detectors (%)")
    ax.set_title(title)
    ax.set_xticks(x)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output, dpi=300)
    plt.close(fig)


def plot_detectors_needing_many_reps(stats_df: pd.DataFrame, threshold: int, output: str):
    # get index of detectors not satisfying the threshold
    not_satisfied = stats_df[f"{threshold}"]
    not_satisfied = not_satisfied[~not_satisfied].index

    # return if all detectors satisfy the threshold
    if not_satisfied.empty:
        return

    # get average count
    avg = stats_df["mean"].mean()

    # filter dataset
    df = stats_df[["mean"]].loc[not_satisfied].copy()
    df = df.sort_values(by="mean", ascending=True)

    # X as categorical labels (strings)
    x_labels = df.index.astype(str).to_list()
    y = df["mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_labels, y)
    ax.text(0.05, 0.95, f"Average count: {avg:.2f}", transform=ax.transAxes, ha="left", va="top")

    ax.set_xlabel("Detector")
    ax.set_ylabel("Vehicles detected (mean)")
    ax.set_title(f"Detectors requiring more than {threshold} simulations")

    # Make labels readable when there are many detectors
    ax.tick_params(axis="x", labelrotation=90)
    ax.margins(x=0.01)
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / output, dpi=300)
    plt.close(fig)


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
plot_detectors_needing_many_reps(stats, many_rep_threshold, "detectors_over_15.png")

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
