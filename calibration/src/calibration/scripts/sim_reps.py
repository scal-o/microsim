import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calibration.replications import load_or_compute_required_replications
from calibration.utils import load_spsa_config

# set test parameters
confidence_level = 0.95
max_sim_n = 30

# load config
config, _, _ = load_spsa_config()

OUTPUT_DIR = config["PLOTS"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# load cached stats if available, otherwise compute and cache them
stats = load_or_compute_required_replications(
    config,
    confidence_level=confidence_level,
    max_sim_n=max_sim_n,
    tol_factor=0.1,
)


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


def plot_detectors_over_repl_threshold(stats_df: pd.DataFrame, threshold: int, output: str):
    # get detector ids not satisfying the threshold
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

    # X as categorical labels (detector ids as strings)
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


num = [stats[str(n)].astype(bool).sum() for n in range(1, max_sim_n + 1)]

# create plots
plot_det_prop(num, len(stats), r"Detectors within 95% confidence level", "stat_signif.png")
plot_detectors_over_repl_threshold(stats, 15, "detectors_over_15.png")
plot_detectors_over_repl_threshold(stats, 20, "detectors_over_20.png")

# filter links with average flow < 5% of the global average
p = stats["mean"] > (0.1 * stats["mean"].mean())
filt_stats = stats[p]
filt_num = [filt_stats[str(n)].astype(bool).sum() for n in range(1, max_sim_n + 1)]
plot_det_prop(
    filt_num,
    len(filt_stats),
    r"Meaningful detectors within 95% confidence level",
    "stat_signif_filt.png",
)

# prop_adj = []
# for n in range(1, max_sim_n + 1):
#     prop_adj.append((filt_stats[f"{n}"]).sum() / filt_stats.shape[0])
