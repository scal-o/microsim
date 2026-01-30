import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration import gof, spsa, utils

# define run parameters
OD_MULTIPLIER = 0.9
RUN_WEIGHTS = {"counts": 0.5, "density": 1, "speeds": 0.0}
# RUN_WEIGHTS = {"counts": 1.0}


@click.group("spsa")
def spsa_cli():
    pass


@spsa_cli.command("run")
def run_calibration() -> None:
    """Run the SPSA calibration process."""

    # load configs
    config, sim_setup, spsa_setup = utils.load_spsa_config()

    # load true values and initial od matrix
    df_true = utils.load_true_counts(config, sim_setup)
    input_od = utils.load_start_od(config, sim_setup)

    # scale down the initial od matrix
    input_od.iloc[:, 2:] = input_od.iloc[:, 2:] * OD_MULTIPLIER

    # initialize goodness-of-fit calculator
    gof_calc = gof.Gof()
    gof_calc.set_excluded_from_config(config, 15)  # exclude detectors with >15 required sims
    gof_calc.update_weights(RUN_WEIGHTS)

    # run SPSA calibration
    spsa.run_spsa(config, sim_setup, spsa_setup, df_true, input_od, gof_calc)


@spsa_cli.command("plot")
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    help="Path to the SPSA results pickle file.",
)
@click.option(
    "--dir",
    "data_dir",
    type=click.Path(exists=True),
    help="Directory containing SPSA results pickle files to plot. If provided, --data is ignored.",
)
@click.option("--show", "-s", is_flag=True)
def plot_results(data: str, data_dir: str, show: bool) -> None:
    """Plot the results of the SPSA calibration process."""

    if data_dir:
        data_files = list(Path(data_dir).glob("*.pckl"))
        if not data_files:
            click.echo(f"No pickle files found in directory: {data_dir}")
            return
    else:
        data_files = [Path(data)]

    for data_path in data_files:
        results = pickle.load(open(data_path, "rb"))

        ## ======================================
        ## plot RMSN history
        # load rmsn history and components
        rmsns = results["rmsn_history"]
        rmsns_comp = pd.DataFrame.from_records(results["rmsn_components_history"])
        n = len(rmsns)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            range(n),
            rmsns,
            label="Weighted RMSN",
            color="black",
        )
        for col in rmsns_comp.columns:
            ax.plot(
                range(n),
                rmsns_comp[col],
                label=f"RMSN {col}",
                linestyle="--",
            )

        ax.legend()
        ax.set_xlabel("Iteration")
        ax.grid(True)
        fig.savefig(data_path.parent / f"{data_path.stem}.png", dpi=300)
        plt.show() if show else plt.close()

        ## ======================================
        ## plot predicted vs actual for best RMSN iteration
        # find best iteration index
        idx_best = rmsns.index(results["Best_RMSN"])

        df_history = results.get("df_history", [])
        if not df_history:
            click.echo(
                "No simulation history available in results; skipping predicted vs actual plot."
            )
            continue

        n_dfs = len(df_history)

        # choose the closest available df_history index if the exact one is missing
        if idx_best < n_dfs:
            chosen_idx = idx_best
        else:
            # if not all dfs are available, snapshots were taken every 10 iterations
            mapped = round(idx_best / 10)
            chosen_idx = max(0, min(int(mapped), n_dfs - 1))
            approx_iter = chosen_idx * 10
            print(
                f"df_history appears subsampled; using snapshot index {chosen_idx} (approx iteration {approx_iter})"
            )

        df_best = df_history[chosen_idx]

        # load true data
        config, sim_setup, _ = utils.load_spsa_config()
        df_true = utils.load_true_counts(config, sim_setup).fillna(0)

        # combine true and predicted data
        df_combined = pd.concat([df_true, df_best], axis=1)

        metrics = [
            ("true_counts", "simulated_counts"),
            ("true_speeds", "simulated_speeds"),
            ("true_density", "simulated_density"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (truth_col, pred_col) in zip(axes.flatten(), metrics):
            if truth_col not in df_combined.columns or pred_col not in df_combined.columns:
                ax.set_visible(False)
                continue

            x = df_combined[truth_col]
            y = df_combined[pred_col]
            mask = x.notna() & y.notna()
            if mask.sum() == 0:
                ax.set_visible(False)
                continue
            x = x[mask]
            y = y[mask]

            vmax = float(max(x.max(), y.max()))
            vmin = 0.0 - vmax * 0.05  # add 5% padding
            vmax = vmax * 1.05  # add 5% padding

            # set equal axes and limits
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_aspect("equal")

            # GEH shading for counts subplot
            is_counts = truth_col == "true_counts" and pred_col == "simulated_counts"
            if is_counts:
                xs = np.linspace(0.0, vmax, 400)
                x_ = np.linspace(vmin, 0.0, 10)
                sqrt_term = np.sqrt(16 * xs + 25)
                y_upper = (4 * xs + 25 + 5 * sqrt_term) / 4.0
                y_lower = (4 * xs + 25 - 5 * sqrt_term) / 4.0
                ax.fill_between(xs, y_upper, vmax, color="lightcoral", alpha=0.25, label="GEH>5")
                ax.fill_between(xs, vmin, y_lower, color="lightcoral", alpha=0.25)
                ax.fill_between(x_, vmin, vmax, color="lightcoral", alpha=0.25)
                ax.plot(xs, y_upper, color="red", linewidth=0.8, linestyle="--")
                ax.plot(xs, y_lower, color="red", linewidth=0.8, linestyle="--")
                ax.legend()

            # scatter points
            ax.scatter(x, y, s=8, color="tab:blue", alpha=0.7)
            ax.plot([vmin, vmax], [vmin, vmax], color="red", linewidth=1, zorder=4)

            ax.set_xlabel(truth_col.replace("true_", "").capitalize())
            ax.set_ylabel(pred_col.replace("simulated_", "").capitalize())
            ax.set_title(pred_col.replace("simulated_", "").capitalize())
            ax.grid(True)

        fig.suptitle(f"Predicted vs Actual (best RMSN at iteration {idx_best})")
        plt.savefig(data_path.parent / f"{data_path.stem}_pred_vs_actual_{idx_best}.png", dpi=300)
        plt.show() if show else plt.close()
