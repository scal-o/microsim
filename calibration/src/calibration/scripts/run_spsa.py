import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
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
    required=True,
)
@click.option(
    "--dir",
    type=click.Path(exists=True),
    help="Directory containing SPSA results pickle files to plot. If provided, --data is ignored.",
)
def plot_results(data: str, data_dir: str) -> None:
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

        # load rmsn history and components
        rmsns = results["rmsn_history"]
        rmsns_comp = pd.DataFrame.from_records(results["rmsn_components_history"])
        n = len(rmsns)

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(n),
            rmsns,
            label="Weighted RMSN",
            color="black",
        )
        for col in rmsns_comp.columns:
            plt.plot(
                range(n),
                rmsns_comp[col],
                label=f"RMSN {col}",
                linestyle="--",
            )

        plt.legend()
        plt.xlabel("Iteration")
        plt.savefig(data_path.parent / f"{data_path.stem}.png", dpi=300)
        plt.show()
