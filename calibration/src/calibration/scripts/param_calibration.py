"""Script to perform automatic parameter calibration using Optuna and SUMO simulations."""

import click
import optuna

from calibration import spsa, utils


def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna to minimize.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.

    Returns
    -------
    float
        The objective value to minimize (RMSN).
    """

    a = trial.suggest_int("a", 1, 100)
    c = trial.suggest_float("c", 0.01, 1.0)
    A = trial.suggest_int("A", 10, 100)
    alpha = trial.suggest_float("alpha", 0.1, 1.0)
    gamma = trial.suggest_float("gamma", 0.1, 1.0)

    spsa_setup = {
        "a": a,
        "c": c,
        "A": A,
        "alpha": alpha,
        "gamma": gamma,
        "N": 20,
        "G": 1,
        "seg": 5,
    }

    # load configs
    config, sim_setup, _ = utils.load_spsa_config(
        "calibration/configs/config.json",
        "calibration/configs/simulation_setups.json",
        "calibration/configs/spsa_setups.json",
    )

    # load true vals and od
    df_true = utils.load_true_counts(config, sim_setup)
    input_od = utils.load_start_od(config, sim_setup)
    # # scale down the initial od matrix (found with starting rmsn analysis)
    # input_od.iloc[:, 2:] = input_od.iloc[:, 2:] * 0.6
    # Run SPSA calibration with the current parameters
    results = spsa.run_spsa(config, sim_setup, spsa_setup, df_true, input_od)
    rmsn = results["Best_RMSN"]

    return rmsn


@click.command("calibrate")
def main():
    """Main entry point for the calibration script."""
    print("Starting calibration...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best = study.best_params
    print("Best parameters found:")
    for key, value in best.items():
        print(f"  {key}: {value}")

    with open("results/best_params.txt", "w") as f:
        for key, value in best.items():
            f.write(f"{key}: {value}\n")

    print("Calibration completed.")
