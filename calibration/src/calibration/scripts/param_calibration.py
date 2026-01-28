"""Script to perform automatic parameter calibration using Optuna and SUMO simulations."""

import click
import optuna

from calibration import gof, spsa, utils


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

    a = trial.suggest_int("a", 1, 150)
    A = trial.suggest_int("A", 10, 100)
    c = trial.suggest_float("c", 0.01, 1.0)

    spsa_setup = {
        "a": a,
        "c": c,
        "A": A,
        "alpha": 0.602,
        "gamma": 0.101,
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
    gof_calc = gof.Gof()
    gof_calc.set_excluded_from_config(config, 15)  # exclude detectors with >15 required sims
    gof_calc.update_weights({"counts": 0.65, "density": 0.25, "speeds": 0.1})
    # scale down the initial od matrix (found with starting rmsn analysis)
    input_od.iloc[:, 2:] = input_od.iloc[:, 2:] * 0.6
    # run SPSA calibration with the current parameters
    results = spsa.run_spsa(config, sim_setup, spsa_setup, df_true, input_od, gof_calc, trial)
    rmsn = results["Best_RMSN"]

    return rmsn


@click.command("calibrate")
def main():
    """Main entry point for the calibration script."""
    print("Starting calibration...")

    # define Median Pruner for Optuna
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15, interval_steps=1)
    db_url = "sqlite:///results/optuna_study.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage=db_url,
        load_if_exists=True,
        study_name="spsa_a_c_optimization_weighted_rmsn",
    )
    study.optimize(objective, n_trials=100)

    best = study.best_params
    print("Best parameters found:")
    for key, value in best.items():
        print(f"  {key}: {value}")

    with open("results/best_params.txt", "a") as f:
        f.write("Best parameters found:\n")
        for key, value in best.items():
            f.write(f"{key}: {value}\n")
        f.write("With corresponding RMSN: ")
        f.write(f"{study.best_value}\n\n")

    print("Calibration completed.")
