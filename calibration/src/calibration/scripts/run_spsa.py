import click

from calibration import gof, spsa, utils

# define run parameters
OD_MULTIPLIER = 0.6
RUN_WEIGHTS = {"counts": 0.65, "density": 0.25, "speeds": 0.1}


@click.command("spsa")
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
