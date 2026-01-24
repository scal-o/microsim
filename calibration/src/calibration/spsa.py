"""
SPSA (Simultaneous Perturbation Stochastic Approximation) optimization for calibration.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from calibration import parsing, simulations, utils


@dataclass(frozen=True)
class SPSAConfig:
    a: float
    c: float
    A: float
    alpha: float
    gamma: float
    G: int
    N: int
    seg: float

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        return cls(
            a=config_dict["a"],
            c=config_dict["c"],
            A=config_dict["A"],
            alpha=config_dict["alpha"],
            gamma=config_dict["gamma"],
            G=config_dict["G"],
            N=config_dict["N"],
            seg=config_dict["seg"],
        )


def run_parse_cleanup(
    config: dict[str, Path],
    sim_setup: dict[str, Any],
    input_od: pd.DataFrame,
) -> pd.DataFrame:
    """Helper function to run simulations, parse outputs, and clean up temporary files."""

    utils.od_to_file(config, sim_setup, input_od)
    simulations.run_multiple_simulations(config, sim_setup)
    df_simulated = parsing.parse_multiple_runs_data(config, sim_setup)
    utils.cleanup_results(config)
    return df_simulated


def run_spsa(
    config: dict[str, Path],
    sim_setup: dict[str, Any],
    spsa_setup: dict[str, Any],
    df_true: pd.DataFrame,
    input_od: pd.DataFrame,
) -> dict[str, Any]:
    """Run the SPSA algorithm for calibrating the OD matrix.

    Parameters
    ----------
    config: dictionary containing config params
    sim_setup: dictionary containing sim params
    spsa_setup : dictionary containing SPSA setup params
    df_true : true counts DataFrame
    input_od : input OD matrix DataFrame

    Returns
    -------
    Dictionary containing:
        - 'Best_OD': best OD matrix found
        - 'Best_RMSN': best RMSN value
        - 'Best_simulatedCounts': simulated counts corresponding to the best OD
        - 'rmsn_history': list of RMSN values over iterations
        - 'ak_history': list of ak values over iterations
        - 'ck_history': list of ck values over iterations
        - 'g_history': list of gradient norms over iterations

    The dictionary is also picked and saved to the results directory specified in config.
    """

    # create params dataclass
    params = SPSAConfig.from_dict(spsa_setup)

    # create original matrix to be used as source of truth after perturbations
    ODbase = utils.od_to_matrix(input_od.iloc[:, 2])
    OD = ODbase.copy()

    # initial simulation to get starting conditions
    # run, parse outputs, and clean up
    print("Simulation 0 started")
    df_simulated = run_parse_cleanup(config, sim_setup, input_od)
    print("Simulation 0 completed")

    # clean true data
    df_true = df_true.fillna(0)

    # evaluate goodness of fit
    y = utils.gof_eval(df_true, df_simulated)
    print("Starting RMSN = ", y)
    print("========================================")

    # initialize histories and best values
    Best_OD: pd.DataFrame = input_od.iloc[:, 2]
    Best_RMSN: float = 100
    Best_simulatedCounts: pd.Series = df_simulated["simulated_counts"]

    # # SPSA iterations
    list_ak = []
    list_ck = []
    list_g = []

    rmsn: list[float] = []
    rmsn.append(y)

    # initialize od copies to perturb
    OD_plus = input_od.copy()
    OD_minus = input_od.copy()
    OD_min = input_od.copy()

    # spsa core loop
    for iteration in range(1, params.N + 1):
        # calculating gain sequence parameters
        ak = params.a / ((iteration + params.A) ** params.alpha)
        ck = params.c / (iteration**params.gamma)
        list_ak.append(ak)
        list_ck.append(ck)
        g_hat_it = pd.DataFrame()
        for ga in range(0, params.G):
            delta = (
                2 * np.random.binomial(n=1, p=0.5, size=input_od.shape[0]) - 1
            )  # Bernoulli distribution
            m = np.mean(OD)

            # perturbation
            for i in range(1, int(np.fix(OD.max() / params.seg)) + 1):  #!!
                for f in range(0, OD.shape[1]):
                    for e in range(0, OD.shape[0]):
                        if OD[e, f] > 0:
                            q = i * params.seg  # upper limit
                            p = q - params.seg  # lower limit
                            if OD[e, f] > p and OD[e, f] <= q:
                                if OD[e, f] == ODbase[e, f]:
                                    # propotional perturbation right side
                                    OD[e, f] = OD[e, f] + (ck * delta[e * OD.shape[0] + f]) * q / m

            del p, q

            # update OD_plus dataframe
            OD_plus.iloc[:, 2] = pd.DataFrame(OD).stack().values

            # run simulation with positive perturbation
            print("Simulation %d . %d . plus perturbation" % (iteration, ga))
            df_simulated = run_parse_cleanup(config, sim_setup, OD_plus)
            y = utils.gof_eval(df_true, df_simulated)
            yplus = np.asarray(y)

            # reset OD matrix
            OD = ODbase.copy()

            for i in range(1, int(np.fix(OD.max() / params.seg)) + 1):  #!!
                for f in range(0, OD.shape[1]):
                    for e in range(0, OD.shape[0]):
                        if OD[e, f] > 0:
                            q = i * params.seg
                            p = q - params.seg
                            if OD[e, f] > p and OD[e, f] <= q:
                                if OD[e, f] == ODbase[e, f]:
                                    # propotional perturbation left side
                                    OD[e, f] = OD[e, f] - (ck * delta[e * OD.shape[0] + f]) * q / m
            del p, q

            # update OD_minus dataframe
            OD_minus.iloc[:, 2] = pd.DataFrame(OD).stack().values

            # run simulation with negative perturbation
            print("Simulation %d . %d . minus perturbation" % (iteration, ga))
            df_simulated = run_parse_cleanup(config, sim_setup, OD_minus)
            y = utils.gof_eval(df_true, df_simulated)
            yminus = np.asarray(y)

            # reset OD matrix
            OD = ODbase.copy()

            # evaluate the gradient
            g_hat_tem = pd.DataFrame((yplus - yminus) / (2 * ck * delta))
            # append the estimated gradient to the iteration dataframe
            g_hat_it = pd.concat([g_hat_it, g_hat_tem], axis=1)

        # average gradient over perturbations
        g_hat = g_hat_it.mean(axis=1)
        list_g.append(abs(g_hat).mean())

        for i in range(1, int(np.fix(OD.max() / params.seg)) + 1):  #!!
            for f in range(0, OD.shape[1]):
                for e in range(0, OD.shape[0]):
                    if OD[e, f] > 4:
                        q = i * params.seg
                        p = q - params.seg
                        if OD[e, f] > p and OD[e, f] <= q:
                            if OD[e, f] == ODbase[e, f]:
                                OD[e, f] = OD[e, f] - ((ak * g_hat[e * OD.shape[0] + f] * q) / m)
                            diff = (OD[e, f] - ODbase[e, f]) / ODbase[e, f]
                            # limit the change to within +/-15%
                            if diff < -0.15:
                                OD[e, f] = ODbase[e, f] * 0.85
                            if diff > 0.15:
                                OD[e, f] = ODbase[e, f] * 1.15

        # update the base OD matrix
        ODbase = OD.copy()
        # update the OD_min dataframe
        OD_min.iloc[:, 2] = pd.DataFrame(OD).stack().values

        # run simulation with updated OD
        print("Simulation %d . %d . minimization" % (iteration, ga))
        df_simulated = run_parse_cleanup(config, sim_setup, OD_min)
        y_min = utils.gof_eval(df_true, df_simulated)

        rmsn.append(y_min)

        print("Iteration NO. %d done" % iteration)
        print("RMSN = ", y_min)
        print("Iterations remaining = %d" % (params.N - iteration))
        print("========================================")

        # check for best values
        if y_min < Best_RMSN:
            Best_OD = OD_min.iloc[:, 2]
            Best_RMSN = y_min
            Best_simulatedCounts = df_simulated["simulated_counts"]

    # create results dictionary
    results = {
        "Best_OD": Best_OD,
        "Best_RMSN": Best_RMSN,
        "Best_simulatedCounts": Best_simulatedCounts,
        "rmsn_history": rmsn,
        "ak_history": list_ak,
        "ck_history": list_ck,
        "g_history": list_g,
    }

    # save results to pickle file
    with open(config["RESULTS"] / "spsa_results.pckl", "wb") as f:  # for overall results
        pickle.dump(results, f)

    return results
