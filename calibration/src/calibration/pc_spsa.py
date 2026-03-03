"""
PC-SPSA (Principal Components - Simultaneous Perturbation Stochastic Approximation) optimization for calibration.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from calibration.gof import Gof
from calibration.spsa import run_parse_cleanup

if TYPE_CHECKING:
    import optuna


@dataclass(frozen=True)
class PCSPSAConfig:
    a: float
    c: float
    A: float
    alpha: float
    gamma: float
    G: int
    N: int
    variance: float

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
            variance=config_dict["variance"],
        )


def fit_pca(config: dict[str, Path]) -> PCA:
    """Fit PCA on the starting OD matrix and return the fitted PCA object."""

    samples_matrix = pd.read_csv(config["NETWORK"] / "od_pca_matrix.csv", header=None, sep=";")

    pca = PCA()
    pca.fit(samples_matrix)

    return pca


def run_pc_spsa(
    config: dict[str, Path],
    sim_setup: dict[str, Any],
    spsa_setup: dict[str, Any],
    df_true: pd.DataFrame,
    input_od: pd.DataFrame,
    pca: sklearn.decomposition.PCA,
    gof_calculator: Gof | None = None,
    trial: optuna.trial.Trial | None = None,
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
    params = PCSPSAConfig.from_dict(spsa_setup)

    # check how many components will be used
    components = np.argmax(np.cumsum(pca.explained_variance_ratio_) > params.variance) + 1

    # apply PCA transformation to the OD matrix
    OD_pca = input_od.iloc[:, 2].values.reshape(1, -1)
    OD_pca = pca.transform(OD_pca)
    theta = OD_pca[0, :components]  # keep only components that meet variance threshold

    # initial simulation to get starting conditions
    # run, parse outputs, and clean up
    print("Simulation 0 started")
    df_simulated = run_parse_cleanup(config, sim_setup, input_od)
    print("Simulation 0 completed")

    # clean true data
    df_true = df_true.fillna(0)

    # evaluate goodness of fit
    # If a GOF calculator is not provided, keep backward-compatible behavior.
    if gof_calculator is None:
        gof_calculator = Gof()
    rmsn_components = gof_calculator.compute_rmsn_components(df_true, df_simulated)
    y = gof_calculator.compute_gof(df_true, df_simulated)
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

    list_dfs = []
    list_dfs.append(df_simulated)
    list_rmsn_components = []
    list_rmsn_components.append(rmsn_components)
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
            # use the same replicate seeds for the + and - evaluations
            seeds_ga = np.random.normal(0, 10000, int(sim_setup["n_sumo_replicate"])).astype(
                "int32"
            )

            delta = (
                2 * np.random.binomial(n=1, p=0.5, size=components) - 1
            )  # Bernoulli distribution

            # plus perturbation
            theta_plus = theta + theta * ck * delta
            theta_plus_full = np.zeros((1, pca.n_components_))
            theta_plus_full[0, :components] = theta_plus

            # update OD_plus dataframe
            od_plus_full = np.maximum(0, pca.inverse_transform(theta_plus_full).flatten())
            OD_plus.iloc[:, 2] = od_plus_full

            # run simulation with positive perturbation
            print("Simulation %d . %d . plus perturbation" % (iteration, ga))
            df_simulated = run_parse_cleanup(config, sim_setup, OD_plus, seeds=seeds_ga)
            y = gof_calculator.compute_gof(df_true, df_simulated)
            yplus = np.asarray(y)

            # minus perturbation
            theta_minus = theta - theta * ck * delta
            theta_minus_full = np.zeros((1, pca.n_components_))
            theta_minus_full[0, :components] = theta_minus

            # update OD_minus dataframe
            od_minus_full = np.maximum(0, pca.inverse_transform(theta_minus_full).flatten())
            OD_minus.iloc[:, 2] = od_minus_full

            # run simulation with negative perturbation
            print("Simulation %d . %d . minus perturbation" % (iteration, ga))
            df_simulated = run_parse_cleanup(config, sim_setup, OD_minus, seeds=seeds_ga)
            y = gof_calculator.compute_gof(df_true, df_simulated)
            yminus = np.asarray(y)

            # evaluate the gradient
            g_hat_tem = pd.DataFrame((yplus - yminus) / (2 * ck * delta))
            # append the estimated gradient to the iteration dataframe
            g_hat_it = pd.concat([g_hat_it, g_hat_tem], axis=1)

        # average gradient over perturbations
        g_hat = g_hat_it.mean(axis=1)
        list_g.append(abs(g_hat).mean())

        theta = theta - theta * ak * g_hat.values  # update in PCA space
        theta_full = np.zeros((1, pca.n_components_))
        theta_full[0, :components] = theta

        # update the OD_min dataframe
        od_min_full = np.maximum(0, pca.inverse_transform(theta_full).flatten())
        OD_min.iloc[:, 2] = od_min_full

        # run simulation with updated OD
        print("Simulation %d . %d . minimization" % (iteration, ga))
        df_simulated = run_parse_cleanup(config, sim_setup, OD_min)
        rmsn_components = gof_calculator.compute_rmsn_components(df_true, df_simulated)
        y_min = gof_calculator.compute_gof(df_true, df_simulated)

        list_dfs.append(df_simulated)
        list_rmsn_components.append(rmsn_components)
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

        # if using optuna, report intermediate objective value
        if trial is not None:
            trial.report(Best_RMSN, iteration)
            # handle pruning based on the intermediate value
            if trial.should_prune():
                # local import
                import optuna

                print("Trial pruned at iteration ", iteration)
                raise optuna.TrialPruned()

    # create results dictionary
    results = {
        "Best_OD": Best_OD,
        "Best_RMSN": Best_RMSN,
        "Best_simulatedCounts": Best_simulatedCounts,
        "rmsn_history": rmsn,
        "ak_history": list_ak,
        "ck_history": list_ck,
        "g_history": list_g,
        "df_history": list_dfs,
        "rmsn_components_history": list_rmsn_components,
    }

    # save results to pickle file
    pckl_path = (
        config["RESULTS"] / ".." / f"pc_spsa_results_a{params.a}_c{params.c}_A{params.A}.pckl"
    )
    print("Saving SPSA results to pickle file: ", pckl_path)
    with open(pckl_path, "wb") as f:  # for overall results
        pickle.dump(results, f)

    return results
