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
from calibration.gof import Gof


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
    seeds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Helper function to run simulations, parse outputs, and clean up temporary files."""

    utils.od_to_file(config, sim_setup, input_od)
    simulations.run_multiple_simulations(config, sim_setup, seeds=seeds)
    df_simulated = parsing.parse_multiple_runs_data(config, sim_setup)
    utils.cleanup_results(config)
    return df_simulated


def run_spsa(
    config: dict[str, Path],
    sim_setup: dict[str, Any],
    spsa_setup: dict[str, Any],
    df_true: pd.DataFrame,
    input_od: pd.DataFrame,
    gof_calculator: Gof | None = None,
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
    # If a GOF calculator is not provided, keep backward-compatible behavior.
    if gof_calculator is None:
        gof_calculator = Gof()
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
        m = np.mean(OD)

        # --- diagnostics: ck influence (plus/minus perturbations) ---
        # Track perturbation magnitudes for each ga and summarize at end of iteration.
        plus_abs_means: list[float] = []
        plus_abs_medians: list[float] = []
        plus_abs_maxs: list[float] = []
        plus_rel_means: list[float] = []
        plus_rel_medians: list[float] = []
        plus_rel_maxs: list[float] = []

        minus_abs_means: list[float] = []
        minus_abs_medians: list[float] = []
        minus_abs_maxs: list[float] = []
        minus_rel_means: list[float] = []
        minus_rel_medians: list[float] = []
        minus_rel_maxs: list[float] = []

        yplus_list: list[float] = []
        yminus_list: list[float] = []

        for ga in range(0, params.G):
            # use the same replicate seeds for the + and - evaluations
            seeds_ga = np.random.normal(0, 10000, int(sim_setup["n_sumo_replicate"])).astype(
                "int32"
            )

            delta = (
                2 * np.random.binomial(n=1, p=0.5, size=input_od.shape[0]) - 1
            )  # Bernoulli distribution

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

            # diagnostics: plus perturbation size vs ODbase
            od_plus = OD.copy()
            od_abs_plus = np.abs(od_plus - ODbase)
            nonzero_mask_plus = ODbase != 0
            od_rel_plus = np.full_like(ODbase, np.nan, dtype=float)
            od_rel_plus[nonzero_mask_plus] = od_abs_plus[nonzero_mask_plus] / np.abs(
                ODbase[nonzero_mask_plus]
            )
            plus_abs_means.append(float(np.nanmean(od_abs_plus)))
            plus_abs_medians.append(float(np.nanmedian(od_abs_plus)))
            plus_abs_maxs.append(float(np.nanmax(od_abs_plus)))
            plus_rel_means.append(float(np.nanmean(od_rel_plus)))
            plus_rel_medians.append(float(np.nanmedian(od_rel_plus)))
            plus_rel_maxs.append(float(np.nanmax(od_rel_plus)))

            df_simulated = run_parse_cleanup(config, sim_setup, OD_plus, seeds=seeds_ga)
            y = gof_calculator.compute_gof(df_true, df_simulated)
            yplus = np.asarray(y)
            yplus_list.append(float(y))

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

            # diagnostics: minus perturbation size vs ODbase
            od_minus = OD.copy()
            od_abs_minus = np.abs(od_minus - ODbase)
            nonzero_mask_minus = ODbase != 0
            od_rel_minus = np.full_like(ODbase, np.nan, dtype=float)
            od_rel_minus[nonzero_mask_minus] = od_abs_minus[nonzero_mask_minus] / np.abs(
                ODbase[nonzero_mask_minus]
            )
            minus_abs_means.append(float(np.nanmean(od_abs_minus)))
            minus_abs_medians.append(float(np.nanmedian(od_abs_minus)))
            minus_abs_maxs.append(float(np.nanmax(od_abs_minus)))
            minus_rel_means.append(float(np.nanmean(od_rel_minus)))
            minus_rel_medians.append(float(np.nanmedian(od_rel_minus)))
            minus_rel_maxs.append(float(np.nanmax(od_rel_minus)))

            df_simulated = run_parse_cleanup(config, sim_setup, OD_minus, seeds=seeds_ga)
            y = gof_calculator.compute_gof(df_true, df_simulated)
            yminus = np.asarray(y)
            yminus_list.append(float(y))

            # reset OD matrix
            OD = ODbase.copy()

            # evaluate the gradient
            g_hat_tem = pd.DataFrame((yplus - yminus) / (2 * ck * delta))
            # append the estimated gradient to the iteration dataframe
            g_hat_it = pd.concat([g_hat_it, g_hat_tem], axis=1)

        # average gradient over perturbations
        g_hat = g_hat_it.mean(axis=1)
        list_g.append(abs(g_hat).mean())

        # --- diagnostics summary for plus/minus ---
        # Summarize across ga samples (G times). If G=1 these are just that single value.
        if len(yplus_list) > 0 and len(yminus_list) > 0:
            yplus_mean = float(np.mean(yplus_list))
            yminus_mean = float(np.mean(yminus_list))
            ydiff_abs = float(np.mean(np.abs(np.array(yplus_list) - np.array(yminus_list))))
            ydiff_abs_norm = float(ydiff_abs / (2 * ck)) if ck != 0 else float("nan")
        else:
            yplus_mean = float("nan")
            yminus_mean = float("nan")
            ydiff_abs = float("nan")
            ydiff_abs_norm = float("nan")

        plus_abs_mean = float(np.mean(plus_abs_means)) if plus_abs_means else float("nan")
        plus_abs_median = float(np.mean(plus_abs_medians)) if plus_abs_medians else float("nan")
        plus_abs_max = float(np.mean(plus_abs_maxs)) if plus_abs_maxs else float("nan")
        plus_rel_mean = float(np.mean(plus_rel_means)) if plus_rel_means else float("nan")
        plus_rel_median = float(np.mean(plus_rel_medians)) if plus_rel_medians else float("nan")
        plus_rel_max = float(np.mean(plus_rel_maxs)) if plus_rel_maxs else float("nan")

        minus_abs_mean = float(np.mean(minus_abs_means)) if minus_abs_means else float("nan")
        minus_abs_median = float(np.mean(minus_abs_medians)) if minus_abs_medians else float("nan")
        minus_abs_max = float(np.mean(minus_abs_maxs)) if minus_abs_maxs else float("nan")
        minus_rel_mean = float(np.mean(minus_rel_means)) if minus_rel_means else float("nan")
        minus_rel_median = float(np.mean(minus_rel_medians)) if minus_rel_medians else float("nan")
        minus_rel_max = float(np.mean(minus_rel_maxs)) if minus_rel_maxs else float("nan")

        # --- diagnostics for this iteration ---
        clip_lower_count = 0  # diff < -15%
        clip_upper_count = 0  # diff > +15%
        od_update_attempts = 0

        # snapshot base OD before applying this iteration's update (for deltas)
        ODbase_before_update = ODbase.copy()

        for i in range(1, int(np.fix(OD.max() / params.seg)) + 1):  #!!
            for f in range(0, OD.shape[1]):
                for e in range(0, OD.shape[0]):
                    if OD[e, f] > 4:
                        q = i * params.seg
                        p = q - params.seg
                        if OD[e, f] > p and OD[e, f] <= q:
                            if OD[e, f] == ODbase[e, f]:
                                OD[e, f] = OD[e, f] - ((ak * g_hat[e * OD.shape[0] + f] * q) / m)
                                od_update_attempts += 1
                            diff = (OD[e, f] - ODbase[e, f]) / ODbase[e, f]
                            # limit the change to within +/-15%
                            if diff < -0.15:
                                clip_lower_count += 1
                                OD[e, f] = ODbase[e, f] * 0.85
                            if diff > 0.15:
                                clip_upper_count += 1
                                OD[e, f] = ODbase[e, f] * 1.15

        # update the base OD matrix
        ODbase = OD.copy()

        # --- diagnostics: OD change stats after update (before running minimization) ---
        od_abs_change = np.abs(ODbase - ODbase_before_update)
        nonzero_mask = ODbase_before_update != 0
        od_rel_change = np.full_like(ODbase_before_update, np.nan, dtype=float)
        od_rel_change[nonzero_mask] = od_abs_change[nonzero_mask] / np.abs(
            ODbase_before_update[nonzero_mask]
        )

        mean_abs = float(np.nanmean(od_abs_change))
        med_abs = float(np.nanmedian(od_abs_change))
        max_abs = float(np.nanmax(od_abs_change))
        mean_rel = float(np.nanmean(od_rel_change))
        med_rel = float(np.nanmedian(od_rel_change))
        max_rel = float(np.nanmax(od_rel_change))

        # update the OD_min dataframe
        OD_min.iloc[:, 2] = pd.DataFrame(OD).stack().values

        # run simulation with updated OD
        print("Simulation %d . %d . minimization" % (iteration, ga))
        df_simulated = run_parse_cleanup(config, sim_setup, OD_min)
        y_min = gof_calculator.compute_gof(df_true, df_simulated)

        rmsn.append(y_min)

        print("Iteration NO. %d done" % iteration)
        print("RMSN = ", y_min)
        print(
            "Clipping: lower=%d upper=%d (attempted_updates=%d)"
            % (clip_lower_count, clip_upper_count, od_update_attempts)
        )
        print(
            "Perturbation (ck) | plus abs mean=%.6f median=%.6f max=%.6f | plus rel mean=%.6f median=%.6f max=%.6f"
            % (
                plus_abs_mean,
                plus_abs_median,
                plus_abs_max,
                plus_rel_mean,
                plus_rel_median,
                plus_rel_max,
            )
        )
        print(
            "Perturbation (ck) | minus abs mean=%.6f median=%.6f max=%.6f | minus rel mean=%.6f median=%.6f max=%.6f"
            % (
                minus_abs_mean,
                minus_abs_median,
                minus_abs_max,
                minus_rel_mean,
                minus_rel_median,
                minus_rel_max,
            )
        )
        print(
            "Objective split | yplus(mean)=%.6f yminus(mean)=%.6f | mean|yplus-yminus|=%.6f | / (2*ck)=%.6f"
            % (yplus_mean, yminus_mean, ydiff_abs, ydiff_abs_norm)
        )
        print(
            "OD change | abs mean=%.6f median=%.6f max=%.6f | rel mean=%.6f median=%.6f max=%.6f"
            % (mean_abs, med_abs, max_abs, mean_rel, med_rel, max_rel)
        )
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
