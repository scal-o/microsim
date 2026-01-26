import numpy as np
import pandas as pd


def compute_rmsn(true: pd.Series, simulated: pd.Series) -> float:
    """
    Compute the Root Mean Square Normalized (RMSN) error between true and simulated values.

    Parameters
    ----------
    true : pd.Series
        Series containing the true values.
    simulated : pd.Series
        Series containing the simulated values.

    Returns
    -------
    float
        The RMSN error.
    """
    n = len(true)
    sum_diff: float = ((simulated - true) ** 2).sum()
    sum_true: float = true.sum()
    RMSN: float = np.sqrt(n * sum_diff) / sum_true
    return RMSN


def compute_rmsn_components(true: pd.DataFrame, simulated: pd.DataFrame) -> dict[str, float]:
    """
    Compute RMSN for counts, speeds, and density.

    Parameters
    ----------
    true : pd.DataFrame
        DataFrame containing the true values with columns 'true_counts', 'true_speeds', 'true_density'.
    simulated : pd.DataFrame
        DataFrame containing the simulated values with columns 'simulated_counts', 'simulated_speeds', 'simulated_density'.

    Returns
    -------
    dict[str, float]
        Dictionary with RMSN values for counts, speeds, and density.
    """
    metrics = {
        "counts": ("true_counts", "simulated_counts"),
        "speeds": ("true_speeds", "simulated_speeds"),
        "density": ("true_density", "simulated_density"),
    }

    rmsn_results = {}
    for key, (true_col, sim_col) in metrics.items():
        rmsn_results[key] = compute_rmsn(true[true_col], simulated[sim_col])

    return rmsn_results


def gof_eval(
    df_true: pd.DataFrame, df_simulated: pd.DataFrame, weights: dict[str, float] = {"counts": 1.0}
) -> float:
    """
    Evaluate the goodness of fit (GoF) between true and simulated data.

    Parameters
    ----------
    df_true : pd.DataFrame
        DataFrame containing the true values.
    df_simulated : pd.DataFrame
        DataFrame containing the simulated values.
    weights : dict[str, float], optional
        Weights for each component (counts, speeds, density) in the GoF calculation.
        If not provided, defaults to {"counts": 1.0}.

    Returns
    -------
    float
        The overall GoF score.
    """
    components = compute_rmsn_components(df_true, df_simulated)
    gof = sum(components[key] * weights.get(key, 0.0) for key in components)

    return gof
