from pathlib import Path

import numpy as np
import pandas as pd

from calibration.replications import load_or_compute_required_replications


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


class Gof:
    """
    Goodness-of-fit (GoF) calculator with configurable internal weights and detector exclusions.

    Notes
    -----
    - Exclusions are applied by filtering rows where index name is 'detector_id'.
    - Weights are stored internally and used by default for GoF computation.
    """

    _DEFAULT_WEIGHTS: dict[str, float] = {"counts": 1.0}

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        excluded_detectors: list[str] | set[str] | None = None,
    ) -> None:
        self._weights: dict[str, float] = dict(self._DEFAULT_WEIGHTS)
        if weights is not None:
            self.update_weights(weights)

        self._excluded_detectors: set[str] = set()
        if excluded_detectors is not None:
            self.set_excluded_detectors(excluded_detectors)

    def update_weights(self, weights: dict[str, float]) -> None:
        """
        Update internal weights used for GoF computation.

        Parameters
        ----------
        weights : dict[str, float]
            Keys are component names: 'counts', 'speeds', 'density'.
            Values must be non-negative.
        """
        for k, v in weights.items():
            if k not in ["counts", "density", "speeds"]:
                raise KeyError(
                    f"Invalid weight key '{k}'; must be one of 'counts', 'speeds', 'density'"
                )
            if v < 0:
                raise ValueError(f"Weight for '{k}' must be non-negative, got {v}")
            self._weights[str(k)] = float(v)

    def set_excluded_detectors(self, detector_ids: list[str] | set[str]) -> None:
        """
        Set excluded detectors by ID. These detectors will be removed from computations.
        """
        self._excluded_detectors = {str(x) for x in detector_ids}

    def set_excluded_from_config(self, config: dict[str, Path], threshold: int = 15) -> None:
        """
        Compute and set excluded detectors based on required replications.

        A detector is excluded if required_sims > threshold.

        Parameters
        ----------
        config : dict[str, Path]
            Calibration config containing at least config["NETWORK"].
        threshold : int, default 15
            Exclude detectors requiring more than this number of replications.
        """
        stats = load_or_compute_required_replications(config)
        excluded = stats.index[stats["required_sims"] > int(threshold)].astype(str).tolist()
        self.set_excluded_detectors(excluded)

    def _apply_exclusions(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._excluded_detectors:
            return df

        if df.index.name != "detector_id":
            raise ValueError(
                "GoF inputs must be indexed by 'detector_id' in order to apply exclusions "
                f"(got index name {df.index.name!r})"
            )

        df = df.loc[~df.index.astype(str).isin(self._excluded_detectors)]
        return df

    def compute_rmsn_components(
        self, true: pd.DataFrame, simulated: pd.DataFrame
    ) -> dict[str, float]:
        """
        Compute RMSN for counts, speeds, and density, applying detector exclusions.
        """
        true_f = self._apply_exclusions(true)
        sim_f = self._apply_exclusions(simulated)

        # check that both datasets contain the same detectors after exclusions
        if not true_f.index.equals(sim_f.index):
            raise ValueError(
                "After applying exclusions, true and simulated data must have the same detectors. "
                f"Got {len(true_f)} true and {len(sim_f)} simulated."
            )

        # align indices
        common_index = true_f.index.intersection(sim_f.index)
        true_f = true_f.loc[common_index]
        sim_f = sim_f.loc[common_index]

        metrics = {
            "counts": ("true_counts", "simulated_counts"),
            "speeds": ("true_speeds", "simulated_speeds"),
            "density": ("true_density", "simulated_density"),
        }

        rmsn_results: dict[str, float] = {}
        for key, (true_col, sim_col) in metrics.items():
            if true_col not in true_f.columns or sim_col not in sim_f.columns:
                raise KeyError(
                    f"Missing columns for '{key}': expected '{true_col}' in true and '{sim_col}' in simulated"
                )
            rmsn_results[key] = compute_rmsn(true_f[true_col], sim_f[sim_col])

        return rmsn_results

    def compute_gof(
        self,
        df_true: pd.DataFrame,
        df_simulated: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Compute weighted GoF score using internal weights (or provided overrides).
        """
        w = self._weights if weights is None else {**self._weights, **weights}
        components = self.compute_rmsn_components(df_true, df_simulated)
        return float(sum(components[key] * w.get(key, 0.0) for key in components))


def compute_rmsn_components(true: pd.DataFrame, simulated: pd.DataFrame) -> dict[str, float]:
    """
    Backward-compatible function wrapper for RMSN components (no exclusions).
    """
    return Gof().compute_rmsn_components(true, simulated)


def gof_eval(
    df_true: pd.DataFrame, df_simulated: pd.DataFrame, weights: dict[str, float] = {"counts": 1.0}
) -> float:
    """
    Backward-compatible function wrapper for GoF evaluation (no exclusions).
    """
    return Gof(weights=weights).compute_gof(df_true, df_simulated)
