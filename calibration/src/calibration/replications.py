from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import t as tdist


def compute_required_replications(
    config: dict[str, Path],
    confidence_level: float = 0.95,
    max_sim_n: int = 30,
    tol_factor: float = 0.1,
    detector_id_col: int = 0,
    drop_first_row: bool = True,
) -> pd.DataFrame:
    """
    Compute required simulation replications per detector from a raw counts CSV.

    Returns a DataFrame indexed by detector_id with columns:
      mean, std, tol, required_sims, plus "1".."max_sim_n"
    """

    # paths to read and write files
    counts_csv_path = config["NETWORK"] / "t_test_counts100.csv"
    cache_csv_path = config["NETWORK"] / "required_replications.csv"

    raw = pd.read_csv(counts_csv_path, sep=",", header=None)
    if drop_first_row and len(raw.index) > 0:
        raw = raw.drop([0], axis=0)

    detector_id = raw.iloc[:, detector_id_col].astype(str)
    counts = raw.drop([detector_id_col], axis=1)
    counts.index = detector_id
    counts = counts.apply(pd.to_numeric, errors="coerce")

    stats = pd.DataFrame(index=counts.index.copy())
    stats["mean"] = counts.mean(axis=1)
    stats["std"] = counts.std(axis=1)
    stats["tol"] = tol_factor * stats["mean"]

    alpha = 1.0 - confidence_level

    for n in range(1, max_sim_n + 1):
        t_val = tdist.ppf(1 - alpha / 2, n)
        required_est = ((t_val * stats["std"]) / stats["tol"]) ** 2
        required_est = required_est.fillna(0)
        stats[str(n)] = (required_est < (n + 1)).astype(bool)

    required_cols = [str(n) for n in range(1, max_sim_n + 1)]
    satisfied = stats[required_cols].astype(bool)

    required_sims = satisfied.apply(
        lambda row: next((i for i, ok in enumerate(row, start=1) if ok), max_sim_n + 1),
        axis=1,
    )
    stats["required_sims"] = required_sims.astype(int)
    stats.index.name = "detector_id"
    stats.to_csv(cache_csv_path, index=True)

    return stats


def load_or_compute_required_replications(
    config: dict[str, Path],
    confidence_level: float = 0.95,
    max_sim_n: int = 30,
    tol_factor: float = 0.1,
) -> pd.DataFrame:
    """
    Load or compute required simulation replications per detector.

    Returns a DataFrame indexed by detector_id.
    """

    cache_csv_path = config["NETWORK"] / "required_replications.csv"

    # if the cached csv exists, read it and return it directly
    if cache_csv_path.exists():
        df = pd.read_csv(cache_csv_path)
        df = df.set_index("detector_id")
        return df

    stats = compute_required_replications(
        config,
        confidence_level=confidence_level,
        max_sim_n=max_sim_n,
        tol_factor=tol_factor,
    )

    return stats
