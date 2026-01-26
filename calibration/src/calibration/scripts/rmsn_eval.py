"""
Script to evaluate starting RMSN with different data types and their combinations,
with different multiples of the starting OD matrix.
"""

import json
from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # ensure non-interactive backend for CLI runs
import matplotlib.pyplot as plt

from calibration import gof, spsa, utils


# click command group
@click.group(name="rmsn")
def rmsn_cli() -> None:
    """RMSN evaluation utilities (evaluate + plotting)."""


## ================================================================================
## general helpers
def load_rmsn_eval_config(config_path: str) -> dict[str, Path]:
    """Load RMSN evaluation configuration from a JSON file."""

    with open(config_path, "r") as f:
        config = json.load(f)

    # convert paths to Path objects
    for key in config:
        config[key] = Path(config[key])

    return config


def simulated_csv_path(eval_results_dir: Path, multiple: float) -> Path:
    # keep naming stable with existing output
    return eval_results_dir / f"simulated_counts_od_multiple_{multiple}.csv"


## ================================================================================
## rmsn evaluation
def load_or_create_matrix(
    config: dict[str, Path], input_od: pd.DataFrame, multiple: float
) -> pd.DataFrame:
    """Load or create a single OD matrix scaled by the given multiple."""

    # check if the cache folder exists
    cache_folder = config["CACHE"]
    od_matrix_path = cache_folder / f"od_matrix_{multiple}.csv"

    # if it exists, load the od matrix
    # else, create it from scratch
    if od_matrix_path.exists():
        od_temp = pd.read_csv(od_matrix_path, sep=" ", header=None)
    else:
        od_temp = input_od.copy()
        od_temp.iloc[:, 2:] = od_temp.iloc[:, 2:] * multiple
        od_temp.iloc[:, 2:] = od_temp.iloc[:, 2:].round().astype(int)

        # create cache folder if needed
        if not cache_folder.exists():
            cache_folder.mkdir(parents=True, exist_ok=True)

        # save the OD matrix
        od_temp.to_csv(od_matrix_path, sep=" ", header=False, index=False)

    # return OD matrix
    return od_temp


def load_or_run_simulation(
    config: dict[str, Path],
    sim_setup: dict[str, Any],
    eval_results_dir: Path,
    multiple: float,
    od_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Load simulated link outputs from cache if present, else run SUMO."""

    out_path = simulated_csv_path(eval_results_dir, multiple)
    if out_path.exists():
        print(f"Using cached simulation outputs for multiplier {multiple}: {out_path}")
        return pd.read_csv(out_path, sep=";", header=0, index_col=0)

    print(f"Running simulations for OD matrix scaled by factor {multiple}")
    df_simulated = spsa.run_parse_cleanup(config, sim_setup, od_matrix)
    df_simulated.to_csv(out_path, sep=";", header=True, index=True)
    return df_simulated


@rmsn_cli.command(name="eval")
def eval_cmd() -> None:
    """Run simulations (with caching) and compute per-multiplier RMSNs."""

    config, sim_setup, _ = utils.load_spsa_config(
        config=Path("calibration/configs/config.json"),
        sim_setup=Path("calibration/configs/simulation_setups.json"),
        spsa_setup=Path("calibration/configs/spsa_setups.json"),
    )
    input_od = utils.load_start_od(config, sim_setup)

    eval_config = load_rmsn_eval_config("calibration/configs/rmsn_eval_config.json")
    config.update(eval_config)

    # define multiples and load / create OD matrices
    multiples = np.arange(0.2, 2.0 + 1e-9, 0.1).round(1).tolist()
    od_matrices = [load_or_create_matrix(config, input_od, m) for m in multiples]

    # create evaluation results folder if it does not exist
    eval_config["EVAL_RESULTS"].mkdir(parents=True, exist_ok=True)

    # load true data once
    df_true = utils.load_true_counts(config, sim_setup).fillna(0)

    summary_rows: list[dict[str, float]] = []

    for multiple, od_matrix in zip(multiples, od_matrices):
        df_simulated = load_or_run_simulation(
            config=config,
            sim_setup=sim_setup,
            eval_results_dir=eval_config["EVAL_RESULTS"],
            multiple=multiple,
            od_matrix=od_matrix,
        )

        # compute per-metric RMSNs
        row = gof.compute_rmsn_components(df_true, df_simulated)
        row["multiplier"] = float(multiple)
        summary_rows.append(row)

    # save summary table
    df_summary = pd.DataFrame.from_records(summary_rows).set_index("multiplier").sort_index()
    df_summary.to_csv(
        eval_config["EVAL_RESULTS"] / "rmsn_summary.csv", sep=";", header=True, index=True
    )
    print(f"Saved RMSN summary to: {eval_config['EVAL_RESULTS'] / 'rmsn_summary.csv'}")


## ================================================================================
## rmsn plotting
def weight_grid(step: float) -> list[tuple[float, float, float]]:
    """Generate (wc, ws, wd) triples with a given stepsize."""
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")

    k = int(round(1.0 / step))
    triples: list[tuple[float, float, float]] = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            wc = i * step
            ws = j * step
            wd = (k - i - j) * step
            triples.append((float(wc), float(ws), float(wd)))
    return triples


def combine_rmsn(df: pd.DataFrame, wc: float, ws: float, wd: float) -> pd.Series:
    """Weighted sum of the three rmsn columns. Normalizes the weights before use."""
    weights = [wc, ws, wd]
    s = float(sum(weights))
    weights = [w / s for w in weights]

    return df["counts"] * weights[0] + df["speeds"] * weights[1] + df["density"] * weights[2]


def plot_line_search(
    x: np.ndarray,
    series_map: dict[str, pd.Series],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Simple line plot helper used for all line-search curves."""
    plt.figure(figsize=(7.5, 4.5), dpi=300)
    for label, ser in series_map.items():
        plt.plot(x, ser.values, marker="o", linewidth=1.8, label=label)
    plt.xlabel("OD multiplier")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@rmsn_cli.command(name="plot")
def plot_cmd() -> None:
    """Grid search over weight combinations and generate plots in EVAL_RESULTS."""

    eval_config = load_rmsn_eval_config("calibration/configs/rmsn_eval_config.json")
    eval_results_dir = eval_config["EVAL_RESULTS"]
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    # --- load summary produced by `rmsn eval` ---
    summary_path = eval_results_dir / "rmsn_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary file: {summary_path}. Run `rmsn eval` first to generate it."
        )
    df_summary = pd.read_csv(summary_path, sep=";", header=0, index_col=0)
    df_summary.index = df_summary.index.astype(float)
    df_summary = df_summary.sort_index()

    def min_normalize(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
        """Min-normalize selected columns (1.0 is best), intended for plotting only."""
        out = df.copy()
        for c in cols:
            m = float(pd.to_numeric(out[c]).min())
            if np.isfinite(m) and not np.isclose(m, 0.0):
                out[c] = out[c].astype(float) / m
            else:
                out[c] = np.nan
        return out

    def label(wc: float, ws: float, wd: float) -> str:
        return f"wC={wc:.2f}, wS={ws:.2f}, wD={wd:.2f}"

    x = df_summary.index.to_numpy(dtype=float)

    # --- (1) single-component plots (raw RMSNs) ---
    single_map = {
        "counts": df_summary["counts"],
        "speeds": df_summary["speeds"],
        "density": df_summary["density"],
    }
    plot_line_search(
        x=x,
        series_map=single_map,
        title="Starting RMSN by data type (line search over OD multiplier)",
        ylabel="RMSN",
        out_path=eval_results_dir / "rmsn_single_components.png",
    )

    # --- (1b) single-component plots (min-normalized / relative degradation) ---
    df_norm = min_normalize(df_summary, ("counts", "speeds", "density"))
    single_norm_map = {
        "counts (normalized)": df_norm["counts"],
        "speeds (normalized)": df_norm["speeds"],
        "density (normalized)": df_norm["density"],
    }
    plot_line_search(
        x=x,
        series_map=single_norm_map,
        title="Starting RMSN by data type (min-normalized: 1.0 = best)",
        ylabel="Relative RMSN (RMSN / min(RMSN))",
        out_path=eval_results_dir / "rmsn_single_components_relative.png",
    )

    # --- (2) grid search over weight combinations ---
    grid = weight_grid(0.1)
    records_raw: list[dict[str, float]] = []
    records_norm: list[dict[str, float]] = []
    for wc, ws, wd in grid:
        combined_raw = combine_rmsn(df_summary, wc, ws, wd)
        best_mult_raw = float(pd.to_numeric(combined_raw.idxmin()))
        best_val_raw = float(combined_raw.min())
        records_raw.append(
            {
                "w_counts": wc,
                "w_speeds": ws,
                "w_density": wd,
                "best_multiplier": best_mult_raw,
                "best_score": best_val_raw,
            }
        )

        combined_norm = combine_rmsn(df_norm, wc, ws, wd)
        best_mult_norm = float(pd.to_numeric(combined_norm.idxmin()))
        best_val_norm = float(combined_norm.min())
        mean_val_norm = float(combined_norm.mean())
        records_norm.append(
            {
                "w_counts": wc,
                "w_speeds": ws,
                "w_density": wd,
                "best_multiplier": best_mult_norm,
                "best_score": best_val_norm,
                "mean_score": mean_val_norm,
            }
        )

    df_grid_raw = pd.DataFrame.from_records(records_raw)
    df_grid_raw = df_grid_raw.sort_values("best_score", ascending=True).reset_index(drop=True)
    df_grid_raw.to_csv(eval_results_dir / "rmsn_weight_grid_results_raw.csv", sep=";", index=False)

    df_grid_norm = pd.DataFrame.from_records(records_norm)
    df_grid_norm = df_grid_norm.sort_values("best_score", ascending=True).reset_index(drop=True)
    df_grid_norm.to_csv(
        eval_results_dir / "rmsn_weight_grid_results_normalized.csv", sep=";", index=False
    )

    # --- (3) plot top-k weight combinations as line search ---
    # Choose tuples using normalized (dimensionless) scores for fair comparison.
    #  - mean_score: overall performance across multipliers

    df_top_mean = df_grid_norm.sort_values("mean_score", ascending=True).head(5)
    top_series_mean: dict[str, pd.Series] = {}
    for _, r in df_top_mean.iterrows():
        wc, ws, wd = float(r["w_counts"]), float(r["w_speeds"]), float(r["w_density"])
        top_series_mean[label(wc, ws, wd)] = combine_rmsn(df_norm, wc, ws, wd)

    plot_line_search(
        x=x,
        series_map=top_series_mean,
        title="Top-5 weights by mean normalized score (overall fit)",
        ylabel="Weighted relative RMSN",
        out_path=eval_results_dir / "rmsn_top_5_weights_by_mean.png",
    )

    df_top_mean = (
        df_grid_norm[df_grid_norm["w_speeds"] == 0.0]
        .sort_values("mean_score", ascending=True)
        .head(5)
    )
    top_series_mean: dict[str, pd.Series] = {}
    for _, r in df_top_mean.iterrows():
        wc, ws, wd = float(r["w_counts"]), float(r["w_speeds"]), float(r["w_density"])
        top_series_mean[label(wc, ws, wd)] = combine_rmsn(df_norm, wc, ws, wd)

    plot_line_search(
        x=x,
        series_map=top_series_mean,
        title="Top-5 weights (fixed w_speeds=0) by mean normalized score (overall fit)",
        ylabel="Weighted relative RMSN",
        out_path=eval_results_dir / "rmsn_top_5_weights_fixed_speeds_by_mean.png",
    )

    top_series_mean: dict[str, pd.Series] = {}
    for _, r in df_top_mean.iterrows():
        wc, ws, wd = float(r["w_counts"]), float(r["w_speeds"]), float(r["w_density"])
        top_series_mean[label(wc, ws, wd)] = combine_rmsn(df_norm[df_norm.index <= 1.0], wc, ws, wd)

    plot_line_search(
        x=x[x <= 1.0],
        series_map=top_series_mean,
        title="Top-5 weights (fixed w_speeds=0, multiplier ≤ 1.0) by mean normalized score (overall fit)",
        ylabel="Weighted relative RMSN",
        out_path=eval_results_dir / "rmsn_top_5_weights_window_fixed_speeds_by_mean.png",
    )

    print(f"Saved plots and grid search CSVs to: {eval_results_dir}")
