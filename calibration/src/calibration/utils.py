import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _run(cmd: list[str | Path], **kwargs: Any) -> None:
    """
    Run a subprocess command with Path support and cross-platform argument handling.
    """
    cmd_str = [str(c) for c in cmd]
    subprocess.run(cmd_str, check=True, **kwargs)


def _ensure_sumo_env(config: dict[str, Path]) -> None:
    """
    Ensure SUMO environment is configured in a standard way:

    - ensures `SUMO_HOME` is set (from env if present, otherwise from `config["SUMO"]`)
    - adds `SUMO_HOME/tools` to PYTHONPATH and to `sys.path`

    This enables importing `sumolib` / `traci` on both Windows and WSL.

    Returns
    -------
    None
    """
    # resolve SUMO_HOME
    if "SUMO_HOME" not in os.environ:
        if "SUMO" not in config:
            raise EnvironmentError(
                "SUMO_HOME environment variable not set and config['SUMO'] missing."
            )
        sumo_home = Path(config["SUMO"])
        os.environ["SUMO_HOME"] = str(sumo_home)
    else:
        sumo_home = Path(os.environ["SUMO_HOME"])

    # add SUMO tools to python import path
    sys.path.append(str(sumo_home / "tools"))


def load_spsa_config(
    config: Path | str = "calibration/configs/config.json",
    sim_setup: Path | str = "calibration/configs/simulation_setups.json",
    spsa_setup: Path | str = "calibration/configs/spsa_setups.json",
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Load paths, simulation setups and algorithm setups


    Parameters
    ----------
    config : Union[Path, str], optional
        Paths to cache, network, and SUMO. The default is "config.json".
    sim_setup : Union[Path, str], optional
        Simulation parameters. The default is "simulation_setups.json".
    spsa_setup : Union[Path, str], optional
        SPSA parameters. The default is "spsa_setups.json".

    Returns
    -------
    config : Dictionary
    sim_setup : Dictionary
    spsa_setup : Dictionary

    """
    config = Path(config) if isinstance(config, str) else config
    config = json.load(open(config))
    conf_dict: dict[str, Path] = {}
    for k, v in config.items():
        conf_dict[k] = Path(v)

    # ensure SUMO_HOME + tools are available (sumolib/traci)
    _ensure_sumo_env(conf_dict)

    sim_setup = Path(sim_setup) if isinstance(sim_setup, str) else sim_setup
    sim_setup = json.load(open(sim_setup))
    sim_dict: dict[str, Any] = {}
    for k, v in sim_setup.items():
        sim_dict[k] = v

    spsa_setup = Path(spsa_setup) if isinstance(spsa_setup, str) else spsa_setup
    spsa_setup = json.load(open(spsa_setup))
    spsa_dict: dict[str, Any] = {}
    for k, v in spsa_setup.items():
        spsa_dict[k] = v

    return conf_dict, sim_setup, spsa_setup


def load_start_od(config: dict[str, Path], sim_setup: dict[str, str]) -> pd.DataFrame:
    """
    Load the initial/prior OD matrix from a text file.

    Parameters
    ----------
    config: dictionary containing config params
    sim_setup: dictionary containing sim params

    Returns
    -------
    od_matrix: Pandas DataFrame containing the initial/prior OD matrix

    """
    file_name = config["NETWORK"] / sim_setup["prior_od"]
    od_matrix = pd.read_csv(
        file_name,
        sep=r"\s+",
        header=None,
        skiprows=5,
    )

    return od_matrix


def load_true_counts(config: dict[str, Path], sim_setup: dict[str, str]) -> pd.DataFrame:
    """
    Load the true loop counts from a text file.

    Parameters
    ----------
    config: dictionary containing config params
    sim_setup: dictionary containing sim params

    Returns
    -------
    true_counts: Pandas DataFrame containing the true loop counts

    """
    file_name = config["NETWORK"] / sim_setup["loop_data"]
    true_counts = pd.read_csv(file_name, header=None)
    true_counts.columns = [
        "detector_id",
        "true_counts",
        "true_speeds",
        "true_density",
    ]  # for counts
    true_counts = true_counts.set_index("detector_id")
    true_counts.index = true_counts.index.map(str)

    return true_counts


def od_to_matrix(od_vector: pd.DataFrame) -> np.ndarray:
    """A simple function that converts an OD vector to OD matrix. We read
    through the vector (row after row) and create a matrix.

    Parameters
    ----------
    od_vector : Pandas DataFrame

    Returns
    -------
    od_matrix : Numpy array
        Two-dimensional numpy array.

    """
    n_zones = int(np.sqrt(od_vector.shape[0]))
    od_matrix = np.zeros((n_zones, n_zones))
    m = 0
    for i in range(0, n_zones):
        for j in range(0, n_zones):
            od_matrix[i, j] = od_vector[m]
            m = m + 1
    return od_matrix


def od_to_file(config: dict[str, Path], sim_setup: dict[str, str], od_matrix: pd.DataFrame) -> None:
    """
    Write the current OD matrix to a text file in a
    SUMO recognisable format.

    Parameters
    ----------
    config: dictionary containing config params
    sim_setup: dictionary containing sim params
    od_matrix: Pandas DF containing the current demand

    Returns
    -------
    None
    """

    header_text = f"$OR;D2 \n* From-Time  To-Time \n{sim_setup['starttime']}.00 {sim_setup['endtime']}.00\n* Factor \n1.00\n"

    file_name = config["CACHE"] / "od_updated.txt"

    with open(file_name, "w") as f:
        f.write(header_text)
        od_matrix.to_csv(f, header=False, index=False, sep=" ")


def cleanup_results(config: dict[str, Path]) -> None:
    """
    Clean up temporary files from previous simulations.

    Parameters
    ----------
    config: dictionary containing config params

    Returns
    -------
    None
    """
    results_dir = config["RESULTS"]
    for file in results_dir.glob("*"):
        if file.is_file():
            file.unlink()
