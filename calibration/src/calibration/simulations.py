"""Functions to run single / multiple SUMO simulations for calibration purposes."""

import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from calibration import utils

MAX_PROCESSES = 18  # maximum number of parallel processes


def run_single_simulation(
    config: dict[str, Path], sim_setup: dict[str, Any], job: tuple[int, int]
) -> None:
    """
    Run a single SUMO simulation.
    Runs the od2trips tool generate trips from the current OD matrix, then runs SUMO.

    Parameters
    ----------
    config : Dictionary
        Necessary paths.
    sim_setup : Dictionary
        Simulation setup parameters.
    job : tuple[int, int]
        Tuple (counter, seed): replication index and random seed to use.

    Returns
    -------
    None.
    """

    counter, seed = job

    # set default rerouting probability
    p_reroute = 0.1

    # run od2trips tool
    try:
        import sumolib
    except Exception as e:
        raise ImportError("Ensure sumolib is in your PYTHONPATH -> utils._ensure_sumo_env") from e

    od2trips_bin = sumolib.checkBinary("od2trips")
    od_updated = config["CACHE"] / "od_updated.txt"
    trips_out = config["CACHE"] / "upd_od_trips.trips.xml"

    od2trips_cmd = [
        od2trips_bin,
        "--no-step-log",
        "--output-prefix",
        str(counter),
        "--spread.uniform",
        "--taz-files",
        config["NETWORK"] / sim_setup["taz"],
        "-d",
        od_updated,
        "-o",
        trips_out,
        "--seed",
        str(seed),
    ]

    utils._run(od2trips_cmd, stdout=subprocess.DEVNULL)

    # patch sim_setup for start and end sim seconds
    sim_setup["start_sim_sec"] = sim_setup["starttime"] * 3600 - 1800
    sim_setup["end_sim_sec"] = sim_setup["endtime"] * 3600 + 1800

    # run SUMO simulation
    sumo_bin = sumolib.checkBinary("sumo")

    route_file = config["CACHE"] / f"{counter}upd_od_trips.trips.xml"
    vehroutes_file = config["CACHE"] / "routes.vehroutes.xml"

    sumo_cmd = [
        sumo_bin,
        "--mesosim",
        "--no-step-log",
        "--output-prefix",
        str(counter),
        "-n",
        config["NETWORK"] / sim_setup["net"],
        "-W",
        "-b",
        str(sim_setup["start_sim_sec"]),
        "-e",
        str(sim_setup["end_sim_sec"]),
        "-r",
        route_file,
        "--vehroutes",
        vehroutes_file,
        "--additional-files",
        config["NETWORK"] / sim_setup["detector"],
        "--xml-validation",
        "never",
        "--device.rerouting.probability",
        str(p_reroute),
        "--seed",
        str(seed),
    ]

    utils._run(sumo_cmd)


def run_multiple_simulations(
    config: dict[str, Path],
    sim_setup: dict[str, Any],
    seeds: np.ndarray | list[int] | None = None,
) -> None:
    """
    Run multiple SUMO simulations in parallel.

    Parameters
    ----------
    config : Dictionary
        Necessary paths.
    sim_setup : Dictionary
        Simulation setup parameters.

    seeds
        Optional list/array of integer seeds (length == n_sumo_replicate). If not
        provided, seeds are generated internally.

    Returns
    -------
    None.
    """

    n_replicates = int(sim_setup["n_sumo_replicate"])

    if seeds is None:
        seeds_list = np.random.normal(0, 10000, n_replicates).astype("int32")
    else:
        seeds_list = np.asarray(list(seeds), dtype="int32")
        if seeds_list.shape[0] != n_replicates:
            raise ValueError(
                f"seeds must have length n_sumo_replicate={n_replicates}, got {seeds_list.shape[0]}"
            )

    # create partial function
    worker_fun = partial(run_single_simulation, config, sim_setup)
    processes = min(MAX_PROCESSES, n_replicates)

    with Pool(processes=processes) as pool:
        list(
            tqdm(
                pool.imap_unordered(worker_fun, enumerate(seeds_list)),
                total=n_replicates,
                desc="Running SUMO simulations",
            )
        )
