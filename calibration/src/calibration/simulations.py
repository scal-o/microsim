"""Functions to run single / multiple SUMO simulations for calibration purposes."""

import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

MAX_PROCESSES = 8  # maximum number of parallel processes


def run_single_simulation(
    config: dict[str, Path], sim_setup: dict[str, Any], counter: int, seed: int
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
    counter : int
        SUMO replication index.
    seed : int
        Random seed to use for the sim.

    Returns
    -------
    None.
    """

    # set default rerouting probability
    p_reroute = 0.1

    # run od2trips tool
    od2trips = (
        f"uv run {config['SUMO']}\\tools\\od2trips\\od2trips.py "
        f"--no-step-log --output-prefix {counter} --spread.uniform "
        f"--taz-files {config['NETWORK'] / sim_setup['taz']} "
        f"-d {config['CACHE']}\\od_updated.txt "
        f"-o {config['CACHE']}\\upd_od_trips.trips.xml --seed {seed}"
    )

    subprocess.run(od2trips)

    # run SUMO simulation
    sumo_run = (
        f"uv run sumo --mesosim --no-step-log --output-prefix {counter} "
        f"-n {config['NETWORK'] / sim_setup['net']} -W "
        f"-b {sim_setup['start_sim_sec']} -e {sim_setup['end_sim_sec']} "
        f"-r {config['CACHE']}\\{counter}upd_od_trips.trips.xml "
        f"--vehroutes {config['CACHE']}\\routes.vehroutes.xml "
        f"--additional-files {config['NETWORK'] / sim_setup['detector']} "
        f"--xml-validation never --device.rerouting.probability {p_reroute} --seed {seed}"
    )

    subprocess.run(sumo_run)


def run_multiple_simulations(config: dict[str, Path], sim_setup: dict[str, Any]) -> None:
    """
    Run multiple SUMO simulations in parallel.

    Parameters
    ----------
    config : Dictionary
        Necessary paths.
    sim_setup : Dictionary
        Simulation setup parameters.

    Returns
    -------
    None.
    """

    n_replicates = sim_setup["n_sumo_replicate"]
    seeds = np.random.normal(0, 10000, n_replicates).astype("int32")

    # create partial function
    worker_fun = partial(run_single_simulation, config, sim_setup)

    with Pool(processes=MAX_PROCESSES) as pool:
        tqdm(
            pool.imap_unordered(worker_fun, enumerate(seeds)),
            total=n_replicates,
            desc="Running SUMO simulations",
        )
