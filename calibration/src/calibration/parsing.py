"""Functions to parse SUMO output files for calibration purposes."""

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from calibration import utils

MAX_PROCESSES = 8  # maximum number of parallel processes


def parse_single_run_data(
    config: dict[str, Path], loop_file: Path | str, endtime: float | int, starttime: float | int
) -> pd.DataFrame:
    """
    Low-level function to parse the results of a single SUMO simulation run.
    Reads the loop detector data and aggregates it on a link level.
    Internally uses SUMO's xml2csv tool to convert the XML output to CSV,
    then reads the CSV with pandas and processes it.

    Parameters
    ----------
    config : Dictionary
        Necessary paths.
    loop_file : Path or String
        SUMO detector data file.
    endtime : Float/Int
        End time of the current interval.
    starttime : Float/Int
        Start time of the current interval.

    Returns
    -------
    Numpy array
        Aggregated counts, speeds, and densities per link.
    """
    loop_file = Path(loop_file)

    # compute simulation end time in secs
    fract = float(endtime)
    integ = int(fract)
    fract = round(fract - integ, 2)
    endSimTime = integ * 60 * 60 + fract * 60

    # compute simulation end time in secs
    fract = float(starttime)
    integ = int(fract)
    fract = round(fract - integ, 2)
    startSimTime = integ * 60 * 60 + fract * 60

    # create data2csv command (portable Windows/WSL)
    xml2csv_py = config["SUMO"] / "tools" / "xml" / "xml2csv.py"
    det_xsd = config["SUMO"] / "data" / "xsd" / "det_e1meso_file.xsd"
    input_xml = config["RESULTS"] / loop_file

    data2csv_cmd = [
        config["PYTHON"],
        xml2csv_py,
        input_xml,
        "--x",
        det_xsd,
    ]

    # convert XML to CSV
    utils._run(data2csv_cmd)

    # output file name
    output_file = config["RESULTS"] / f"{loop_file.stem}.csv"

    # read csv file and filter relevant trips
    df_trips = pd.read_csv(output_file, sep=";", header=0)
    df_trips = df_trips[df_trips["interval_begin"] >= startSimTime]
    df_trips = df_trips[df_trips["interval_end"] < endSimTime]

    # extract detector IDs
    det_id = df_trips["interval_id"]
    det_id = [word.split("_")[1] for word in det_id]
    df_trips["detector_id"] = det_id

    # aggregate counts per link
    temp = pd.DataFrame()
    temp["detector_id"] = df_trips["detector_id"]
    temp["Counts"] = df_trips["interval_entered"]
    temp["Speeds"] = df_trips["interval_speed"]
    temp["Density"] = df_trips["interval_density"]
    temp = temp.fillna(0)
    df_group = temp.groupby("detector_id").agg(np.sum)
    df_group2 = temp.groupby("detector_id").agg(np.average)
    df_group["detector_id"] = df_group.index
    df_group2["detector_id"] = df_group.index
    df_all = pd.DataFrame()
    df_all["counts"] = df_group["Counts"]
    df_all["speeds"] = df_group2["Speeds"]
    df_all["density"] = df_group2["Density"]
    df_all = df_all.reindex()
    return df_all


def parse_multiple_runs_data(config: dict[str, Path], sim_setup: dict[str, Any]) -> pd.DataFrame:
    """
    High-level function to parse the results of multiple SUMO simulation runs.
    Calls the low-level parse_single_run_data function for each run and
    aggregates the results by computing the mean across runs.

    Parameters
    ----------
    config : Dictionary
        Necessary paths.
    sim_setup : Dictionary
        Simulation setup parameters

    Returns
    -------
    Pandas DataFrame
        Aggregated counts, speeds, and densities per link across all runs.
    """

    worker_fun = partial(
        parse_single_run_data,
        config,
        endtime=sim_setup["endtime"],
        starttime=sim_setup["starttime"],
    )
    n_replicates = sim_setup["n_sumo_replicate"]
    file_names = [f"{i}out.xml" for i in range(n_replicates)]

    with Pool(processes=MAX_PROCESSES) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_fun, file_names),
                total=n_replicates,
                desc="Parsing SUMO outputs",
            )
        )

    # concatenate outputs
    counts = pd.concat([df["counts"] for df in results], axis=1, sort=True)
    speeds = pd.concat([df["speeds"] for df in results], axis=1, sort=True)
    density = pd.concat([df["density"] for df in results], axis=1, sort=True)

    # compute link-wise means
    colnames = ["simulated_counts", "simulated_speeds", "simulated_density"]
    df_counts_mean = pd.DataFrame(columns=colnames)
    df_counts_mean["simulated_counts"] = counts.mean(axis=1)
    df_counts_mean["simulated_speeds"] = speeds.mean(axis=1)
    df_counts_mean["simulated_density"] = density.mean(axis=1)

    return df_counts_mean
