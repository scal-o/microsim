"""Functions to parse SUMO output files for calibration purposes."""

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_single_run_data(
    config: dict[str, Path], loop_file: Path | str, endtime: float | int
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

    Returns
    -------
    Numpy array
        Aggregated counts, speeds, and densities per link.
    """

    # compute simulation end time in secs
    fract = float(endtime)
    integ = int(fract)
    fract = round(fract - integ, 2)
    endSimTime = integ * 60 * 60 + fract * 60

    # create data2csv command
    output_file = config["RESULTS"] / "loopDataName.csv"
    data2csv = (
        f"uv run {config['SUMO']}\\tools\\xml\\xml2csv.py "
        f"{config['RESULTS'] / loop_file} "
        f"--x {config['SUMO']}\\data\\xsd\\det_e1meso_file.xsd "
        f"-o {output_file}"
    )

    # convert XML to CSV
    subprocess.run(data2csv)

    # read csv file and filter relevant trips
    df_trips = pd.read_csv(output_file, sep=";", header=0)
    df_trips = df_trips[df_trips["interval_end"] < endSimTime]

    # extract edge IDs from detector IDs
    det_id = df_trips["interval_id"]
    edge_id = [word.split("_")[1] for word in det_id]
    df_trips["EdgeID"] = edge_id

    # aggregate counts per link
    temp = pd.DataFrame()
    temp["EdgeID"] = edge_id
    temp["Counts"] = df_trips["interval_entered"]
    temp["Speeds"] = df_trips["interval_speed"]
    temp["Density"] = df_trips["interval_density"]
    temp = temp.fillna(0)
    df_group = temp.groupby("EdgeID").agg(np.sum)
    df_group2 = temp.groupby("EdgeID").agg(np.average)
    df_group["Edge"] = df_group.index
    df_group2["Edge"] = df_group.index
    df_all = pd.DataFrame()
    df_all["counts"] = df_group["Counts"]
    df_all["speeds"] = df_group2["Speeds"]
    df_all["density"] = df_group2["Density"]
    df_all = df_all.reindex("Edge")
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

    # create empty dataframes to hold results
    counts = pd.DataFrame()
    speeds = pd.DataFrame()
    density = pd.DataFrame()

    # loop over all simulation replicates
    # parse results and concatenate in the dataframes
    for counter in range(sim_setup["n_sumo_replicate"]):
        loop_file = str(counter) + "out.xml"
        df_loop = parse_single_run_data(config, loop_file, sim_setup["endtime"])
        counts = pd.concat([counts, df_loop["counts"]], axis=1, sort=True)
        speeds = pd.concat([speeds, df_loop["speeds"]], axis=1, sort=True)
        density = pd.concat([density, df_loop["density"]], axis=1, sort=True)

    df_counts_mean = counts.astype("int32")
    df_counts_mean = pd.DataFrame()
    df_counts_mean.columns = ["simulated_counts", "simulated_speeds", "simulated_density"]
    df_counts_mean["simulated_counts"] = counts.mean(axis=1)
    df_counts_mean["simulated_speeds"] = speeds.mean(axis=1)
    df_counts_mean["simulated_density"] = density.mean(axis=1)

    return df_counts_mean
