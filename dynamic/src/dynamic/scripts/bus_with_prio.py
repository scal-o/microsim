import click
from tqdm import tqdm

from dynamic.bus_manager import RSBusManager
from dynamic.manager import RunManager
from dynamic.tl_controller import TLSManager


@click.command(name="pt-prio-simulation")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="",
    help="Output directory for simulation results (will be created as a subdirectory of results/dyn. defaults to results/dyn/bus_ptp).",
)
@click.option(
    "--num-runs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of simulation runs to execute.",
)
def pt_prio_simulation(output_dir: str, num_runs: int):
    if output_dir == "":
        output_dir = "bus_ptp"

    for run_idx in range(num_runs):
        run_output_dir = f"{output_dir}/run_{run_idx + 1}"

        # initialize the run manager with the correct output dir
        with RunManager(
            cfg="dynamic/configs/dyn_config_prio.sumocfg", output_prefix=run_output_dir
        ) as ctx:
            # initialize bus manager with request stops mode
            buses = RSBusManager(ctx)
            # initialize the traffic light controller for bus prioritization
            tls_manager = TLSManager(junction_id="GS_60713745", context=ctx)

            print(
                f"Starting simulation with mode: public transport prioritization (run {run_idx + 1}/{num_runs})"
            )

            for _ in tqdm(range(3600), desc=f"Running SUMO (run {run_idx + 1}/{num_runs})"):
                ctx.step()

                # update the bus lists
                buses.update_buses()

                # set the stop durations
                buses.set_stops_duration()

                # update the tls manager
                tls_manager.update_approaching_buses(list(buses.buses.values()))
                tls_manager.step()
