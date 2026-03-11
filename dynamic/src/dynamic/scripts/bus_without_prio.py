import click
from tqdm import tqdm

from dynamic.bus_manager import DDTBusManager, RSBusManager
from dynamic.manager import RunManager


@click.command(name="dwell-time-simulation")
@click.option(
    "--mode",
    type=click.Choice(["ddt", "rs"], case_sensitive=False),
    default="ddt",
    help="Bus stop duration mode: 'ddt' for Dynamic Dwell Time, 'rs' for Request Stops",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="",
    help="Output directory for simulation results (will be created as a subdrectory of results/dyn. defaults to results/dyn/bus_<mode>).",
)
@click.option(
    "--num-runs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of simulation runs to execute.",
)
def dwell_time_simulation(mode: str, output_dir: str, num_runs: int):
    if output_dir == "":
        output_dir = f"bus_{mode}"

    for run_idx in range(num_runs):
        run_output_dir = f"{output_dir}/run_{run_idx + 1}"

        # initialize the run manager with the correct output dir
        with RunManager(output_prefix=run_output_dir) as ctx:
            if mode == "ddt":
                buses = DDTBusManager(ctx)
            else:
                buses = RSBusManager(ctx)

            print(
                "Starting simulation with mode: ",
                "dynamic dwell time" if mode == "ddt" else "request stops",
                f"(run {run_idx + 1}/{num_runs})",
            )

            for _ in tqdm(range(3600), desc=f"Running SUMO (run {run_idx + 1}/{num_runs})"):
                ctx.step()

                # update the bus lists
                buses.update_buses()

                # set the stop durations
                buses.set_stops_duration()
