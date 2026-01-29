import click
from tqdm import tqdm
from dynamic.manager import RunManager


@click.command(name="status-quo-simulation")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="",
    help="Output directory for simulation results (will be created as a subdrectory of results/dyn. defaults to results/dyn/bus_status_quo).",
)
def status_quo_simulation(output_dir: str):
    if output_dir == "":
        output_dir = "bus_status_quo"

    # initialize the run manager with the correct output dir
    with RunManager(output_prefix=output_dir) as ctx:
        print("Starting simulation with status quo bus stop durations")

        for i in tqdm(range(3600), desc="Running SUMO"):
            ctx.step()
