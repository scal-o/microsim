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
@click.option(
    "--num-runs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of simulation runs to execute.",
)
def status_quo_simulation(output_dir: str, num_runs: int):
    if output_dir == "":
        output_dir = "bus_status_quo"

    for run_idx in range(num_runs):
        run_output_dir = f"{output_dir}/run_{run_idx + 1}"

        # initialize the run manager with the correct output dir
        with RunManager(output_prefix=run_output_dir) as ctx:
            print(
                f"Starting simulation with status quo bus stop durations (run {run_idx + 1}/{num_runs})"
            )

            for _ in tqdm(range(3600), desc=f"Running SUMO (run {run_idx + 1}/{num_runs})"):
                ctx.step()
