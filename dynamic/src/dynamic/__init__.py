# defines whether the package will use traci or libsumo to communicate with the simulation
# import traci
import libsumo as traci
import click
from dynamic.scripts import bus_without_prio, bus_status_quo


@click.group(name="bus")
def bus_cli():
    pass


bus_cli.add_command(bus_without_prio.dwell_time_simulation)
bus_cli.add_command(bus_status_quo.status_quo_simulation)
