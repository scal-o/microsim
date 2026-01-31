# defines whether the package will use traci or libsumo to communicate with the simulation
import click

# import libsumo as traci
import traci

from dynamic.scripts import bus_status_quo, bus_with_prio, bus_without_prio


@click.group(name="bus")
def bus_cli():
    pass


bus_cli.add_command(bus_without_prio.dwell_time_simulation)
bus_cli.add_command(bus_status_quo.status_quo_simulation)
bus_cli.add_command(bus_with_prio.pt_prio_simulation)
