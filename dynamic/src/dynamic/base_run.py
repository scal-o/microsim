import traceback
from pathlib import Path

import traci


class BaseRun:
    """
    Base Context manager for sumo simulations controlled via traci.
    Provides several methods to interact with the simulation:
        - get_bus_ids
        - get_bus_stops
        - get_bus_status(busID)
        - get_bus_passengers(busID)
        - get_bus_speed(busID)
        - get_bus_pos(busID)
        - get_vehicle_ids
        - get_vehicle_pos(vehicleID)
        - get_n_boarding(stopID)
        - get_n_alighting(busID, stopID)
        - get_next_stop_id(busID)
        - get_next_stop_distance(busID)
        - get_tls_phase(junctionID)
        - get_tls_duration(junctionID)
    """

    def __init__(self, cfg: Path | str = "dynamic/configs/dyn_config.sumocfg"):
        """
        Initialize the base run class.
        Reads the .sumocfg file from the provided directory to initializes the run parameters:
            - begin
            - end
        """

        self.sumocfg = Path(cfg)

        if not self.sumocfg.exists():
            raise FileNotFoundError(f"dyn_config.sumocfg file not found in {self.cfg_dir}")

        self.begin = 0
        self.end = 3600
        self.curr_step: int = 0

    def __enter__(self):
        """Simple method to start the simulation via traci"""

        try:
            traci.start(["sumo", "-c", str(self.sumocfg)])
            self.curr_step = 0
        except:
            traceback.print_exc()

    def step(self) -> int:
        if self.curr_step < self.end:
            try:
                traci.simulationStep()
                self.curr_step += 1
                return self.curr_step
            except Exception:
                traceback.print_exc()
                return -1
