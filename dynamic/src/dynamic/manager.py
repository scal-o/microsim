from __future__ import annotations

import traceback
from pathlib import Path
from typing import Literal, Self

import sumolib

from dynamic import traci


class RunManager:
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
        - get_n_alighting(stopID, busID)
        - get_next_stop_id(busID)
        - get_next_stop_distance(busID)
        - get_tls_phase(junctionID) wip
        - get_tls_duration(junctionID) wip
    """

    def __init__(self, cfg: Path | str = "dynamic/configs/dyn_config.sumocfg", gui: bool = False):
        """
        Initialize the base run class.
        Reads the .sumocfg file from the provided directory to initializes the run parameters:
            - begin
            - end
        """

        self.sumocfg = Path(cfg)

        if not self.sumocfg.exists():
            raise FileNotFoundError(f"dyn_config.sumocfg file not found at {self.sumocfg}")

        self.gui = gui
        self.begin = 0
        self.end = 3600
        self.curr_step: int = 0

        # initialize cache containers
        # save the ids here so we have to query traci as little as possible
        self._vehicles: list[str] = []
        self._buses: list[str] = []
        self._stops: list[str] = []

    def __enter__(self) -> Self:
        """Simple method to start the simulation via traci"""
        sumo = "sumo-gui" if self.gui else "sumo"
        sumo = sumolib.checkBinary(sumo)
        cmd = [sumo, "-c", str(self.sumocfg)]
        print(cmd)

        try:
            traci.start([sumo, "-c", str(self.sumocfg)])
            self.curr_step = 0
        except:
            traceback.print_exc()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Simple method to close the simulation on exit via traci"""
        traci.close()

    def step(self) -> int:
        if self.curr_step < self.end:
            try:
                # run a simulation step
                traci.simulationStep()
                self.curr_step += 1

                # clear cache containers
                # the vehicle list can change at every step
                self._vehicles = []
                self._buses = []

                return self.curr_step
            except Exception:
                traceback.print_exc()
                return -1

    ## ================================================================================
    ## vehicle methods
    def get_vehicle_ids(self) -> list[str]:
        """Returns list of all vehicles at the current simulation step"""
        if not self._vehicles:
            self._vehicles = traci.vehicle.getIDList()
        return self._vehicles

    def get_vehicle_pos(self, vehicleID: str):
        """Returns the current position of the vehicle as ???"""
        return traci.vehicle.getPosition(vehicleID)

    ## ================================================================================
    ## bus methods
    def get_bus_ids(self) -> list[str]:
        """Returns list of all buses at the current simulation step"""
        if not self._buses:
            self._buses = [
                veh for veh in self.get_vehicle_ids() if traci.vehicle.getTypeID(veh) == "bus"
            ]

        return self._buses

    def get_bus_status(self, busID: str) -> Literal["stop", "drive"]:
        """Returns bus status wrt its current bus stop"""

        # use built in method
        state: bool = traci.vehicle.isAtBusStop(busID)

        if state:
            return "stop"
        else:
            return "drive"

    def get_bus_passengers(self, busID: str) -> list[str]:
        """Returns list of all persons currently on the bus"""
        return traci.vehicle.getPersonIDList(busID)

    def get_bus_speed(self, busID: str):
        """Returns the current speed of the bus in m/s"""
        return traci.vehicle.getSpeed(busID)

    def get_bus_pos(self, busID: str):
        """Returns the current position of the bus as ???"""
        return traci.vehicle.getPosition(busID)

    ## ================================================================================
    ## bus stop methods

    def get_bus_stops(self) -> list[str]:
        """Returns list of all bus stops in the simulation"""
        if not self._stops:
            self._stops = traci.busstop.getIDList()

        return self._stops

    def get_n_boarding(self, stopID: str) -> int:
        """Returns number of person waiting for boarding at the bus stop"""
        return traci.busstop.getPersonCount(stopID)

    def get_n_alighting(self, stopID: str, busID: str) -> int:
        """Returns number of persons alighting from the bus at the bus stop"""

        # to check how many people need to alight, we retrieve the ids of the people on the bus
        persons = self.get_bus_passengers(busID)

        # we then check the travel stage of every person, and their destination
        stages = [traci.person.getStage(p, 0) for p in persons]
        alighting = [s for s in stages if s.destStop == stopID]

        return len(alighting)

    def get_next_stop_id(self, busID: str) -> str:
        """Returns the id of the next stop for the bus"""
        stops = traci.vehicle.getStops(busID)
        stops = [s for s in stops if s.stopFlags & 8]

        if len(stops):
            return stops[0].stoppingPlaceID
        else:
            return ""

    def get_next_stop_distance(self, busID: str) -> float:
        """Returns the distance to the next stop for the bus in meters"""
        stops = traci.vehicle.getStops(busID)
        stops = [s for s in stops if s.stopFlags & 8]

        if len(stops):
            stop = stops[0]
            lane = stop.lane
            edge = traci.lane.getEdgeID(lane)
            endPos = stop.endPos

            return traci.vehicle.getDrivingDistance(busID, edge, endPos)
        else:
            return -1.0
