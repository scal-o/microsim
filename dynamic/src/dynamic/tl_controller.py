from math import ceil

from dynamic import traci
from dynamic.bus import Bus
from dynamic.manager import RunManager

DEFAULT_PHASE_DICT = {
    0: "major",
    1: "interstage",
    2: "major_left",
    3: "interstage",
    4: "bus",
    5: "interstage",
}

DEFAULT_PHASE_GREEN_TIMES = {
    "major": (10, 38),
    "interstage": (3, 3),
    "major_left": (6, 6),
    "bus": (10, 37),
}

# default detector distance:
# computed as the distance a bus traveling at the max speed limit (50 km/h) can cover in:
# - min major phase gt (10s) +
# - interstage phase (3s) +
# - min major left phase gt (6s) +
# - interstage phase (3s)
# = 22s -> (50 km/h = 13.89 m/s) -> 13.89 * 22 = ~300 meters

DEFAULT_DETECTOR_DISTANCE = 300  # meters


class TLSManager:
    def __init__(
        self,
        junction_id: str,
        context: RunManager,
        phase_dict: dict[int, str] = DEFAULT_PHASE_DICT,
        phase_gt: dict[str, tuple[int, int]] = DEFAULT_PHASE_GREEN_TIMES,
        detector_dist: int = DEFAULT_DETECTOR_DISTANCE,
    ):
        # initialize junction id
        self.junction_id = junction_id

        # context
        self.ctx = context

        # phase dict contains the mapping of phase numbers to their names
        self.phase_dict = phase_dict

        # phase green time dict contains the mapping of phase names to their min/max gt
        # "phase name": (10, 15)
        self.phase_gt = phase_gt

        # distance from the junction at which buses are to be detected
        # needed to create a "virtual detector" via traci
        self._detector_distance: int = detector_dist

        # list of buses approaching the intersection
        self._approaching_buses: list[Bus] = []
        # list of buses for which prioritization has already been applied
        self._prioritized_buses: list[Bus] = []

        # current phase caching
        self._current_phase_id = -1
        self._current_phase_updated = -1

        # time in current phase counter
        self._time_in_phase = -1
        self._time_in_phase_updated = -1

    # properties to track current phase and time in phase
    # use caching to avoid repeated traci calls
    @property
    def current_phase_id(self) -> int:
        if self._current_phase_updated < self.ctx.curr_step:
            self._current_phase_id = traci.trafficlight.getPhase(self.junction_id)

        self._current_phase_updated = self.ctx.curr_step
        return self._current_phase_id

    @property
    def current_phase_name(self) -> str:
        return self.phase_dict[self.current_phase_id]

    @property
    def time_in_phase(self) -> int:
        if self._time_in_phase_updated < self.ctx.curr_step:
            self._time_in_phase = traci.trafficlight.getSpentDuration(self.junction_id)

        self._time_in_phase_updated = self.ctx.curr_step
        return self._time_in_phase

    def update_approaching_buses(self, buses: list[Bus]) -> None:
        """Updates internal bus list representations"""

        # removes buses that have already passed the junction
        for bus in self._approaching_buses:
            if bus.next_tls_id != self.junction_id:
                self._approaching_buses.remove(bus)

        # adds newly approaching buses
        for bus in buses:
            if bus.next_tls_id == self.junction_id:
                if bus not in self._prioritized_buses:
                    if bus not in self._approaching_buses:
                        self._approaching_buses.append(bus)

    @property
    def remaining_min_time(self) -> int:
        min_gt = self.phase_gt[self.current_phase_name][0]
        return max(0, min_gt - self.time_in_phase)

    @property
    def remaining_max_time(self) -> int:
        max_gt = self.phase_gt[self.current_phase_name][1]
        return max(0, max_gt - self.time_in_phase)

    def _over_min_time(self):
        return self.remaining_min_time <= 0

    def _over_max_time(self):
        return self.remaining_max_time <= 0

    def _next(self):
        next_phase_id = (self.current_phase_id + 1) % len(self.phase_dict)
        traci.trafficlight.setPhase(self.junction_id, next_phase_id)

    def step(self):
        # interstage phase: do nothing
        if self.current_phase_name == "interstage":
            return

        # under phase min time: do nothing
        if not self._over_min_time():
            return

        for bus in self._approaching_buses:
            if bus.next_tls_distance and bus.next_tls_distance <= self._detector_distance:
                dist_stop = bus.next_stop_distance if bus.next_stop_distance else float("inf")
                dist_tls = bus.next_tls_distance if bus.next_tls_distance else float("inf")
                travel_time = ceil(dist_tls / max(bus.speed, 5.0)) + 2  # avoid division by zero

                # if the bus has to stop before the traffic light:
                # - check its stop duration
                # - if the tls phase is not "bus", do not do anything
                # - if the tls phase is "bus", only take an effect if there is not enough
                #       remaining time to serve the bus (stop duration + margin > remaining time)
                if dist_stop < dist_tls:
                    # if the duration of the bus stop has been set, check it
                    if bus._duration_set_for == bus.next_stop_id:
                        stop_duration = bus.next_stop.duration

                        # if the duration is higher than zero, check the tls phase
                        if stop_duration:
                            if self.current_phase_name != "bus":
                                print(
                                    f"Bus {bus.id} will stop before junction {self.junction_id} during phase {self.current_phase_name}, phase is not bus: do nothing"
                                )
                                # bus will stop before the tls, but phase is not bus: do nothing
                                return
                            else:
                                # bus will stop before the tls, and phase is bus
                                # check if there is enough remaining time to serve the bus
                                if stop_duration + travel_time <= self.remaining_max_time:
                                    print(
                                        f"Bus {bus.id} will stop before junction {self.junction_id} during phase {self.current_phase_name}, enough time to serve"
                                    )
                                    return
                                # otherwise, check if the total time needed is more than the time it would take
                                # to go through a full cycle
                                else:
                                    tot_time = (
                                        sum(min_t for min_t, max_t in self.phase_gt.values()) - 10
                                    )
                                    if stop_duration + travel_time >= tot_time:
                                        print(
                                            f"Bus {bus.id} will stop before junction {self.junction_id} during phase {self.current_phase_name}, but not enough time in this cycle"
                                        )
                                        # not enough time in this cycle, go to next phase
                                        self._next()
                                        return

                                    print(
                                        f"Bus {bus.id} will stop before junction {self.junction_id} during phase {self.current_phase_name}, extending phase to serve the bus"
                                    )
                                    # otherwise, extend the current phase to serve the bus
                                    traci.trafficlight.setPhaseDuration(
                                        self.junction_id, stop_duration + travel_time
                                    )
                                    return

                # bus will not stop before the traffic light
                if dist_stop > dist_tls:
                    traci.vehicle.setColor(bus.id, (255, 0, 0, 255))  # set bus color to red

                    print(
                        f"Bus {bus.id} approaching junction {self.junction_id} in {travel_time}s during phase {self.current_phase_name}"
                    )

                    if self.current_phase_name == "bus":
                        # if there is still enough time to serve the bus, do nothing
                        if travel_time <= self.remaining_max_time:
                            print(
                                f"Bus {bus.id} can be served during phase {self.current_phase_name}, do nothing"
                            )
                            return
                        else:
                            print(
                                f"Bus {bus.id} cannot be fully served during phase {self.current_phase_name}, extending phase"
                            )
                            traci.trafficlight.setPhaseDuration(self.junction_id, travel_time)
                            return
                    elif self.current_phase_name == "major":
                        print(
                            f"Bus {bus.id} approaching during major phase, switching to bus phase"
                        )
                        # if the phase is major and we have already checked the minimum gt,
                        # switch phases
                        self._next()
                        return
                    else:
                        print(
                            f"Bus {bus.id} approaching junction {self.junction_id} during phase {self.current_phase_name}, but not major or bus: do nothing"
                        )
                        # phase is neither major nor bus: do nothing
                        return

        # over phase max time: go next
        if self._over_max_time():
            self._next()
            return
