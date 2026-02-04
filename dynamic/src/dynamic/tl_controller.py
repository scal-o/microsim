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
# = 22s -> (30 km/h = 8.5 m/s) -> 8.5 * 22 = ~200 meters
DEFAULT_DETECTOR_DISTANCE = 200  # meters

# since we don't use the actual bus speed (as it can fluctuate quite badly) we add a tolerance
# to the times we calculate
DEFAULT_TIME_TOLERANCE = 15  # seconds


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

        # current phase caching
        self._current_phase_id = -1
        self._current_phase_updated = -1

        # time in current phase counter
        self._time_in_phase = -1
        self._time_in_phase_updated = -1

        # get max and min cycle times
        self._max_cycle_time = sum(max_t for min_t, max_t in self.phase_gt.values())
        self._min_cycle_time = sum(min_t for min_t, max_t in self.phase_gt.values())

        # prioritization plan
        self.prioritization_plan: dict[str, int] = {}

    ## ============================================================
    ## current phase
    ## ============================================================
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
                if bus not in self._approaching_buses:
                    self._approaching_buses.append(bus)

    @property
    def closest_bus(self) -> Bus | None:
        """Returns the closest bus to the junction"""

        curr_min = float("inf")
        curr_bus = None
        for bus in self._approaching_buses:
            dist = bus.next_tls_distance
            if dist and dist < curr_min:
                curr_min = dist
                curr_bus = bus

        return curr_bus

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

    def _apply_bus_prio(self, bus: Bus) -> dict[str, int]:
        dist_stop = bus.next_stop_distance if bus.next_stop_distance else float("inf")
        dist_tls = bus.next_tls_distance
        # compute the travel time using an approx speed of around 30 km/h = 7 m/s
        travel_time = ceil(dist_tls / 8.5) + 2
        min_travel_time = travel_time - DEFAULT_TIME_TOLERANCE
        max_travel_time = travel_time + DEFAULT_TIME_TOLERANCE

        # the bus still has to come across the stop
        # in this case, check if the stop duration has already been set
        # if it is: add the stop time to the travel time, apply pre-prioritization strategy, add bus to the prioritized list
        # if it is not: skip prioritization for now
        if dist_stop < dist_tls:
            if bus._duration_set_for != bus.next_stop_id:
                print(f"Bus {bus.id} approaching {self.junction_id}, stop duration not set yet")
                return {}

            stop_duration = bus.next_stop.duration
            min_travel_time += stop_duration
            max_travel_time += stop_duration

        print("***")
        print("Developing prioritization plan for bus ", bus.id)
        print("Bus ", bus.id, " approaching junction in ", travel_time, " seconds")

        # apply prioritization logic
        # phase: major
        # if the bus is too close (travel time < time to switch) -> switch immediately
        # if the bus is not too close, set the duration to the difference between travel time and switch time
        if self.current_phase_name == "major":
            print("Current phase: major")
            # switch time = 2*interstage + major_left
            switch_time = 12
            if max_travel_time <= switch_time:
                print("Travel time lower than time needed for the switch: starting switch")

                return {"major": self.phase_gt[self.current_phase_name][0]}
            else:
                # set the new duration to the maximum between the remaining max time and the needed time
                major_extension = min_travel_time - switch_time
                major_extension = min(
                    major_extension, self.phase_gt["major"][1] - self.time_in_phase
                )
                print("Travel time greater than switch time, extending major phase")

                # set the minimum duration for the next bus phase
                # if the bus needs more time than its normal max green time, extend it
                bus_duration = max_travel_time - major_extension - switch_time

                major_duration = self.time_in_phase + major_extension
                return {"major": major_duration, "bus": bus_duration}

        # phase: bus
        # if the bus is close enough (travel time < remaining max green time) -> do nothing
        # if the bus is not close enough:
        # if travel time > min cycle time - bus phase -> switch immediately
        # else: extend bus phase duration
        elif self.current_phase_name == "bus":
            print("Current phase: bus")
            if max_travel_time <= self.remaining_max_time:
                print("Travel time lower than remaining max time, serving bus normally")
                return {"bus": self.phase_gt["bus"][1]}
            elif min_travel_time >= self._min_cycle_time - 10 + self.remaining_min_time:
                print("Travel time is enough to get a quick cycle in, switching now")

                # in this case, the switch time is 3*interstage + major_left + offset to let queue dissolve
                switch_time = 15

                # we set the major phase duration to the remaining time after the switch
                major_duration = min_travel_time - switch_time
                major_duration = max(self.phase_gt["major"][0], major_duration)

                return {"bus": 0, "major": major_duration}
            else:
                print("Extending bus phase to serve the bus")
                return {"bus": max_travel_time + self._time_in_phase}

    def _apply_intersection_sync(self):
        pass

    def step(self):
        # check if there are buses to prioritize
        if not self.prioritization_plan:
            if self.closest_bus:
                self.prioritization_plan = self._apply_bus_prio(self.closest_bus)
                if self.prioritization_plan:
                    print("*** New prioritization plan: ", self.prioritization_plan)
                    traci.vehicle.setColor(self.closest_bus.id, (255, 0, 0))

        # if we are not over the in green time, return anyways
        if not self._over_min_time():
            return

        # fourth step: if a prioritization scheme is given, apply it
        if self.prioritization_plan:
            if self.current_phase_name in self.prioritization_plan:
                print(
                    "=== Remaining time in phase: ",
                    self.prioritization_plan[self.current_phase_name] - self.time_in_phase,
                )
                if self.time_in_phase >= self.prioritization_plan[self.current_phase_name]:
                    self._next()
                    del self.prioritization_plan[self.current_phase_name]
                    return
                else:
                    return
            else:
                if self._over_max_time():
                    self._next()
                    return

        # over phase max time: go next
        if self._over_max_time():
            self._next()
            return
