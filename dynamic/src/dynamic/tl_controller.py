from dynamic import traci
from dynamic.bus import Bus
from dynamic.bus_manager import BusManager
from dynamic.manager import RunManager

traci
BusManager


class TLSManager:
    def __init__(
        self,
        junction_id: str,
        phase_dict: dict[int, str],
        phase_gt: dict[str, tuple[int, int]],
        detector_dist: int,
        context: RunManager,
    ):
        # initialize junction id
        self.junction_id = junction_id

        # phase dict contains the mapping of phase numbers to their names
        self.phase_dict = phase_dict

        # phase green time dict contains the mapping of phase names to their min/max gt
        # "phase name": (10, 15)
        self.phase_gt = phase_gt

        # distance from the junction at which buses are to be detected
        # needed to create a "virtual detector" via traci
        self._detector_distance = detector_dist

        # current phase and elapsed time since phase start
        self.current_phase_id = 0
        self.current_phase_name = ""
        self.time_in_phase = 0

        # list of buses approaching the intersection
        self._approaching_buses: list[Bus] = []
        # lists of buses for which prioritization has already been applied
        self._prioritized_buses: list[Bus] = []

    def update_approaching_buses(self, buses: list[Bus]) -> None:
        """Adds new approaching buses to the internal list"""
        for bus in buses:
            if bus.next_tls_id == self.junction_id:
                if bus not in self._prioritized_buses:
                    if bus not in self._approaching_buses:
                        self._approaching_buses.append(bus)

    def _update_phase_timer(self):
        current_phase_id = traci.trafficlight.getPhase(self.junction_id)
        if current_phase_id == self.current_phase_id:
            self.time_in_phase += 1
        else:
            self.current_phase_id = current_phase_id
            self.current_phase_name = self.phase_dict[self.current_phase_id]
            self.time_in_phase = 0

    def _over_min_time(self):
        min_gt = self.phase_gt[self.current_phase_name][0]
        return self.time_in_phase >= min_gt

    def _over_max_time(self):
        max_gt = self.phase_gt[self.current_phase_name][1]
        return self.time_in_phase >= max_gt

    def _next(self):
        next_phase_id = (self.current_phase_id + 1) % len(self.phase_dict)
        traci.trafficlight.setPhase(self.junction_id, next_phase_id)

    def step(self):
        # update phase counter
        self._update_phase_timer()

        # interstage phase: do nothing
        if self.current_phase_name == "interstage":
            return

        # under phase min time: do nothing
        if not self._over_min_time():
            return

        for bus in self._approaching_buses:
            if bus.next_tls_distance <= self._detector_distance:
                # implement the prioritization strategy
                pass

        # TODO: move at the END OF THE STEP
        # this way we can set bus prio mt = inf and always prioritize
        # over phase max time: go next
        if self._over_max_time():
            self._next()
            return

        #

        pass
