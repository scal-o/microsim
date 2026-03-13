from dynamic import traci
from dynamic.manager import RunManager


class Bus:
    """Simple class with info for a single bus"""

    def __init__(self, bus_id: str, context: RunManager):
        self.id = bus_id
        self.ctx = context

        # next stop info
        self._next_stop = None
        self._next_stop_updated = -1

        # passengers info
        self._passengers = []
        self._passengers_updated = -1
        self._passengers_before_stop = []
        self._passengers_before_stop_updated = -1

        self._alighting = 0
        self._boarding = 0

        # bus status info
        self._stopped = False
        self._stopped_updated = -1
        self._stop_start_time = -1

        # stop duration control
        # prevents updating the stop duration multiple times for the same stop
        self._duration_set_for = None
        self._last_duration = 0

        # next tls info
        self._next_tls = None
        self._next_tls_updated = -1

    def is_at_stop(self) -> bool:
        """Returns whether the bus is currently at a stop or not"""

        if self._stopped_updated < self.ctx.curr_step:
            was_stopped = self._stopped
            self._stopped = traci.vehicle.isAtBusStop(self.id)

            if self._stopped and not was_stopped:
                self._stop_start_time = self.ctx.curr_step
            elif not self._stopped and was_stopped:
                self._stop_start_time = -1

        self._stopped_updated = self.ctx.curr_step
        return self._stopped

    def update(self):
        """Method to update the bus info, should be called every step"""

        # we update the next stop and passengers info every step so it is always up to date when we need it
        self.next_stop
        self.next_tls
        self.passengers
        self.passengers_before_stop
        self.n_alighting
        self.n_boarding

    @property
    def next_stop(self) -> traci._vehicle.StopData | None:
        """Returns the next stop data of the bus (None if no next stop)"""

        # check cached data so we only update once per step
        if self._next_stop_updated < self.ctx.curr_step:
            stops = traci.vehicle.getStops(self.id)
            stops = [s for s in stops if s.stopFlags & 8]

            if len(stops):
                self._next_stop = stops[0]
            else:
                self._next_stop = None

            self._next_stop_updated = self.ctx.curr_step

        return self._next_stop

    @property
    def next_stop_id(self) -> str | None:
        """Returns the ID of the next stop of the bus (None if no next stop)"""

        if self.next_stop:
            return self.next_stop.stoppingPlaceID

        return None

    @property
    def next_stop_distance(self) -> float | None:
        """Returns the distance to the next stop of the bus (None if no next stop)"""
        if self.next_stop:
            lane = self.next_stop.lane
            edge = traci.lane.getEdgeID(lane)
            endPos = self.next_stop.endPos
            return traci.vehicle.getDrivingDistance(self.id, edge, endPos)

        return None

    @property
    def passengers(self) -> list[str]:
        """Returns list of person IDs currently on the bus"""

        if self._passengers_updated < self.ctx.curr_step:
            self._passengers = traci.vehicle.getPersonIDList(self.id)

        self._passengers_updated = self.ctx.curr_step
        return self._passengers

    @property
    def passengers_before_stop(self) -> list[str]:
        if not self.is_at_stop():
            if self._passengers_before_stop_updated < self.ctx.curr_step:
                self._passengers_before_stop = traci.vehicle.getPersonIDList(self.id)

        self._passengers_before_stop_updated = self.ctx.curr_step
        return self._passengers_before_stop

    @property
    def n_alighting(self) -> int:
        """Returns number of persons alighting from the bus at the next stop (0 if no next stop)"""

        if not self.next_stop:
            return 0

        if not self.is_at_stop():
            # we then check the travel stage of every person, and their destination
            stages = [traci.person.getStage(p, 0) for p in self.passengers]
            alighting = [s for s in stages if s.destStop == self.next_stop_id]
            self._alighting = len(alighting)

        return self._alighting

    @property
    def n_boarding(self) -> int:
        """Returns number of persons boarding the bus at the next stop (0 if no next stop)"""

        if not self.next_stop:
            return 0

        if not self.is_at_stop():
            self._boarding = traci.busstop.getPersonCount(self.next_stop_id)
        else:
            self._boarding = len(set(self.passengers) - set(self.passengers_before_stop))

        return self._boarding

    def set_stop_duration(self, duration: float) -> None:
        """Sets the duration of the next stop of the bus (does nothing if no next stop)"""

        if not self.next_stop:
            return
        # sets the bus stop duration and freezes it so it cannot be updated again for this stop
        # reset the cache so the next calls to self.next_stop will have the updated data
        if (
            self.next_stop and self._duration_set_for != self.next_stop_id
        ) or duration > self._last_duration:
            edge = traci.lane.getEdgeID(self.next_stop.lane)
            startpos = self.next_stop.startPos
            endpos = self.next_stop.endPos

            elapsed = self.ctx.curr_step - self._stop_start_time
            remaining = max(0, duration - elapsed)

            traci.vehicle.setStop(
                self.id, edgeID=edge, startPos=startpos, pos=endpos, duration=remaining
            )

            print(
                f"Bus: {self.id}, stop: {self.next_stop_id}, alighting: {self.n_alighting}, boarding: {self.n_boarding}, duration: {duration}"
            )

            self._duration_set_for = self.next_stop_id
            self._last_duration = duration
            self._next_stop_updated = -1

    @property
    def next_tls(self) -> tuple[str, int, float, int] | None:
        """
        Returns tuple (tlsID, tlsIndex, distance, state) for the next
        traffic light the bus will encounter
        """

        if self._next_tls_updated < self.ctx.curr_step:
            tls = traci.vehicle.getNextTLS(self.id)
            if len(tls):
                self._next_tls = tls[0]
            else:
                self._next_tls = None

            self._next_tls_updated = self.ctx.curr_step
        return self._next_tls

    @property
    def next_tls_id(self) -> str | None:
        """Returns the ID of the next traffic light the bus will encounter (None if no next tls)"""
        if self.next_tls:
            return self.next_tls[0]
        return None

    @property
    def next_tls_distance(self) -> float | None:
        """Returns the distance of the next traffic light the bus will encounter (None if no next tls)"""
        if self.next_tls:
            return self.next_tls[2]
        return None

    @property
    def speed(self) -> float:
        """Returns the current speed of the bus in m/s"""
        return traci.vehicle.getSpeed(self.id)
