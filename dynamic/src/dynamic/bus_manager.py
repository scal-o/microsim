from dynamic.bus import Bus
from dynamic.manager import RunManager


class BusManager:
    def __init__(self, context: RunManager):
        self.ctx = context

        self.buses = {}

    def update_buses(self):
        """Method to update the list of active buses"""

        bus_ids = self.ctx.get_bus_ids()

        # create new bus objects as needed
        for bus_id in bus_ids:
            self.buses.setdefault(bus_id, Bus(bus_id, self.ctx))

        # remove buses that are no longer active
        for bus_id in list(self.buses.keys()):
            if bus_id not in bus_ids:
                del self.buses[bus_id]

    def set_stops_duration(self):
        """Method to set the duration of each bus stop"""
        pass

    def compute_dynamic_dwell_time(self, bus: Bus) -> float:
        """Method to get the dynamic dwell time for a bus, based on number of boarding/alighting passengers"""
        n_boarding = bus.n_boarding
        n_alighting = bus.n_alighting

        duration = 15 + 0.5 * (n_boarding + n_alighting)
        return duration


class DDTBusManager(BusManager):
    def set_stops_duration(self):
        """Sets the stop durations based on dynamic dwell time model"""
        for bus in self.buses.values():
            if bus.is_at_stop():
                duration = self.compute_dynamic_dwell_time(bus)
                bus.set_stop_duration(duration)


class RSBusManager(BusManager):
    def set_stops_duration(self):
        """Sets the stop durations based on requests / waiting passengers"""
        for bus in self.buses.values():
            if bus.is_at_stop():
                # so we don't set it while it is at the stop
                return
            elif bus.next_stop and bus.next_stop_distance < 60:
                # if nobody is boarding or alighting, the dynamic dwell time will default to 15 seconds
                # if this is the case, set it to 0 instead
                duration = self.compute_dynamic_dwell_time(bus)
                if duration != 15:
                    bus.set_stop_duration(duration)
                else:
                    bus.set_stop_duration(0)
