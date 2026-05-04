# SPDX-License-Identifier: LGPL-3.0-or-later

import jupedsim.native as py_jps


class Timer:
    """
    Timer can be used to measure the time spent in a code block. It is used
    internally to measure the time spent in different stages of the simulation.
    """

    def __init__(self, sim_object: py_jps.Simulation, timer_log_level: int = 0):
        self._obj = sim_object
        self._obj.set_timer_log_level(timer_log_level)
        self.push_timer("Total Simulation Time")
        self._timer_dict = None
        self._prev_iteration_time = 0
        self._prev_op_dec_time = 0
        pass

    def __repr__(self) -> str:
        """Prints the timer entries to the console. By default only the contributions with more than 0.01% are printed."""
        # even if the timer has been stopped this should not change things
        self._obj.pop_timer("Total Simulation Time")
        timer_dict = self._obj.get_durations()
        if timer_dict.get("Total Simulation Time") is not None:
            ref = timer_dict["Total Simulation Time"]
            timer_dict.pop("Total Simulation Time")
        else:
            ref = timer_dict.get("Total Iteration")
            timer_dict.pop("Total Iteration")
        out = f"JuPedSim Timings:\n{'-' * 53}\n"
        for name, duration in timer_dict.items():
            if duration / ref * 100 < 0.01:
                continue
            out += f"{name:<30}| {duration / 1000000:8.2f} s ({duration / ref * 100:5.2f}%)\n"

        out += "-----------------------------------------------------\n"
        out += f"{('Total Simulation Time'):<30}| {ref / 1000000:8.2f} s\n"
        out += ""
        return out

    def elapsed_time_us(self, key: str) -> int:
        """
        Returns:
            Elapsed time in microseconds.
        """
        return self._obj.get_duration(key)

    def push_timer(self, name: str) -> None:
        """
        Pushes a timer with the given name. The timer will be stopped when the corresponding pop_timer is called.

        Args:
            name: Name of the timer to be pushed.
        """
        self._obj.push_timer(name)

    def pop_timer(self, name: str) -> None:
        """
        Pops a timer with the given name. The timer will be stopped and the elapsed time will be recorded.

        Args:
            name: Name of the timer to be popped.
        """
        self._obj.pop_timer(name)

    @property
    def iteration_duration_us(self) -> int:
        """
        Returns:
            Elapsed time of the iteration in microseconds.
        """
        current_iteration_time = self._obj.get_duration("Total Iteration")
        iteration_duration = current_iteration_time - self._prev_iteration_time
        self._prev_iteration_time = current_iteration_time
        return iteration_duration

    @property
    def operational_level_duration_us(self) -> int:
        """
        Returns:
            Elapsed time per iteration of the operational level in microseconds.
        """
        current_op_dec_time = self._obj.get_duration(
            "Operational Decision System"
        )
        op_dec_duration = current_op_dec_time - self._prev_op_dec_time
        self._prev_op_dec_time = current_op_dec_time
        return op_dec_duration

    def set_timer_instance(self, timer_object: py_jps.Trace) -> None:
        """
        Sets the timer object to be used for tracing. This can be used to set a custom timer object that is not the default one.

        Args:
            timer_object: Timer object to be used for tracing.
        """
        self._obj = timer_object

    def __str__(self) -> str:
        return self.__repr__()


class Profiler:
    """
    Profiler can be used to enable and disable the profiler and to dump the profiler traces to a file.
    """

    def __init__(self):
        self._profiler = py_jps.Trace.instance()
        self.enable()

    def enable(self) -> None:
        """Enables the profiler."""
        self._profiler.enable()

    def disable(self) -> None:
        """Disables the profiler."""
        self._profiler.disable()

    def dump_and_reset(self, filename: str) -> None:
        """Dumps the profiler traces to a file and resets the profiler.

        Args:
            filename: Name of the file to dump the traces to.
        """
        self._profiler.dump_and_reset(filename)

    def push_probe(self, name: str) -> None:
        """Pushes a probe with the given name. The probe will be stopped when the corresponding pop_probe is called.

        Args:
            name: Name of the probe to be pushed.
        """
        self._profiler.push_probe(name)

    def pop_probe(self) -> None:
        """Pops the last pushed probe. The probe will be stopped and the elapsed time will be recorded."""
        self._profiler.pop_probe()
