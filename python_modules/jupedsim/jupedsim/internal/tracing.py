# SPDX-License-Identifier: LGPL-3.0-or-later

import jupedsim.native as py_jps


class Trace:
    """
    Timer can be used to measure the time spent in a code block. It is used
    internally to measure the time spent in different stages of the simulation.
    """

    def __init__(self, timer_object: py_jps.Trace=None):
        self._timer = timer_object if timer_object is not None else py_jps.Trace()
        self._timer_dict = None
        self._prev_iteration_time = 0
        self._prev_op_dec_time = 0
        pass

    def __repr__(self) -> str:
        """Prints the timer entries to the console. By default only the contributions with more than 0.01% are printed."""
        # even if the timer has been stopped this should not change things
        self._timer.pop_timer("Total Simulation Time")
        timer_dict = self._timer.get_timer_entries()
        if (timer_dict.get("Total Simulation Time") != None):
            ref = timer_dict["Total Simulation Time"]
            timer_dict.pop("Total Simulation Time")
        else:
            ref = timer_dict.get("TotalIteration")
            timer_dict.pop("TotalIteration")
        out=f"JuPedSim Timings:\n{'-'*53}\n"
        for name, duration in timer_dict.items():
            if duration/ref*100 < 0.01:
                continue
            tabs = ""
            if len(name) < 12:
                tabs = "\t\t\t"
            elif len(name) < 20:
                tabs = "\t\t"
            else:
                tabs = "\t"
            out += f"{name}: {tabs} {duration/1000000:8.2f} s ({duration/ref*100:5.2f}%)\n"
   
        out += "-----------------------------------------------------\n"
        out += f"Total Simulation Time: \t\t {ref/1000000:8.2f} s\n"
        out += ""
        return out

    def elapsed_time_us(self, key: str) -> int:
        """
        Returns:
            Elapsed time in microseconds.
        """
        return self._timer.get_timer_entry(key)
    
    def push_timer(self, name: str) -> None:
        """
        Pushes a timer with the given name. The timer will be stopped when the corresponding pop_timer is called.

        Args:
            name: Name of the timer to be pushed.
        """
        print("Warning: you are pushing to copy of the C++ timer. " \
        "This will not affect the original timer and the timings will not be recorded.")

    def pop_timer(self, name: str) -> None:
        """
        Pops a timer with the given name. The timer will be stopped and the elapsed time will be recorded.

        Args:
            name: Name of the timer to be popped.
        """
        self._timer.pop_timer(name)
    
    @property
    def iteration_duration_us(self) -> int:
        """
        Returns:
            Elapsed time of the iteration in microseconds.
        """
        current_iteration_time = self._timer.get_timer_entry("Total Iteration")
        iteration_duration = current_iteration_time - self._prev_iteration_time
        self._prev_iteration_time = self._timer.get_timer_entry("Total Iteration")
        return iteration_duration

    @property
    def operational_level_duration_us(self) -> int:
        """
        Returns:
            Elapsed time per iteration of the operational level in microseconds.
        """
        current_op_dec_time = self._timer.get_timer_entry("Operational Decision System")
        op_dec_duration = current_op_dec_time - self._prev_op_dec_time
        self._prev_op_dec_time = self._timer.get_timer_entry("Operational Decision System")
        return op_dec_duration
    
    def set_timer_instance(self, timer_object: py_jps.Trace) -> None:
        """
        Sets the timer object to be used for tracing. This can be used to set a custom timer object that is not the default one.

        Args:
            timer_object: Timer object to be used for tracing.
        """
        self._timer = timer_object

    def __str__(self) -> str:
        return self.__repr__()
    

    def enable_tracing(self) -> None:
        #self._timer.enable_tracing()
        pass