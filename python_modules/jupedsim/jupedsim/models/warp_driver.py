# SPDX-License-Identifier: LGPL-3.0-or-later
"""WarpDriver Model.

Based on Wolinski, Lin, and Pettré (2016) -- probabilistic collision
avoidance using warped intrinsic fields.

The WarpDriver model carries simulation-global state (the precomputed
intrinsic collision-probability field controlled by ``sigma`` and a random
number generator), so it is passed to the simulation as an instance:

.. code:: python

    sim = jupedsim.Simulation(
        model=jupedsim.WarpDriverModel(sigma=0.3, rng_seed=42),
        geometry=...,
    )
    sim.add_agent(
        journey_id,
        stage_id,
        jupedsim.WarpDriverModelState(position=(1.0, 1.0)),
    )

.. warning::

    The model instance is consumed by the ``Simulation`` constructor and must
    not be reused afterwards.

:class:`WarpDriverModelState` exposes the complete per-agent state of the
model as keyword-only constructor arguments with sensible defaults:
``position``, ``orientation``, ``radius``, ``desired_speed``, ``stuck_time``,
``anchor_x``, ``anchor_y``, ``detour_time``, ``detour_side``,
``time_horizon``, ``step_size``, ``time_uncertainty``,
``velocity_uncertainty_x``, ``velocity_uncertainty_y`` and ``num_samples``.
"""

import jupedsim.native as py_jps

WarpDriverModel = py_jps.WarpDriverModel
WarpDriverModelState = py_jps.WarpDriverModelState

__all__ = [
    "WarpDriverModel",
    "WarpDriverModelState",
]
