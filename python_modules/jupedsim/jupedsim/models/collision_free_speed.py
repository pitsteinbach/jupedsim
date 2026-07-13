# SPDX-License-Identifier: LGPL-3.0-or-later
"""Collision Free Speed Model.

A general description of the Collision Free Speed Model can be found in the
originating publication https://arxiv.org/abs/1512.05597. A more detailed
description can be found at
https://pedestriandynamics.org/models/collision_free_speed_model/

Use ``Simulation(model=ModelType.COLLISION_FREE_SPEED, ...)`` and add agents
with :class:`CollisionFreeSpeedModelState`:

.. code:: python

    sim.add_agent(
        journey_id,
        stage_id,
        jupedsim.CollisionFreeSpeedModelState(
            position=(1.0, 1.0), desired_speed=1.4
        ),
    )

:class:`CollisionFreeSpeedModelState` exposes the complete per-agent state of
the model as keyword-only constructor arguments with sensible defaults:
``position``, ``orientation``, ``time_gap``, ``desired_speed``, ``radius``,
``strength_neighbor_repulsion``, ``range_neighbor_repulsion``,
``strength_geometry_repulsion`` and ``range_geometry_repulsion``.
"""

# SPDX-License-Identifier: LGPL-3.0-or-later
import jupedsim.native as py_jps

CollisionFreeSpeedModelState = py_jps.CollisionFreeSpeedModelState


class CollisionFreeSpeedModel:
    def __init__(self) -> None:
        self._native = py_jps.CollisionFreeSpeedModel()

    def compute_next(self, dt, state, routing, geometry, neighborhood_search):
        return self._native._compute_next_state(
            dt,
            state,
            routing,
            geometry._obj,
            neighborhood_search._obj,
        )
