# SPDX-License-Identifier: LGPL-3.0-or-later
"""Collision Free Speed Model V2.

Variant of the Collision Free Speed Model (https://arxiv.org/abs/1512.05597)
in which all repulsion parameters are per-agent instead of global.

Use ``Simulation(model=ModelType.COLLISION_FREE_SPEED_V2, ...)`` and add
agents with :class:`CollisionFreeSpeedModelV2State`:

.. code:: python

    sim.add_agent(
        journey_id,
        stage_id,
        jupedsim.CollisionFreeSpeedModelV2State(
            position=(1.0, 1.0), strength_neighbor_repulsion=9.0
        ),
    )

:class:`CollisionFreeSpeedModelV2State` exposes the complete per-agent state
of the model as keyword-only constructor arguments with sensible defaults:
``position``, ``orientation``, ``strength_neighbor_repulsion``,
``range_neighbor_repulsion``, ``strength_geometry_repulsion``,
``range_geometry_repulsion``, ``time_gap``, ``desired_speed`` and ``radius``.
"""

import jupedsim.native as py_jps

CollisionFreeSpeedModelV2State = py_jps.CollisionFreeSpeedModelV2State

__all__ = ["CollisionFreeSpeedModelV2State"]
