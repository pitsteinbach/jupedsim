# SPDX-License-Identifier: LGPL-3.0-or-later
"""Collision Free Speed Model V3.

Variant of the Collision Free Speed Model with anisotropic neighbor
interaction and relaxed heading dynamics.

Use ``Simulation(model=ModelType.COLLISION_FREE_SPEED_V3, ...)`` and add
agents with :class:`CollisionFreeSpeedModelV3State`:

.. code:: python

    sim.add_agent(
        journey_id,
        stage_id,
        jupedsim.CollisionFreeSpeedModelV3State(
            position=(1.0, 1.0), desired_speed=1.4
        ),
    )

:class:`CollisionFreeSpeedModelV3State` exposes the complete per-agent state
of the model as keyword-only constructor arguments with sensible defaults:
``position``, ``orientation``, ``strength_neighbor_repulsion``,
``range_neighbor_repulsion``, ``strength_geometry_repulsion``,
``range_geometry_repulsion``, ``range_x_scale``, ``range_y_scale``,
``theta_max_upper_bound``, ``agent_buffer``, ``time_gap``, ``desired_speed``,
``radius`` and ``heading_angle``.
"""

import jupedsim.native as py_jps

CollisionFreeSpeedModelV3State = py_jps.CollisionFreeSpeedModelV3State

__all__ = ["CollisionFreeSpeedModelV3State"]
