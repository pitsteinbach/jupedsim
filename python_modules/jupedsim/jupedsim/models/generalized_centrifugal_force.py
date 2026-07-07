# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generalized Centrifugal Force Model.

See the scientific publication for more details about this model:
https://arxiv.org/abs/1008.4297

Use ``Simulation(model=ModelType.GENERALIZED_CENTRIFUGAL_FORCE, ...)`` and
add agents with :class:`GeneralizedCentrifugalForceModelState`:

.. code:: python

    sim.add_agent(
        journey_id,
        stage_id,
        jupedsim.GeneralizedCentrifugalForceModelState(
            position=(1.0, 1.0), desired_speed=1.0
        ),
    )

:class:`GeneralizedCentrifugalForceModelState` exposes the complete per-agent
state of the model as keyword-only constructor arguments with sensible
defaults: ``position``, ``orientation``, ``speed``, ``desired_direction``
(e0), ``orientation_delay``, ``mass``, ``tau``, ``desired_speed`` (v0),
``a_v``, ``a_min``, ``b_min``, ``b_max``, ``strength_neighbor_repulsion``,
``strength_geometry_repulsion``, ``max_neighbor_interaction_distance``,
``max_geometry_interaction_distance``,
``max_neighbor_interpolation_distance``,
``max_geometry_interpolation_distance``, ``max_neighbor_repulsion_force`` and
``max_geometry_repulsion_force``.
"""

import jupedsim.native as py_jps

GeneralizedCentrifugalForceModelState = (
    py_jps.GeneralizedCentrifugalForceModelState
)

__all__ = ["GeneralizedCentrifugalForceModelState"]
