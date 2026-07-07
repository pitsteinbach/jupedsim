# SPDX-License-Identifier: LGPL-3.0-or-later
"""Operational models of JuPedSim.

Stateless built-in models are selected with :class:`ModelType` when creating
a :class:`~jupedsim.simulation.Simulation`. Models with simulation-global
state (:class:`~jupedsim.models.anticipation_velocity_model.AnticipationVelocityModel`,
:class:`~jupedsim.models.warp_driver.WarpDriverModel`) and custom Python
models (:class:`~jupedsim.models.custom_model.CustomOperationalModel`
subclasses) are passed as instances instead.
"""

import jupedsim.native as py_jps

ModelType = py_jps.ModelType
"""Selects one of the stateless built-in operational models.

Members: ``COLLISION_FREE_SPEED``, ``COLLISION_FREE_SPEED_V2``,
``COLLISION_FREE_SPEED_V3``, ``GENERALIZED_CENTRIFUGAL_FORCE``,
``SOCIAL_FORCE``.
"""

__all__ = ["ModelType"]
