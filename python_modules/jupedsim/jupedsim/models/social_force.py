# SPDX-License-Identifier: LGPL-3.0-or-later
"""Social Force Model.

See the scientific publication for more details about this model:
https://doi.org/10.1038/35035023

Use ``Simulation(model=ModelType.SOCIAL_FORCE, ...)`` and add agents with
:class:`SocialForceModelState`:

.. code:: python

    sim.add_agent(
        journey_id,
        stage_id,
        jupedsim.SocialForceModelState(position=(1.0, 1.0), mass=75.0),
    )

:class:`SocialForceModelState` exposes the complete per-agent state of the
model as keyword-only constructor arguments with sensible defaults:
``position``, ``velocity``, ``mass`` (m), ``desired_speed`` (v0),
``reaction_time`` (tau), ``agent_scale`` (A), ``obstacle_scale`` (A),
``force_distance`` (B), ``radius`` (r), ``body_force`` (k) and
``friction`` (kappa).
"""

import jupedsim.native as py_jps

SocialForceModelState = py_jps.SocialForceModelState

__all__ = ["SocialForceModelState"]
