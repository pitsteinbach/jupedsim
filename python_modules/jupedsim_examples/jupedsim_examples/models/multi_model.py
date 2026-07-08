# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from typing import Protocol, Type, runtime_checkable

from jupedsim.models.custom_model import (
    CustomModelAgentState,
    CustomOperationalModel,
)


@runtime_checkable
class MultiModelAgentState(CustomModelAgentState, Protocol):
    """Extended state protocol for agents in a multi-model simulation.

    All state types used together in a :class:`MultiModel` must satisfy this
    protocol so that sub-models can safely read neighbor properties during
    force/interaction calculations without knowing the neighbor's concrete
    state type.

    Required in addition to ``position`` (inherited from
    :class:`CustomModelAgentState`):

    - ``velocity``: current velocity vector, used e.g. for friction terms in
      social-force-style models.
    - ``radius``: agent body radius, used for collision / minimum-distance
      checks between agents of different sub-model types.
    """

    position: tuple[float, float]
    velocity: tuple[float, float]
    radius: float


class MultiModel(CustomOperationalModel):
    """Dispatcher that routes each agent to the appropriate sub-model.

    Each agent's sub-model is determined by the *type* of its state object.
    Register sub-models at construction time::

        model = MultiModel({
            SFMState:  PythonSocialForceModel(),
            CFSMState: PythonCFSM(),
        })

    All state types used as keys must satisfy :class:`MultiModelAgentState` so
    that neighbor interactions across sub-model boundaries can access the
    minimal shared interface (``position``, ``velocity``, ``radius``).

    ``compute_new_position`` and ``check_model_constraint`` are both dispatched
    by state type. Unknown state types raise ``KeyError``.
    """

    def __init__(self, routes: dict[Type[MultiModelAgentState], CustomOperationalModel]):
        super().__init__()
        self._routes = routes

    def _sub_model_for(self, agent):
        state_type = type(agent.model)
        try:
            return self._routes[state_type]
        except KeyError:
            raise KeyError(
                f"No sub-model registered for state type '{state_type.__name__}'. "
                f"Registered types: {[t.__name__ for t in self._routes]}"
            ) from None

    def compute_new_position(self, dt, agent, geometry, neighborhood_search):
        return self._sub_model_for(agent)._compute_new_position(
            dt, agent, geometry, neighborhood_search
        )

    def check_model_constraint(self, agent, neighborhood_search, geometry):
        self._sub_model_for(agent)._check_model_constraint(
            agent, neighborhood_search, geometry
        )
