# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, replace

from jupedsim.models.custom_model import CustomModelAgentState, CustomOperationalModel
from jupedsim_examples.models.collision_free_speed import CollisionFreeSpeedModel
from jupedsim_examples.models.pysocial_force import (
    PythonSocialForceModel,
    PythonSocialForceModelState,
)


@dataclass
class MultiModelAgentState(CustomModelAgentState):
    model_type: str
    states: dict

    def __getattr__(self, name: str):
        try:
            states = object.__getattribute__(self, "states")
            model_type = object.__getattribute__(self, "model_type")
        except AttributeError:
            raise AttributeError(name)
        try:
            return getattr(states[model_type], name)
        except KeyError:
            raise AttributeError(name)


class MultiModel(CustomOperationalModel):
    def __init__(self):
        super().__init__()
        self._routes = {
            "SFM": PythonSocialForceModel(),
            "CFSM": CollisionFreeSpeedModel(),
        }

    def compute_next(self, dt, state, routing, geometry, neighborhood_search):
        model_type = state.model_type
        sub_state = state.states[model_type]
        new_sub_state = self._routes[model_type].compute_next(
            dt, sub_state, routing, geometry, neighborhood_search
        )
        return replace(state, states={**state.states, model_type: new_sub_state})

    def check_model_constraint(self, agent, neighborhood_search, geometry):
        pass
