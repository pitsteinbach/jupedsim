# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jupedsim.agent import Agent
    from jupedsim.geometry import Geometry
    from jupedsim.neighborhood import NeighborhoodSearch


@dataclass(kw_only=True)
class CustomModelAgentUpdate:
    """Update returned by a custom operational model.

    ``position`` and ``orientation`` update JuPedSim's canonical agent fields.
    ``model`` replaces the custom per-agent model state when it is not ``None``.
    This is the *only* supported way to change an agent's model state -- return a
    new state object here rather than mutating ``ped.model`` in place (see
    :class:`CustomOperationalModel`).
    """

    position: tuple[float, float] | None = None
    orientation: tuple[float, float] | None = None
    model: Any | None = None


@dataclass(kw_only=True)
class CustomModelAgentParameters:
    """Parameters required to create an agent for a custom model.

    ``model`` is the agent's initial per-agent model state. It should be an
    immutable object -- a ``@dataclass(frozen=True)`` is strongly recommended --
    because the simulation shares it live with your model during each step (see
    :class:`CustomOperationalModel`). It defaults to ``None`` (no state); supply
    your own state type instead of relying on a mutable default.
    """

    position: tuple[float, float] = (0.0, 0.0)
    orientation: tuple[float, float] = (0.0, 0.0)
    journey_id: int = 0
    stage_id: int = 0
    model: Any = None


class CustomOperationalModel(ABC):
    """Base class for operational models implemented in Python.

    Subclasses implement :meth:`compute_new_position` and optionally
    :meth:`check_model_constraint`. Constraint violations should be reported by
    raising an exception.

    .. warning::

        **Per-agent model state is live and shared -- never mutate it in place.**

        The ``ped.model`` object you receive (and every neighbor's ``.model``
        returned from a neighborhood query) is the agent's *live* state, shared
        by reference with the running simulation for performance. JuPedSim
        advances agents in two phases per step: it first *computes* every
        agent's update from the current state of all agents, then *applies* all
        updates together. Mutating ``ped.model`` (or a neighbor's) during the
        compute phase changes state that other agents are still reading in the
        same step, silently breaking the compute-then-apply ordering and
        producing order-dependent results.

        The only correct way to change state is to return a new state object via
        :class:`CustomModelAgentUpdate` (``model=...``). Make your state type
        immutable -- a ``@dataclass(frozen=True)`` -- so accidental in-place
        writes raise immediately instead of corrupting the simulation. See the
        ``pysocial_force`` example for the reference pattern.
    """

    @abstractmethod
    def compute_new_position(
        self,
        dt: float,
        ped: Agent,
        geometry: Geometry,
        neighborhood_search: NeighborhoodSearch,
    ) -> CustomModelAgentUpdate:
        """Compute one update for ``ped``."""

    def check_model_constraint(
        self,
        ped: Agent,
        neighborhood_search: NeighborhoodSearch,
        geometry: Geometry,
    ) -> None:
        """Raise an exception when ``ped`` violates this model's constraints."""

    def _compute_new_position(
        self,
        dt,
        ped,
        geometry,
        neighborhood_search,
    ) -> CustomModelAgentUpdate:
        from jupedsim.agent import Agent
        from jupedsim.geometry import Geometry
        from jupedsim.neighborhood import NeighborhoodSearch

        return self.compute_new_position(
            dt,
            Agent(ped),
            Geometry(geometry),
            NeighborhoodSearch(neighborhood_search),
        )

    def _check_model_constraint(
        self,
        ped,
        neighborhood_search,
        geometry,
    ) -> None:
        from jupedsim.agent import Agent
        from jupedsim.geometry import Geometry
        from jupedsim.neighborhood import NeighborhoodSearch

        self.check_model_constraint(
            Agent(ped),
            NeighborhoodSearch(neighborhood_search),
            Geometry(geometry),
        )
