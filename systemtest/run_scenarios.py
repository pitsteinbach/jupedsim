# SPDX-License-Identifier: LGPL-3.0-or-later
"""QA baseline: run one deterministic scenario per operational model and dump
trajectories as '<iteration>,<agent_id>,<repr(x)>,<repr(y)>' lines."""

import os
import sys

import jupedsim as jps
import shapely
from jupedsim_examples.models.pysocial_force import (
    PythonSocialForceModel,
    PythonSocialForceModelState,
)

ROOM = shapely.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
EXIT_POLY = [(19, 9), (19, 11), (20, 11), (20, 9)]
DT = 0.05
MAX_ITERATIONS = 500


def grid_positions():
    # 4x4 grid, 2 m spacing (GCFM requires generous inter-agent distance).
    return [
        (2.0 + col * 2.0, 7.0 + row * 2.0)
        for row in range(4)
        for col in range(4)
    ]


def run(model, make_state, out_path):
    sim = jps.Simulation(model=model, geometry=ROOM, dt=DT)
    exit_id = sim.add_exit_stage(EXIT_POLY)
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))
    for pos in grid_positions():
        sim.add_agent(journey_id, exit_id, make_state(pos))

    lines = []
    for it in range(1, MAX_ITERATIONS + 1):
        if sim.agent_count() == 0:
            break
        sim.iterate()
        for agent in sim.agents():
            x, y = agent.position
            lines.append(f"{it},{int(agent.id)},{x!r},{y!r}\n")
    with open(out_path, "w") as f:
        f.writelines(lines)


def scenarios():
    yield (
        "CollisionFreeSpeedModel",
        jps.ModelType.COLLISION_FREE_SPEED,
        lambda pos: jps.CollisionFreeSpeedModelState(position=pos),
    )
    yield (
        "CollisionFreeSpeedModelV2",
        jps.ModelType.COLLISION_FREE_SPEED_V2,
        lambda pos: jps.CollisionFreeSpeedModelV2State(position=pos),
    )
    yield (
        "CollisionFreeSpeedModelV3",
        jps.ModelType.COLLISION_FREE_SPEED_V3,
        # Pin the historic per-agent defaults of the removed
        # CollisionFreeSpeedModelV3AgentParameters dataclass.
        lambda pos: jps.CollisionFreeSpeedModelV3State(
            position=pos,
            orientation=(0.0, 0.0),
            radius=0.2,
            strength_neighbor_repulsion=8.0,
            range_neighbor_repulsion=0.1,
            strength_geometry_repulsion=5.0,
            range_geometry_repulsion=0.02,
        ),
    )
    yield (
        "GeneralizedCentrifugalForceModel",
        jps.ModelType.GENERALIZED_CENTRIFUGAL_FORCE,
        # Pin the historic default orientation of the removed
        # GeneralizedCentrifugalForceModelAgentParameters dataclass.
        lambda pos: jps.GeneralizedCentrifugalForceModelState(
            position=pos, orientation=(1.0, 0.0)
        ),
    )
    yield (
        "AnticipationVelocityModel",
        jps.AnticipationVelocityModel(rng_seed=1234),
        lambda pos: jps.AnticipationVelocityModelState(position=pos),
    )
    yield (
        "SocialForceModel",
        jps.ModelType.SOCIAL_FORCE,
        lambda pos: jps.SocialForceModelState(position=pos),
    )
    yield (
        "WarpDriverModel",
        jps.WarpDriverModel(),
        lambda pos: jps.WarpDriverModelState(position=pos),
    )
    yield (
        "PythonSocialForceModel",
        PythonSocialForceModel(),
        lambda pos: PythonSocialForceModelState(
            position=pos, velocity=(0.0, 0.0)
        ),
    )


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for name, model, make_state in scenarios():
        out_path = os.path.join(out_dir, f"{name}.txt")
        run(model, make_state, out_path)
        print(f"{name}: wrote {out_path}")


if __name__ == "__main__":
    main()
