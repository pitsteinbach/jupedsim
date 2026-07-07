# SPDX-License-Identifier: LGPL-3.0-or-later
import jupedsim as jps
import pytest

# Expected defaults are pinned to the C++ Agent struct initializers of each
# operational model. Constructing a state with only a position must yield
# exactly these values.
STATE_DEFAULTS = [
    (
        jps.CollisionFreeSpeedModelState,
        {
            "orientation": (0.0, 0.0),
            "time_gap": 1.0,
            "desired_speed": 1.2,
            "radius": 0.2,
            "strength_neighbor_repulsion": 8.0,
            "range_neighbor_repulsion": 0.1,
            "strength_geometry_repulsion": 5.0,
            "range_geometry_repulsion": 0.02,
        },
    ),
    (
        jps.CollisionFreeSpeedModelV2State,
        {
            "orientation": (0.0, 0.0),
            "time_gap": 1.0,
            "desired_speed": 1.2,
            "radius": 0.2,
            "strength_neighbor_repulsion": 8.0,
            "range_neighbor_repulsion": 0.1,
            "strength_geometry_repulsion": 5.0,
            "range_geometry_repulsion": 0.02,
        },
    ),
    (
        jps.CollisionFreeSpeedModelV3State,
        {
            "orientation": (1.0, 0.0),
            "strength_neighbor_repulsion": 8.0,
            "range_neighbor_repulsion": 0.1,
            "strength_geometry_repulsion": 5.0,
            "range_geometry_repulsion": 0.02,
            "range_x_scale": 20.0,
            "range_y_scale": 8.0,
            "theta_max_upper_bound": 1.57,
            "agent_buffer": 0.0,
            "time_gap": 1.0,
            "desired_speed": 1.2,
            "radius": 0.2,
            "heading_angle": 0.0,
        },
    ),
    (
        jps.GeneralizedCentrifugalForceModelState,
        {
            "orientation": (1.0, 0.0),
            "speed": 0.0,
            "desired_direction": (0.0, 0.0),
            "orientation_delay": 0,
            "mass": 1.0,
            "tau": 0.5,
            "desired_speed": 1.2,
            "a_v": 1.0,
            "a_min": 0.2,
            "b_min": 0.2,
            "b_max": 0.4,
            "strength_neighbor_repulsion": 0.3,
            "strength_geometry_repulsion": 0.2,
            "max_neighbor_interaction_distance": 2.0,
            "max_geometry_interaction_distance": 2.0,
            "max_neighbor_interpolation_distance": 0.1,
            "max_geometry_interpolation_distance": 0.1,
            "max_neighbor_repulsion_force": 9.0,
            "max_geometry_repulsion_force": 3.0,
        },
    ),
    (
        jps.SocialForceModelState,
        {
            "velocity": (0.0, 0.0),
            "mass": 80.0,
            "desired_speed": 0.8,
            "reaction_time": 0.5,
            "agent_scale": 2000.0,
            "obstacle_scale": 2000.0,
            "force_distance": 0.08,
            "radius": 0.3,
            "body_force": 120000.0,
            "friction": 240000.0,
        },
    ),
    (
        jps.AnticipationVelocityModelState,
        {
            "orientation": (0.0, 0.0),
            "strength_neighbor_repulsion": 8.0,
            "range_neighbor_repulsion": 0.1,
            "wall_buffer_distance": 0.1,
            "anticipation_time": 1.0,
            "reaction_time": 0.3,
            "velocity": (0.0, 0.0),
            "time_gap": 1.06,
            "desired_speed": 1.2,
            "radius": 0.2,
            "pushout_strength": 0.3,
        },
    ),
    (
        jps.WarpDriverModelState,
        {
            "orientation": (0.0, 0.0),
            "radius": 0.15,
            "desired_speed": 1.2,
            "stuck_time": 0.0,
            "anchor_x": 0.0,
            "anchor_y": 0.0,
            "detour_time": 0.0,
            "detour_side": 1,
            "time_horizon": 2.0,
            "step_size": 0.5,
            "time_uncertainty": 0.5,
            "velocity_uncertainty_x": 0.2,
            "velocity_uncertainty_y": 0.2,
            "num_samples": 20,
        },
    ),
]


@pytest.mark.parametrize(
    "state_cls, expected_defaults",
    STATE_DEFAULTS,
    ids=[cls.__name__ for cls, _ in STATE_DEFAULTS],
)
def test_state_defaults_match_cpp_struct_initializers(
    state_cls, expected_defaults
):
    """State ctor defaults must match the C++ Agent struct initializers."""
    state = state_cls(position=(0, 0))
    assert state.position == (0, 0)
    for field, expected in expected_defaults.items():
        actual = getattr(state, field)
        assert actual == pytest.approx(expected), (
            f"{state_cls.__name__}.{field}: expected {expected}, got {actual}"
        )
