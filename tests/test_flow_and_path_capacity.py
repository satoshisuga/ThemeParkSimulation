from themepark.config import GATE_X, GATE_Y, SimulationConfig
from themepark.engine import Simulation
from themepark.models import AgentState
from themepark.pathing import edge_id


def test_visitors_enter_at_a_constant_interval() -> None:
    sim = Simulation(SimulationConfig(visitor_count=5, entry_interval_steps=2))

    sim.step_once()
    assert sim.entered_count == 1

    sim.step_once()
    assert sim.entered_count == 1

    sim.step_once()
    assert sim.entered_count == 2

    for _ in range(7):
        sim.step_once()

    assert sim.entered_count == 5


def test_default_movement_speed_is_three_times_the_one_second_baseline() -> None:
    assert SimulationConfig().movement_speed == 0.36


def test_single_visitor_reaches_upper_center_ride_in_expected_time() -> None:
    sim = Simulation(SimulationConfig(visitor_count=1, entry_interval_steps=999))
    visitor = sim.visitors[0]
    visitor.state = AgentState.MOVING
    visitor.target_attraction_id = 1
    visitor.route = sim.path_network.route_to_attraction((GATE_X, GATE_Y), 1)

    for _ in range(249):
        sim._move_visitors()

    assert visitor.state == AgentState.MOVING

    sim._move_visitors()

    assert visitor.state == AgentState.WAITING
    assert len(sim.attractions[1].queue) == 1


def test_path_segment_entries_are_limited_per_step() -> None:
    sim = Simulation(
        SimulationConfig(
            visitor_count=4,
            entry_interval_steps=999,
            path_lane_count=3,
        )
    )
    first_segment_id = edge_id("gate", "south_junction")
    for visitor in sim.visitors:
        visitor.state = AgentState.MOVING
        visitor.x = GATE_X
        visitor.y = GATE_Y
        visitor.target_attraction_id = 1
        visitor.route = sim.path_network.route_to_attraction((GATE_X, GATE_Y), 1)

    sim._move_visitors()

    entered = [
        visitor
        for visitor in sim.visitors
        if visitor.current_segment_id == first_segment_id
    ]
    blocked = sim.visitors[3]
    assert len(entered) == 3
    assert [visitor.path_lane_index for visitor in entered] == [0, 1, 2]
    assert blocked.x == GATE_X
    assert blocked.y == GATE_Y
    assert blocked.current_segment_id is None

    sim._move_visitors()

    assert blocked.current_segment_id == first_segment_id
    assert blocked.x != GATE_X or blocked.y != GATE_Y
