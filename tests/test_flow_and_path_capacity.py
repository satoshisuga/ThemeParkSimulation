from themepark.config import GATE_X, GATE_Y, SimulationConfig
from themepark.engine import Simulation
from themepark.models import AgentState
from themepark.pathing import edge_id
from themepark.smooth_payload import build_state_payload


def _entry_events(sim: Simulation) -> list[tuple[int, int]]:
    events: dict[int, int] = {}
    for visitor in sim.visitors:
        if visitor.entered_at is not None:
            events[visitor.entered_at] = events.get(visitor.entered_at, 0) + 1
    return sorted(events.items())


def test_visitors_enter_in_random_groups_at_random_intervals() -> None:
    sim = Simulation(SimulationConfig(visitor_count=100, seed=123))

    for _ in range(25):
        sim.step_once()

    events = _entry_events(sim)
    assert events[0][0] == 0
    for _, entered_count in events:
        assert sim.config.entry_group_min <= entered_count <= sim.config.entry_group_max
    for (previous_step, _), (current_step, _) in zip(events, events[1:]):
        interval = current_step - previous_step
        assert sim.config.entry_interval_min_steps <= interval <= sim.config.entry_interval_max_steps

    assert len(events) > 4


def test_entry_flow_is_seeded() -> None:
    def collect_entry_events(seed: int) -> list[tuple[int, int]]:
        sim = Simulation(SimulationConfig(visitor_count=100, seed=seed))
        for _ in range(26):
            sim.step_once()
        return _entry_events(sim)

    assert collect_entry_events(42) == collect_entry_events(42)


def test_default_movement_speed_is_slightly_faster() -> None:
    config = SimulationConfig()
    assert config.entry_interval_min_steps == 1
    assert config.entry_interval_max_steps == 4
    assert config.entry_group_min == 1
    assert config.entry_group_max == 3
    assert config.movement_speed == 1.15
    assert config.attraction_exit_spacing == 1.6
    assert config.attraction_loading_wait_steps == 60


def test_single_visitor_reaches_upper_center_ride_in_expected_time() -> None:
    sim = Simulation(
        SimulationConfig(
            visitor_count=1,
            entry_interval_min_steps=999,
            entry_interval_max_steps=999,
        )
    )
    visitor = sim.visitors[0]
    visitor.state = AgentState.MOVING
    visitor.target_attraction_id = 1
    visitor.route = sim.path_network.route_to_attraction((GATE_X, GATE_Y), 1)

    for _ in range(78):
        sim._move_visitors()

    assert visitor.state == AgentState.MOVING

    sim._move_visitors()

    assert visitor.state == AgentState.WAITING
    assert len(sim.attractions[1].queue) == 1


def test_path_segment_entries_are_limited_per_step() -> None:
    sim = Simulation(
        SimulationConfig(
            visitor_count=4,
            entry_interval_min_steps=999,
            entry_interval_max_steps=999,
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


def test_attraction_waits_for_capacity_before_boarding() -> None:
    sim = Simulation(
        SimulationConfig(
            visitor_count=8,
            attraction_loading_wait_steps=60,
        )
    )
    attraction = sim.attractions[1]
    attraction.queue.append(0)
    sim.visitors[0].state = AgentState.WAITING
    sim.visitors[0].queue_entered_at = 0

    sim._board_waiting_visitors()

    assert attraction.riders == []
    assert list(attraction.queue) == [0]
    assert attraction.loading_started_at == 0

    for visitor_id in range(1, attraction.capacity):
        attraction.queue.append(visitor_id)
        sim.visitors[visitor_id].state = AgentState.WAITING
        sim.visitors[visitor_id].queue_entered_at = 1

    sim._step = 1
    sim._board_waiting_visitors()

    assert len(attraction.riders) == attraction.capacity
    assert attraction.loading_started_at is None


def test_attraction_boards_after_loading_timeout() -> None:
    sim = Simulation(
        SimulationConfig(
            visitor_count=1,
            attraction_loading_wait_steps=2,
        )
    )
    attraction = sim.attractions[1]
    attraction.queue.append(0)
    sim.visitors[0].state = AgentState.WAITING
    sim.visitors[0].queue_entered_at = 0

    sim._board_waiting_visitors()
    sim._step = 1
    sim._board_waiting_visitors()

    assert attraction.riders == []

    sim._step = 2
    sim._board_waiting_visitors()

    assert attraction.riders == [0]
    assert attraction.loading_started_at is None


def test_completed_riders_are_spaced_at_attraction_exit() -> None:
    sim = Simulation(
        SimulationConfig(
            visitor_count=5,
            attraction_exit_spacing=1.2,
        )
    )
    attraction = sim.attractions[1]
    attraction.riders = [0, 1, 2, 3, 4]
    attraction.cycle_remaining_steps = 0
    for visitor_id in attraction.riders:
        sim.visitors[visitor_id].state = AgentState.RIDING

    sim._finish_completed_rides()

    assert [
        round(sim.visitors[visitor_id].release_offset_x, 1)
        for visitor_id in range(5)
    ] == [-0.7, 0.0, 0.7, -0.7, 0.0]
    assert [
        sim.visitors[visitor_id].release_offset_y
        for visitor_id in range(5)
    ] == [0.0, 0.0, 0.0, 1.2, 1.2]


def test_release_spacing_is_reflected_in_smooth_payload() -> None:
    sim = Simulation(SimulationConfig(visitor_count=1))
    visitor = sim.visitors[0]
    visitor.state = AgentState.MOVING
    visitor.x = 50.0
    visitor.y = 20.0
    visitor.release_offset_x = 1.2
    visitor.release_offset_y = 0.4

    payload = build_state_payload(sim)

    assert payload["visitors"][0]["x"] == 51.2
    assert payload["visitors"][0]["y"] == 20.4


def test_attraction_cycle_remaining_time_is_in_smooth_payload() -> None:
    sim = Simulation(SimulationConfig(visitor_count=1))
    attraction = sim.attractions[1]
    attraction.riders = [0]
    attraction.cycle_remaining_steps = 75

    payload = build_state_payload(sim)
    attraction_payload = payload["attractions"][1]

    assert attraction_payload["nextStartStatus"] == "running"
    assert attraction_payload["nextStartRemainingSeconds"] == 75


def test_attraction_loading_remaining_time_is_in_smooth_payload() -> None:
    sim = Simulation(SimulationConfig(visitor_count=1, attraction_loading_wait_steps=60))
    attraction = sim.attractions[1]
    attraction.queue.append(0)
    attraction.loading_started_at = 10
    sim._step = 40

    payload = build_state_payload(sim)
    attraction_payload = payload["attractions"][1]

    assert attraction_payload["nextStartStatus"] == "loading"
    assert attraction_payload["nextStartRemainingSeconds"] == 30


def test_release_spacing_remains_visible_on_first_path_segment() -> None:
    sim = Simulation(SimulationConfig(visitor_count=1))
    visitor = sim.visitors[0]
    visitor.state = AgentState.MOVING
    visitor.x = GATE_X
    visitor.y = GATE_Y
    visitor.target_attraction_id = 1
    visitor.route = sim.path_network.route_to_attraction((GATE_X, GATE_Y), 1)
    visitor.release_offset_y = 1.6

    sim._move_visitors()
    payload = build_state_payload(sim)

    assert visitor.current_segment_id == edge_id("gate", "south_junction")
    assert visitor.release_offset_y == 1.6
    assert payload["visitors"][0]["y"] < visitor.y - 0.8
