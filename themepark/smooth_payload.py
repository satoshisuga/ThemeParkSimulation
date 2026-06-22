from __future__ import annotations

from math import cos, pi, sin
from typing import Any

from themepark.config import GATE_X, GATE_Y, MAP_HEIGHT, MAP_WIDTH
from themepark.congestion import actual_estimated_waits, wait_steps_to_minutes
from themepark.models import AgentState
from themepark.pathing import attraction_entrances_payload, path_edges_payload
from themepark.presets import freshness_key_for


STATE_COLORS = {
    AgentState.CHOOSING: "#5b5f97",
    AgentState.MOVING: "#00a6a6",
    AgentState.WAITING: "#f28f3b",
    AgentState.RIDING: "#7cb518",
    AgentState.EXITING: "#c44536",
}


def build_state_payload(sim, *, max_visitors: int = 1200) -> dict[str, Any]:
    metrics = sim.current_metrics()
    actual_waits = actual_estimated_waits(sim.attractions)
    displayed_waits = sim.congestion.displayed_wait_steps
    queue_positions = _queue_positions(sim)
    rider_positions = _rider_positions(sim)
    active_visitors = [
        visitor
        for visitor in sim.visitors
        if visitor.state not in {AgentState.NOT_ENTERED, AgentState.EXITED}
    ]
    visible_visitors = _sample_visitors(active_visitors, max_visitors)

    return {
        "step": sim.step_count,
        "formattedTime": sim.formatted_time(),
        "finished": sim.finished,
        "finishedReason": sim.finished_reason,
        "map": {"width": MAP_WIDTH, "height": MAP_HEIGHT},
        "paths": path_edges_payload(),
        "entrances": attraction_entrances_payload(),
        "metrics": {
            "activeVisitors": metrics.active_visitors,
            "exitedVisitors": metrics.exited_visitors,
            "completedRides": metrics.completed_rides,
            "meanWaitMinutes": _wait_minutes(
                metrics.mean_wait_per_ride_steps,
                sim.config.step_seconds,
            ),
            "meanSatisfaction": metrics.mean_satisfaction,
            "queueImbalance": metrics.queue_imbalance,
            "choiceSynchronization": metrics.choice_synchronization,
            "oscillationAmplitude": metrics.oscillation_amplitude,
        },
        "attractions": [
            {
                "id": attraction.id,
                "name": attraction.name,
                "x": attraction.x,
                "y": attraction.y,
                "popularity": attraction.popularity,
                "queue": len(attraction.queue),
                "riders": len(attraction.riders),
                "actualWaitMinutes": _wait_minutes(
                    actual_waits[attraction.id],
                    sim.config.step_seconds,
                ),
                "displayedWaitMinutes": _wait_minutes(
                    displayed_waits[attraction.id],
                    sim.config.step_seconds,
                ),
            }
            for attraction in sim.attractions
        ],
        "visitors": [
            _visitor_payload(visitor, sim, queue_positions, rider_positions)
            for visitor in visible_visitors
        ],
        "visitorDisplay": {
            "shown": len(visible_visitors),
            "active": len(active_visitors),
            "total": len(sim.visitors),
        },
        "gate": {"x": GATE_X, "y": GATE_Y},
        "queueHistory": _queue_history_payload(sim),
        "stateLegend": [
            {"state": state.value, "label": _state_label(state), "color": color}
            for state, color in STATE_COLORS.items()
        ],
        "config": sim.config.to_dict(),
        "informationFreshness": freshness_key_for(
            sim.config.information_update_interval_steps,
            sim.config.information_delay_steps,
        ),
    }


def _visitor_payload(visitor, sim, queue_positions, rider_positions) -> dict[str, Any]:
    x, y = visitor.x, visitor.y
    if visitor.state == AgentState.WAITING:
        x, y = queue_positions.get(visitor.id, (x, y))
    elif visitor.state == AgentState.RIDING:
        x, y = rider_positions.get(visitor.id, (x, y))
    target = None
    if visitor.target_attraction_id is not None:
        target = sim.attractions[visitor.target_attraction_id].name
    return {
        "id": visitor.id,
        "state": visitor.state.value,
        "stateLabel": _state_label(visitor.state),
        "x": x,
        "y": y,
        "color": STATE_COLORS.get(visitor.state, "#64748b"),
        "hasInfo": visitor.has_congestion_info,
        "rideCount": visitor.ride_count,
        "satisfaction": visitor.satisfaction,
        "target": target,
    }


def _queue_positions(sim) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for attraction in sim.attractions:
        for rank, visitor_id in enumerate(attraction.queue):
            lane = rank % 12
            row = rank // 12
            positions[visitor_id] = (
                attraction.x - 3.0 + lane * 0.55,
                attraction.y + 2.4 + row * 0.42,
            )
    return positions


def _rider_positions(sim) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for attraction in sim.attractions:
        count = max(len(attraction.riders), 1)
        for index, visitor_id in enumerate(attraction.riders):
            angle = 2 * pi * index / count
            positions[visitor_id] = (
                attraction.x + 1.2 * cos(angle),
                attraction.y + 1.2 * sin(angle),
            )
    return positions


def _queue_history_payload(sim, limit: int = 180) -> list[dict[str, Any]]:
    history = sim.queue_length_history[-limit:]
    if not history:
        history = [(sim.step_count, tuple(len(attraction.queue) for attraction in sim.attractions))]
    return [
        {"step": step, "queues": list(queue_lengths)}
        for step, queue_lengths in history
    ]


def _sample_visitors(visitors: list, max_visitors: int) -> list:
    if len(visitors) <= max_visitors:
        return visitors
    stride = max(1, len(visitors) // max_visitors)
    return visitors[::stride][:max_visitors]


def _wait_minutes(wait_steps: float | None, step_seconds: int) -> int | None:
    return wait_steps_to_minutes(wait_steps, step_seconds)


def _state_label(state: AgentState) -> str:
    labels = {
        AgentState.CHOOSING: "選択中",
        AgentState.MOVING: "移動中",
        AgentState.WAITING: "待機中",
        AgentState.RIDING: "搭乗中",
        AgentState.EXITING: "退場移動",
    }
    return labels.get(state, state.value)
