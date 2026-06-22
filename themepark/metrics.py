from __future__ import annotations

from statistics import fmean

import numpy as np

from themepark.config import SimulationConfig
from themepark.models import AgentState, Attraction, MetricSnapshot, VisitorAgent


def optional_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(fmean(values))


def queue_imbalance(queue_lengths: tuple[int, ...]) -> float:
    if not queue_lengths:
        return 0.0
    values = np.asarray(queue_lengths, dtype=np.float64)
    return float(values.std() / max(values.mean(), 1.0))


def oscillation_amplitude(
    queue_length_history: list[tuple[int, tuple[int, ...]]],
    current_step: int,
    window_steps: int,
) -> float | None:
    window = [
        lengths
        for step, lengths in queue_length_history
        if current_step - window_steps <= step <= current_step
    ]
    if len(window) < 2:
        return None
    values = np.asarray(window, dtype=np.float64)
    if values.sum() < 5:
        return None
    p90 = np.percentile(values, 90, axis=0)
    p10 = np.percentile(values, 10, axis=0)
    means = values.mean(axis=0)
    amplitudes = (p90 - p10) / (means + 1.0)
    return float(amplitudes.mean())


def satisfaction_delta(
    visitor: VisitorAgent,
    attraction_id: int,
    config: SimulationConfig,
    map_diagonal: float,
) -> float:
    return float(
        config.like_reward_weight * visitor.preferences[attraction_id]
        - config.wait_penalty_weight * (visitor.last_queue_wait_steps / config.wait_reference_steps)
        - config.move_penalty_weight * (visitor.current_trip_distance / map_diagonal)
    )


def make_snapshot(
    *,
    step: int,
    visitors: list[VisitorAgent],
    attractions: list[Attraction],
    config: SimulationConfig,
    completed_rides: int,
    total_completed_wait_steps: int,
    choice_synchronization: float | None,
    queue_length_history: list[tuple[int, tuple[int, ...]]],
) -> MetricSnapshot:
    queue_lengths = tuple(len(attraction.queue) for attraction in attractions)
    active_states = {
        AgentState.CHOOSING,
        AgentState.MOVING,
        AgentState.WAITING,
        AgentState.RIDING,
        AgentState.EXITING,
    }
    active_visitors = sum(visitor.state in active_states for visitor in visitors)
    exited_visitors = sum(visitor.state == AgentState.EXITED for visitor in visitors)
    exited_stays = [
        float(visitor.exited_at - visitor.entered_at)
        for visitor in visitors
        if visitor.exited_at is not None and visitor.entered_at is not None
    ]
    experienced = [visitor for visitor in visitors if visitor.ride_count > 0]
    info_satisfaction = [
        visitor.satisfaction for visitor in experienced if visitor.has_congestion_info
    ]
    no_info_satisfaction = [
        visitor.satisfaction for visitor in experienced if not visitor.has_congestion_info
    ]
    mean_wait = (
        total_completed_wait_steps / completed_rides
        if completed_rides > 0
        else None
    )
    return MetricSnapshot(
        step=step,
        active_visitors=active_visitors,
        exited_visitors=exited_visitors,
        completed_rides=completed_rides,
        mean_wait_per_ride_steps=float(mean_wait) if mean_wait is not None else None,
        mean_stay_steps=optional_mean(exited_stays),
        mean_satisfaction=optional_mean([visitor.satisfaction for visitor in experienced]),
        mean_satisfaction_info=optional_mean(info_satisfaction),
        mean_satisfaction_no_info=optional_mean(no_info_satisfaction),
        queue_imbalance=queue_imbalance(queue_lengths),
        choice_synchronization=choice_synchronization,
        oscillation_amplitude=oscillation_amplitude(
            queue_length_history,
            step,
            config.recent_oscillation_window_steps,
        ),
        queue_lengths=queue_lengths,
    )
