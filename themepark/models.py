from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from themepark.pathing import RouteStep


class AgentState(StrEnum):
    NOT_ENTERED = "not_entered"
    CHOOSING = "choosing"
    MOVING = "moving"
    WAITING = "waiting"
    RIDING = "riding"
    EXITING = "exiting"
    EXITED = "exited"


@dataclass(slots=True)
class VisitorAgent:
    id: int
    state: AgentState
    x: float
    y: float

    preferences: NDArray[np.float64]
    distance_weight: float
    congestion_weight: float
    has_congestion_info: bool

    visited: NDArray[np.bool_]
    ride_count: int = 0
    target_attraction_id: int | None = None
    route: list["RouteStep"] = field(default_factory=list)
    route_index: int = 0
    current_segment_id: str | None = None
    path_lane_index: int = 0

    entered_at: int | None = None
    exited_at: int | None = None
    target_chosen_at: int | None = None
    queue_entered_at: int | None = None

    perceived_wait_steps_at_choice: float = 0.0
    current_trip_distance: float = 0.0
    last_queue_wait_steps: int = 0

    total_wait_steps: int = 0
    total_move_distance: float = 0.0
    satisfaction: float = 0.0


@dataclass(slots=True)
class Attraction:
    id: int
    name: str
    x: float
    y: float
    popularity: float
    capacity: int
    service_duration_steps: int

    queue: deque[int] = field(default_factory=deque)
    riders: list[int] = field(default_factory=list)
    cycle_remaining_steps: int = 0


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    step: int
    active_visitors: int
    exited_visitors: int
    completed_rides: int
    mean_wait_per_ride_steps: float | None
    mean_stay_steps: float | None
    mean_satisfaction: float | None
    mean_satisfaction_info: float | None
    mean_satisfaction_no_info: float | None
    queue_imbalance: float
    choice_synchronization: float | None
    oscillation_amplitude: float | None
    queue_lengths: tuple[int, ...]
