from __future__ import annotations

from collections import defaultdict, deque
from math import hypot
from typing import Any

import numpy as np

from themepark.config import (
    DEFAULT_ATTRACTION_SPECS,
    GATE_X,
    GATE_Y,
    SimulationConfig,
)
from themepark.congestion import CongestionInfo, actual_estimated_waits
from themepark.decision import MAP_DIAGONAL, choose_attraction
from themepark.metrics import make_snapshot, satisfaction_delta
from themepark.models import AgentState, Attraction, MetricSnapshot, VisitorAgent
from themepark.pathing import DEFAULT_PATH_NETWORK, PathNetwork
from themepark.rng import CommonRandoms, make_common_randoms, make_rng_bundle


class Simulation:
    def __init__(self, config: SimulationConfig) -> None:
        config.validate()
        self.config = config
        self._rngs = make_rng_bundle(config.seed)
        self._common = make_common_randoms(config, len(DEFAULT_ATTRACTION_SPECS))
        self.path_network: PathNetwork = DEFAULT_PATH_NETWORK
        self.attractions = self._build_attractions()
        self.visitors = self._build_visitors(self._common)
        self.congestion = CongestionInfo(config, len(self.attractions))
        self._step = 0
        self.entered_count = 0
        self.completed_rides = 0
        self.total_completed_wait_steps = 0
        self.finished_reason: str | None = None
        self.last_choice_synchronization: float | None = None
        self.metric_history: list[MetricSnapshot] = []
        self.queue_length_history: list[tuple[int, tuple[int, ...]]] = []
        self.time_series_records: list[dict[str, Any]] = []

    @property
    def finished(self) -> bool:
        return self.finished_reason is not None

    @property
    def step_count(self) -> int:
        return self._step

    def step(self) -> None:
        self.step_once()

    def step_many(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        for _ in range(count):
            if self.finished:
                break
            self.step_once()

    def step_once(self) -> None:
        if self.finished:
            return
        self.last_choice_synchronization = None
        self._admit_visitors()
        self._advance_ride_cycles()
        self._finish_completed_rides()
        self._board_waiting_visitors()
        self.congestion.record_actual(self._step, self.attractions)
        self.congestion.update_displayed_if_due(self._step)
        self._choose_destinations_synchronously()
        self._move_visitors()
        if self._step % self.config.metric_record_interval_steps == 0:
            self._record_metrics()
        self._step += 1
        self._update_finished_reason()

    def run_until_finished(self) -> None:
        while not self.finished:
            self.step_once()

    def current_metrics(self) -> MetricSnapshot:
        queue_lengths = tuple(len(attraction.queue) for attraction in self.attractions)
        history = [*self.queue_length_history, (self._step, queue_lengths)]
        return make_snapshot(
            step=self._step,
            visitors=self.visitors,
            attractions=self.attractions,
            config=self.config,
            completed_rides=self.completed_rides,
            total_completed_wait_steps=self.total_completed_wait_steps,
            choice_synchronization=self.last_choice_synchronization,
            queue_length_history=history,
        )

    def summary(self) -> dict[str, Any]:
        snapshot = self.current_metrics()
        mean_queue_imbalance = self._mean_metric("queue_imbalance")
        mean_sync = self._mean_optional_metric("choice_synchronization")
        return {
            "finished_reason": self.finished_reason or "running",
            "completed_at_step": self._step,
            "exited_visitors": snapshot.exited_visitors,
            "completed_rides": self.completed_rides,
            "mean_wait_per_ride_steps": snapshot.mean_wait_per_ride_steps,
            "mean_stay_steps": snapshot.mean_stay_steps,
            "mean_satisfaction": snapshot.mean_satisfaction,
            "mean_satisfaction_info": snapshot.mean_satisfaction_info,
            "mean_satisfaction_no_info": snapshot.mean_satisfaction_no_info,
            "mean_queue_imbalance": mean_queue_imbalance,
            "mean_choice_synchronization": mean_sync,
            "oscillation_amplitude": snapshot.oscillation_amplitude,
        }

    def formatted_time(self) -> str:
        total_seconds = self._step * self.config.step_seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def queue_history_dataframe(self):
        from themepark.serialization import queue_history_dataframe

        return queue_history_dataframe(self)

    def visitor_dataframe(self):
        from themepark.serialization import visitor_dataframe

        return visitor_dataframe(self)

    def _build_attractions(self) -> list[Attraction]:
        return [
            Attraction(
                id=spec.id,
                name=spec.name,
                x=spec.x,
                y=spec.y,
                popularity=spec.popularity,
                capacity=spec.capacity,
                service_duration_steps=spec.service_duration_steps,
            )
            for spec in DEFAULT_ATTRACTION_SPECS
        ]

    def _build_visitors(self, common: CommonRandoms) -> list[VisitorAgent]:
        popularities = np.asarray(
            [spec.popularity for spec in DEFAULT_ATTRACTION_SPECS],
            dtype=np.float64,
        )
        visitors: list[VisitorAgent] = []
        for visitor_id in range(self.config.visitor_count):
            preferences = np.clip(
                popularities + common.preference_base[visitor_id] * self.config.preference_diversity,
                0.0,
                1.0,
            ).astype(np.float64)
            distance_weight = max(
                0.0,
                self.config.distance_weight_mean
                + self.config.distance_weight_sd * common.z_distance[visitor_id],
            )
            congestion_weight = max(
                0.0,
                self.config.congestion_weight_mean
                + self.config.congestion_weight_sd * common.z_congestion[visitor_id],
            )
            visitors.append(
                VisitorAgent(
                    id=visitor_id,
                    state=AgentState.NOT_ENTERED,
                    x=GATE_X,
                    y=GATE_Y,
                    preferences=preferences,
                    distance_weight=float(distance_weight),
                    congestion_weight=float(congestion_weight),
                    has_congestion_info=bool(
                        common.info_draw[visitor_id] < self.config.information_rate
                    ),
                    visited=np.zeros(len(DEFAULT_ATTRACTION_SPECS), dtype=np.bool_),
                )
            )
        return visitors

    def _admit_visitors(self) -> None:
        if self.entered_count >= self.config.visitor_count:
            return
        if self._step % self.config.entry_interval_steps == 0:
            admit_count = self.config.gate_capacity_per_step
        else:
            admit_count = 0
        admit_count = min(admit_count, self.config.visitor_count - self.entered_count)
        for _ in range(admit_count):
            visitor = self.visitors[self.entered_count]
            visitor.state = AgentState.CHOOSING
            visitor.x = GATE_X
            visitor.y = GATE_Y
            visitor.route.clear()
            visitor.route_index = 0
            visitor.current_segment_id = None
            visitor.path_lane_index = 0
            visitor.entered_at = self._step
            self.entered_count += 1

    def _advance_ride_cycles(self) -> None:
        for attraction in self.attractions:
            if attraction.riders and attraction.cycle_remaining_steps > 0:
                attraction.cycle_remaining_steps -= 1

    def _finish_completed_rides(self) -> None:
        for attraction in self.attractions:
            if not attraction.riders or attraction.cycle_remaining_steps > 0:
                continue
            riders = list(attraction.riders)
            attraction.riders.clear()
            for visitor_id in riders:
                visitor = self.visitors[visitor_id]
                visitor.visited[attraction.id] = True
                visitor.ride_count += 1
                visitor.satisfaction += satisfaction_delta(
                    visitor,
                    attraction.id,
                    self.config,
                    MAP_DIAGONAL,
                )
                visitor.state = AgentState.CHOOSING
                visitor.target_attraction_id = None
                visitor.route.clear()
                visitor.route_index = 0
                visitor.current_segment_id = None
                visitor.path_lane_index = 0
                visitor.queue_entered_at = None
                visitor.current_trip_distance = 0.0

    def _board_waiting_visitors(self) -> None:
        for attraction in self.attractions:
            if attraction.riders or not attraction.queue:
                continue
            boarding_count = min(attraction.capacity, len(attraction.queue))
            for _ in range(boarding_count):
                visitor_id = attraction.queue.popleft()
                visitor = self.visitors[visitor_id]
                wait_steps = (
                    self._step - visitor.queue_entered_at
                    if visitor.queue_entered_at is not None
                    else 0
                )
                visitor.last_queue_wait_steps = int(wait_steps)
                visitor.total_wait_steps += int(wait_steps)
                visitor.state = AgentState.RIDING
                attraction.riders.append(visitor_id)
                self.completed_rides += 1
                self.total_completed_wait_steps += int(wait_steps)
            if attraction.riders:
                attraction.cycle_remaining_steps = attraction.service_duration_steps

    def _choose_destinations_synchronously(self) -> None:
        choosing = [
            visitor
            for visitor in self.visitors
            if visitor.state == AgentState.CHOOSING
        ]
        if not choosing:
            return
        decisions: dict[int, tuple[int | None, float]] = {}
        choice_counts: dict[int, int] = defaultdict(int)
        for visitor in choosing:
            if visitor.ride_count >= self.config.rides_to_exit or bool(np.all(visitor.visited)):
                decisions[visitor.id] = (None, 0.0)
                continue
            decision_index = min(visitor.ride_count, self._common.base_choice_noise.shape[1] - 1)
            result = choose_attraction(
                visitor=visitor,
                attractions=self.attractions,
                displayed_wait_steps=self.congestion.displayed_wait_steps,
                config=self.config,
                noise_row=self._common.base_choice_noise[visitor.id, decision_index],
                tie_rng=self._rngs.tie_break,
            )
            if result is None:
                decisions[visitor.id] = (None, 0.0)
            else:
                decisions[visitor.id] = (result.attraction_id, result.perceived_wait_steps)
                choice_counts[result.attraction_id] += 1

        chosen_count = sum(choice_counts.values())
        if chosen_count >= 5:
            self.last_choice_synchronization = max(choice_counts.values()) / chosen_count
        else:
            self.last_choice_synchronization = None

        for visitor in choosing:
            target_id, perceived_wait = decisions[visitor.id]
            visitor.target_chosen_at = self._step
            visitor.perceived_wait_steps_at_choice = perceived_wait
            visitor.current_trip_distance = 0.0
            if target_id is None:
                visitor.state = AgentState.EXITING
                visitor.target_attraction_id = None
                visitor.route = self.path_network.route_to_gate((visitor.x, visitor.y))
                visitor.route_index = 0
                visitor.current_segment_id = None
                visitor.path_lane_index = 0
            else:
                visitor.state = AgentState.MOVING
                visitor.target_attraction_id = target_id
                visitor.route = self.path_network.route_to_attraction(
                    (visitor.x, visitor.y),
                    target_id,
                )
                visitor.route_index = 0
                visitor.current_segment_id = None
                visitor.path_lane_index = 0

    def _move_visitors(self) -> None:
        arrivals: dict[int, list[int]] = defaultdict(list)
        segment_entries: dict[str, int] = defaultdict(int)
        for visitor in self.visitors:
            if visitor.state == AgentState.MOVING:
                if visitor.target_attraction_id is None:
                    continue
                arrived = self._move_along_route(visitor, segment_entries)
                if arrived:
                    arrivals[visitor.target_attraction_id].append(visitor.id)
            elif visitor.state == AgentState.EXITING:
                arrived = self._move_along_route(visitor, segment_entries)
                if arrived:
                    visitor.state = AgentState.EXITED
                    visitor.exited_at = self._step
                    visitor.target_attraction_id = None
                    visitor.route.clear()
                    visitor.route_index = 0
                    visitor.current_segment_id = None
                    visitor.path_lane_index = 0

        for attraction_id, visitor_ids in arrivals.items():
            ordered_ids = visitor_ids
            if len(visitor_ids) > 1:
                ordered_ids = [int(value) for value in self._rngs.queue.permutation(visitor_ids)]
            attraction = self.attractions[attraction_id]
            for visitor_id in ordered_ids:
                visitor = self.visitors[visitor_id]
                visitor.state = AgentState.WAITING
                visitor.queue_entered_at = self._step
                visitor.route.clear()
                visitor.route_index = 0
                visitor.current_segment_id = None
                visitor.path_lane_index = 0
                attraction.queue.append(visitor_id)

    def _move_along_route(
        self,
        visitor: VisitorAgent,
        segment_entries: dict[str, int],
    ) -> bool:
        remaining = self.config.movement_speed
        while remaining > 1e-9:
            if visitor.route_index >= len(visitor.route):
                return True
            route_step = visitor.route[visitor.route_index]
            if not self._enter_route_segment(visitor, route_step.segment_id, segment_entries):
                return False
            distance = self._move_toward(visitor, route_step.x, route_step.y, remaining)
            remaining -= distance
            if hypot(visitor.x - route_step.x, visitor.y - route_step.y) <= 1e-9:
                visitor.route_index += 1
                visitor.current_segment_id = None
            else:
                return False
        return visitor.route_index >= len(visitor.route)

    def _enter_route_segment(
        self,
        visitor: VisitorAgent,
        segment_id: str | None,
        segment_entries: dict[str, int],
    ) -> bool:
        if segment_id is None or visitor.current_segment_id == segment_id:
            return True
        entered_count = segment_entries[segment_id]
        if entered_count >= self.config.path_lane_count:
            return False
        segment_entries[segment_id] = entered_count + 1
        visitor.current_segment_id = segment_id
        visitor.path_lane_index = entered_count
        return True

    def _move_toward(
        self,
        visitor: VisitorAgent,
        target_x: float,
        target_y: float,
        max_distance: float,
    ) -> float:
        dx = target_x - visitor.x
        dy = target_y - visitor.y
        distance = hypot(dx, dy)
        if distance <= 1e-12:
            return 0.0
        if distance <= max_distance:
            move_distance = distance
            visitor.x = target_x
            visitor.y = target_y
        else:
            move_distance = max_distance
            visitor.x += max_distance * dx / distance
            visitor.y += max_distance * dy / distance
        visitor.total_move_distance += move_distance
        visitor.current_trip_distance += move_distance
        return move_distance

    def _record_metrics(self) -> None:
        queue_lengths = tuple(len(attraction.queue) for attraction in self.attractions)
        self.queue_length_history.append((self._step, queue_lengths))
        actual_waits = actual_estimated_waits(self.attractions)
        displayed_waits = self.congestion.displayed_wait_steps
        for attraction in self.attractions:
            self.time_series_records.append(
                {
                    "step": self._step,
                    "attraction_id": attraction.id,
                    "attraction_name": attraction.name,
                    "queue_length": len(attraction.queue),
                    "riders": len(attraction.riders),
                    "actual_wait_steps": actual_waits[attraction.id],
                    "displayed_wait_steps": displayed_waits[attraction.id],
                }
            )
        self.metric_history.append(self.current_metrics())

    def _update_finished_reason(self) -> None:
        if all(visitor.state == AgentState.EXITED for visitor in self.visitors):
            self.finished_reason = "all_visitors_exited"
        elif self._step >= self.config.max_steps:
            self.finished_reason = "max_steps_reached"

    def _mean_metric(self, field_name: str) -> float | None:
        values = [getattr(snapshot, field_name) for snapshot in self.metric_history]
        if not values:
            return None
        return float(sum(values) / len(values))

    def _mean_optional_metric(self, field_name: str) -> float | None:
        values = [
            getattr(snapshot, field_name)
            for snapshot in self.metric_history
            if getattr(snapshot, field_name) is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))
