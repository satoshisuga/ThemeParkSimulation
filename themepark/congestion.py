from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from themepark.config import SimulationConfig
from themepark.models import Attraction


def actual_estimated_wait_steps(attraction: Attraction) -> float:
    current_cycle_wait = attraction.cycle_remaining_steps if attraction.riders else 0
    full_cycles_ahead = len(attraction.queue) // max(attraction.capacity, 1)
    return float(current_cycle_wait + full_cycles_ahead * attraction.service_duration_steps)


def actual_estimated_waits(attractions: list[Attraction]) -> tuple[float, ...]:
    return tuple(actual_estimated_wait_steps(attraction) for attraction in attractions)


def wait_steps_to_minutes(wait_steps: float | None, step_seconds: int) -> int | None:
    if wait_steps is None:
        return None
    return int(ceil(wait_steps * step_seconds / 60))


@dataclass(slots=True)
class CongestionInfo:
    config: SimulationConfig
    attraction_count: int
    displayed_wait_steps: tuple[float, ...] = field(init=False)
    history: list[tuple[int, tuple[float, ...]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.displayed_wait_steps = tuple(0.0 for _ in range(self.attraction_count))

    def record_actual(self, step: int, attractions: list[Attraction]) -> tuple[float, ...]:
        waits = actual_estimated_waits(attractions)
        self.history.append((step, waits))
        return waits

    def update_displayed_if_due(self, step: int) -> None:
        if step % self.config.information_update_interval_steps != 0:
            return
        if not self.history:
            return
        source_step = max(0, step - self.config.information_delay_steps)
        self.displayed_wait_steps = self.waits_at_or_before(source_step)

    def waits_at_or_before(self, source_step: int) -> tuple[float, ...]:
        for recorded_step, waits in reversed(self.history):
            if recorded_step <= source_step:
                return waits
        return self.history[0][1]
