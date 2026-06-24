from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


MAP_WIDTH = 100.0
MAP_HEIGHT = 100.0
GATE_X = 50.0
GATE_Y = 100.0


@dataclass(frozen=True, slots=True)
class AttractionSpec:
    id: int
    name: str
    x: float
    y: float
    popularity: float
    capacity: int
    service_duration_steps: int


DEFAULT_ATTRACTION_SPECS: tuple[AttractionSpec, ...] = (
    AttractionSpec(0, "スカイコースター", 12, 15, 1.00, 8, 360),
    AttractionSpec(1, "スプラッシュライド", 50, 10, 0.90, 8, 340),
    AttractionSpec(2, "スペースフライト", 88, 15, 0.85, 10, 320),
    AttractionSpec(3, "ホラーハウス", 12, 46, 0.70, 10, 280),
    AttractionSpec(4, "シアター", 88, 46, 0.55, 16, 360),
    AttractionSpec(5, "カルーセル", 12, 84, 0.45, 12, 200),
    AttractionSpec(6, "観覧車", 88, 84, 0.35, 12, 240),
)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    seed: int = 42
    max_steps: int = 36000
    step_seconds: int = 1

    visitor_count: int = 500
    gate_capacity_per_step: int = 1
    entry_interval_steps: int = 2
    rides_to_exit: int = 5
    movement_speed: float = 0.36
    path_lane_count: int = 3

    information_rate: float = 0.5
    information_update_interval_steps: int = 120
    information_delay_steps: int = 0

    preference_weight: float = 1.0
    preference_diversity: float = 0.45
    distance_weight_mean: float = 0.25
    distance_weight_sd: float = 0.05
    congestion_weight_mean: float = 1.1
    congestion_weight_sd: float = 0.20
    choice_noise: float = 0.05
    no_info_prior_strength: float = 0.0

    wait_reference_steps: float = 1800.0
    wait_norm_max: float = 3.0

    like_reward_weight: float = 1.0
    wait_penalty_weight: float = 0.7
    move_penalty_weight: float = 0.1

    metric_record_interval_steps: int = 60
    recent_oscillation_window_steps: int = 1800

    def validate(self) -> None:
        attraction_count = len(DEFAULT_ATTRACTION_SPECS)
        if self.visitor_count < 1:
            raise ValueError("visitor_count must be at least 1")
        if self.gate_capacity_per_step < 1:
            raise ValueError("gate_capacity_per_step must be at least 1")
        if self.path_lane_count < 1:
            raise ValueError("path_lane_count must be at least 1")
        if not 1 <= self.rides_to_exit <= attraction_count:
            raise ValueError("rides_to_exit must be between 1 and attraction_count")
        if self.movement_speed <= 0:
            raise ValueError("movement_speed must be positive")
        if not 0 <= self.information_rate <= 1:
            raise ValueError("information_rate must be between 0 and 1")
        if self.information_update_interval_steps < 1:
            raise ValueError("information_update_interval_steps must be at least 1")
        if self.information_delay_steps < 0:
            raise ValueError("information_delay_steps must be non-negative")
        if self.entry_interval_steps < 1:
            raise ValueError("entry_interval_steps must be at least 1")
        if self.distance_weight_sd < 0:
            raise ValueError("distance_weight_sd must be non-negative")
        if self.congestion_weight_sd < 0:
            raise ValueError("congestion_weight_sd must be non-negative")
        if self.preference_diversity < 0:
            raise ValueError("preference_diversity must be non-negative")
        if self.choice_noise < 0:
            raise ValueError("choice_noise must be non-negative")
        if self.wait_reference_steps <= 0:
            raise ValueError("wait_reference_steps must be positive")
        if self.wait_norm_max <= 0:
            raise ValueError("wait_norm_max must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.step_seconds < 1:
            raise ValueError("step_seconds must be at least 1")
        if self.metric_record_interval_steps < 1:
            raise ValueError("metric_record_interval_steps must be at least 1")
        if self.recent_oscillation_window_steps < 1:
            raise ValueError("recent_oscillation_window_steps must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **updates: Any) -> "SimulationConfig":
        config = replace(self, **updates)
        config.validate()
        return config

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "SimulationConfig":
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in values.items() if key in allowed})
        config.validate()
        return config
