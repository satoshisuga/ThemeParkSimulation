from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np
from numpy.typing import NDArray

from themepark.config import DEFAULT_ATTRACTION_SPECS, MAP_HEIGHT, MAP_WIDTH, SimulationConfig
from themepark.models import Attraction, VisitorAgent


MAP_DIAGONAL = hypot(MAP_WIDTH, MAP_HEIGHT)


@dataclass(frozen=True, slots=True)
class DecisionResult:
    attraction_id: int
    perceived_wait_steps: float
    utilities: tuple[float, ...]


def normalized_distance(visitor: VisitorAgent, attraction: Attraction) -> float:
    return hypot(visitor.x - attraction.x, visitor.y - attraction.y) / MAP_DIAGONAL


def perceived_wait_norm(
    visitor: VisitorAgent,
    attraction: Attraction,
    displayed_wait_steps: float,
    config: SimulationConfig,
) -> float:
    if visitor.has_congestion_info:
        return float(np.clip(displayed_wait_steps / config.wait_reference_steps, 0, config.wait_norm_max))
    return float(config.no_info_prior_strength * attraction.popularity)


def calculate_utilities(
    visitor: VisitorAgent,
    attractions: list[Attraction],
    displayed_wait_steps: tuple[float, ...],
    config: SimulationConfig,
    noise_row: NDArray[np.float64],
) -> NDArray[np.float64]:
    utilities = np.full(len(attractions), -np.inf, dtype=np.float64)
    for attraction in attractions:
        if visitor.visited[attraction.id]:
            continue
        distance_norm = normalized_distance(visitor, attraction)
        wait_norm = perceived_wait_norm(
            visitor,
            attraction,
            displayed_wait_steps[attraction.id],
            config,
        )
        utilities[attraction.id] = (
            config.preference_weight * visitor.preferences[attraction.id]
            - visitor.distance_weight * distance_norm
            - visitor.congestion_weight * wait_norm
            + config.choice_noise * noise_row[attraction.id]
        )
    return utilities


def choose_attraction(
    visitor: VisitorAgent,
    attractions: list[Attraction],
    displayed_wait_steps: tuple[float, ...],
    config: SimulationConfig,
    noise_row: NDArray[np.float64],
    tie_rng: np.random.Generator,
) -> DecisionResult | None:
    if visitor.ride_count >= config.rides_to_exit or bool(np.all(visitor.visited)):
        return None
    utilities = calculate_utilities(visitor, attractions, displayed_wait_steps, config, noise_row)
    finite_indices = np.flatnonzero(np.isfinite(utilities))
    if finite_indices.size == 0:
        return None
    best_value = utilities[finite_indices].max()
    tied = finite_indices[np.isclose(utilities[finite_indices], best_value)]
    if tied.size == 1:
        attraction_id = int(tied[0])
    else:
        attraction_id = int(tie_rng.choice(tied))
    attraction = attractions[attraction_id]
    if visitor.has_congestion_info:
        perceived_wait = displayed_wait_steps[attraction_id]
    else:
        perceived_wait = config.no_info_prior_strength * attraction.popularity * config.wait_reference_steps
    return DecisionResult(
        attraction_id=attraction_id,
        perceived_wait_steps=float(perceived_wait),
        utilities=tuple(float(value) for value in utilities),
    )


def default_popularities() -> tuple[float, ...]:
    return tuple(spec.popularity for spec in DEFAULT_ATTRACTION_SPECS)
