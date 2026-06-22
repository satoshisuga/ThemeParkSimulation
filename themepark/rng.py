from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from themepark.config import DEFAULT_ATTRACTION_SPECS, SimulationConfig


@dataclass(frozen=True, slots=True)
class CommonRandoms:
    preference_base: NDArray[np.float64]
    z_distance: NDArray[np.float64]
    z_congestion: NDArray[np.float64]
    info_draw: NDArray[np.float64]
    base_choice_noise: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RngBundle:
    population: np.random.Generator
    runtime: np.random.Generator
    queue: np.random.Generator
    tie_break: np.random.Generator


def make_rng_bundle(seed: int) -> RngBundle:
    seed_sequence = np.random.SeedSequence(seed)
    population_ss, runtime_ss, queue_ss, tie_ss = seed_sequence.spawn(4)
    return RngBundle(
        population=np.random.default_rng(population_ss),
        runtime=np.random.default_rng(runtime_ss),
        queue=np.random.default_rng(queue_ss),
        tie_break=np.random.default_rng(tie_ss),
    )


def make_common_randoms(
    config: SimulationConfig,
    attraction_count: int = len(DEFAULT_ATTRACTION_SPECS),
) -> CommonRandoms:
    rng = make_rng_bundle(config.seed).population
    max_decisions = max(1, config.rides_to_exit + 1)
    return CommonRandoms(
        preference_base=rng.uniform(
            low=-0.5,
            high=0.5,
            size=(config.visitor_count, attraction_count),
        ).astype(np.float64),
        z_distance=rng.standard_normal(config.visitor_count).astype(np.float64),
        z_congestion=rng.standard_normal(config.visitor_count).astype(np.float64),
        info_draw=rng.random(config.visitor_count).astype(np.float64),
        base_choice_noise=rng.uniform(
            low=-1.0,
            high=1.0,
            size=(config.visitor_count, max_decisions, attraction_count),
        ).astype(np.float64),
    )
