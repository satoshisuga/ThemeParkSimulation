from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from themepark.config import SimulationConfig


FIXED_CHOICE_NOISE = 0.05

INFORMATION_FRESHNESS_OPTIONS: dict[str, dict[str, int | str]] = {
    "realtime": {
        "label": "リアルタイム",
        "information_update_interval_steps": 1,
        "information_delay_steps": 0,
    },
    "slightly_old": {
        "label": "少し古い",
        "information_update_interval_steps": 120,
        "information_delay_steps": 60,
    },
    "old": {
        "label": "かなり古い",
        "information_update_interval_steps": 300,
        "information_delay_steps": 120,
    },
}


def freshness_key_for(update_interval_steps: int, delay_steps: int) -> str:
    for key, option in INFORMATION_FRESHNESS_OPTIONS.items():
        if (
            option["information_update_interval_steps"] == update_interval_steps
            and option["information_delay_steps"] == delay_steps
        ):
            return key
    return "slightly_old"


def freshness_updates(freshness_key: str) -> dict[str, int]:
    option = INFORMATION_FRESHNESS_OPTIONS.get(
        freshness_key,
        INFORMATION_FRESHNESS_OPTIONS["slightly_old"],
    )
    return {
        "information_update_interval_steps": int(option["information_update_interval_steps"]),
        "information_delay_steps": int(option["information_delay_steps"]),
    }


@dataclass(frozen=True, slots=True)
class Preset:
    key: str
    label: str
    updates: dict[str, Any]
    observation: str


PRESETS: dict[str, Preset] = {
    "A": Preset(
        key="A",
        label="A: 情報なし・人気集中",
        updates={
            "seed": 42,
            "information_rate": 0.0,
            "information_update_interval_steps": 120,
            "information_delay_steps": 60,
            "congestion_weight_mean": 0.0,
            "preference_diversity": 0.45,
            "choice_noise": FIXED_CHOICE_NOISE,
        },
        observation="人気上位施設に行列が集まりやすく、行列の偏りが大きくなります。",
    ),
    "B": Preset(
        key="B",
        label="B: ほどよい情報共有",
        updates={
            "seed": 42,
            "information_rate": 0.5,
            "information_update_interval_steps": 120,
            "information_delay_steps": 60,
            "congestion_weight_mean": 1.1,
            "preference_diversity": 0.45,
            "choice_noise": FIXED_CHOICE_NOISE,
        },
        observation="混雑情報を使う人が一部にいることで、行列が分散する場合があります。",
    ),
    "C": Preset(
        key="C",
        label="C: 全員が同じ古い情報を見る",
        updates={
            "seed": 42,
            "information_rate": 1.0,
            "information_update_interval_steps": 300,
            "information_delay_steps": 120,
            "congestion_weight_mean": 1.8,
            "preference_diversity": 0.15,
            "choice_noise": FIXED_CHOICE_NOISE,
        },
        observation="空いていると表示された施設へ集団で向かい、混雑先が入れ替わりやすくなります。",
    ),
    "D": Preset(
        key="D",
        label="D: 多様性が一斉行動を崩す",
        updates={
            "seed": 42,
            "information_rate": 1.0,
            "information_update_interval_steps": 300,
            "information_delay_steps": 120,
            "congestion_weight_mean": 1.8,
            "preference_diversity": 0.65,
            "choice_noise": FIXED_CHOICE_NOISE,
        },
        observation="好みや混雑回避度に差が出ることで、一斉選択が弱まる場合があります。",
    ),
    "E": Preset(
        key="E",
        label="E: 効率と満足度は同じか",
        updates={
            "seed": 42,
            "information_update_interval_steps": 120,
            "information_delay_steps": 60,
            "congestion_weight_mean": 1.2,
            "preference_diversity": 0.35,
            "choice_noise": FIXED_CHOICE_NOISE,
            "like_reward_weight": 1.2,
            "wait_penalty_weight": 0.55,
            "move_penalty_weight": 0.1,
        },
        observation="情報所持率を変えたとき、待ち時間と満足度の最良条件がずれるかを見ます。",
    ),
    "F": Preset(
        key="F",
        label="F: 情報の新しさ",
        updates={
            "seed": 42,
            "information_rate": 1.0,
            "information_update_interval_steps": 120,
            "information_delay_steps": 60,
            "congestion_weight_mean": 1.3,
            "preference_diversity": 0.35,
            "choice_noise": FIXED_CHOICE_NOISE,
        },
        observation="情報の新しさ、更新頻度、遅延が集団行動をどう変えるかを比べます。",
    ),
}


def apply_preset(config: SimulationConfig, preset_key: str) -> SimulationConfig:
    preset = PRESETS[preset_key]
    updates = {**preset.updates, "choice_noise": FIXED_CHOICE_NOISE}
    return config.with_updates(**updates)


INFO_RATE_SWEEP = tuple(round(value / 10, 1) for value in range(11))
COMPARISON_SEEDS = (11, 22, 33, 44, 55)
