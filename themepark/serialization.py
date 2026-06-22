from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd

from themepark.config import SimulationConfig

if TYPE_CHECKING:
    from themepark.engine import Simulation


SCHEMA_VERSION = 1


def config_payload(config: SimulationConfig) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, **config.to_dict()}


def config_to_json(config: SimulationConfig) -> str:
    return json.dumps(config_payload(config), ensure_ascii=False, indent=2)


def result_payload(sim: "Simulation") -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": sim.config.to_dict(),
        "finished_reason": sim.finished_reason or "running",
        "completed_at_step": sim.step_count,
        "summary": sim.summary(),
    }


def result_to_json(sim: "Simulation") -> str:
    return json.dumps(result_payload(sim), ensure_ascii=False, indent=2)


def queue_history_dataframe(sim: "Simulation") -> pd.DataFrame:
    columns = [
        "step",
        "attraction_id",
        "attraction_name",
        "queue_length",
        "riders",
        "actual_wait_steps",
        "displayed_wait_steps",
    ]
    if not sim.time_series_records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(sim.time_series_records, columns=columns)


def visitor_dataframe(sim: "Simulation") -> pd.DataFrame:
    rows = [
        {
            "agent_id": visitor.id,
            "has_information": visitor.has_congestion_info,
            "entered_at": visitor.entered_at,
            "exited_at": visitor.exited_at,
            "ride_count": visitor.ride_count,
            "total_wait_steps": visitor.total_wait_steps,
            "total_move_distance": visitor.total_move_distance,
            "satisfaction": visitor.satisfaction,
        }
        for visitor in sim.visitors
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "agent_id",
            "has_information",
            "entered_at",
            "exited_at",
            "ride_count",
            "total_wait_steps",
            "total_move_distance",
            "satisfaction",
        ],
    )


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")
