from __future__ import annotations

import json
from math import cos, pi, sin
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from themepark.config import GATE_X, GATE_Y, MAP_HEIGHT, MAP_WIDTH
from themepark.congestion import actual_estimated_waits, wait_steps_to_minutes
from themepark.models import AgentState
from themepark.pathing import attraction_entrances_payload, path_edges_payload

if TYPE_CHECKING:
    from themepark.engine import Simulation


STATE_COLORS = {
    AgentState.CHOOSING: "#5b5f97",
    AgentState.MOVING: "#00a6a6",
    AgentState.WAITING: "#f28f3b",
    AgentState.RIDING: "#7cb518",
    AgentState.EXITING: "#c44536",
}

LINE_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#0f766e",
)


def format_optional(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_wait_minutes(wait_steps: float | None, step_seconds: int = 10) -> str:
    minutes = wait_steps_to_minutes(wait_steps, step_seconds)
    if minutes is None:
        return "-"
    return f"{minutes}分"


def create_park_canvas_html(sim: "Simulation", max_visitors: int = 1200) -> str:
    actual_waits = actual_estimated_waits(sim.attractions)
    displayed_waits = sim.congestion.displayed_wait_steps
    queue_positions = _queue_positions(sim)
    rider_positions = _rider_positions(sim)
    visible_visitors = _visible_visitors(sim, max_visitors)
    attractions = [
        {
            "id": attraction.id,
            "name": attraction.name,
            "x": attraction.x,
            "y": attraction.y,
            "popularity": attraction.popularity,
            "queue": len(attraction.queue),
            "riders": len(attraction.riders),
            "actualWait": format_wait_minutes(
                actual_waits[attraction.id],
                sim.config.step_seconds,
            ),
            "displayedWait": format_wait_minutes(
                displayed_waits[attraction.id],
                sim.config.step_seconds,
            ),
        }
        for attraction in sim.attractions
    ]
    visitors = []
    for visitor in visible_visitors:
        x, y = visitor.x, visitor.y
        if visitor.state == AgentState.WAITING:
            x, y = queue_positions.get(visitor.id, (x, y))
        elif visitor.state == AgentState.RIDING:
            x, y = rider_positions.get(visitor.id, (x, y))
        target = "-"
        if visitor.target_attraction_id is not None:
            target = sim.attractions[visitor.target_attraction_id].name
        visitors.append(
            {
                "id": visitor.id,
                "state": visitor.state.value,
                "stateLabel": _state_label(visitor.state),
                "x": x,
                "y": y,
                "color": STATE_COLORS.get(visitor.state, "#64748b"),
                "hasInfo": visitor.has_congestion_info,
                "rideCount": visitor.ride_count,
                "satisfaction": round(visitor.satisfaction, 2),
                "target": target,
            }
        )
    payload = {
        "mapWidth": MAP_WIDTH,
        "mapHeight": MAP_HEIGHT,
        "paths": path_edges_payload(),
        "entrances": attraction_entrances_payload(),
        "gate": {"x": GATE_X, "y": GATE_Y},
        "attractions": attractions,
        "visitors": visitors,
        "stateLegend": [
            {"label": _state_label(state), "color": color}
            for state, color in STATE_COLORS.items()
        ],
        "shownVisitors": len(visitors),
        "totalVisitors": len(
            [
                visitor
                for visitor in sim.visitors
                if visitor.state not in {AgentState.NOT_ENTERED, AgentState.EXITED}
            ]
        ),
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _CANVAS_TEMPLATE.replace("__PARK_DATA__", data_json)


def create_park_figure(sim: "Simulation", max_visitors: int = 1200) -> go.Figure:
    fig = go.Figure()
    actual_waits = actual_estimated_waits(sim.attractions)
    displayed_waits = sim.congestion.displayed_wait_steps
    queue_positions = _queue_positions(sim)
    rider_positions = _rider_positions(sim)
    visible_visitors = _visible_visitors(sim, max_visitors)
    visitor_trace = go.Scatter if len(visible_visitors) <= 1000 else go.Scattergl

    for state in STATE_COLORS:
        xs: list[float] = []
        ys: list[float] = []
        hover: list[str] = []
        for visitor in visible_visitors:
            if visitor.state != state:
                continue
            x, y = visitor.x, visitor.y
            if visitor.state == AgentState.WAITING:
                x, y = queue_positions.get(visitor.id, (x, y))
            elif visitor.state == AgentState.RIDING:
                x, y = rider_positions.get(visitor.id, (x, y))
            target = "-"
            if visitor.target_attraction_id is not None:
                target = sim.attractions[visitor.target_attraction_id].name
            xs.append(x)
            ys.append(y)
            hover.append(
                "<br>".join(
                    [
                        f"ID: {visitor.id}",
                        f"状態: {visitor.state.value}",
                        f"情報: {'あり' if visitor.has_congestion_info else 'なし'}",
                        f"体験数: {visitor.ride_count}",
                        f"満足度: {visitor.satisfaction:.2f}",
                        f"目的地: {target}",
                    ]
                )
            )
        if xs:
            fig.add_trace(
                visitor_trace(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": STATE_COLORS[state],
                        "opacity": 0.72,
                    },
                    name=_state_label(state),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=[attraction.x for attraction in sim.attractions],
            y=[attraction.y for attraction in sim.attractions],
            mode="markers+text",
            marker={
                "size": [
                    16 + 10 * attraction.popularity for attraction in sim.attractions
                ],
                "color": [len(attraction.queue) for attraction in sim.attractions],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "行列"},
                "line": {"color": "#111827", "width": 1},
            },
            text=[f"{attraction.id}: {attraction.name}" for attraction in sim.attractions],
            textposition="top center",
            name="アトラクション",
            customdata=[
                [
                    attraction.name,
                    attraction.popularity,
                    len(attraction.queue),
                    len(attraction.riders),
                    format_wait_minutes(actual_waits[attraction.id], sim.config.step_seconds),
                    format_wait_minutes(displayed_waits[attraction.id], sim.config.step_seconds),
                ]
                for attraction in sim.attractions
            ],
            hovertemplate=(
                "%{customdata[0]}<br>"
                "人気度: %{customdata[1]:.2f}<br>"
                "行列: %{customdata[2]}人<br>"
                "搭乗中: %{customdata[3]}人<br>"
                "実推定待ち: %{customdata[4]}<br>"
                "表示待ち: %{customdata[5]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[GATE_X],
            y=[GATE_Y],
            mode="markers+text",
            marker={"size": 18, "color": "#111827", "symbol": "diamond"},
            text=["ゲート"],
            textposition="top center",
            name="ゲート",
            hovertemplate="ゲート<extra></extra>",
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[0, MAP_WIDTH], fixedrange=False, zeroline=False)
    fig.update_yaxes(range=[MAP_HEIGHT, 0], scaleanchor="x", scaleratio=1, zeroline=False)
    return fig


def create_queue_history_figure(sim: "Simulation", mode: str = "top3") -> go.Figure:
    frame = sim.queue_history_dataframe()
    fig = go.Figure()
    if frame.empty:
        fig.update_layout(
            title="施設別行列の時系列",
            xaxis_title="step",
            yaxis_title="行列人数",
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        return fig
    selected_ids = _selected_attraction_ids(sim, frame, mode)
    display_frame = _thin_history(frame[frame["attraction_id"].isin(selected_ids)])
    for index, attraction_id in enumerate(selected_ids):
        attraction = sim.attractions[attraction_id]
        part = display_frame[display_frame["attraction_id"] == attraction_id]
        fig.add_trace(
            go.Scatter(
                x=part["step"],
                y=part["queue_length"],
                mode="lines",
                line={"color": LINE_COLORS[index % len(LINE_COLORS)], "width": 2},
                name=f"{attraction.id}: {attraction.name}",
                hovertemplate="step=%{x}<br>行列=%{y}人<extra></extra>",
            )
        )
    fig.update_layout(
        title="施設別行列の時系列",
        xaxis_title="step",
        yaxis_title="行列人数",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def create_comparison_figure(frame: pd.DataFrame, y_column: str, title: str) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    grouped = (
        frame.groupby("information_rate", as_index=False)[y_column]
        .mean(numeric_only=True)
        .sort_values("information_rate")
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["information_rate"],
            y=grouped[y_column],
            mode="lines+markers",
            line={"color": "#2563eb", "width": 3},
            marker={"size": 8},
            hovertemplate="情報所持率=%{x:.1f}<br>値=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="情報所持率",
        yaxis_title=title,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def _visible_visitors(sim: "Simulation", max_visitors: int):
    visitors = [
        visitor
        for visitor in sim.visitors
        if visitor.state not in {AgentState.NOT_ENTERED, AgentState.EXITED}
    ]
    if len(visitors) <= max_visitors:
        return visitors
    stride = int(np.ceil(len(visitors) / max_visitors))
    return visitors[::stride]


def _queue_positions(sim: "Simulation") -> dict[int, tuple[float, float]]:
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


def _rider_positions(sim: "Simulation") -> dict[int, tuple[float, float]]:
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


def _selected_attraction_ids(sim: "Simulation", frame: pd.DataFrame, mode: str) -> list[int]:
    if mode == "all":
        return [attraction.id for attraction in sim.attractions]
    if mode == "popular3":
        return [
            attraction.id
            for attraction in sorted(sim.attractions, key=lambda item: item.popularity, reverse=True)[:3]
        ]
    latest = frame[frame["step"] == frame["step"].max()]
    return [
        int(value)
        for value in latest.sort_values("queue_length", ascending=False)["attraction_id"].head(3)
    ]


def _thin_history(frame: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    steps = sorted(frame["step"].unique())
    stride = int(np.ceil(len(steps) / max_points))
    keep_steps = set(steps[::stride])
    return frame[frame["step"].isin(keep_steps)]


def _state_label(state: AgentState) -> str:
    labels = {
        AgentState.CHOOSING: "選択中",
        AgentState.MOVING: "移動中",
        AgentState.WAITING: "待機中",
        AgentState.RIDING: "搭乗中",
        AgentState.EXITING: "退場移動",
    }
    return labels.get(state, state.value)


_CANVAS_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .wrap {
      position: relative;
      width: 100%;
      height: 560px;
      border: 1px solid #e5e7eb;
      background: #f8fafc;
      box-sizing: border-box;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    .tooltip {
      position: absolute;
      display: none;
      max-width: 260px;
      padding: 8px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.96);
      color: #0f172a;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
      pointer-events: none;
      white-space: nowrap;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <canvas id="park"></canvas>
    <div id="tooltip" class="tooltip"></div>
  </div>
  <script id="park-data" type="application/json">__PARK_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("park-data").textContent);
    const canvas = document.getElementById("park");
    const tooltip = document.getElementById("tooltip");
    const ctx = canvas.getContext("2d");
    const hoverTargets = [];

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw(rect.width, rect.height);
    }

    function toScreen(point, width, height) {
      const pad = 42;
      const plotW = width - pad * 2;
      const plotH = height - pad * 2;
      return {
        x: pad + (point.x / data.mapWidth) * plotW,
        y: pad + (point.y / data.mapHeight) * plotH,
      };
    }

    function draw(width, height) {
      hoverTargets.length = 0;
      ctx.clearRect(0, 0, width, height);
      drawBackground(width, height);
      drawPaths(width, height);
      drawAttractions(width, height);
      drawGate(width, height);
      drawVisitors(width, height);
      drawLegend(width, height);
    }

    function drawBackground(width, height) {
      const pad = 42;
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#dbe4ef";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, pad, width - pad * 2, height - pad * 2);
      ctx.strokeStyle = "#e5e7eb";
      for (let i = 1; i < 5; i += 1) {
        const x = pad + (width - pad * 2) * i / 5;
        const y = pad + (height - pad * 2) * i / 5;
        ctx.beginPath();
        ctx.moveTo(x, pad);
        ctx.lineTo(x, height - pad);
        ctx.moveTo(pad, y);
        ctx.lineTo(width - pad, y);
        ctx.stroke();
      }
      ctx.fillStyle = "#334155";
      ctx.font = "600 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillText("パーク地図", 14, 24);
      ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = "#64748b";
      ctx.fillText(`表示中 ${data.shownVisitors} / 園内 ${data.totalVisitors} 人`, 92, 24);
    }

    function drawPaths(width, height) {
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = 15;
      ctx.strokeStyle = "#d7c9aa";
      for (const edge of data.paths) {
        const start = toScreen(edge.from, width, height);
        const end = toScreen(edge.to, width, height);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#b9a476";
      for (const edge of data.paths) {
        const start = toScreen(edge.from, width, height);
        const end = toScreen(edge.to, width, height);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    function drawAttractions(width, height) {
      const maxQueue = Math.max(1, ...data.attractions.map((item) => item.queue));
      for (const attraction of data.attractions) {
        const point = toScreen(attraction, width, height);
        const radius = 8 + attraction.popularity * 6;
        const queueRatio = attraction.queue / maxQueue;
        const hue = 205 - queueRatio * 155;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = "#0f172a";
        ctx.stroke();

        ctx.fillStyle = "#0f172a";
        ctx.font = "600 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`${attraction.id}: ${attraction.name}`, point.x, point.y - radius - 7);
        ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        ctx.fillStyle = "#475569";
        ctx.fillText(`待ち ${attraction.queue}`, point.x, point.y + radius + 13);

        hoverTargets.push({
          kind: "attraction",
          x: point.x,
          y: point.y,
          r: radius + 7,
          html: [
            `<strong>${attraction.name}</strong>`,
            `人気度: ${attraction.popularity.toFixed(2)}`,
            `行列: ${attraction.queue}人`,
            `搭乗中: ${attraction.riders}人`,
            `実推定待ち: ${attraction.actualWait}`,
            `表示待ち: ${attraction.displayedWait}`,
          ].join("<br>")
        });
      }
      ctx.textAlign = "start";
    }

    function drawGate(width, height) {
      const point = toScreen(data.gate, width, height);
      ctx.save();
      ctx.translate(point.x, point.y);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = "#111827";
      ctx.fillRect(-8, -8, 16, 16);
      ctx.restore();
      ctx.fillStyle = "#111827";
      ctx.font = "600 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("ゲート", point.x, point.y - 15);
      ctx.textAlign = "start";
    }

    function drawVisitors(width, height) {
      for (const visitor of data.visitors) {
        const point = toScreen(visitor, width, height);
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = visitor.color;
        ctx.globalAlpha = 0.74;
        ctx.fill();
        ctx.globalAlpha = 1;
        hoverTargets.push({
          kind: "visitor",
          x: point.x,
          y: point.y,
          r: 7,
          html: [
            `<strong>ID: ${visitor.id}</strong>`,
            `状態: ${visitor.stateLabel}`,
            `情報: ${visitor.hasInfo ? "あり" : "なし"}`,
            `体験数: ${visitor.rideCount}`,
            `満足度: ${visitor.satisfaction}`,
            `目的地: ${visitor.target}`,
          ].join("<br>")
        });
      }
    }

    function drawLegend(width, height) {
      let x = 14;
      const y = height - 18;
      ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      for (const item of data.stateLegend) {
        ctx.beginPath();
        ctx.arc(x + 5, y - 4, 5, 0, Math.PI * 2);
        ctx.fillStyle = item.color;
        ctx.fill();
        ctx.fillStyle = "#334155";
        ctx.fillText(item.label, x + 16, y);
        x += ctx.measureText(item.label).width + 44;
      }
    }

    canvas.addEventListener("mousemove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let found = null;
      for (let i = hoverTargets.length - 1; i >= 0; i -= 1) {
        const target = hoverTargets[i];
        const dx = x - target.x;
        const dy = y - target.y;
        if (dx * dx + dy * dy <= target.r * target.r) {
          found = target;
          break;
        }
      }
      if (!found) {
        tooltip.style.display = "none";
        return;
      }
      tooltip.innerHTML = found.html;
      tooltip.style.display = "block";
      tooltip.style.left = `${Math.min(x + 12, rect.width - 280)}px`;
      tooltip.style.top = `${Math.max(8, y + 12)}px`;
    });

    canvas.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""
