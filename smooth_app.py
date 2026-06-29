from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from themepark.config import SimulationConfig
from themepark.engine import Simulation
from themepark.presets import (
    FIXED_CHOICE_NOISE,
    INFORMATION_FRESHNESS_OPTIONS,
    PRESETS,
    apply_preset,
    freshness_updates,
)
from themepark.serialization import (
    config_to_json,
    dataframe_to_csv_bytes,
    result_to_json,
)
from themepark.smooth_payload import build_state_payload


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STUDENT_UPDATE_KEYS = {
    "information_rate",
    "congestion_weight_mean",
    "preference_diversity",
}


class SmoothSimulationController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sim = Simulation(apply_preset(SimulationConfig(), "B"))
        self._running = False
        self._steps_per_second = 10.0
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def state(self) -> dict[str, Any]:
        with self._lock:
            payload = build_state_payload(self._sim)
            payload["running"] = self._running
            payload["stepsPerSecond"] = self._steps_per_second
            return payload

    def set_control(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if "running" in values:
                self._running = bool(values["running"]) and not self._sim.finished
            if "stepsPerSecond" in values:
                self._steps_per_second = _clamp_float(
                    values["stepsPerSecond"],
                    1.0,
                    120.0,
                )
            return self.state()

    def reset(self, values: dict[str, Any]) -> dict[str, Any]:
        preset_key = str(values.get("preset", "B"))
        updates = values.get("updates", {})
        if not isinstance(updates, dict):
            updates = {}
        with self._lock:
            config = apply_preset(SimulationConfig(), preset_key)
            safe_updates = {
                key: value
                for key, value in updates.items()
                if key in STUDENT_UPDATE_KEYS
            }
            if "information_freshness" in updates:
                safe_updates.update(freshness_updates(str(updates["information_freshness"])))
            safe_updates["choice_noise"] = FIXED_CHOICE_NOISE
            config = config.with_updates(**safe_updates)
            self._sim = Simulation(config)
            self._running = False
            return self.state()

    def step(self, count: int = 1) -> dict[str, Any]:
        with self._lock:
            self._running = False
            self._sim.step_many(max(1, min(count, 500)))
            return self.state()

    def download(self, name: str) -> tuple[bytes, str, str]:
        with self._lock:
            if name == "config.json":
                return config_to_json(self._sim.config).encode("utf-8"), "application/json", name
            if name == "result.json":
                return result_to_json(self._sim).encode("utf-8"), "application/json", name
            if name == "timeseries.csv":
                return dataframe_to_csv_bytes(self._sim.queue_history_dataframe()), "text/csv", name
            if name == "visitors.csv":
                return dataframe_to_csv_bytes(self._sim.visitor_dataframe()), "text/csv", name
            raise KeyError(name)

    def stop(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=2)

    def _run_loop(self) -> None:
        last = time.monotonic()
        accumulator = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            dt = now - last
            last = now
            with self._lock:
                if self._running and not self._sim.finished:
                    accumulator += dt * self._steps_per_second
                    steps = int(accumulator)
                    if steps > 0:
                        self._sim.step_many(min(steps, 40))
                        accumulator -= steps
                    if self._sim.finished:
                        self._running = False
                else:
                    accumulator = 0.0
            time.sleep(0.01)


def make_handler(controller: SmoothSimulationController):
    class SmoothRequestHandler(BaseHTTPRequestHandler):
        server_version = "ThemeParkSmooth/1.0"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._send_json(controller.state())
                elif path == "/api/presets":
                    self._send_json(
                        {
                            "presets": [
                                {
                                    "key": key,
                                    "label": preset.label,
                                    "observation": preset.observation,
                                    "updates": preset.updates,
                                }
                                for key, preset in PRESETS.items()
                            ],
                            "freshnessOptions": [
                                {
                                    "key": key,
                                    "label": str(option["label"]),
                                    "information_update_interval_steps": option[
                                        "information_update_interval_steps"
                                    ],
                                    "information_delay_steps": option["information_delay_steps"],
                                }
                                for key, option in INFORMATION_FRESHNESS_OPTIONS.items()
                            ],
                        }
                    )
                elif path.startswith("/download/"):
                    data, content_type, filename = controller.download(path.rsplit("/", 1)[-1])
                    self._send_bytes(data, content_type, filename)
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:  # noqa: BLE001
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                data = self._read_json()
                if path == "/api/control":
                    self._send_json(controller.set_control(data))
                elif path == "/api/reset":
                    self._send_json(controller.reset(data))
                elif path == "/api/step":
                    self._send_json(controller.step(int(data.get("count", 1))))
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:  # noqa: BLE001
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_bytes(
            self,
            data: bytes,
            content_type: str,
            filename: str | None = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if filename is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            data = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return SmoothRequestHandler


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    controller = SmoothSimulationController()
    server = ThreadingHTTPServer((host, port), make_handler(controller))
    print(f"Smooth simulation app: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        controller.stop()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smooth Canvas theme park simulation app")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    run(args.host, args.port)


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


INDEX_HTML = r"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>テーマパーク人工社会シミュレーション</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #111827;
      --muted: #64748b;
      --line: #dbe4ef;
      --soft: #f8fafc;
      --accent: #2563eb;
      --good: #15803d;
      --warn: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #ffffff;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 20px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }
    .question {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
    }
    .layout {
      display: grid;
      grid-template-columns: 280px minmax(520px, 1fr) 390px;
      min-height: calc(100vh - 66px);
    }
    aside {
      padding: 14px;
      border-right: 1px solid var(--line);
      background: var(--soft);
    }
    main {
      padding: 14px;
      min-width: 0;
    }
    .right {
      padding: 14px;
      border-left: 1px solid var(--line);
      min-width: 0;
    }
    label {
      display: block;
      margin: 11px 0 4px;
      color: #334155;
      font-size: 12px;
      font-weight: 600;
    }
    select, input, button {
      width: 100%;
      min-height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
    }
    input { padding: 0 8px; }
    button {
      cursor: pointer;
      font-weight: 700;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: white;
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
    }
    .downloads {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    .downloads a {
      display: grid;
      place-items: center;
      min-height: 32px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: white;
      color: #1e3a8a;
      font-size: 12px;
      font-weight: 700;
      text-decoration: none;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .metric {
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
    }
    .metric .label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
    }
    .metric .value {
      margin-top: 2px;
      font-size: 18px;
      font-weight: 800;
    }
    canvas {
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--soft);
    }
    #parkCanvas { height: calc(100vh - 158px); min-height: 520px; }
    #queueCanvas { height: 310px; }
    .status-line {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .section-title {
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 800;
    }
    .observation {
      min-height: 58px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: #334155;
      font-size: 13px;
      line-height: 1.45;
    }
    .tooltip {
      position: fixed;
      display: none;
      z-index: 10;
      max-width: 280px;
      padding: 8px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: rgba(255,255,255,.97);
      color: #0f172a;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 10px 24px rgba(15,23,42,.12);
      pointer-events: none;
      white-space: nowrap;
    }
    @media (max-width: 1180px) {
      .layout { grid-template-columns: 260px 1fr; }
      .right {
        grid-column: 1 / -1;
        border-left: 0;
        border-top: 1px solid var(--line);
      }
      #parkCanvas { height: 560px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>テーマパーク人工社会シミュレーション</h1>
      <div class="question">みんなが混雑を避けると、社会全体も必ず良くなるのか？</div>
    </div>
    <div id="connection">接続中</div>
  </header>
  <div class="layout">
    <aside>
      <div class="section-title">操作</div>
      <label for="preset">プリセット</label>
      <select id="preset"></select>
      <div class="button-row">
        <button id="play" class="primary">再生</button>
        <button id="pause">一時停止</button>
      </div>
      <div class="button-row">
        <button id="step">1ステップ</button>
        <button id="reset">反映してリセット</button>
      </div>
      <label for="speed">速度 <span id="speedLabel"></span></label>
      <input id="speed" type="range" min="1" max="80" step="1" value="10">

      <label for="information_rate">混雑情報所持率</label>
      <input id="information_rate" type="number" min="0" max="1" step="0.05">
      <label for="congestion_weight_mean">混雑回避度</label>
      <input id="congestion_weight_mean" type="number" min="0" max="3" step="0.05">
      <label for="preference_diversity">選好多様性</label>
      <input id="preference_diversity" type="number" min="0" max="1.5" step="0.05">
      <label for="information_freshness">情報の新しさ</label>
      <select id="information_freshness"></select>

      <div class="downloads">
        <a href="/download/config.json">設定JSON</a>
        <a href="/download/result.json">結果JSON</a>
        <a href="/download/timeseries.csv">時系列CSV</a>
        <a href="/download/visitors.csv">来場者CSV</a>
      </div>
    </aside>
    <main>
      <div class="metric-grid">
        <div class="metric"><div class="label">経過時刻</div><div class="value" id="time">-</div></div>
        <div class="metric"><div class="label">パーク内人数</div><div class="value" id="active">-</div></div>
        <div class="metric"><div class="label">平均待ち時間</div><div class="value" id="wait">-</div></div>
        <div class="metric"><div class="label">平均満足度</div><div class="value" id="satisfaction">-</div></div>
      </div>
      <canvas id="parkCanvas"></canvas>
      <div class="status-line">
        <span id="stepStatus">step -</span>
        <span id="visitorStatus">表示 -</span>
      </div>
    </main>
    <section class="right">
      <div class="section-title">施設別行列</div>
      <canvas id="queueCanvas"></canvas>
      <div class="metric-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 10px;">
        <div class="metric"><div class="label">行列の偏り</div><div class="value" id="imbalance">-</div></div>
        <div class="metric"><div class="label">一斉選択度</div><div class="value" id="sync">-</div></div>
      </div>
      <div class="section-title" style="margin-top: 12px;">観察ポイント</div>
      <div class="observation" id="observation"></div>
    </section>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const parkCanvas = document.getElementById("parkCanvas");
    const parkCtx = parkCanvas.getContext("2d");
    const queueCanvas = document.getElementById("queueCanvas");
    const queueCtx = queueCanvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const visitorVisuals = new Map();
    const hoverTargets = [];
    let appState = null;
    let presets = [];
    let freshnessOptions = [];
    let currentPreset = "B";
    let controlsDirty = false;
    let resetInFlight = false;
    let lastFetchAt = 0;
    const pollMs = 180;
    const colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#4d7c0f", "#7c3aed", "#0f766e"];

    function fitCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { width: rect.width, height: rect.height };
    }

    function toScreen(point, width, height) {
      const pad = 42;
      const plotW = width - pad * 2;
      const plotH = height - pad * 2;
      return {
        x: pad + (point.x / appState.map.width) * plotW,
        y: pad + (point.y / appState.map.height) * plotH,
      };
    }

    function fmt(value, digits = 2) {
      return value === null || value === undefined ? "-" : Number(value).toFixed(digits);
    }

    function formatDurationSeconds(seconds) {
      const total = Math.max(0, Math.ceil(Number(seconds)));
      const minutes = Math.floor(total / 60);
      const rest = String(total % 60).padStart(2, "0");
      return `${minutes}:${rest}`;
    }

    function nextStartLabel(attraction) {
      const seconds = attraction.nextStartRemainingSeconds;
      if (seconds === null || seconds === undefined) {
        return "";
      }
      if (attraction.nextStartStatus === "loading") {
        return `開始まで ${formatDurationSeconds(seconds)}`;
      }
      if (attraction.nextStartStatus === "ready") {
        return "開始待ち";
      }
      return `残り ${formatDurationSeconds(seconds)}`;
    }

    async function loadPresets() {
      const response = await fetch("/api/presets");
      const data = await response.json();
      presets = data.presets;
      freshnessOptions = data.freshnessOptions || [];
      const select = document.getElementById("preset");
      select.innerHTML = "";
      for (const preset of presets) {
        const option = document.createElement("option");
        option.value = preset.key;
        option.textContent = preset.label;
        select.appendChild(option);
      }
      select.value = currentPreset;
      const freshnessSelect = document.getElementById("information_freshness");
      freshnessSelect.innerHTML = "";
      for (const option of freshnessOptions) {
        const choice = document.createElement("option");
        choice.value = option.key;
        choice.textContent = option.label;
        freshnessSelect.appendChild(choice);
      }
      select.addEventListener("change", () => {
        currentPreset = select.value;
        const preset = presets.find((item) => item.key === currentPreset);
        document.getElementById("observation").textContent = preset ? preset.observation : "";
        fillPresetInputs(preset);
        markControlsDirty();
      });
      bindParameterControls();
    }

    async function fetchState() {
      try {
        const response = await fetch("/api/state");
        const state = await response.json();
        mergeState(state);
        document.getElementById("connection").textContent = state.running ? "再生中" : "一時停止";
      } catch (error) {
        document.getElementById("connection").textContent = "接続エラー";
      }
    }

    function mergeState(state, options = {}) {
      appState = state;
      lastFetchAt = performance.now();
      for (const visitor of state.visitors) {
        const existing = visitorVisuals.get(visitor.id);
        if (existing) {
          existing.fromX = existing.x;
          existing.fromY = existing.y;
          existing.toX = visitor.x;
          existing.toY = visitor.y;
          existing.color = visitor.color;
          existing.payload = visitor;
          existing.updatedAt = lastFetchAt;
        } else {
          visitorVisuals.set(visitor.id, {
            x: visitor.x,
            y: visitor.y,
            fromX: visitor.x,
            fromY: visitor.y,
            toX: visitor.x,
            toY: visitor.y,
            color: visitor.color,
            payload: visitor,
            updatedAt: lastFetchAt,
          });
        }
      }
      const liveIds = new Set(state.visitors.map((visitor) => visitor.id));
      for (const id of visitorVisuals.keys()) {
        if (!liveIds.has(id)) {
          visitorVisuals.delete(id);
        }
      }
      updateDom(state, options);
    }

    function updateDom(state, options = {}) {
      document.getElementById("time").textContent = state.formattedTime;
      document.getElementById("active").textContent = state.metrics.activeVisitors;
      document.getElementById("wait").textContent = state.metrics.meanWaitMinutes === null ? "-" : `${state.metrics.meanWaitMinutes}分`;
      document.getElementById("satisfaction").textContent = fmt(state.metrics.meanSatisfaction);
      document.getElementById("imbalance").textContent = fmt(state.metrics.queueImbalance);
      document.getElementById("sync").textContent = fmt(state.metrics.choiceSynchronization);
      document.getElementById("stepStatus").textContent = `step ${state.step}`;
      document.getElementById("visitorStatus").textContent = `表示 ${state.visitorDisplay.shown} / 園内 ${state.visitorDisplay.active} 人`;
      const preset = presets.find((item) => item.key === currentPreset);
      document.getElementById("observation").textContent = preset ? preset.observation : "";
      fillConfigInputs(state.config, state.informationFreshness, options);
    }

    function fillConfigInputs(config, freshnessKey, options = {}) {
      if ((controlsDirty || resetInFlight) && !options.forceConfigSync) {
        return;
      }
      for (const key of ["information_rate", "congestion_weight_mean", "preference_diversity"]) {
        const input = document.getElementById(key);
        if (document.activeElement !== input) {
          input.value = config[key];
        }
      }
      const freshnessSelect = document.getElementById("information_freshness");
      if (document.activeElement !== freshnessSelect) {
        freshnessSelect.value = freshnessKey || freshnessKeyForValues(config);
      }
    }

    function fillPresetInputs(preset) {
      if (!preset) return;
      const updates = preset.updates || {};
      for (const key of ["information_rate", "congestion_weight_mean", "preference_diversity"]) {
        if (updates[key] !== undefined) {
          document.getElementById(key).value = updates[key];
        }
      }
      document.getElementById("information_freshness").value = freshnessKeyForValues(updates);
    }

    function freshnessKeyForValues(values) {
      const interval = Number(values.information_update_interval_steps);
      const delay = Number(values.information_delay_steps);
      const option = freshnessOptions.find((item) => (
        item.information_update_interval_steps === interval &&
        item.information_delay_steps === delay
      ));
      return option ? option.key : "slightly_old";
    }

    function bindParameterControls() {
      for (const key of ["information_rate", "congestion_weight_mean", "preference_diversity", "information_freshness"]) {
        const input = document.getElementById(key);
        input.addEventListener("input", markControlsDirty);
        input.addEventListener("change", markControlsDirty);
      }
    }

    function markControlsDirty() {
      controlsDirty = true;
    }

    function draw() {
      if (appState) {
        drawPark();
        drawQueue();
      }
      requestAnimationFrame(draw);
    }

    function drawPark() {
      const { width, height } = fitCanvas(parkCanvas);
      hoverTargets.length = 0;
      parkCtx.clearRect(0, 0, width, height);
      drawMapBackground(width, height);
      drawPaths(width, height);
      drawAttractions(width, height);
      drawGate(width, height);
      drawVisitors(width, height);
      drawLegend(width, height);
    }

    function drawMapBackground(width, height) {
      const pad = 42;
      parkCtx.fillStyle = "#f8fafc";
      parkCtx.fillRect(0, 0, width, height);
      parkCtx.strokeStyle = "#dbe4ef";
      parkCtx.strokeRect(pad, pad, width - pad * 2, height - pad * 2);
      parkCtx.strokeStyle = "#e5e7eb";
      for (let i = 1; i < 5; i += 1) {
        const x = pad + (width - pad * 2) * i / 5;
        const y = pad + (height - pad * 2) * i / 5;
        parkCtx.beginPath();
        parkCtx.moveTo(x, pad);
        parkCtx.lineTo(x, height - pad);
        parkCtx.moveTo(pad, y);
        parkCtx.lineTo(width - pad, y);
        parkCtx.stroke();
      }
    }

    function drawPaths(width, height) {
      parkCtx.save();
      parkCtx.lineCap = "round";
      parkCtx.lineJoin = "round";
      parkCtx.lineWidth = 21;
      parkCtx.strokeStyle = "#d7c9aa";
      for (const edge of appState.paths) {
        const start = toScreen(edge.from, width, height);
        const end = toScreen(edge.to, width, height);
        parkCtx.beginPath();
        parkCtx.moveTo(start.x, start.y);
        parkCtx.lineTo(end.x, end.y);
        parkCtx.stroke();
      }
      parkCtx.lineWidth = 3;
      parkCtx.strokeStyle = "#b9a476";
      for (const edge of appState.paths) {
        const start = toScreen(edge.from, width, height);
        const end = toScreen(edge.to, width, height);
        parkCtx.beginPath();
        parkCtx.moveTo(start.x, start.y);
        parkCtx.lineTo(end.x, end.y);
        parkCtx.stroke();
      }
      parkCtx.restore();
    }

    function drawAttractions(width, height) {
      const maxQueue = Math.max(1, ...appState.attractions.map((item) => item.queue));
      parkCtx.textAlign = "center";
      for (const attraction of appState.attractions) {
        const point = toScreen(attraction, width, height);
        const radius = 8 + attraction.popularity * 6;
        const queueRatio = attraction.queue / maxQueue;
        const hue = 205 - queueRatio * 155;
        const nextStartText = nextStartLabel(attraction);
        parkCtx.beginPath();
        parkCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        parkCtx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        parkCtx.fill();
        parkCtx.lineWidth = 1.2;
        parkCtx.strokeStyle = "#0f172a";
        parkCtx.stroke();
        const labelLines = [
          {
            text: `${attraction.id}: ${attraction.name}`,
            font: "600 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color: "#0f172a",
          },
        ];
        if (nextStartText) {
          labelLines.push({
            text: nextStartText,
            font: "600 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color: "#334155",
          });
        }
        labelLines.push({
          text: `待ち ${attraction.queue}`,
          font: "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
          color: "#475569",
        });
        const labelGap = 13;
        const labelStartY = point.y - radius - 7 - (labelLines.length - 1) * labelGap;
        for (let index = 0; index < labelLines.length; index += 1) {
          const line = labelLines[index];
          parkCtx.font = line.font;
          parkCtx.fillStyle = line.color;
          parkCtx.fillText(line.text, point.x, labelStartY + index * labelGap);
        }
        hoverTargets.push({
          x: point.x,
          y: point.y,
          r: radius + 8,
          html: [
            `<strong>${attraction.name}</strong>`,
            `行列: ${attraction.queue}人`,
            `搭乗中: ${attraction.riders}人`,
            `次回開始: ${nextStartText || "-"}`,
            `実推定待ち: ${attraction.actualWaitMinutes ?? "-"}分`,
            `表示待ち: ${attraction.displayedWaitMinutes ?? "-"}分`,
          ].join("<br>")
        });
      }
      parkCtx.textAlign = "start";
    }

    function drawGate(width, height) {
      const point = toScreen(appState.gate, width, height);
      parkCtx.save();
      parkCtx.translate(point.x, point.y);
      parkCtx.rotate(Math.PI / 4);
      parkCtx.fillStyle = "#111827";
      parkCtx.fillRect(-8, -8, 16, 16);
      parkCtx.restore();
      parkCtx.fillStyle = "#111827";
      parkCtx.font = "600 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      parkCtx.textAlign = "center";
      parkCtx.fillText("ゲート", point.x, point.y - 15);
      parkCtx.textAlign = "start";
    }

    function drawVisitors(width, height) {
      const progress = Math.min(1, (performance.now() - lastFetchAt) / pollMs);
      const eased = progress * progress * (3 - 2 * progress);
      for (const visual of visitorVisuals.values()) {
        visual.x = visual.fromX + (visual.toX - visual.fromX) * eased;
        visual.y = visual.fromY + (visual.toY - visual.fromY) * eased;
        const point = toScreen({ x: visual.x, y: visual.y }, width, height);
        parkCtx.beginPath();
        parkCtx.arc(point.x, point.y, 3.2, 0, Math.PI * 2);
        parkCtx.globalAlpha = 0.78;
        parkCtx.fillStyle = visual.color;
        parkCtx.fill();
        parkCtx.globalAlpha = 1;
        const visitor = visual.payload;
        hoverTargets.push({
          x: point.x,
          y: point.y,
          r: 7,
          html: [
            `<strong>ID: ${visitor.id}</strong>`,
            `状態: ${visitor.stateLabel}`,
            `情報: ${visitor.hasInfo ? "あり" : "なし"}`,
            `体験数: ${visitor.rideCount}`,
            `満足度: ${fmt(visitor.satisfaction)}`,
            `目的地: ${visitor.target ?? "-"}`,
          ].join("<br>")
        });
      }
    }

    function drawLegend(width, height) {
      let x = 14;
      const y = height - 18;
      parkCtx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      for (const item of appState.stateLegend) {
        parkCtx.beginPath();
        parkCtx.arc(x + 5, y - 4, 5, 0, Math.PI * 2);
        parkCtx.fillStyle = item.color;
        parkCtx.fill();
        parkCtx.fillStyle = "#334155";
        parkCtx.fillText(item.label, x + 16, y);
        x += parkCtx.measureText(item.label).width + 44;
      }
    }

    function drawQueue() {
      const { width, height } = fitCanvas(queueCanvas);
      const pad = 34;
      queueCtx.clearRect(0, 0, width, height);
      queueCtx.fillStyle = "#f8fafc";
      queueCtx.fillRect(0, 0, width, height);
      const history = appState.queueHistory;
      if (!history || history.length < 2) return;
      const maxY = Math.max(1, ...history.flatMap((row) => row.queues));
      const minStep = history[0].step;
      const maxStep = history[history.length - 1].step;
      queueCtx.strokeStyle = "#dbe4ef";
      queueCtx.strokeRect(pad, 18, width - pad * 1.5, height - pad * 1.8);
      for (let attractionId = 0; attractionId < appState.attractions.length; attractionId += 1) {
        queueCtx.beginPath();
        for (let index = 0; index < history.length; index += 1) {
          const row = history[index];
          const x = pad + ((row.step - minStep) / Math.max(1, maxStep - minStep)) * (width - pad * 1.8);
          const y = 18 + (1 - row.queues[attractionId] / maxY) * (height - pad * 2);
          if (index === 0) queueCtx.moveTo(x, y);
          else queueCtx.lineTo(x, y);
        }
        queueCtx.strokeStyle = colors[attractionId % colors.length];
        queueCtx.lineWidth = attractionId < 3 ? 2 : 1.2;
        queueCtx.stroke();
      }
      queueCtx.fillStyle = "#64748b";
      queueCtx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      queueCtx.fillText(`最大 ${maxY}人`, pad, 14);
      queueCtx.fillText(`step ${minStep} - ${maxStep}`, pad, height - 10);
    }

    function collectUpdates() {
      const updates = {};
      const numberKeys = ["information_rate", "congestion_weight_mean", "preference_diversity"];
      for (const key of numberKeys) updates[key] = Number(document.getElementById(key).value);
      updates.information_freshness = document.getElementById("information_freshness").value;
      return updates;
    }

    async function postJson(path, payload, options = {}) {
      try {
        const response = await fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const state = await response.json();
        if (options.forceConfigSync) {
          controlsDirty = false;
        }
        mergeState(state, options);
      } finally {
        if (options.forceConfigSync) {
          resetInFlight = false;
        }
      }
    }

    document.getElementById("play").addEventListener("click", () => postJson("/api/control", { running: true }));
    document.getElementById("pause").addEventListener("click", () => postJson("/api/control", { running: false }));
    document.getElementById("step").addEventListener("click", () => postJson("/api/step", { count: 1 }));
    document.getElementById("reset").addEventListener("click", () => {
      resetInFlight = true;
      postJson(
        "/api/reset",
        { preset: currentPreset, updates: collectUpdates() },
        { forceConfigSync: true },
      );
    });
    document.getElementById("speed").addEventListener("input", (event) => {
      const value = Number(event.target.value);
      document.getElementById("speedLabel").textContent = `x${value}`;
      postJson("/api/control", { stepsPerSecond: value });
    });

    parkCanvas.addEventListener("mousemove", (event) => {
      const rect = parkCanvas.getBoundingClientRect();
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
      tooltip.style.left = `${Math.min(event.clientX + 12, window.innerWidth - 300)}px`;
      tooltip.style.top = `${Math.max(8, event.clientY + 12)}px`;
    });
    parkCanvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
    window.addEventListener("resize", () => {
      if (appState) {
        drawPark();
        drawQueue();
      }
    });

    async function boot() {
      await loadPresets();
      document.getElementById("speedLabel").textContent = `x${document.getElementById("speed").value}`;
      await fetchState();
      setInterval(fetchState, pollMs);
      requestAnimationFrame(draw);
    }
    boot();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
